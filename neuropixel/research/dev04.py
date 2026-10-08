"""Fresh-body development replication; no historical weights/test scoring."""
import hashlib
import json
import pathlib
import shutil
import time

import torch
from neuropixel.research.data import ResearchRoleTask, frozen_dataset
from neuropixel.research.models import ResearchNCA
from neuropixel.research import qtrain_budget as budget
from neuropixel.research.qtrain import evaluate_queries
from neuropixel.research.readout_experiment import save_json, save_state, sha

REGISTRATION='NP-DEV04-20261008'
EVAL_MASKS=list(range(95000,95008))
TRAIN_MASKS=list(range(94000,94008))


def inventory():
    return [dict(run_id=f'DEV04_partition{split}_seed{400+4*j+k}',partition_seed=split,
                 init_seed=400+4*j+k,policy='paired',school_weight=.3,updates=16384)
            for j,split in enumerate((101,102,103)) for k in range(4)]


def triples_digest(triples):
    raw=(json.dumps([list(t) for t in sorted(triples)],separators=(',',':'))+'\n').encode()
    return hashlib.sha256(raw).hexdigest()


class DevelopmentRoleTask(ResearchRoleTask):
    """Repartition only old development; original test remains fixed/inaccessible."""
    def __init__(self,partition_seed):
        if partition_seed not in (101,102,103):raise ValueError('Registered development partition required')
        super().__init__(seed=0)
        pool=sorted(self.train_triples+self.validation_triples)
        if len(pool)!=1056 or len(set(pool))!=1056:raise ValueError('Original development pool differs')
        generator=torch.Generator(device='cpu').manual_seed(partition_seed)
        order=torch.randperm(len(pool),generator=generator).tolist()
        self.validation_triples=tuple(pool[i] for i in order[:132])
        self.train_triples=tuple(pool[i] for i in order[132:])
        self.split_seed=partition_seed
        self._triples_cpu.update(train=torch.tensor(self.train_triples,dtype=torch.long),
                                 validation=torch.tensor(self.validation_triples,dtype=torch.long))
        self.check_partitions()

    def check_partitions(self):
        train,val,test=map(set,(self.train_triples,self.validation_triples,self.test_triples))
        if (len(train),len(val),len(test))!=(924,132,264) or train&val or train&test or val&test:
            raise ValueError('Development/test separation differs')

    def sample(self,batch,split='train',*args,**kwargs):
        if split=='test':raise ValueError('Original test scoring/sampling prohibited in DEV04')
        return super().sample(batch,split,*args,**kwargs)

    def partition_record(self):
        return dict(partition_seed=self.split_seed,
                    counts={k:len(getattr(self,k+'_triples')) for k in ('train','validation','test')},
                    composition_sha256={k:triples_digest(getattr(self,k+'_triples')) for k in ('train','validation','test')},
                    test_original_seed=0,test_sampled=False)


def panels(task,n_probe=2048,n_validation=4096):
    offset=task.split_seed-101
    return dict(probe=frozen_dataset(task,'train',n_probe,96001+offset),
                validation=frozen_dataset(task,'validation',n_validation,97001+offset))


def new_objects(config):
    """Only constructor initialization; no input checkpoint is accepted."""
    budget.require_ram()
    torch.manual_seed(config['init_seed'])
    model=ResearchNCA(35,(7,7),c_id=16,c=48,hidden=128,steps=config.get('steps',16),
                      fire_rate=.5,tied=True,reinject=True,freeze_pad=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
    sampler=torch.Generator().manual_seed(93002)
    query_rng=torch.Generator().manual_seed(93003)
    return model,optimizer,sampler,query_rng


def check_resume(payload,config,plan_hash,birth,environment):
    if (payload.get('registration_id')!=REGISTRATION or payload.get('config')!=config
        or payload.get('plan_sha256')!=plan_hash or payload.get('birth_model_digest')!=birth
        or payload.get('new_initialization') is not True or payload.get('historical_checkpoint_loaded') is not False):
        raise ValueError('Foreign/historical checkpoint or fresh-birth identity differs')
    update=payload['update']
    if not 0<=update<=config['updates'] or update%128:raise ValueError('Invalid durable new-body update')
    budget.verify_environment(payload['environment'],environment)
    if update:budget.validate_state(payload,update,config,plan_hash)
    elif payload['training_state_digest']!=budget.state_digest({k:payload[k] for k in budget.STATE_FIELDS}):
        raise ValueError('Initial state digest differs')


def fit_body(config,task,datasets,output,plan_hash,environment,endpoints=(8192,16384),stop_after=None,archive_callback=None,expected_birth=None):
    """Durable fresh fitting; skips closed bodies before admission/initialization."""
    output=pathlib.Path(output);final=output/'result.json'
    if final.exists():
        result=json.loads(final.read_text(encoding='utf-8'))
        if (result['registration_id']!=REGISTRATION or result['config']!=config or result['plan_sha256']!=plan_hash
            or result['last_update']!=config['updates'] or sha(output/'checkpoint.pt')!=result['checkpoint_sha256']):raise ValueError('Closed fresh-body identity differs')
        return result
    budget.require_ram();task.check_partitions()
    if task.split_seed!=config['partition_seed']:raise ValueError('Current task partition differs from registered body configuration')
    model,opt,sampler,query_rng=new_objects(config)
    birth=budget.state_digest(model.state_dict());checkpoint=output/'checkpoint.pt'
    if expected_birth is not None and birth!=expected_birth:raise ValueError('Fresh model initialization differs from prospective admission hash')
    curve,evaluations,resources=[],[],[];elapsed=0.;last=0
    if checkpoint.exists():
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        check_resume(payload,config,plan_hash,birth,environment)
        if payload['partition']!=task.partition_record():raise ValueError('Resume partition membership differs')
        budget.restore(model,opt,sampler,query_rng,payload,payload['training_state_digest'])
        last=payload['update'];curve=payload['curve'];evaluations=payload['evaluations'];resources=payload['resources'];elapsed=payload['elapsed_update_seconds']
        if [w['update'] for w in curve]!=list(range(128,last+1,128)):raise ValueError('Missing/repeated durable prefix')
    def persist(update):
        state=budget.training_state(model,opt,sampler,query_rng)
        payload=dict(**state,registration_id=REGISTRATION,config=config,plan_sha256=plan_hash,
            birth_model_digest=birth,new_initialization=True,historical_checkpoint_loaded=False,
            environment=environment,update=update,training_state_digest=budget.state_digest(state),
            curve=curve,evaluations=evaluations,resources=resources,elapsed_update_seconds=elapsed,
            partition=task.partition_record())
        save_state(checkpoint,payload)
        save_json(output/'progress.json',dict(registration_id=REGISTRATION,config=config,plan_sha256=plan_hash,
            update=update,birth_model_digest=birth,new_initialization=True,historical_checkpoint_loaded=False,
            checkpoint_sha256=sha(checkpoint),test_accessed=False))
    def endpoint(update):
        item=dict(update=update,panels={},test_accessed=False)
        for panel,data in datasets.items():
            item['panels'][panel]=evaluate_queries(model,data,output/'decisions',f'u{update}_{panel}',EVAL_MASKS,True)
        evaluations.append(item);persist(update);snapshot=output/f'checkpoint_u{update}.pt'
        if snapshot.exists() and snapshot.read_bytes()!=checkpoint.read_bytes():raise ValueError('Different preserved body endpoint')
        if not snapshot.exists():shutil.copyfile(checkpoint,snapshot)
    if not checkpoint.exists():persist(0)
    if last in endpoints and last not in [e['update'] for e in evaluations]:endpoint(last)
    model.train();window=[];witness=hashlib.sha256();role_counts=torch.zeros(4,dtype=torch.long)
    for update in range(last+1,config['updates']+1):
        if update%128==1:resources.append(dict(update=update,available_ram_gib=budget.require_ram()))
        start=time.monotonic();value=budget.update_once(model,opt,task,sampler,query_rng,config['policy'],config['school_weight']);elapsed+=time.monotonic()-start
        window.append(value);role_counts+=torch.bincount(value['roles'],minlength=4)
        for name in ('canvas','target','roles'):witness.update(value[name].numpy().tobytes())
        if update%128==0:
            curve.append(dict(update=update,answer_loss_mean=sum(v['answer_loss'] for v in window)/128,
                school_loss_mean=sum(v['school_loss'] for v in window)/128,total_loss_mean=sum(v['total_loss'] for v in window)/128,
                last_unclipped_gradient_norm=value['gradient_norm'],role_counts=role_counts.tolist(),
                batch_witness_sha256=witness.hexdigest(),elapsed_update_seconds=elapsed))
            window=[];role_counts.zero_();witness=hashlib.sha256();persist(update)
            if update in endpoints:endpoint(update)
            if archive_callback and update%1024==0 and update<config['updates']:
                before=budget.state_digest(budget.training_state(model,opt,sampler,query_rng));archive_callback()
                if budget.state_digest(budget.training_state(model,opt,sampler,query_rng))!=before:raise ValueError('Archival changed actual training state')
            print(json.dumps(dict(event='dev04_body_progress',case=config['run_id'],update=update)),flush=True)
            if stop_after and update>=stop_after:return None
    result=dict(registration_id=REGISTRATION,status='completed',config=config,plan_sha256=plan_hash,
        last_update=config['updates'],new_initialization=True,historical_checkpoint_loaded=False,
        birth_model_digest=birth,parameter_count=sum(p.numel() for p in model.parameters()),partition=task.partition_record(),
        checkpoint_sha256=sha(checkpoint),environment=environment,curve=curve,evaluations=evaluations,resources=resources,test_accessed=False)
    save_json(final,result)
    if archive_callback:archive_callback()
    return result
