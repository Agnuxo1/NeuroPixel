import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
now=datetime.now(timezone.utc).isoformat()
paths=['results/research/TASK4_review/recovery_inventory.json','results/research/TASK4_review/saved_evidence_audit.json','results/research/TASK4_review/control_recount.json','docs/research/TASK4_historical_capabilities_audit_20261009.md']
for p in paths[1:3]:
 j=json.loads((ROOT/p).read_bytes());assert j['status']=='passed' and not j['issues'] and j['model_loads']==0 and j['training_updates']==0
receipt=dict(status='closed_bounded_historical_claim_audit',closed_utc=now,artifacts_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},all_capabilities_demonstrated=False,external_replication=False,programme_complete=False)
(ROOT/'results/research/TASK4_review/closure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
p=ROOT/'docs/research/programme_20261008.json';j=json.loads(p.read_bytes())
for t in j['tasks']:
 if t['number']==4:t.update(status=receipt['status'],closed_utc=now,report=paths[-1],closure_receipt='results/research/TASK4_review/closure_receipt.json',historical_training_repeated=False,checks=3968,all_capabilities_demonstrated=False)
 if t['number']==5:t.update(status='in_progress',phase='prospective_scanner_causal_and_external_visual_protocol_preparation',started_utc=now)
j['updated_utc']=now;p.write_text(json.dumps(j,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
cp=ROOT/'coord/recovery/status-audit-20261007/checkpoint.md';entry=f'\n\n## Tarea4 auditoría cerrada — {now}\n763originalfiles/401580342bytes/15authenticatedcommits/providers terminalsuccess, allMerkle/blob/manifestsmatched;984savedarraychecks149NPZs +2984independentcontrolchecks0issues. BindingcomplexNP9.628906%vsTF49.355469% difference−39.726563pp IC[−41.254103,−38.199022]; bAbINP50.4%vsTF88.66%,−38.26ppIC[−52.385695,−24.134305]. MemoryNP38/48vsGRU48/48;46/48required,main/interferenceunexecuted. Ninepoolsunion1319/1320eligible(notactualexposure),8copyoldtensorisolation/aliasnegative,324repairrowsmatched,15Bfullrawstatesrecountedgrowth4.293–18.431/pairgain1.557–6.059singleexposedscene. NoTorchload/training/newpredictions/localGPU; historyqualified/report/closurepreserved. Task5nowprospectiveprotocolprep scanner+externalvisual; allcapabilities/Nobel/thirdpartyreplication/physicalenergy unestablished. Deadline13:49:31Z.\n';cp.write_bytes(cp.read_bytes().rstrip(b'\r\n')+entry.encode('utf-8'))
print(json.dumps(receipt))
