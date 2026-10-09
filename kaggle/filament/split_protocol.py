"""Manifiestos temporales anidados; no importa torch ni carga imagenes."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

def fingerprint(meta):
    return hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def chunks(items, k):
    q, r = divmod(len(items), k)
    start = 0
    for i in range(k):
        end = start + q + (i < r)
        yield items[start:end]
        start = end

def times(meta):
    return {f: datetime.strptime(f[:14], '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc).timestamp()/86400
            for f in {a['file'] for a in meta['ann']}}

def build(meta, fold, k=5, inner_fold=0, gap_days=5):
    if not 0 <= fold < k or not 0 <= inner_fold < k or gap_days < 0:
        raise ValueError('fold/inner_fold/gap invalidos')
    ts = times(meta)
    files = sorted(ts, key=lambda f: (ts[f], f))
    test = set(list(chunks(files, k))[fold])
    if not test:
        raise ValueError('bloque externo vacio')
    eligible = [f for f in files if f not in test and min(abs(ts[f]-ts[t]) for t in test) >= gap_days]
    cal = set(list(chunks(eligible, k))[inner_fold])
    if not cal:
        raise ValueError('calibracion vacia')
    train = {f for f in eligible if f not in cal and min(abs(ts[f]-ts[c]) for c in cal) >= gap_days}
    groups = {'train': train, 'calibration': cal, 'test': test}
    used = train | cal | test
    groups['excluded'] = set(files)-used
    out = {'version': 1, 'metadata_sha256': fingerprint(meta), 'fold': fold, 'k': k,
           'inner_fold': inner_fold, 'gap_days': gap_days,
           'selection_policy': 'checkpoint and postprocess on calibration only; test once after selection',
           'files': {g: sorted(fs) for g, fs in groups.items()},
           'indices': {g: [i for i,a in enumerate(meta['ann']) if a['file'] in fs] for g,fs in groups.items()}}
    validate(meta, out)
    return out

def validate(meta, manifest):
    if manifest['version'] != 1 or manifest['metadata_sha256'] != fingerprint(meta):
        raise ValueError('metadatos distintos del manifiesto')
    ts = times(meta)
    groups = {g: set(manifest['files'][g]) for g in ('train','calibration','test','excluded')}
    if any(not groups[g] for g in ('train','calibration','test')):
        raise ValueError('particion vacia')
    seen = set()
    img_sets = {}
    for g, fs in groups.items():
        if seen & fs:
            raise ValueError('solape de archivos')
        seen |= fs
        expected = [i for i,a in enumerate(meta['ann']) if a['file'] in fs]
        if expected != manifest['indices'][g]:
            raise ValueError('indices incorrectos o anotadores separados')
        img_sets[g] = {meta['ann'][i]['img'] for i in expected}
    if seen != set(ts):
        raise ValueError('cobertura incompleta')
    for a,b in (('train','calibration'),('train','test'),('calibration','test')):
        if img_sets[a] & img_sets[b]:
            raise ValueError('solape de imagenes')
        if min(abs(ts[x]-ts[y]) for x in groups[a] for y in groups[b]) < manifest['gap_days']:
            raise ValueError('purga temporal insuficiente')
    return {g: {'images':len(groups[g]), 'annotations':len(manifest['indices'][g])} for g in groups}

def build_interleaved(meta, fold, k=5, seed=0, inner_fold=0):
    """Etiquetas separadas, sin purga temporal; mide contexto disponible en test intercalado."""
    if not 0 <= fold < k or not 0 <= inner_fold < k:
        raise ValueError('fold invalido')
    files = sorted(times(meta))
    random.Random(seed).shuffle(files)
    test = set(list(chunks(files,k))[fold])
    remaining = [f for f in files if f not in test]
    cal = set(list(chunks(remaining,k))[inner_fold])
    train = set(remaining)-cal
    groups = {'train':train,'calibration':cal,'test':test,'excluded':set()}
    out = {'version':1,'metadata_sha256':fingerprint(meta),'fold':fold,'k':k,'seed':seed,
           'inner_fold':inner_fold,'gap_days':0,'protocol':'interleaved_grouped',
           'selection_policy':'checkpoint and postprocess on calibration only; test once after selection',
           'files':{g:sorted(fs) for g,fs in groups.items()},
           'indices':{g:[i for i,a in enumerate(meta['ann']) if a['file'] in fs] for g,fs in groups.items()}}
    validate(meta,out)
    return out

def load(meta, path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    validate(meta, manifest)
    return manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--meta', type=Path, default=Path(__file__).parent/'cache/meta.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('blocks','interleaved'), default='blocks')
    a = p.parse_args()
    meta = json.loads(a.meta.read_text(encoding='utf-8'))
    a.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for fold in range(5):
        m = build(meta, fold) if a.mode == 'blocks' else build_interleaved(meta,fold)
        (a.output/f'fold-{fold}.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
        summary[str(fold)] = validate(meta, m)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))
