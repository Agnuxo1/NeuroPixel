"""FIL-015: vecinos por fecha, sin leer pixeles ni etiquetas de segmentacion."""
from __future__ import annotations
import argparse
import bisect
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
import split_protocol

def timestamp(file):
    return datetime.strptime(file[:14], '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc).timestamp()/86400

def pair_targets(targets, sources, max_days=2):
    ordered = sorted((timestamp(f),f) for f in set(sources))
    dates = [t for t,_ in ordered]
    pairs = []
    for f in sorted(set(targets)):
        t = timestamp(f)
        left = bisect.bisect_left(dates,t)-1
        right = bisect.bisect_right(dates,t)
        prev = ordered[left] if left >= 0 else None
        nxt = ordered[right] if right < len(ordered) else None
        # Exact-time images are not used as temporal context; neither is the target itself.
        row = {'target':f,'previous':None,'next':None,'nearest_days':None}
        neighbors = [n for n in (prev,nxt) if n]
        if neighbors:
            row['nearest_days'] = min(abs(t-n[0]) for n in neighbors)
        for name,n in (('previous',prev),('next',nxt)):
            if n and abs(t-n[0]) <= max_days:
                row[name] = {'file':n[1], 'delta_days':n[0]-t}
        pairs.append(row)
    return pairs

def summary(rows):
    distances = [r['nearest_days'] for r in rows if r['nearest_days'] is not None]
    n = len(rows)
    return {'images':n,'median_nearest_days': statistics.median(distances) if distances else None,
            'within_1_day_fraction':sum(d<=1 for d in distances)/n if n else 0,
            'within_2_days_fraction':sum(d<=2 for d in distances)/n if n else 0,
            'usable_context_fraction':sum(bool(r['previous'] or r['next']) for r in rows)/n if n else 0,
            'both_sides_fraction':sum(bool(r['previous'] and r['next']) for r in rows)/n if n else 0}

def main():
    root = Path(__file__).parent
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-days',type=float,default=2)
    a = p.parse_args()
    if a.max_days <= 0:
        p.error('--max-days debe ser positivo')
    meta = json.loads((root/'cache/meta.json').read_text(encoding='utf-8'))
    sources = sorted({ann['file'] for ann in meta['ann']})
    test_dir = root/'data/MAGFiLO_1.0_Kaggle_2026/test/test_images'
    targets = sorted(f.name for f in test_dir.iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png'))
    if not targets:
        raise ValueError('sin imagenes test locales')
    a.output.mkdir(parents=True,exist_ok=True)
    report = {'metadata_sha256':split_protocol.fingerprint(meta),'max_days':a.max_days,
              'context_policy':'image-only train neighbors; no masks; target excluded; exact timestamp excluded',
              'competition_test':summary(pair_targets(targets,sources,a.max_days)),'blocks':{},'interleaved':{}}
    (a.output/'competition-test-neighbors.json').write_text(json.dumps(pair_targets(targets,sources,a.max_days),indent=2),encoding='utf-8')
    for fold in range(5):
        manifest = split_protocol.build(meta,fold)
        groups = {}
        payload = {}
        for group in ('train','calibration','test'):
            rows = pair_targets(manifest['files'][group],manifest['files']['train'],a.max_days)
            allowed = set(manifest['files']['train'])
            for row in rows:
                for side in ('previous','next'):
                    if row[side]:
                        assert row[side]['file'] in allowed and row[side]['file'] != row['target']
            payload[group] = rows
            groups[group] = summary(rows)
        report['blocks'][str(fold)] = groups
        (a.output/f'fold-{fold}-neighbors.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
        inter = split_protocol.build_interleaved(meta,fold)
        split_protocol.validate(meta,inter)
        (a.output/f'interleaved-fold-{fold}.json').write_text(json.dumps(inter,indent=2),encoding='utf-8')
        ipayload = {g:pair_targets(inter['files'][g],inter['files']['train'],a.max_days)
                    for g in ('train','calibration','test')}
        report['interleaved'][str(fold)] = {g:summary(rows) for g,rows in ipayload.items()}
        (a.output/f'interleaved-fold-{fold}-neighbors.json').write_text(json.dumps(ipayload,indent=2),encoding='utf-8')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
