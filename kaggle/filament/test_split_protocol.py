"""Comprobacion ligera de particiones reales y rechazo de manifiestos corruptos."""
import ast
import json
from copy import deepcopy
from pathlib import Path
import split_protocol as s

def main():
    here = Path(__file__).parent
    meta = json.loads((here/'cache/meta.json').read_text(encoding='utf-8'))
    ast.parse((here/'train_fil.py').read_text(encoding='utf-8'))
    covered = set()
    for fold in range(5):
        manifest = s.build(meta, fold)
        assert not covered.intersection(manifest['files']['test'])
        covered.update(manifest['files']['test'])
        for corruption in ('hash','overlap','index'):
            altered = deepcopy(manifest)
            if corruption == 'hash':
                altered['metadata_sha256'] = 'invalid'
            elif corruption == 'overlap':
                altered['files']['train'].append(altered['files']['test'][0])
            else:
                altered['indices']['train'].append(altered['indices']['test'][0])
            try:
                s.validate(meta, altered)
            except ValueError:
                pass
            else:
                raise AssertionError(corruption)
    assert covered == {a['file'] for a in meta['ann']}
    inter_covered = set()
    for fold in range(5):
        x = s.build_interleaved(meta,fold)
        assert x == s.build_interleaved(meta,fold)
        assert not inter_covered.intersection(x['files']['test'])
        inter_covered.update(x['files']['test'])
        s.validate(meta,x)
    assert inter_covered == covered
    print('PASS: five folds, disjoint/exhaustive heldout images, temporal gaps, annotation grouping, 15 corrupt manifests rejected, training syntax')

if __name__ == '__main__':
    main()
