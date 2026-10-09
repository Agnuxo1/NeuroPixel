"""Validate a full recursive Git tree snapshot against its actual Merkle hashes."""
import hashlib
import json
import pathlib


def load_snapshot(path,repository='Agnuxo1/NeuroPixel'):
    snapshot=json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    tree=snapshot['tree'];commit=snapshot['commit']
    if snapshot['repository']!=repository or tree.get('truncated') or tree['sha']!=commit['tree']['sha']:
        raise ValueError('Incomplete/mismatched snapshot commit-tree binding')
    entries={};children={}
    for row in tree['tree']:
        name=row['path'];parts=pathlib.PurePosixPath(name)
        if parts.is_absolute() or '..' in parts.parts or '\\' in name or name in entries or str(parts)!=name:
            raise ValueError('Duplicate/unsafe Git tree path')
        if row['type'] not in ('blob','tree','commit') or len(row['sha'])!=40:
            raise ValueError('Unsupported Git object metadata')
        entries[name]=row
        parent=str(parts.parent);parent='' if parent=='.' else parent
        children.setdefault(parent,[]).append(row)
    for parent in children:
        if parent and (parent not in entries or entries[parent]['type']!='tree'):
            raise ValueError('Git tree parent missing')
    directories={''}|{p for p,row in entries.items() if row['type']=='tree'}
    for directory in directories:
        rows=children.get(directory,[])
        rows=sorted(rows,key=lambda row:pathlib.PurePosixPath(row['path']).name.encode('utf-8')+(b'/' if row['type']=='tree' else b''))
        body=b''
        for row in rows:
            mode=row['mode'].lstrip('0')
            body+=mode.encode('ascii')+b' '+pathlib.PurePosixPath(row['path']).name.encode('utf-8')+b'\0'+bytes.fromhex(row['sha'])
        digest=hashlib.sha1(b'tree '+str(len(body)).encode()+b'\0'+body).hexdigest()
        expected=tree['sha'] if directory=='' else entries[directory]['sha']
        if digest!=expected:raise ValueError('Recursive Git Merkle tree differs: '+directory)
    return snapshot
