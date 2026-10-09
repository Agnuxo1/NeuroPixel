"""Official UCI pixel data, known-range scaling and fixed internal DEV groups."""
import hashlib,io,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
URL='https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip'
ZIP_SHA='0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0'
TRAIN_SHA='250ca30ac14cffbb3830e81761326916a2f7d215209a0bf103a2d2579921d4f4'
TEST_SHA='544512cfdfe87c0f7be1db8304c334a051e507a7e803c73f6a706b7e34d9cc9e'
def original(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=path.read_bytes() if path.exists() else urllib.request.urlopen(URL,timeout=60).read(2*1024**2)
    if hashlib.sha256(raw).hexdigest()!=ZIP_SHA:raise ValueError('Official original ZIP differs')
    if not path.exists():path.write_bytes(raw)
    return raw
def parse(raw,name,expected):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:a=np.loadtxt(io.BytesIO(z.read(name)),delimiter=',',dtype=np.int16)
    if hashlib.sha256(a.astype('<i2').tobytes()).hexdigest()!=expected:raise ValueError('Canonical official pixels/labels differ')
    if a.shape[1]!=65 or not np.isin(a[:,-1],np.arange(10)).all() or (a[:,:64]<0).any() or (a[:,:64]>16).any():raise ValueError('Data contract')
    return a
def identity(rows):return [hashlib.sha256(row[:64].astype('<i2').tobytes()).hexdigest() for row in rows]
def development(raw):
    a=parse(raw,'optdigits.tra',TRAIN_SHA);rng=np.random.default_rng(888000);val=[];train=[]
    if len(set(identity(a)))!=len(a):raise ValueError('Unexpected duplicate image groups before selection')
    for label in range(10):
        ids=np.flatnonzero(a[:,-1]==label);order=rng.permutation(ids);val.extend(order[:30].tolist());train.extend(order[30:].tolist())
    return a,np.array(sorted(train),dtype=np.int64),np.array(sorted(val),dtype=np.int64)
def pixels(rows):
    gray=rows[:,:64].astype(np.float32).reshape(-1,1,8,8)/np.float32(16)
    return np.repeat(gray,3,axis=1),rows[:,-1].astype(np.int64)+1
def final(raw,gate):
    if gate.get('status')!='all_ten_selected_models_sealed_before_official_test' or len(gate.get('selected_models',[]))!=10:raise ValueError('All ten immutable selections required before test parsing')
    return parse(raw,'optdigits.tes',TEST_SHA)
