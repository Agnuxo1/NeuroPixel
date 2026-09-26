import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuropixel import codec, hrr  # noqa: E402
from neuropixel.model import NeuroPixel, TinyTransformer  # noqa: E402
from neuropixel.safety import GpuStatus, choose_device  # noqa: E402
from neuropixel.task import RoleTask  # noqa: E402


def test_codec_roundtrip_png():
    ids = np.array([0, 1, 255, 256, 65535, 2**24 - 1, 199_997])
    assert (codec.rgb_to_ids(codec.ids_to_rgb(ids)) == ids).all()
    img = codec.pack_image(ids)
    assert (codec.from_png(codec.to_png(img), ids.size) == ids).all()
    assert tuple(codec.ids_to_rgb([0])[0]) == (0, 0, 0)  # vacío = negro


def test_hrr_bind_unbind_and_order():
    g = torch.Generator().manual_seed(0)
    mem = hrr.random_vectors(50, 1024, g)
    agent, patient, dog, man = mem[0], mem[1], mem[2], mem[3]
    s1 = hrr.bind(agent, dog) + hrr.bind(patient, man)   # el perro ... al hombre
    s2 = hrr.bind(agent, man) + hrr.bind(patient, dog)   # el hombre ... al perro
    assert hrr.cleanup(hrr.unbind(s1, agent), mem).item() == 2
    assert hrr.cleanup(hrr.unbind(s2, agent), mem).item() == 3
    assert not torch.allclose(s1, s2)


def test_task_heldout_disjoint_and_valid():
    t = RoleTask(8, 8, seed=0)
    assert not set(t.train_triples) & set(t.test_triples)
    canvas, target = t.sample(32, "test", torch.Generator().manual_seed(1))
    assert canvas.shape == (32, 8, 8)
    assert ((canvas != 0).sum((1, 2)) == 9).all()        # 4 parejas + consulta
    qr, qc = t.query_pos
    for b in range(32):
        role = canvas[b, qr, qc]
        r, c = (canvas[b, :-1] == role).nonzero()[0].tolist()
        assert canvas[b, r, c + 1] == target[b]           # la respuesta está a la derecha del papel


def test_models_forward_backward():
    t = RoleTask(8, 8)
    canvas, target = t.sample(4)
    for m in (NeuroPixel(len(t.v), t.out_pos, steps=4), TinyTransformer(len(t.v), 8, 8, t.out_pos)):
        out = m(canvas, trace=isinstance(m, NeuroPixel))
        assert out["logits"].shape == (4, len(t.v))
        torch.nn.functional.cross_entropy(out["logits"], target).backward()
    assert out is not None


def test_empty_canvas_is_black():
    t = RoleTask(8, 8)
    m = NeuroPixel(len(t.v), t.out_pos, steps=3)
    out = m(torch.zeros(2, 8, 8, dtype=torch.long), trace=True)
    assert out["frames"][:, 0].abs().max() == 0          # sin datos, estado inicial a cero


def test_safety_refuses_busy_gpu(monkeypatch):
    import neuropixel.safety as s
    monkeypatch.setattr(s, "gpu_status", lambda: GpuStatus(98, 16649, 24576))
    monkeypatch.setattr(s.torch.cuda, "is_available", lambda: True)
    assert choose_device("cuda", threads=2).type == "cpu"
    assert choose_device("cpu", threads=2).type == "cpu"


def test_scanner_lens_shapes():
    from neuropixel.scanner import lens
    t = RoleTask(8, 8)
    m = NeuroPixel(len(t.v), t.out_pos, steps=3).eval()
    canvas, _ = t.sample(1)
    fr = m(canvas, trace=True)["frames"][0]
    word, conf = lens(m, fr)
    assert word.shape == (4, 8, 8) and (word != 0).all() and (conf <= 1).all()


def test_grounded_dictionary_fixed_colors():
    t = RoleTask(8, 8)
    rgb, mask = t.v.grounded()
    m = NeuroPixel(len(t.v), t.out_pos, steps=2, grounded=(rgb, mask))
    d = m.dictionary()
    i = t.v.idx["río"]
    assert torch.allclose(d[i, :3], rgb[i])                 # el río es azul y no se aprende
    canvas, target = t.sample(4)
    torch.nn.functional.cross_entropy(m(canvas)["logits"], target).backward()
    assert m.embed.weight.grad is not None
    assert d[0].abs().max() == 0                             # vacío sigue siendo negro


def test_vectorized_sampler_matches_structure():
    t = RoleTask(8, 8, seed=0)
    c, y = t.sample(256, "test", torch.Generator().manual_seed(3))
    names = t.v.tokens
    from neuropixel.task import NOUNS
    test_set = set(t.test_triples)
    for b in range(256):
        found = {}
        for r in range(7):
            for col in range(7):
                tok = names[int(c[b, r, col])]
                if tok in ("AGENTE", "ACCION", "PACIENTE"):
                    found[tok] = names[int(c[b, r, col + 1])]
        tri = (NOUNS.index(found["AGENTE"]), t.v.tokens.index(found["ACCION"]) - t.v.idx["muerde"],
               NOUNS.index(found["PACIENTE"]))
        assert tri in test_set                              # solo combinaciones de test
    assert ((c != 0).sum((1, 2)) == 9).all()


def test_both_models_accept_train_kwargs():
    t = RoleTask(8, 8)
    c, _ = t.sample(2)
    for m in (NeuroPixel(len(t.v), t.out_pos, steps=2), TinyTransformer(len(t.v), 8, 8, t.out_pos)):
        m(c, lens_every=0)


def test_camera_pixels_replace_words():
    t = RoleTask(8, 8)
    c, y, rgb, cam = t.sample_camera(64, "test", torch.Generator().manual_seed(0), cam_roles={3: 1.0},
                                     query_role=3, place_pool=[t.v.idx["playa"]])
    assert cam.sum() == 64 and (c[cam] == 0).all()           # la playa ya no es palabra
    assert (y == t.v.idx["playa"]).all()
    m = NeuroPixel(len(t.v), t.out_pos, steps=2, grounded=t.v.grounded())
    out = m(c, rgb=rgb, cam=cam, trace=True)
    assert out["frames"][:, 0][cam.unsqueeze(1).expand(-1, out["frames"].shape[2], -1, -1)].abs().sum() > 0
