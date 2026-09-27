"""Descarga CIFAR-10 (~170 MB, Universidad de Toronto) y crea data/cifar10.npz para la cartilla.

    python scripts/prepare_cifar.py
"""
from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def main():
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    arc = data / "cifar-10-python.tar.gz"
    if not arc.exists():
        print("descargando", URL)
        urllib.request.urlretrieve(URL, arc)
    with tarfile.open(arc) as tf:
        tf.extractall(data)
    src = data / "cifar-10-batches-py"

    def lb(p):
        d = pickle.load(open(p, "rb"), encoding="latin1")
        return d["data"].reshape(-1, 3, 32, 32).astype(np.uint8), np.array(d["labels"], np.int64)
    xs, ys = zip(*[lb(src / f"data_batch_{i}") for i in range(1, 6)])
    xt, yt = lb(src / "test_batch")
    np.savez(data / "cifar10.npz", x_train=np.concatenate(xs), y_train=np.concatenate(ys), x_test=xt, y_test=yt)
    print("creado", data / "cifar10.npz")


if __name__ == "__main__":
    main()
