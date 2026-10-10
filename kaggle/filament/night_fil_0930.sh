#!/usr/bin/env bash
# Noche 30-09/01-10 (Fran: "lanza todas las pruebas"): tras FIL-011c, protocolo estricto de Codex (fold 0, train/calibracion/test externo),
# mismas condiciones (8k, lr 1e-3, semilla 0, --lowmem): base | consenso suave | consenso ponderado por acuerdo | objetos pequenos.
cd "$(dirname "$0")"
Q=D:/PROJECTS/.cognition/gpu_queue/gpuq.py
M="--split-manifest work/cv-protocol-20260930/fold-0.json"
C="--lowmem --batch 8 --crop 512 --steps 24 --pos-frac 0.4 --w-pos 3 --eval-every 2000 --vram-cap 12 --force-gpu --c 48 --hidden 128 --ms learned --iters 8000 --lr 1e-3 --seed 0"
run() { name=$1; shift; python $Q run --name "filament:$name" --vram 10 --ram 2 --cwd "$PWD" -- python train_fil.py $M $C "$@" --name $name > runs_$name.log 2>&1; grep -h RESULT runs_$name.log | cut -c1-400; }
run cv0_base
run cv0_cons --consensus
run cv0_consw --consensus --consensus-w 0.7
run cv0_small --small-frac 0.3
