#!/usr/bin/env bash
# Noche 01->02 oct (RES-001): protocolo estricto de Codex, fold 0, 8k, lr 1e-3, semilla 0, --lowmem (mismas condiciones que cv0_*).
# P4 EMA | P5 control U-Net (w=11, ~100k params) | P6 Skeleton Recall | P7 perdida auxiliar en pasos intermedios.
cd "$(dirname "$0")"
Q=D:/PROJECTS/.cognition/gpu_queue/gpuq.py
M="--split-manifest work/cv-protocol-20260930/fold-0.json"
C="--lowmem --batch 8 --crop 512 --steps 24 --pos-frac 0.4 --w-pos 3 --eval-every 2000 --vram-cap 12 --force-gpu --c 48 --hidden 128 --ms learned --iters 8000 --lr 1e-3 --seed 0"
run() { name=$1; shift; python $Q run --name "filament:$name" --vram 10 --ram 2 --cwd "$PWD" -- python train_fil.py $M $C "$@" --name $name > runs_$name.log 2>&1; grep -h RESULT runs_$name.log | cut -c1-400; }
run cv0_ema --ema 0.999
run cv0_unet --unet 11
run cv0_skel --skel-w 0.5
run cv0_aux --aux-w 0.3 --aux-steps 8,16
