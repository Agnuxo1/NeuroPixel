#!/usr/bin/env bash
# SAT-001b (idea de Fran): modelos de 8000 it + 4000 it más (lr 5e-4), mismas condiciones para los tres.
# Se encola mientras corre ms_fsr_small, antes de ms_learned_big (FIFO de gpuq).
cd "$(dirname "$0")"
Q=D:/PROJECTS/.cognition/gpu_queue/gpuq.py
C="--batch 8 --crop 512 --steps 24 --pos-frac 0.4 --w-pos 3 --eval-every 1000 --vram-cap 12 --force-gpu --c 48 --hidden 128 --iters 4000 --lr 5e-4"
until [ -f runs_ms_fsr_small.log ]; do sleep 60; done
sleep 120
for m in fsr bilinear learned; do
  python $Q run --name "filament:SAT +4k ms_$m" --vram 10 --ram 4 --cwd "$PWD" -- \
    python train_fil.py $C --ms $m --init ms_$m --name ms_${m}_plus4k > runs_ms_${m}_plus4k.log 2>&1
done
grep -h RESULT runs_ms_*_plus4k.log | cut -c1-200
