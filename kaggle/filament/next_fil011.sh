#!/usr/bin/env bash
# FIL-011 (Fran 28-09, primera tarea del 29-09): consenso suave desde cero hasta 40.000 it, validacion cada 2.000,
# para ver si sigue mejorando o se estanca (curva unica, sin reinicios de lr). ~6,5 h en la 3090.
cd "$(dirname "$0")"
python D:/PROJECTS/.cognition/gpu_queue/gpuq.py run --name "filament:FIL-011 ms_learned_cons_40k" --vram 10 --ram 8 --cwd "$PWD" -- \
  python train_fil.py --batch 8 --crop 512 --steps 24 --pos-frac 0.4 --w-pos 3 --eval-every 2000 --vram-cap 12 --force-gpu \
  --c 48 --hidden 128 --ms learned --consensus --iters 40000 --lr 2e-3 --name ms_learned_cons_40k > runs_ms_learned_cons_40k.log 2>&1
grep -E '"it"|RESULT|Trace' runs_ms_learned_cons_40k.log | cut -c1-200
