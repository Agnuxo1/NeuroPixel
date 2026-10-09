#!/usr/bin/env bash
# Corregida la geometria del disco (01-10): FIL-007 (filtros) y FIL-008 (SDO) se midieron con radio ~11 % inflado. Repetir SDO con la alineacion correcta y probar PIL (P10),
# protocolo estricto de Codex fold 0, 8k, lr 1e-3, semilla 0, --lowmem. Empieza a las 22:00 junto a night_fil_1001b.sh (FIFO de gpuq).
cd "$(dirname "$0")"
until [ "$(date +%H)" -ge 22 ] && [ "$(date +%H)" -lt 23 ]; do sleep 60; done
Q=D:/PROJECTS/.cognition/gpu_queue/gpuq.py
M="--split-manifest work/cv-protocol-20260930/fold-0.json"
C="--lowmem --batch 8 --crop 512 --steps 24 --pos-frac 0.4 --w-pos 3 --eval-every 2000 --vram-cap 12 --force-gpu --c 48 --hidden 128 --ms learned --iters 8000 --lr 1e-3 --seed 0"
run() { name=$1; shift; python $Q run --name "filament:$name" --vram 10 --ram 2 --cwd "$PWD" -- python train_fil.py $M $C "$@" --name $name > runs_$name.log 2>&1; grep -h RESULT runs_$name.log | cut -c1-400; }
run cv0_sdo2 --sdo
run cv0_pil --pil
