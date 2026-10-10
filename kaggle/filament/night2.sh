#!/usr/bin/env bash
# Segunda parte de la noche 2026-09-27/28 (Claude): ensamble + envío con gate, y estudio de saturación.
# Se encola en gpuq DESPUÉS de ms_fsr_big (espera a que exista su ticket/log).
cd "$(dirname "$0")"
Q=D:/PROJECTS/.cognition/gpu_queue/gpuq.py
C="--batch 8 --crop 512 --steps 24 --pos-frac 0.4 --w-pos 3 --eval-every 2000 --vram-cap 12 --force-gpu"
until [ -f runs_ms_fsr_big.log ]; do sleep 60; done

# 1) mejor learned (original o continuado) + np_big_reposo_cont; envío si val > 0,4061
BEST=$(python -c "
import json;from pathlib import Path
c={r:json.loads(Path(f'runs/{r}/result.json').read_text())['best_postproc']['PQ'] for r in ('ms_learned','ms_learned_cont') if Path(f'runs/{r}/result.json').exists()}
print(max(c,key=c.get))")
echo "mejor learned: $BEST"
python $Q run --name "filament:FIL-002 ens $BEST+big" --vram 6 --ram 5 --cwd "$PWD" -- \
  python ensemble_fil.py --runs $BEST np_big_reposo_cont --val --test > runs_ens_ms.log 2>&1
python $Q run --name "filament:FIL-002 ens $BEST solo-val" --vram 6 --ram 5 --cwd "$PWD" -- \
  python ensemble_fil.py --runs $BEST --val > runs_ens_ms_solo.log 2>&1
ENS=runs/ens_${BEST}+np_big_reposo_cont
GATE=$(python -c "
import json;from pathlib import Path
e=json.loads(Path('$ENS/val.json').read_text())['PQ']
s=json.loads(Path('runs/ens_$BEST/val.json').read_text())['PQ']
print('ens' if e>max(s,0.4061) else ('solo' if s>0.4061 else 'no'), e, s)")
echo "gate: $GATE"
case "$GATE" in
  ens*) kaggle competitions submit -c filament-segmentation-2026 -f $ENS/submission.csv \
          -m "NeuroPixel ensemble: multiscale learned canvas ($BEST) + big reposo canvas; $GATE" 2>&1 | tail -1 ;;
  solo*) python $Q run --name "filament:FIL-002 pred $BEST" --vram 6 --ram 5 --cwd "$PWD" -- \
           python predict_fil.py --run $BEST > runs_pred_$BEST.log 2>&1
         kaggle competitions submit -c filament-segmentation-2026 -f runs/$BEST/submission.csv \
           -m "NeuroPixel multiscale learned canvas ($BEST); $GATE" 2>&1 | tail -1 ;;
  *) echo "sin envío (gate no superado)";;
esac

# 2) saturación en tamaño (FSR c=24 frente a 48 y 96) y learned grande
python $Q run --name "filament:SAT ms_fsr_small" --vram 6 --ram 4 --cwd "$PWD" -- \
  python train_fil.py $C --c 24 --hidden 64 --ms fsr --iters 16000 --lr 2e-3 --name ms_fsr_small > runs_ms_fsr_small.log 2>&1
python $Q run --name "filament:SAT ms_learned_big" --vram 14 --ram 4 --cwd "$PWD" -- \
  python train_fil.py $C --c 96 --hidden 256 --ms learned --iters 8000 --lr 2e-3 --vram-cap 16 --name ms_learned_big > runs_ms_learned_big.log 2>&1
grep -h RESULT runs_ms_fsr_small.log runs_ms_learned_big.log | cut -c1-200
