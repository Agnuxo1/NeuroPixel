#!/usr/bin/env bash
# Ensamble de 3 modelos largos y distintos (learned_cont, cons_cont, cons_40k): val+test por gpuq; envio solo si val > 0,4293.
cd "$(dirname "$0")"
python D:/PROJECTS/.cognition/gpu_queue/gpuq.py run --name "filament:FIL-ENS3 cont+cons_cont+cons_40k" --vram 6 --ram 5 --cwd "$PWD" -- \
  python ensemble_fil.py --runs ms_learned_cont ms_learned_cons_cont ms_learned_cons_40k --val --test > runs_ens3.log 2>&1
grep -E "VAL|TEST|Trace" runs_ens3.log | cut -c1-250
E="runs/ens_ms_learned_cont+ms_learned_cons_cont+ms_learned_cons_40k"
if python -c "import json,sys;sys.exit(0 if json.load(open('$E/val.json'))['PQ']>0.4293 else 1)"; then
  kaggle competitions submit -c filament-segmentation-2026 -f "$E/submission.csv" -m "NeuroPixel ensemble of 3 long multiscale canvases (individual + soft-consensus 24k + consensus 40k)" 2>&1 | tail -1
else echo "gate no superado (val <= 0,4293): no se envia"; fi
