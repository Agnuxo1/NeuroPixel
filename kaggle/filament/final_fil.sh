#!/usr/bin/env bash
# Cierre de la tanda (Fran 28-09): esperar semillas 1-2, ensamble final con gate y envio; despues Biohub.
cd "$(dirname "$0")"
until grep -qE "RESULT|Traceback" runs_ms_learned_s2.log 2>/dev/null; do sleep 120; done
grep -h RESULT runs_ms_learned_s1.log runs_ms_learned_s2.log | cut -c1-200
python D:/PROJECTS/.cognition/gpu_queue/gpuq.py run --name "filament:FIL-FINAL ens4" --vram 6 --ram 6 --cwd "$PWD" -- \
  python ensemble_fil.py --runs ms_learned_cont ms_learned_cons_cont ms_learned_s1 ms_learned_s2 --val --test > runs_ens4.log 2>&1
grep -E "VAL|TEST|Trace" runs_ens4.log | cut -c1-250
E="runs/ens_ms_learned_cont+ms_learned_cons_cont+ms_learned_s1+ms_learned_s2"
if python -c "import json,sys;sys.exit(0 if json.load(open('$E/val.json'))['PQ']>0.4293 else 1)"; then
  kaggle competitions submit -c filament-segmentation-2026 -f "$E/submission.csv" -m "NeuroPixel ensemble of 4 multiscale learned canvases (2 long + 2 seeds)" 2>&1 | tail -1
  until kaggle competitions submissions -c filament-segmentation-2026 2>&1 | sed -n 3p | grep -qE "COMPLETE|ERROR"; do sleep 20; done
  kaggle competitions submissions -c filament-segmentation-2026 2>&1 | sed -n 3p | grep -oE "(COMPLETE|ERROR).*"
else echo "gate no superado: se mantiene el envio de 0,4293 (publico 0,35)"; fi
