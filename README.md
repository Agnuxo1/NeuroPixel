# NeuroPixel

**A neural network whose entire state lives on a canvas of pixels.** Words enter as learned
"colours" (a dictionary), activity spreads with local cellular-automaton rules, and the answer is
read by translating the colour of an output pixel back through *the same* dictionary. Every step of
the "thinking" is visible and can be decoded pixel by pixel (the *dictionary scanner*).

*Proyecto de Francisco Angulo de Lafuente. Conclusiones detalladas (en español):
[docs/FASE3_CONCLUSIONES.md](docs/FASE3_CONCLUSIONES.md).*

![Scaling curves](docs/img/scaling.png)

## Key results (synthetic compositional task, always evaluated on unseen combinations)

Task: sentences made of role/filler pixel pairs (agent, action, patient, place) placed at random
positions on an 8×8 canvas; a query pixel asks for one role. Test sentences use agent–action–patient
combinations never seen in training.

| Finding | NeuroPixel | Baseline |
|---|---|---|
| **Scaling** (unseen combinations) | 49 % → **99.7 %** from 5 k to 236 k parameters (30 k: 95.4–98.4 %, 2 seeds) | Transformer flat at **54 %** from 13 k to 811 k parameters (it learns a shortcut and swaps agent/patient) |
| **Self-repair** (50 % of the state erased mid-thought) | 79–94 %; with "rest-state" training: **99.4 %**, stable from 8 to 64 thinking steps | Transformer 25–33 % |
| **Cost per answer** vs a local LLM | 30 k params, 99.5 %, **5.7 ms**, 56 MFLOP (CPU) | Qwen2-494M (3-shot, CPU): 94 %, 279 ms, 158 GFLOP (~2 800× more compute) |
| **New word from 5 examples** (only one dictionary row trained) | up to 98.6 % on the new word, 92.6 % kept on old knowledge | Transformer 50–61 % |
| **Growing canvases** (continual learning) | New canvas created when the current ones "cannot read" the input (novelty, ART-like): 3 topics → 3 canvases, **93 %** average | Single canvas trained sequentially: 66 % (catastrophic forgetting) |
| **Persistent memory** (facts shown one by one, queried after N blank frames) | 99 % up to 8 frames, 70 % at 16 | GRU 77 % (less precise, but decays more slowly) |
| **Non-local binding** (role and filler far apart, 12×12) | 66 % → 74 % (with size and rest-state training) | Transformer 56 % (48 k and 320 k) — **not solved yet** |

### Honest caveats

- Synthetic, small tasks; not yet tested on natural language or large vocabularies.
- Most numbers come from a single seed (the 30 k / 44 k scaling points have two).
- Baselines are small transformers/GRU with the same training budget; a transformer with more
  data, other positional encodings or longer training might solve the task.
- Vision: on CIFAR-10 the pure canvas does not perceive objects (16–19 %); with a small "retina"
  front-end 42–62 % vs 72–76 % for a CNN. The goal of this project is not to beat CNNs.
- An attempt to anchor dictionary colours to real perceptual colours did not transfer zero-shot.

## How it works

```
word ──dictionary──▶ pixel "colour" (c_id channels)        identity (kept, re-injected every step)
                          │
canvas state s ──3×3 local rule (depthwise perception + 1×1 MLP), residual, stochastic updates──▶ s'
                          │                                   (T steps; rest-state: T random + damage)
output pixel state ──same dictionary (tied weights)──▶ answer
every pixel, every step ──same dictionary──▶ "what is this pixel thinking" (scanner)
```

- **School loss** (`--lens-aux`): every input pixel must keep "saying" its own word through the
  dictionary — makes the canvas legible and improves learning.
- **Rest-state training**: random number of steps and random damage during training → stable,
  self-repairing dynamics.
- **Growing canvases**: freeze a canvas and create a new one when the fraction of unreadable input
  pixels exceeds a threshold; route queries with the scanner (resonance).

## Repository

| Path | Contents |
|---|---|
| `neuropixel/model.py` | `NeuroPixel` (canvas), `Retina`, `TinyTransformer` baseline |
| `neuropixel/task.py` | Role task, held-out combinations, camera-pixel inputs |
| `neuropixel/phase3.py` | Streaming memory, non-local task, GRU baseline, vocabulary expansion |
| `neuropixel/cartilla.py` | CIFAR-10 "primer" task (image + word on the same canvas) |
| `neuropixel/scanner.py` | Dictionary scanner rendering |
| `neuropixel/codec.py`, `hrr.py` | Token↔RGB24 pixel codec; holographic reduced representations |
| `neuropixel/safety.py` | Resource guards (CPU by default, GPU only if free, VRAM cap) |
| `scripts/train.py`, `train_cartilla.py`, `battery.py`, `phase3.py` | Experiments |
| `scripts/night.py`, `sweep.py` | GPU job queue with temperature watchdog |
| `results/` | Raw JSON results, reports and plots of all experiments |
| `docs/`, `ROADMAP.md` | Conclusions (Spanish), curriculum and work lines |

## Quick start

```bash
pip install torch numpy pillow matplotlib
python -m pytest
python scripts/train.py --model neuropixel --iters 2000            # CPU, small
python scripts/phase3.py scale --arg np30k                          # GPU (checks it is free)
python scripts/phase3.py stable --arg reposo                        # rest-state training
python scripts/report_phase3.py                                     # rebuild the report
```

CIFAR-10 experiments expect `data/cifar10.npz` (not included): run `python scripts/prepare_cifar.py`.

## License

MIT — see [LICENSE](LICENSE).
