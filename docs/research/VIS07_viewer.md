# Inspect neural states and shared decoding

The local viewer exposes the sixteen raw neural channels, an RGB projection with selectable channels, all-cell shared-decoder predictions, output-cell confidence and state RMS. It resets every input and caps updates at the trained depth of eight. It uses the fixed selected seed240 model and original TRAIN examples; this is inspection, not a new test accuracy result.

## Verified archive mode

Run from the repository:

```powershell
python scripts/live_VIS07_viewer.py --offline-replay --output results/research/VIS07_live_viewer/my_archive_session
```

Open `http://127.0.0.1:8767`. Use a fresh output directory for each session. The default session lasts ten minutes and releases its resources afterward. This mode reads actual archived GPU states and complete logits for T0–8; it performs **no neural GPU inference**. The interface labels that distinction prominently. Fifty HTTP checks verified exact state/logit serialization, all-channel RMS, reset, depth cap and local-session admission. Browser checks verified the controls and completed eight-step playback. [HTTP evidence](../../results/research/VIS07_live_viewer/offline_20261009A/http_verification.json).

## Verified live physical GPU mode

The live mode uses the verified vector fragment renderer, fixed exported weights, learned retina and shared decoder. Each requested step computes a new GPU state. Startup first compares the complete T0–8 trajectory to the original GPU archive through 27 registered state/logit/decision checks, then resets to T0. No label token enters the model. All ten fixed TRAIN images can be selected.

Run under a shared GPU reservation, with at least 8 GiB available host RAM throughout, two CPU threads and an available physical NVIDIA context:

```powershell
python scripts/live_VIS07_viewer.py --output results/research/VIS07_live_viewer/my_live_session
```

On the project machine, use the shared `gpuq.py` FIFO before this command. The integration verifier passed **27 startup trajectory gates and 52 HTTP checks** on the physical RTX 3090, including all nine states, logits, decisions, ten TRAIN resets, depth cap and clean resource release. [Physical receipt](../../results/research/VIS07_live_viewer/live_20261009A/http_verification.json). A separate CPU/Mesa llvmpipe execution passed the same registered gates: [software archive](../../results/research/VIS07_viewer_software_review/37936500542/recovery_receipt.json). That software execution and archive playback remain separately labelled; neither is an outside investigator's independent replication.

## Interpretation

Colours are a three-channel projection with a fixed scale derived from the registered trajectory. Display saturation does not clip model state; raw values, range and RMS remain available. Scanner labels outside (4,4) apply the shared decoder but were not individually supervised or validated as causal meanings. Slowed interface playback is not an inference-speed benchmark. Neither display similarity nor confidence establishes biological equivalence, consciousness or a complete causal explanation.

The server binds only to loopback, rejects mutation requests without its ephemeral local-session token, and does not transmit states to external services. The token is not archived or published.
