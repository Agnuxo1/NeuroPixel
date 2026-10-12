# All-study results and immutable evidence

| Study | Report | Original execution / verification |
|---|---|---|
| Historical H1 and later capabilities | [Historical audit](TASK4_historical_capabilities_audit_20261009.md) | Fifteen immutable archives,763files/401580342bytes;3968newauditchecks. Negative/unexecuted panels retained. |
| READ03 | [Report](READ03_results.md) |36readoutfits on12frozenbodies,10616832exactdecisions; development only. |
| DEV04 | [Report](DEV04_results.md) |Original37845945011+failed-start recovery37871777801; replay37876687599; originalprecisionfailed. |
| GLOB05 | [Report](GLOB05_results.md) |37887762153, twelve new global controls,3538944exactdecisions; oldfitsreused. |
| SCN06 | [Report](SCN06_results.md) |Canonical37896280187; redundant37896335193notanotherreplicate; five learned projection matrices. |
| VIS07 | [Report](VIS07_results.md) |Main37908785698, ten fixedfits; functionalreplay37911162172retains10continuousfailures; diagnostic37913653666subset. |
| Real trained trajectory | [Animation provenance](../animations/manifest.json) |CPUsource37915284630andseparateactualGPUcapture, registeredseed240/fixedTRAINclass0. |
| RENDER08 | [Report](RENDER08_results.md) |35fullparitychecks/147timingrows, scalarnegative, all7workloads. |
| RENDER09 | [Report](RENDER09_results.md) |56fullparitychecks/196timingrows, vector/scalar/strongCUDAcomparators, conditionalresults. |
| RENDER10 advanced graphics | [Report](RENDER10_results.md) |OpenGL 4.6/SSBO/shared tiles: 21 Torch-free output checks across seven workloads; 77 full benchmark gates passed; 245 registered timing rows complete; 143 original rows preserved without repeat. Interrupted sessions qualify interpretation. |
| RENDER11 sustained board energy | [Report](RENDER11_results.md) | 105 admitted keys; 103 original rows unchanged, zero repeats; saved recovery 87c60227. Board energy only; wall energy unmeasured. |

All raw archives retained by the project and their original Gitblobs are included/indexed by the publication manifest; the item-5 CPU archive and historical reassessment outputs are unavailable. Separate recipes retain their own source commits, plans, run IDs, hashes and failed attempts. Early status notes and technical retries are not scientific replications. The original test264compositions stays excluded in new studies. Publication in main does not convert an internal check into external replication.

## Final engineering update — 9 October 2026

The recovered RENDER10 cohort is complete:245 registered rows,77 output gates,143 original rows preserved exactly and0 repeated timing keys. OpenGL4.6 compute measures2.084x forC16/grid128 relative to the bestCUDA comparator and loses forC48/grid128; interruption separates sessions and qualifies interpretation. Live physical viewer integration passed27 trajectory gates plus52 HTTP checks; CPU/Mesa passed the same separate internal endpoint. RENDER11 originally retained 103/105 energy blocks after the RAM floor stopped it. Its saved recovery is admitted with 105 unique keys, 103 unchanged original rows and zero repeated closed measurements; GPU-board energy is descriptive and whole-system wall energy remains unmeasured. See RENDER10_results.md, VIS07_viewer.md and RENDER11_results.md.
