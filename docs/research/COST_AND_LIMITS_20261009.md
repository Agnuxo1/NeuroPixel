# Measured cost and unresolved claims

VIS07 recorded training plus DEV selection for all ten fixed fits: NCA413.318s wall/826.410s processCPU; CNN34.237s wall/68.427s processCPU. Near nominal parameters5056vs5039 did not equalize operations or time: NCA cost about12.07× more training walltime in this recipe. Data download, package installation, provider scheduling, checkpoint/Git archival and subsequent verification add cost and are retained in workflow/provider evidence; they are not silently counted as zero.

GPU benchmarks RENDER08/09 include resident-input seeding,16updates and complete scanner, explicit synchronization, device/host timings, recorded setup and shader compilation. They exclude training, network transport, installation and new input/output transfer from the primary resident timing. The full cost of deploying/training a complete system is larger. Technical timing blocks, masks, pixels and endpoints are not independent scientific replicates.

NVML is an actual whole-device energy counter on the measuredRTX3090; short-call zero deltas have insufficient temporal resolution. Per-inference energy remains unknown until an adequate sustained-block protocol. GPU device energy also excludesCPU,RAM,PSU/wallpower. VirtualCPU runner energy was not measured; no TDP×time estimate is presented as measurement.

Learned checkpoint storage, datasets, raw predictions, states and archives are retained with sizes andSHA256 in the publication manifest. Copying experts preserves old weights at growing storage cost; this does not prove effective continual acquisition/routing. Caches and selected best DEV checkpoints are part of the pipeline, not free intelligence.

Scientific boundary: originalH1 unsupported; DEV04precision targetfailed; public/exposed datasets and internal replays do not create blind external replication. The pixel-based vision result is useful on the declared UCI task. Rendering fidelity and a conditional measured speedup do not establish a biological brain, consciousness, clinically meaningful brain scanning or a Nobel-level discovery. Attribution of learned mechanisms, all-time stability, fullwallenergy and independent reproduction remain open.

No photonic simulation was added without an equal-input/output/precision and full-cost hypothesis. This preserves the separate opticalNeuro3D project and its evidence.
