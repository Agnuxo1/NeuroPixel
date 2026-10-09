# RENDER10: OpenGL4.6 graphics-compute, shared tiles and SSBOs

New separately labelled backend using actualOpenGL4.6 compute shaders,4×4×4workgroups, shared pixel/feature tiles, vec4weights inSSBO and explicit memory barriers. It processes image states through the graphics API; it is not fragment rasterization. Old scalar/vector sources and negative results remain immutable.

Engineering qualification uses all seven original RENDER09 input/weight archives and their already captured nativeCUDAstate/logit arrays. All21state/logit/categorical checks passed onRTX3090, with the same frozen continuous rule and exact decisions. The actual shared-memory bound is queried fromthe active driver. Two pre-execution API/compiler faults and memory-admission refusals are retained; none is labelled a completed parity experiment or a speedresult.

Now freeze a same-device comparison ofCUDAeager,CUDAgraph, scalarfragment, vectorfragment and graphics-compute. Reuse the same seven workloads, random seeds, sixteenupdates, exactfull-output gate beforetiming, seven technicalblocks andthreecalls perblock. No precisionchange, expandedbudget, favourablegeometryselection or omittedcomparators. Report allmethods/cases andtechnicalranges; conditionalhardwaremeasurements are not independentreplications.

This is a runtime comparison, not a classification experiment or trainingbyrendering. The datasetvisionaccuracy remains a separate CPUtraining study withverified learnedGPUinference. Per-inference energy in these shortblocks isnot validated; zero NVMLcounterdeltas are not zeroenergy. Longblockenergy requires itsown protocol. Fullwallenergy andbiologicalequivalence remainunestablished.

OneGPUFIFOactor, CPU2, actualavailableRAM≥8, bootstrap9,requestedVRAM4. Applicableadvancedgraphics technology is tested againstequal numericaloutputs; no irrelevant raytracing, meshwork or upscaling is added to alter the task.
