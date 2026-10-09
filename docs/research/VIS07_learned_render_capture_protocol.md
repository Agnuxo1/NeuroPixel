# Pesos aprendidos: captura GPU y fidelidad acotada

Antes deejecutar: usar los cinco modelos originales seleccionadosporDEV, exportadosporhash. Primeras64filasTESToficialyaexpuesto porsemilla, sinselección porresultado. Comparar RGB→retina→NCA8→decoderentreCUDAFP32nativo yrenderfragmentRGBAvector; decisionesexactas/NLLmediaerror≤1e-4 ytodos loserrorespointwise1e-4+1e-5|referencia|reportados sinocultarfallos. Verificartambién lasdecisiones CUDAcontralasoriginalesCPU. No se sustituye eltest completo por320escenas ni se infiere otroaccuracybenchmark desdeestas verificaciones.

Guardar framesreales0–8deun ejemplofijo: primera filaTRAINclase0, semilla240. Captura desde losbuffers realesGPU, no imagenAI. Colores/gráficas posteriores sonproyecciones visuales deestosdatos; no anatomía/circuito causal identificado. Elstep8eselpunto validado; noproclamar estabilidaddeunloopindefinido. Conservartoda discrepancia y noampliartolerancias ni cambiar muestra.

Entrenamientos0, FIFO/RAM8real/bootstrap9,CPU2/VRAM2; fuentes/hash/trained-checkpoint-identity guardados. En casodefallo preservar archivosparciales. Se publicarán estosGIFs yloslímites explícitos enREADME porautorizaciónhumana.
