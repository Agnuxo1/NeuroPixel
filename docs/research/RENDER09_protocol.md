# RENDER09: optimización vectorizada con control escalar conservado

RENDER08 quedó completo ynegativo para velocidad: siete workloads,35gates y147timingrows; renderescalar3.14–12.25veces más lento queel mejorCUDA. No se modifica, borra o mezcla ese resultado.

Nueva implementación separada:coeficientes empaquetados para cuatrooutputs/cuatroinputs, fetchRGBA yDOT4, muestrascompartidas depercepción depthwise. Mismooperador nativo yFP32, nuevosource hash. Preflight físicoGPU136checks passed; ninguna tolerancia ampliada. No se reentrena ni cambia inicialización/dato original delbenchmark.

Mismos siete workloads/seed85200+índice/16updates ygate completo antesdetiming. Comparar dentro deestaejecuciónCUDAeager,CUDAgraph, renderescalaroriginal yrendervector. Siete bloques/tresinferencias/ordenalternado, mismoscriterios ydatos. Guardar todos loscasos ysalidas, inferirventaja sólo ensucoordenadamedida. Losbloques no son réplicas científicas, noCI poblacional ni universalidad. Mejora respecto alescalar no implica superarCUDA.

Este benchmark mide tiempo dehost ydevice. Las lecturas cortasNVMLdeRENDER08 tienen resolución insuficiente yno sostienen diferencias deenergía porinferencia: una lectura cero no representa energía cero. Conservar datos brutos ymarcar esasatribuciones como desconocidas. Medición energética sostenida requiereotro protocolo, cadenciaverificada ybloques largos. GPUdevice tampoco esenergía totaldepared.

OriginalNCA/renderescalar/fuentesDEV04GLOB05 permanecen intactas. Se puedenatribuir costesdecompilación/preparación porseparado; no ocultarlos. CPU2/FIFO/RAM8real/bootstrap9/VRAM4 ydeadline13:49:31UTC.
