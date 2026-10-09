# VIS07 profundidad: diagnóstico acotado de la incidencia

El primer replay funcional reprodujo98.835decisionesexactas; las30condicionesdebase/transformación pasaron logits/NLL bajoeltargetoriginal. Diezcomparacionesdelogits prolongadosT16/T32 fallaron1e-4+1e-5|referencia|. Preservar elrecibo fallido yno ampliartolerancia.

Se ejecutauna única comprobación nueva, sin entrenamiento, conlas primeras128filasTESTyaexpuestas ylos cinco modelos seleccionados originales. Comparar enel mismo procesoCPU elcaminoModuleoriginal yelcaminoF.conv2dfuncional aT0/4/8/16/32; ambos contra losvalores archivados yeltargetoriginal. Guardar todos losarrays/discrepancias/direcciones categóricas ymodeloCPU.

Siamboscaminos coinciden localmente pero difieren delarchivo, eso descartaunerror observable entreesosdoscaminos bajo eseentorno; no identifica por sí solo CPU, librería, alineación ocausa única. Reportar como compatibilidadcontinuadafallida, conhuella deentorno. ElcrecimientoRMS ydeclive deexactitud aT16/32son datosfinítos originales,no prueba deconvergencia ni divergencia infinita. Clasificación válidaaT8 nodebe relabelarse estabilidad arbitraria.

CPU2/RAM8/softwareoriginal, sin modelos nuevos, sin consultas nuevas a holdout intacto, sinselección deinputs por resultados.
