# VIS07: inferencia visual externa y controles de profundidad

Receta nueva tras preflight admitida: ambos modelos memorizaron50/50imágenes TRAIN en512updates, NCA5056/CNN5039parámetros. Esa memorización no es generalización. ZIP/proof originales fijados; no se reutilizan pesos piloto.

Cinco semillas nuevas pareadas240–244. TRAIN3523/DEV300 extraídos delTRAIN oficialUCI3823, TESToficial1797, sin imágenes idénticas entre grupos. Fuente/pixelescala/arquitecturas/licencia delpreflight intactas. Inicialización nueva en cada familia, mismo orden completo de40épocas porsemilla; batch64,AdamLR.001,clipgrad1,fire1,8updates NCA, supervisiónuna etiqueta porimagen. Cero búsqueda deLR, aumento deépocas/semillas, pseudoetiquetas oteacherauxiliar. Cerca en número nominal, no igualdad funcional/FLOPs: retina dilatada yCNN tienen geometrías distintas.

Cada checkpoint se selecciona exclusivamente porDEV: mayorcorrectas/300, luego menorNLL media, luego época más temprana. Guardar cuarenta filas y Adam/model/RNG; no escoger la mejorsemilla. Sellar los diez modelos porhash ypublicar recibo degate antes deparsear/evaluar TEST. Consumir acceso durablemente antes del primer score. TESTpublicoexpuesto no es custodia ciega ni replicación por terceros. Eltest original264composiciones permanece excluido.

Primario: exactitudNCA–CNN pareada porsemilla en TESTbase, media sobrecinco realizaciones, ICt95df4 exploratorio sobrealeatoriedad de entrenamiento condicionada a ese corpus. NoCI poblacional de escritores ni ajuste múltiple. Gatecompetencia NCA≥90%en lascinco semillas; conservar fallos. NLL/macroexactitud ytodos losmodelos también se reportan.

Diagnósticos prospectivos, sinselección: mitad intensidad; desplazamiento1columna condatoscrop (puede eliminar información, no universalinvariancia). NCA entrenadaT8 evaluada coninput sostenido enT0/4/8/16/32 yRMS porimagen; T0 es ablacóndestructiva de inferencia, no baseline entrenado sin recurrencia. Todosreusan1797ejemplos yno generan nuevas réplicas. Convergencia/memoria general/causalidad semántica no se infieren de esos endpoints.

EntrenamientoCPU2/RAM8/Python3.12.14/Torch2.6.0+cpu/NumPy2.2.6/SciPy1.15.1/psutil6.1.1/Pillow11.3, unworker. Inferencia posterior porrenderer se compara contra los pesos seleccionados ylas predicciones archivadas, sinreentrenar. Entrenamientoactual usaPyTorchCPU; noproclamamos entrenamiento porrenderizado. Preservar negativos, parciales ycostes. Ningúnresultado implica calidadNobel, SOTA general niH1rescatada.
