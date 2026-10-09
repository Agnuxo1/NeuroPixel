# Conciliación de continuaciones NeuroPixel

Conciliación completada el 8 de octubre de 2026. No se fusionan resultados científicos ni se repiten entrenamientos.

| Identidad | A06.1 solicitada | Estudio cloud |
|---|---|---|
| Commit ejecutado | a009f2a9b520a3c7333d092cd3d4f0026b17c335 | 08d0d52edd05da6835e71479f3ba4399fcbeabae |
| Plan SHA256 | 4bc25ed4acce0c95f41d0a595289f854e0330fe237aee1210a8787378cdf6b01 | 20f4e9b4385c95c5f206ac26702bca96ac4232930ea343fd718c4de5bec6048a |
| Ejecución | 37588312570 | 37566497890 |
| Archivo | results/research/06_A06_1_recovery/archive | results/research/06_cloud_reference |
| Fuentes experimentales | growth.py; research_growth.py; research_analyze_item6.py | growth_ablation.py; research_growth_ablation.py; analizadores separados |
| Python registrado | 3.12.15 | 3.12.8 |
| Torch CPU registrado | 2.6.0+cpu | 2.6.0+cpu |
| psutil registrado | 6.1.1 | 7.2.2 |

La auditoría recuperó 400 archivos del archivo cloud fijado al commit `15e76456bc2b4cce5faec0b08fb5288fe7844547`, sin discrepancias de tamaño/hash. Las 37 fuentes de ejecución cloud concuerdan con su manifiesto; 17 archivos son idénticos a sus equivalentes locales, incluyendo modelo, entrenador, datos, especificación de ablaciones y protocolo histórico. El controlador del núcleo incorpora un adaptador de procedencia para una dependencia histórica ausente; su implementación difiere. El módulo/controlador de crecimiento es otra implementación. Ninguno se reemplaza silenciosamente.

Los 26 configs nominales, 34 casos y dataset final del núcleo coinciden. Los bytes de los checkpoints y los resultados registrados difieren entre ejecuciones. Esto no identifica por sí solo una causa: hay que separar serialización, realización numérica, entorno y operaciones del controlador. El recibo alinea las 34 filas; no les atribuye independencia adicional ni convierte la diferencia de resultados en un nuevo contraste prospectivo.

Ejemplo: tying medio es +2,014 pp en A06.1 y −0,244 pp en el informe cloud. No se escoge el valor favorable ni se promedian. Las semillas 20/21 no se contabilizan como cuatro inicializaciones independientes. En cloud, novedad reprodujo el banco fijo; en A06.1 su tercer estado y sus métricas difieren. Las diferencias de capacidad y señales léxicas de routing siguen limitando ambos estudios.

La copia local conserva el cierre exacto de A06.1 con sus límites numéricos. El ledger cloud del commit `3434635b9074d9e834604331b6dc27565eaf713c` declara investigaciones 7–15 cerradas y 16 activa. Sus informes ya están recuperados y verificados por hash, pero sus cierres no se importan automáticamente: la tarea 4 debe verificar alcance, artefactos y controles relevantes. La tarea 3 se abre como estudio nuevo de optimización, competencia y precisión, nunca como una reinterpretación de H1 o una repetición encubierta de A06.1.

Evidencia: `coord/recovery/reconciliation-20261008/source_comparison.json`, `reconciliation_receipt.json`, fuentes cloud fijadas por commit y los dos archivos originales separados. La procedencia y los resultados negativos del punto 5 permanecen intactos. No se modificó main ni una copia histórica.
