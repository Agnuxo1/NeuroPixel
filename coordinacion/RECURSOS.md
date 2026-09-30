# Recursos y reservas

Actualizada: 2026-09-29 10:32 Europe/Madrid.

## Inventario local

| Recurso | Capacidad | Margen operativo acordado |
|---|---:|---|
| GPU | RTX 3090, 24.576 MiB | reservar 3,5 GiB para pantalla/sistema; no superar 83 C sostenidos |
| RAM | 23,7 GiB | conservar al menos 8 GiB disponibles; no iniciar trabajo nuevo por debajo de ese umbral |
| CPU | 20 hilos logicos | maximo 8 hilos por agente; total habitual <=16; bajar a 4 con GPU saturada |
| Disco de trabajo | D: y E: | datos, caches, entornos, modelos, temporales y resultados permanecen en D: o E:; evitar C: |

## Reservas activas

| Recurso | Tarea | Responsable | Inicio conocido | Limite | Liberacion esperada | Estado |
|---|---|---|---|---|---|---|
| GPU | SOIL-001 | Claude | antes de 19:04 | 6 GiB | al completar folds 4-5 | liberada 20:13 (6/6 folds; result.json) |
| GPU | FIL-001 | Claude | antes de 17:42 | 14 GiB | al completar 6000 it | liberada 20:40 (6000/6000; result.json) |
| GPU | FIL-005 | Claude | turno anterior | tope 12 GiB | cerrada | liberada antes de 2026-09-29; no es reserva actual |
| GPU | BIO-001 E1 detector | proceso iniciado por Claude; supervisa Codex | 2026-09-29 09:07 | gpuq 14 GiB VRAM / 6 GiB RAM | 09:26 | liberada; 8 epocas completas, resultado interno no validado en pipeline |
| GPU | BIO-001 D1 NP3D | proceso iniciado por Claude; supervisa Codex | 2026-09-29 09:26:30 | gpuq 6 GiB VRAM / 6 GiB RAM | 09:48:29 | liberada; AUC CV 0,75, solo diagnostica por contaminacion/negativos no verificados |
| GPU | BIO-001 D1 puntuacion de candidatos | proceso iniciado por Claude; supervisa Codex | 2026-09-29 09:48:39 | GPU fuera de gpuq, ~1,9 GiB RAM | antes de 09:51:18 | proceso finalizado; `cand_*_scored.parquet` auditados, solo diagnostico; no promocionar |
| GPU | BIO-001 E1 prediccion cruda | proceso iniciado por Claude; supervisa Codex | 2026-09-29 09:51:18 | gpuq 12 GiB VRAM / 6 GiB RAM | 09:58:29 | liberada; resultado raw adverso; no continuar E1 |
| GPU | BIO-001 D1 score en cola | proceso iniciado por Claude; supervisa Codex | 2026-09-29 09:58:46 | gpuq 4 GiB VRAM / 4 GiB RAM | 10:04:19 | liberada; publico y train puntuados, solo diagnostico |
| GPU | BIO-001 blend010 raw | Codex; JEV remoto 0,92 | 2026-09-29 10:10:13 | gpuq 12 GiB VRAM / 8 GiB RAM | 10:13:57 | liberada; delta edge micro -0,00208, no continuar |
| GPU | BIO-001 B detector 0,955 serial | Codex; JEV remoto 1,0 | 2026-09-29 10:18:19 | gpuq 12 GiB VRAM / 8 GiB RAM | 10:29:09 | liberada; CSV completo y verificado; se cerro solo el bloqueo final de `plt.show()` tras guardar todos los artefactos |
| RAM/CPU | descarga Biohub | proceso iniciado por Claude; supervisa Codex | antes de 09:21 | red/CPU ligera | al completar clips | activa; RAM libre 5,83 GiB a las 09:21 |

No hay carga GPU larga reservada a las 10:32; comprobar gpuq y uso real antes de iniciar otra.
Se mantiene una sola carga GPU larga a la vez y el suelo de 8 GiB de RAM libre. Excepcion: una
prueba corta de menos de 10 minutos si la suma medida queda por debajo de 20 GiB y la GPU por debajo
de 80 C antes de empezar.

## Protocolo de reserva

Antes de ejecutar, añadir una fila con tarea, agente, VRAM/RAM/hilos, hora, duracion estimada y
condicion de corte. Renovar cada 90 minutos con un hito. Al acabar, marcar `liberada` y enlazar el
resultado. No finalizar procesos de otro agente sin autorizacion de Fran.

## Computo externo gratuito o promocional

| Plataforma | Oferta oficial observada | Uso recomendado | Limitacion clave |
|---|---|---|---|
| Kaggle Notebooks | P100/T4/TPU; cuota semanal limitada | solo ejecutar el cuaderno final, inferencia, generar submission y entregar al concurso | prohibido gastar esta cuota en entrenamiento |
| Google Colab gratuito | acceso sin coste a GPU/TPU | prototipos y replicas pequenas | recursos y limites fluctuan; sesiones pueden terminar |
| Lightning AI | creditos gratuitos para cuentas elegibles y Studio CPU gratuito | entrenamientos medianos persistentes | elegibilidad y creditos pueden cambiar |
| Hugging Face ZeroGPU | 5 min/dia para cuenta gratuita; hasta 2 Spaces ZeroGPU | demos e inferencias cortas | Gradio, cuotas cortas; no sirve para entrenamientos largos |
| Google Cloud Trial | 300 USD durante 90 dias para nuevos clientes | corrida puntual grande con presupuesto cerrado | GPU no es always-free; requiere cuenta de facturacion |

Fuentes oficiales verificadas el 2026-09-27:

- https://www.kaggle.com/docs/efficient-gpu-usage
- https://www.kaggle.com/docs/notebooks
- https://research.google.com/colaboratory/faq.html
- https://lightning.ai/docs/overview/ai-studio/
- https://huggingface.co/docs/hub/spaces-zerogpu
- https://docs.cloud.google.com/free/docs/free-cloud-features

No crear cuentas, activar facturacion, subir datasets privados ni consumir creditos sin una tarea
aprobada y registro de coste/cuota. El entrenamiento se hace en RTX 3090 local, Colab u otra nube
externa. Kaggle se reserva para ejecucion final, inferencia, archivo de submission y entrega.

Variables de cache y temporales de herramientas nuevas deben apuntar a una carpeta identificada en
D: o E:. Antes de una descarga pesada se registra origen, tamaño aproximado y destino. No se limpia
C: ni se mueve contenido existente sin autorizacion expresa de Fran.

## Programa horario estimado — 2026-09-27 22:04 Europe/Madrid

La cola viva informa `COLA (0)`: las filas `prevista` todavia deben ser reclamadas/encoladas por
su responsable. Horas recalculadas con el tiempo real de ms_learned (4552 s de entrenamiento) y
15-30 min de cierre/evaluacion. Si el proceso actual no libera RAM antes de las 22:30, todo el
programa se desplaza conservando el orden.

| Orden | Prueba | Estado real | Ventana estimada | Duracion | Condicion |
|---:|---|---|---|---:|---|
| 0 | FIL-002 ms_learned | HECHA rc=0; PQ posprocesado 0,4061 | 20:39-22:07 | 1 h 29 min | resultado valido; turno liberado |
| 1 | FIL-002 ms_fsr | ACTIVA desde 22:08; 1000/8000 a las 22:11 | 22:08-22:50/23:00 | ~45-55 min total | GPU exclusiva de Claude |
| 2 | FIL-002 ms_bilinear | prevista tras FSR | 23:00-23:50/00:00 | ~45-60 min | control obligatorio; misma receta y presupuesto |
| 3 | Comparacion FIL-002 | prevista, CPU | 00:00-00:30 | ~30 min | pausa GPU; comparar PQ, SQ, RQ, TP/FP/FN y ruido |
| 4 | FIL-003 posprocesado por pliegues | lista, no encolada | desde 00:30 | ~2-3 h | ejecutar solo con RAM libre >=6 GiB; no necesita GPU larga |
| 5 | FIL-INSTANCIAS-001 | condicional, no encolada | despues de FIL-002/FIL-003 | diseño + ~1,5-2 h por corrida | activar solo si multiescala no supera el ruido |
| 6 | SOIL-002 leave-one-phone-out/hibrido | pendiente, no encolada | despues del bloque Filamentos | por medir | primero auditoria determinista; GPU solo con protocolo validado |

Acuerdo operativo Codex-Claude 22:13: Claude conserva la GPU de forma exclusiva hasta cerrar FSR
y bilinear. Codex no lanza GPU y solo prepara trabajo CPU cuando RAM libre sea >=6 GiB. Tras
bilinear se fuerza una pausa de comparacion antes de reservar la siguiente corrida larga. Coincide
con la politica JEV `single_gpu_queue`.

Actualizacion 2026-09-28 10:07: por instruccion de Fran, Codex deja de usar CPU/RAM local para
pruebas pesadas. El PC queda para coordinacion ligera y la RTX 3090 conserva la cola exclusiva de
Claude. JEV conectado (`provenance=jev`, jev-1.13.0) asigna FIL-003 y SOIL-002 a Lightning CPU
(confianzas 0,97 y 0,98) y la replica fuerte de NCA-001 a Colab GPU (0,95). Ninguna figura ACTIVA
hasta que se publique su reserva de inicio, evitando duplicacion.

## Activacion de computo externo — 2026-09-28 08:30 Europe/Madrid

- Fran autoriza usar recursos GPU/TPU externos gratuitos y cuentas existentes, manteniendo los
  experimentos reproducibles y sin exponer credenciales.
- Acceso a Google Colab verificado en navegador con una sesion ya iniciada. Existen cuadernos previos
  de compatibilidad GPU/TPU, incluido uno de Solar Filament; no se creo cuenta, no se subieron datos
  y no se consumio acelerador en esta comprobacion.
- Orden operativo provisional: (1) Colab para entrenamiento o una semilla independiente; (2) Lightning
  si hay creditos; (3) otras nubes gratuitas reproducibles; (4) ZeroGPU solo para humo/inferencia de
  minutos. Kaggle queda fuera de esta cola de entrenamiento: solo ejecucion final y entrega.
- Minimax, Kimi, Manus u otros agentes con computo propio se reservan para una tarea acotada cuando
  puedan devolver codigo, logs, checkpoint y resultado verificable. Su respuesta no cuenta como mejora
  si no entrega artefactos reproducibles.
- Antes de la primera corrida externa: preparar un paquete portable sin secretos, comprobar las reglas
  del concurso y registrar plataforma, cuenta/perfil no sensible, acelerador, cuota, duracion y salida.
- JEV no estuvo disponible en esta decision (`remote_decision=false`, fallback local, esfuerzo alto).
  Reconsultar antes de elegir la primera corrida larga externa.

### Reserva externa viva

| Plataforma | Tarea | Responsable | Inicio | Recurso | Condicion de cierre |
|---|---|---|---|---|---|
| Google Colab | NCA-001 destilacion temporal 24->6 | Codex | 2026-09-28 08:45 | GPU gratuita, tipo por confirmar | guardar resultado/checkpoint y liberar runtime; Claude no duplica |
| Lightning Studio | NCA-002 lienzo reversible | Codex | 2026-09-28 09:31-09:41 | Tesla T4 16 GB; cerrada | tres gates superados; estudio devuelto a 4 x CPU gratuita |
| Lightning Studio | FIL-SEED-001 semillas 1-2 | Claude | cerrada 2026-09-28 11:15 | sin resultado; L4 403, T4 BF16 >64 min sin llegar a 1k | datos borrados del studio; semillas pasan a la cola local tras learned+4k |
| Lightning Studio | FIL-003 y despues SOIL-002 | Codex | programadas, sin runtime activo | CPU gratuita externa | reclamar cada ejecucion antes de iniciarla; no usar CPU local |
| Google Colab | NCA-001 replica con profesor fuerte | Codex | programada, sin runtime activo | GPU gratuita por confirmar | empaquetar checkpoint, reclamar y liberar al terminar; no duplicar |
| Google Colab | FIL-006 ms_learned_cont | Codex | detenida a 1k ~13:56 | T4 sin entrenamiento; runtime conservado para transferencia | checkpoints 235/131 KiB y ZIP 337 KiB existen; descarga local bloqueada por dialogo Guardar de Chrome; no liberar hasta preservar |

Cierre 09:05: Tesla T4 verificada; dos corridas completadas en ~8,5 min totales. No hay computo activo.
Cuaderno guardado en Drive; resultado factual copiado a `.cognition/colab/nca001_run2_result.json`.

Reserva Lightning 09:31: estudio verificado en cuenta existente con opciones gratuitas GPU/TPU.
JEV conectado (`provenance=jev`, jev-1.13.0) eligio NCA-002 con confianza 0,99 y esfuerzo alto
con confianza 0,91. No usa datos privados, GPU local ni GPU de Kaggle.

Cierre Lightning 09:41: Tesla T4 de 16 GB usada brevemente con saldo gratuito (5 creditos iniciales,
tarifa 0,55 creditos/h facturada por segundo). NCA-002 termino y el estudio volvio a 4 x CPU gratuita;
no queda GPU Lightning consumiendo saldo. Resultado en `.cognition/lightning/nca002_result.json`.

## Contencion de RAM — 2026-09-28 14:46 Europe/Madrid

Tras el reinicio se midieron 23,7 GiB de RAM total y 7,36 GiB libres. JEV remoto valido
(`status=connected`, `provenance=jev`, jev-1.13.0) eligio `finish_fil005_then_idle` con confianza
0,97 y una guardia de 8 GiB libres con confianza 0,58.

- Solo FIL-005 conserva reserva: PID 43900, aproximadamente 1,64 GiB RAM y 11.611 MiB VRAM.
- Al cerrar FIL-005, reposo. No autoarrancar FIL-SEED-001 ni ninguna otra prueba.
- FIL-003, SOIL-002, NCA-001 fuerte, Colab y Lightning quedan en cola sin runtime reservado.
- No abrir navegadores o sesiones externas nuevas durante la contencion.
- El task Traffic Flow debe cerrar en checkpoint seguro y quedar en cola.

## Reanudacion autorizada — 2026-09-28 15:44 Europe/Madrid

Fran levanta la pausa tras coordinarse con Claude. La guardia de 8 GiB y `single_gpu_queue` siguen
vigentes. Estado observado: FIL-ENS-001 usa la GPU con tope 6 GiB y FIL-007 construye cache CPU con
4 hilos/~1 GiB; RAM libre 7,12 GiB, por lo que no se admite una tercera carga. JEV valido
(`provenance=jev`) prioriza FIL-007 (0,96), pero exige piloto 1k antes de autorizar 8k (0,87).

