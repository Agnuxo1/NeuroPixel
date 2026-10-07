# Recursos y reservas

Actualizada: 2026-09-27 19:20 Europe/Madrid.

## Inventario local

| Recurso | Capacidad | Margen operativo acordado |
|---|---:|---|
| GPU | RTX 3090, 24.576 MiB | reservar 3,5 GiB para pantalla/sistema; no superar 83 C sostenidos |
| RAM | 23,7 GiB | conservar al menos 6 GiB disponibles; parar lanzamientos nuevos por debajo de 5 GiB |
| CPU | 20 hilos logicos | maximo 8 hilos por agente; total habitual <=16; bajar a 4 con GPU saturada |
| Disco de trabajo | D: y E: | datos, caches, entornos, modelos, temporales y resultados permanecen en D: o E:; evitar C: |

## Reservas activas

| Recurso | Tarea | Responsable | Inicio conocido | Limite | Liberacion esperada | Estado |
|---|---|---|---|---|---|---|
| GPU | SOIL-001 | Claude | antes de 19:04 | 6 GiB | al completar folds 4-5 | liberada 20:13 (6/6 folds; result.json) |
| GPU | FIL-001 | Claude | antes de 17:42 | 14 GiB | al completar 6000 it | activa, 3000/6000 registrados |
| GPU | FIL-002 | Claude | en cola gpuq desde 19:36 | 10 GiB, RAM 4 GiB | 3 corridas x ~2 h (learned, fsr, bilinear), secuenciales | pendiente: arranca cuando gpuq vea la GPU libre (tras FIL-001) |
| RAM/CPU | ambas | PREEXISTENTE | — | compartido | junto con GPU | RAM ya bajo el margen objetivo |

Las dos reservas actuales superponen 20 GiB nominales y dejan poco margen. Se toleran solo porque
ya estaban ejecutandose. A partir de su cierre: una sola carga GPU larga a la vez. Excepcion: una
prueba corta de menos de 10 minutos si la suma medida queda por debajo de 20 GiB y la GPU por debajo
de 80 C antes de empezar.

## Protocolo de reserva

Antes de ejecutar, añadir una fila con tarea, agente, VRAM/RAM/hilos, hora, duracion estimada y
condicion de corte. Renovar cada 90 minutos con un hito. Al acabar, marcar `liberada` y enlazar el
resultado. No finalizar procesos de otro agente sin autorizacion de Fran.

## Computo externo gratuito o promocional

| Plataforma | Oferta oficial observada | Uso recomendado | Limitacion clave |
|---|---|---|---|
| Kaggle Notebooks | P100 gratis; cuota semanal normalmente 30 h o variable | entrenamientos reproducibles ligados al concurso | cuota compartida y disponibilidad no garantizada |
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
aprobada y registro de coste/cuota. Preferencia: RTX 3090 local, luego Kaggle; nube adicional solo
cuando permita paralelizar una prueba independiente y reproducible.

Variables de cache y temporales de herramientas nuevas deben apuntar a una carpeta identificada en
D: o E:. Antes de una descarga pesada se registra origen, tamaño aproximado y destino. No se limpia
C: ni se mueve contenido existente sin autorizacion expresa de Fran.


## 2026-10-07 — SCI-006 isolated CPU preflight reservation

A standard GitHub-hosted ubuntu-24.04 runner in this public repository is reserved for one sequential preflight, at most 30 minutes. Two CPU threads, one inter-op thread and at least 8 GiB available RAM are required. No GPU job, paid runner, billing change, artifact storage upload or dependency cache is used. Work stays in the isolated Linux runner workspace; the Windows D:/E: storage rule concerns the disconnected Windows device. Standard public runner CPU time is free under the official documentation checked for this action. Source/environment validation precedes every affected scientific run. This preflight does not authorize the pending item-6 training freeze. The runner releases its resources automatically on completion; operational status is visible in this branch's Actions run.

Sources: https://docs.github.com/en/actions/reference/runners/github-hosted-runners and https://docs.github.com/en/billing/concepts/product-billing/github-actions .
