# Reproducción de la receta DEV04

Este documento reproduce únicamente DEV04. El replay comprueba checkpoints ya entrenados y no realiza updates. El cierre científico exige doce casos, veinticuatro lecturas, todos los endpoints, ejecuciones terminales y proofs vinculados a los originales. La tarea 3 amplia y la replicación externa requieren verificación separada.

## Entorno

La referencia específica es `DEV04_environment_reference.json`, derivada de registros originales. Python **3.12.14**; Torch **2.6.0+cpu**; NumPy **2.2.6**; SciPy **1.15.1**; psutil **6.1.1**; Pillow **11.3.0**. El registro científico incluye `pytest: null`: los contratos usan unittest. `container_environment.json` documenta otro entorno histórico y se conserva separado.

Priorizar Ubuntu 24.04 x86_64, como los ejecutores registrados. Cada replay fija dos threads y algoritmos deterministas. Mantener al menos **8 GiB disponibles** antes de cualquier carga neuronal; el guard rechaza la admisión si falta RAM. No rebajar el guard ni modificar versiones para obtener un replay favorable. Las diferencias de host se registran; las decisiones siguen siendo exactas y el error de NLL media debe ser ≤1e-4.

En un entorno aislado creado con Python 3.12.14:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.2.6 scipy==1.15.1 psutil==6.1.1 pillow==11.3.0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Antes de ejecutar, conservar las fuentes, el plan, los ZIP originales y los hashes. El plan es `docs/research/DEV04_execution_plan.json`, SHA-256 `2484f3c5917e474ca3755bcdbbaca7d322077615ba47d51653116042d980f20a`. Sus 43 archivos gobernantes deben existir y coincidir byte por byte.

## Datos y procedencia

Los casos son `DEV04_partition101_seed400`–`403`, `DEV04_partition102_seed404`–`407` y `DEV04_partition103_seed408`–`411`. Cada uno tiene cuerpo nuevo, su partición y tres datasets originales, checkpoints de cuerpo a 8.192/16.384 y checkpoints de ambas lecturas a 1.024/4.096/8.192, con optimizadores y streams RNG. Cuerpo, diccionario y decoder se congelan antes de entrenar las lecturas.

Originales: `results/research/DEV04_recovery/37845945011/originals/<case>/`. Cohorte extraída: `results/research/DEV04_recovery/37845945011/cohort/<case>/`. Los recibos identifican la ejecución real de cada caso: nueve provienen de `37845945011`/fuente `b9431e9b927ba5b6539a172d4b27fe6199776462`; los fallos de arranque 400/402/403 se recuperan en `37871777801`/fuente `81cf94e8f2a089b0d695f5fd465c6daaff50c6f1`, con los mismos archivos científicos. No son semillas o réplicas adicionales.

`snapshot_DEV04_git.py` valida el commit fijado por una referencia del conector, sus bytes SHA1 y todo el árbol Merkle en una caché bare aislada. `recover_DEV04_snapshot.py --tree-snapshot <snapshot> --case <case>` verifica blob, ZIP, manifiesto y cada archivo antes de extraer. La opción `--git-object-cache <bare-cache>` permite reutilizar objetos verificados. Cada máquina genera su propio recibo de extracción con su ruta; los originales científicos permanecen inmutables.

Las 264 composiciones del test original se mantienen excluidas. El generador rechaza su muestreo. Este universo de desarrollo ya estuvo expuesto históricamente; las políticas se solapan entre sí. Los cuerpos nuevos impiden reutilizar pesos históricos, pero no convierten el estudio en un test final externo.

## Replay independiente por caso

Ejemplo desde la raíz de este checkout, con RAM y software admitidos:

```bash
python scripts/replay_DEV04.py \
  --case DEV04_partition101_seed400 \
  --case-folder results/research/DEV04_recovery/37845945011/cohort/DEV04_partition101_seed400 \
  --output results/research/DEV04_independent_replay/DEV04_partition101_seed400/replay.json
```

Repetir secuencialmente para los doce casos; no usar `research_DEV04.py` para recalcular los archivos cerrados. Cada replay completo exige **811.008 decisiones**; el total es **9.732.096**. Es un denominador de verificación, no un número de réplicas. El replay comprueba inicialización, pertenencias, regeneración de datasets, finitud, optimizadores/RNG, todos los endpoints, igualdad de controles ACTO/LUGAR, mismo stream de batches/máscaras en las dos lecturas y conservación de los originales. Los fallos se conservan y diagnostican; no se amplía la tolerancia ni se reemplaza el resultado discrepante.

La verificación automática interna usa `verify_DEV04_archived_cases.py`, exige que termine su observador predecesor y admite como máximo un entrenamiento activo junto a un único replay en serie y publica proofs inmutables por caso. Su fuente operativa registrada es `0ef7446ff8a0b872b3d2f85d5deb11d387ac4375`, ejecución `37876687599`. Un replay hecho por este proyecto sigue siendo verificación interna, aunque el código del recuento sea distinto. El observador previo 37872427770/fuente 125feea39a8c940a7a911b453f11b6e0db0c3fd6 se canceló sin proofs neuronales cerrados; su recibo se conserva. El caso 403 se verifica al final por disponibilidad, sin selección por rendimiento; los doce casos permanecen obligatorios antes de efectos. Una reproducción externa requiere responsable, entorno, ejecución y resultados obtenidos independientemente y documentados.

## Análisis y cierre

Tras recuperar la cohorte y completar su replay, conservar un snapshot Git del verificador y un snapshot del conector con `runs` y `jobs` para las tres ejecuciones. Ejecutar:

```bash
python scripts/collect_DEV04_replay_receipts.py \
  --tree-snapshot <verifier-tree.json> \
  --runs-snapshot <terminal-runs-and-jobs.json> \
  --verification-run 37876687599 \
  --verification-source 0ef7446ff8a0b872b3d2f85d5deb11d387ac4375
python scripts/summarize_DEV04.py \
  --verification-run 37876687599 \
  --verification-source 0ef7446ff8a0b872b3d2f85d5deb11d387ac4375
python scripts/render_DEV04_report.py
python scripts/close_DEV04.py
```

El collector rechaza workflows, fuentes, intentos, jobs o inputs distintos. El analizador verifica los originales y sus proofs antes de estimar efectos. El estimador promedia cuatro diferencias pareadas dentro de cada política y las tres medias con igual peso; ICt95 exploratorio df2 sin truncar. Gate: atención ≥95% en probe y ≥90% en validación en los doce casos. Precisión: semianchura ≤5 pp y límite inferior positivo. Conservar negativos; no aumentar semillas o presupuesto retrospectivamente.

Las figuras requieren matplotlib y son derivados. La tabla y el recuento entero no dependen de ella. El cierre exige los hashes del análisis, informe y recuento ortogonal; no cierra automáticamente la tarea 3 amplia, modifica H1, demuestra energía o certifica originalidad.

Licencia del código: MIT, copyright 2026 Francisco Angulo de Lafuente, texto original en `LICENSE`. Conservar ese archivo y los avisos de dependencias. No se atribuye una licencia inventada a datos o artículos ajenos.
