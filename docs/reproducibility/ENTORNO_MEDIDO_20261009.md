# Entorno medido de NeuroPixel (2026-10-09)

Estado: **medido en esta máquina; no probado desde cero.** La instalación limpia en un entorno nuevo todavía no se ha hecho. Este documento describe el entorno en el que se generaron los resultados, no garantiza que otra máquina reproduzca los mismos números.

## Hardware y sistema

| Elemento | Valor medido |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GB VRAM |
| Driver NVIDIA | 581.29 |
| CUDA (runtime reportado por el driver) | 13.0 |
| CUDA con el que se compiló torch | 12.4 |
| Sistema operativo | Windows 10 Pro 10.0.19045 |
| Python | 3.13.7 |
| RAM total | 23,7 GiB (visible para el sistema) |

## Dependencias

`requirements-measured-20261009.txt` (raíz del repositorio) recoge las versiones instaladas de los paquetes que usan los scripts y los tests: torch 2.6.0+cu124, numpy 2.2.6, psutil 6.1.1, matplotlib 3.10.0, pillow 10.4.0, scipy 1.15.1, scikit-image 0.25.2, scikit-learn 1.4.0, networkx 3.4.2, opencv-python 4.12.0.88, pandas 2.2.3, sympy 1.13.1, pytest 9.1.1, y las dependencias del visor (glfw 2.10.0, moderngl 5.12.0).

Torch con CUDA 12.4 requiere el índice de PyTorch para esa variante (`--index-url https://download.pytorch.org/whl/cu124`); el fichero no lo incluye.

## Condiciones que condicionan la ejecución

- **Guard RAM8.** Los scripts de medición rechazan arrancar con menos de 8 GiB de RAM disponible (`psutil.virtual_memory().available`). En la sesión del 2026-10-09 la RAM libre estuvo entre 4 y 5 GiB, por debajo de ese umbral.
- **Energía.** El alcance de energía medido por NVML es el dispositivo GPU completo. Excluye CPU, RAM, PSU y energía de pared. No hay medidor de pared en este equipo.
- **Contadores de Windows.** El "Medidor de energía" aparece en `typeperf -q`, pero no se puede leer sin permisos de rendimiento elevados. No se ha cambiado ninguna configuración del sistema para ello.

## Pendiente de este documento

- Instalación limpia desde cero en un entorno nuevo, con registro de cada paso y de los fallos.
- Prueba de los tests (`tests/test_core.py`, 13 tests recogidos, no ejecutados en esta sesión).
- Verificación de hashes de las entradas del protocolo R11 tras la instalación.
