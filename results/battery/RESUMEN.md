# Batería nocturna NeuroPixel

## T1 Autorreparación (acierto con combinaciones nuevas)

| ejecución | f0.0_capa1 | f0.0_t4 | f0.0_t4_+8pasos | f0.0_t8 | f0.0_t8_+8pasos | f0.1_capa1 | f0.1_t4 | f0.1_t4_+8pasos | f0.1_t8 | f0.1_t8_+8pasos | f0.3_capa1 | f0.3_t4 | f0.3_t4_+8pasos | f0.3_t8 | f0.3_t8_+8pasos | f0.5_capa1 | f0.5_t4 | f0.5_t4_+8pasos | f0.5_t8 | f0.5_t8_+8pasos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu30k_np_s0 |  | 0.9325 | 0.7175 | 0.9325 | 0.7175 |  | 0.911 | 0.704 | 0.913 | 0.701 |  | 0.858 | 0.6615 | 0.8735 | 0.645 |  | 0.804 | 0.622 | 0.833 | 0.6005 |
| gpu30k_np_s1 |  | 0.9315 | 0.7565 | 0.9315 | 0.7565 |  | 0.92 | 0.749 | 0.9235 | 0.7495 |  | 0.884 | 0.7095 | 0.8925 | 0.708 |  | 0.835 | 0.663 | 0.8495 | 0.684 |
| gpu30k_np_s2 |  | 0.9535 | 0.583 | 0.9535 | 0.583 |  | 0.945 | 0.5765 | 0.947 | 0.587 |  | 0.91 | 0.5685 | 0.9165 | 0.591 |  | 0.8875 | 0.57 | 0.891 | 0.596 |
| gpu30k_np_lens_s0 |  | 0.9795 | 0.886 | 0.9795 | 0.886 |  | 0.977 | 0.886 | 0.981 | 0.8855 |  | 0.959 | 0.8585 | 0.9685 | 0.861 |  | 0.9265 | 0.7925 | 0.94 | 0.816 |
| gpu30k_tf_s0 | 0.5455 |  |  |  |  | 0.498 |  |  |  |  | 0.4035 |  |  |  |  | 0.3085 |  |  |  |  |
| gpu30k_tf_s1 | 0.551 |  |  |  |  | 0.501 |  |  |  |  | 0.4045 |  |  |  |  | 0.312 |  |  |  |  |
| gpu30k_tf_s2 | 0.5535 |  |  |  |  | 0.489 |  |  |  |  | 0.382 |  |  |  |  | 0.2725 |  |  |  |  |

## T2 Palabra nueva ('delfín'), solo diccionario

| ejecución | antes_viejo | k1_nueva | k1_viejo | k5_nueva | k5_viejo | k20_nueva | k20_viejo |
|---|---|---|---|---|---|---|---|
| gpu30k_np_s0 | 0.9325 | 0.062 | 0.9305 | 0.955 | 0.856 | 0.873 | 0.873 |
| gpu30k_np_lens_s0 | 0.9795 | 0.6 | 0.9645 | 0.945 | 0.949 | 0.977 | 0.9225 |
| gpu30k_tf_s0 | 0.5455 | 0.14 | 0.542 | 0.46 | 0.543 | 0.732 | 0.536 |

## T3 Aprendizaje continuo y lienzos que crecen

| modelo | A_tras_A | B_tras_A | secuencial_A_tras_B | secuencial_B_tras_B | crecer_A | crecer_A_elige_lienzo_correcto | crecer_B | crecer_B_elige_lienzo_correcto | segundos |
|---|---|---|---|---|---|---|---|---|---|
| neuropixel | 0.9635 | 0.3835 | 0.477 | 0.9715 | 0.903 | 0.874 | 0.889 | 0.6 | 1363.7 |
| transformer | 0.6085 | 0.488 | 0.507 | 0.686 | 0.536 | 0.825 | 0.6185 | 0.61 | 287.4 |

Coherencia de cúmulos durante el aprendizaje de A (neuropixel): 2000: 0.1962, 4000: 0.2442, 6000: 0.2146, 8000: 0.2322, 10000: 0.2195, 12000: 0.2322, 14000: 0.2244, 16000: 0.2255, 18000: 0.2147, 20000: 0.2169

## T4 Consultar todos los lienzos / enrutar por escáner

```
{
 "neuropixel": {
  "A_confianza": 0.903,
  "A_fusion": 0.9015,
  "A_oraculo": 0.9635,
  "A_escaner": 0.9345,
  "A_escaner_elige_bien": 0.9605,
  "A_coherencia_lienzoA_lienzoB": [
   0.2264,
   0.3116
  ],
  "B_confianza": 0.889,
  "B_fusion": 0.915,
  "B_oraculo": 0.9715,
  "B_escaner": 0.9635,
  "B_escaner_elige_bien": 0.9905,
  "B_coherencia_lienzoA_lienzoB": [
   0.2237,
   0.3045
  ]
 },
 "transformer": {
  "A_confianza": 0.536,
  "A_fusion": 0.537,
  "A_oraculo": 0.6085,
  "B_confianza": 0.6185,
  "B_fusion": 0.6185,
  "B_oraculo": 0.686
 },
 "gpu_todos_a_la_vez": {
  "N2": {
   "una_pasada_ms": 23.3,
   "N_pasadas_ms": 24.2
  },
  "N8": {
   "una_pasada_ms": 81.6,
   "N_pasadas_ms": 92.7
  },
  "N32": {
   "una_pasada_ms": 321.7,
   "N_pasadas_ms": 370.8
  }
 }
}
```
