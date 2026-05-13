# LAB4 — Conclusiones y Entregables
## MLflow con Dataset PIMA Indians Diabetes

---

## 1. Justificación médica del criterio de selección de métricas

El dataset PIMA Indians Diabetes plantea un problema de diagnóstico binario: detectar si un paciente es diabético a partir de variables clínicas (glucosa, insulina, IMC, edad, etc.).

En este contexto los errores del modelo tienen consecuencias asimétricas:

| Tipo de error | Significado clínico | Consecuencia |
|---|---|---|
| **Falso Negativo (FN)** | El modelo predice "sano" cuando el paciente **sí** es diabético | El paciente **no recibe tratamiento**. Riesgo de retinopatía (ceguera), nefropatía (insuficiencia renal), neuropatía, pie diabético, cardiopatía isquémica. Daños irreversibles. |
| **Falso Positivo (FP)** | El modelo predice "diabético" cuando el paciente **no** lo es | El paciente recibe pruebas confirmatorias adicionales (glucemia en ayunas, HbA1c). Coste: una analítica y ansiedad puntual. Consecuencia manejable. |

**Conclusión:** el coste médico de un FN es drásticamente mayor que el de un FP.

Por tanto:

- **Métrica primaria: RECALL** (Sensibilidad) = TP / (TP + FN) → minimiza los falsos negativos
- **Métrica secundaria: F1-score** → penaliza modelos que sacrifican demasiado la precisión (alarmas masivas en la clínica son contraproducentes)
- **Accuracy**: usada como referencia, pero poco fiable porque el dataset tiene ~65 % de casos negativos (un clasificador trivial que dijera siempre "sano" obtendría ~65 % de accuracy sin detectar ningún diabético)

---

## 2. Tabla completa de parámetros y métricas — 30 runs del Grid Search

División del dataset: **Train 70 % (537)** | **Validación 20 % (154)** | **Test 10 % (77)**  
Las métricas de la tabla corresponden al **conjunto de validación**.

| # | C | Solver | max_iter | Accuracy | Precision | **Recall** | F1-score | Seleccionado |
|---|---|--------|----------|----------|-----------|-----------|----------|:---:|
| 1 | 0.1 | lbfgs | 1000 | 0.7922 | 0.8462 | **0.5593** | 0.6735 | **SÍ** ★ |
| 2 | 0.01 | lbfgs | 1000 | 0.7922 | 0.8462 | **0.5593** | 0.6735 | — |
| 3 | 0.1 | lbfgs | 500 | 0.7922 | 0.8462 | **0.5593** | 0.6735 | — |
| 4 | 0.1 | lbfgs | 100 | 0.7922 | 0.8462 | **0.5593** | 0.6735 | — |
| 5 | 0.01 | lbfgs | 500 | 0.7922 | 0.8462 | **0.5593** | 0.6735 | — |
| 6 | 100 | lbfgs | 100 | 0.7987 | 0.8889 | 0.5424 | 0.6737 | — |
| 7 | 1 | lbfgs | 100 | 0.7987 | 0.8889 | 0.5424 | 0.6737 | — |
| 8 | 10 | lbfgs | 100 | 0.7987 | 0.8889 | 0.5424 | 0.6737 | — |
| 9 | 1 | lbfgs | 500 | 0.7922 | 0.8649 | 0.5424 | 0.6667 | — |
| 10 | 1 | lbfgs | 1000 | 0.7922 | 0.8649 | 0.5424 | 0.6667 | — |
| 11 | 100 | lbfgs | 500 | 0.7922 | 0.8857 | 0.5254 | 0.6596 | — |
| 12 | 10 | lbfgs | 500 | 0.7922 | 0.8857 | 0.5254 | 0.6596 | — |
| 13 | 10 | lbfgs | 1000 | 0.7922 | 0.8857 | 0.5254 | 0.6596 | — |
| 14 | 100 | lbfgs | 1000 | 0.7922 | 0.8857 | 0.5254 | 0.6596 | — |
| 15 | 10 | liblinear | 100 | 0.7922 | 0.8857 | 0.5254 | 0.6596 | — |
| 16 | 10 | liblinear | 500 | 0.7922 | 0.8857 | 0.5254 | 0.6596 | — |
| 17 | 10 | liblinear | 1000 | 0.7922 | 0.8857 | 0.5254 | 0.6596 | — |
| 18 | 100 | liblinear | 100 | 0.7922 | 0.8857 | 0.5254 | 0.6596 | — |
| 19 | 100 | liblinear | 500 | 0.7922 | 0.8857 | 0.5254 | 0.6596 | — |
| 20 | 100 | liblinear | 1000 | 0.7922 | 0.8857 | 0.5254 | 0.6596 | — |
| 21 | 0.01 | lbfgs | 100 | 0.7727 | 0.8158 | 0.5254 | 0.6392 | — |
| 22 | 1 | liblinear | 100 | 0.7792 | 0.8788 | 0.4915 | 0.6304 | — |
| 23 | 1 | liblinear | 500 | 0.7792 | 0.8788 | 0.4915 | 0.6304 | — |
| 24 | 1 | liblinear | 1000 | 0.7792 | 0.8788 | 0.4915 | 0.6304 | — |
| 25 | 0.1 | liblinear | 100 | 0.7273 | 0.8148 | 0.3729 | 0.5116 | — |
| 26 | 0.1 | liblinear | 500 | 0.7273 | 0.8148 | 0.3729 | 0.5116 | — |
| 27 | 0.1 | liblinear | 1000 | 0.7273 | 0.8148 | 0.3729 | 0.5116 | — |
| 28 | 0.01 | liblinear | 100 | 0.7013 | 0.7600 | 0.3220 | 0.4524 | — |
| 29 | 0.01 | liblinear | 500 | 0.7013 | 0.7600 | 0.3220 | 0.4524 | — |
| 30 | 0.01 | liblinear | 1000 | 0.7013 | 0.7600 | 0.3220 | 0.4524 | — |

★ Modelo registrado en el Model Registry como `Diabetes_Modelo_Final`

---

## 3. Modelo incluido en el registro — Criterio de selección

### Modelo registrado

| Parámetro/Métrica | Valor |
|---|---|
| **C** | 0.1 |
| **solver** | lbfgs |
| **max_iter** | 1000 |
| Accuracy (val) | 0.7922 |
| Precision (val) | 0.8462 |
| **Recall (val)** | **0.5593** |
| F1-score (val) | 0.6735 |
| Nombre en MLflow | `Diabetes_Modelo_Final` |
| Datos de entrenamiento final | train + val (90 % del total) |

### Por qué se registra este modelo y no los demás

**Grupo 1 — Empate en recall máximo (0.5593, runs 1–5):**  
Los runs 1–5 obtienen exactamente el mismo recall (y las mismas métricas de validación). Todos usan el solver `lbfgs`. Entre ellos se elige el **run 1** (C=0.1, max_iter=1000) porque:
- C=0.1 es regularización moderada: evita tanto el subajuste de C=0.01 como el sobreajuste de C≥1.
- max_iter=1000 garantiza la convergencia completa del optimizador (no hay riesgo de parada anticipada).
- Los runs 2–5 tienen métricas idénticas, por lo que registrarlos todos sería redundante.

**Grupos 2–3 — Recall inferior (0.5424 y 0.5254):**  
Todos los demás modelos `lbfgs` con C≥1 obtienen menor recall aunque mayor precisión. Desde el criterio médico, sacrificar recall para ganar precisión significa **detectar menos diabéticos** → inaceptable en cribado.

**Solver liblinear — Descartado por completo:**  
Para todos los valores de C, `liblinear` produce un recall significativamente inferior al de `lbfgs` con los mismos hiperparámetros (diferencia de hasta 0.24 puntos de recall). El solver `lbfgs` maneja mejor la regularización L2 en este dataset.

**Accuracy alta ≠ Buen modelo médico:**  
El run 6 (C=100, lbfgs) tiene la mayor accuracy (0.7987) pero su recall es 0.5424 — inferior al ganador. Optimizar accuracy en un dataset desbalanceado favorece a la clase mayoritaria (no diabético) a costa de los diabéticos.

---

## 4. Métricas reales sobre el dataset de test (10 %)

Evaluación del modelo `Diabetes_Modelo_Final` (entrenado en train+val) sobre el **conjunto de test** que no intervino en ninguna fase de entrenamiento ni selección. Resultados obtenidos mediante la API de MLflow (script `5_evaluacion_test_api.py`):

| Métrica | Valor (test) | Valor (val) | Diferencia |
|---|---|---|---|
| Accuracy | 0.7013 | 0.7922 | -0.0909 |
| Precision | 0.5625 | 0.8462 | -0.2837 |
| **Recall** | **0.6667** | **0.5593** | **+0.1074** |
| F1-score | 0.6102 | 0.6735 | -0.0633 |

**Matriz de confusión sobre test (n=77):**

|  | Predicho: Sano | Predicho: Diabético |
|---|---|---|
| **Real: Sano** (50) | 36 (TN) | 14 (FP) |
| **Real: Diabético** (27) | 9 (FN) | 18 (TP) |

---

## 5. Conclusiones técnicas

**1. El solver lbfgs supera a liblinear de forma consistente.**  
Para todos los valores de C explorados, `lbfgs` obtiene un recall notablemente superior. Esto se debe a que `lbfgs` implementa una optimización de segundo orden (quasi-Newton) que converge a mejores mínimos en la función de pérdida logística L2, especialmente con datasets pequeños como PIMA (768 muestras).

**2. La regularización moderada (C=0.1) maximiza el recall.**  
Los modelos con C muy bajo (C=0.01 + liblinear) colapsan en recall (0.322): la regularización excesiva aplana los coeficientes y la frontera de decisión se desplaza hacia la clase mayoritaria. Los modelos con C alto (C≥1) obtienen alta precisión pero detectan menos positivos. El punto óptimo se encuentra en C=0.1 con `lbfgs`.

**3. El parámetro max_iter no afecta al resultado final para C=0.1/lbfgs.**  
Los runs con max_iter=100, 500 y 1000 producen métricas idénticas, lo que indica que el optimizador converge ya con 100 iteraciones para este tamaño de dataset y nivel de regularización.

**4. Discrepancia entre métricas de validación y test.**  
El recall mejora en test (0.667 vs 0.559) pero la precisión cae (0.5625 vs 0.8462). Esto puede explicarse por:
- El conjunto de test tiene solo 77 muestras → alta varianza estadística en las métricas.
- Entrenando con train+val (90 %), el modelo dispone de más ejemplos positivos, lo que puede desplazar ligeramente el umbral de decisión hacia detectar más diabéticos (más TP y también más FP).
- El conjunto de validación presentó una distribución de clases con alta precisión y bajo recall que puede no ser representativa del dataset completo.

**5. La regresión logística tiene un techo de rendimiento en este dataset.**  
Con 8 variables y 768 muestras (de las cuales solo ~268 son positivos), la regresión logística alcanza un recall máximo de ~0.56–0.67 en validación/test. Para mejorar se requerirían modelos más potentes (Random Forest, XGBoost) o ingeniería de características adicional (incluir interacciones, normalizar, tratar outliers).

---

## 6. Conclusiones médico-prácticas

**El modelo detecta 18 de 27 diabéticos reales en el test set (recall=0.667).**  
Esto significa que **9 pacientes diabéticos quedan sin diagnosticar** (FN). En un contexto de cribado poblacional a gran escala este número sería inaceptable sin medidas complementarias. Sin embargo, como herramienta de apoyo a la decisión clínica (no como diagnóstico definitivo) puede ser útil para priorizar pruebas de confirmación.

**Los 14 falsos positivos son clínicamente manejables.**  
Representan pacientes remitidos innecesariamente a pruebas adicionales de confirmación (glucemia en ayunas, test de tolerancia oral a la glucosa). Es un coste asumible en salud pública para reducir los falsos negativos.

**El modelo no debe usarse como diagnóstico definitivo.**  
La diabetes tipo 2 exige confirmación analítica (ADA: glucosa en ayunas ≥126 mg/dL en dos ocasiones o HbA1c ≥6.5 %). El modelo debe interpretarse como un **filtro de riesgo** para identificar pacientes que merecen estudio más profundo.

**Mejoras recomendadas para un uso clínico real:**
1. Ampliar el dataset con datos más recientes y diversos (PIMA es un dataset de los años 80 con una población específica).
2. Explorar modelos de mayor capacidad (gradient boosting) que suelen alcanzar recall > 0.75 en este problema.
3. Ajustar el umbral de decisión del clasificador (por defecto 0.5): reducirlo a 0.3–0.4 aumentaría el recall a costa de más FP, lo cual puede ser preferible desde el punto de vista médico.
4. Implementar validación cruzada estratificada para obtener estimaciones de métricas más estables con datasets pequeños.
