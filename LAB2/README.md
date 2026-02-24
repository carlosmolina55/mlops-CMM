# 🏭 Predicción de Fallos en Productos Industriales y Detección de Data Drift (MLOps)

**Autor:** Carlos Molina Martínez  
**Asignatura:** MLOps (Machine Learning Operations)

---

## 📖 Descripción del Proyecto
Este proyecto aborda un problema clásico de Machine Learning aplicado a la industria: predecir si un producto fallará basándose en sus características y pruebas de laboratorio sometidas a estrés (absorción de fluidos en un entorno simulado). 

Más allá de la simple predicción, el proyecto tiene un fuerte enfoque en **MLOps**, evaluando la calidad y homogeneidad de los datos a través de técnicas de **Validación Adversaria** para detectar el *Data Drift* (desviación de datos) entre los entornos de entrenamiento y producción.

---

## 📊 Fases del Proyecto

### 1. Análisis Exploratorio de Datos (EDA)
Antes del modelado, se realizó un EDA para comprender la naturaleza de los datos, descubriendo tres puntos críticos que guiaron el resto de la práctica:
* **Falta de solapamiento:** Los códigos de producto (`product_code`) en el set de entrenamiento (A-E) son completamente distintos a los de test/producción (F-I). Esto obligó a descartar esta variable para evitar el sobreajuste.
* **Valores nulos:** Presencia de hasta un 8.5% de valores faltantes en las mediciones de los sensores, lo que motivó la elección de algoritmos robustos ante datos incompletos.
* **Desbalanceo de clases:** La gran mayoría de los productos no fallan, por lo que se descartó el *Accuracy* como métrica principal en favor del **ROC-AUC**.

### 2. Tarea 1: Modelo Predictivo (LightGBM)
Para predecir la probabilidad de fallo, se implementó un modelo basado en Gradient Boosting (**LightGBM**). Sus ventajas para este caso de uso incluyen:
* Manejo nativo de valores nulos sin necesidad de imputación previa.
* Soporte nativo y eficiente para variables categóricas.
* Entrenamiento estable evaluado mediante **Validación Cruzada Estratificada (5 Folds)**, garantizando la robustez del modelo y promediando las predicciones finales para el conjunto de test (*Ensembling*).

### 3. Tarea 2: Detección de Data Drift (Validación Adversaria)
Para comprobar si el modelo es apto para producción o si los datos han cambiado (*Data Drift*), se implementó un modelo de validación adversaria. 
* **Metodología:** Se unieron los datos de *Train* y *Test*, etiquetándolos con `0` y `1` respectivamente, y se entrenó un clasificador para intentar distinguirlos.
* **Hallazgos:** Se detectó un *Data Drift* extremo (ROC-AUC cercano a 1.0) causado por "variables proxy". Los atributos fijos (`attribute_2`, `attribute_3`) delataban a qué código de producto pertenecía cada fila. 
* **Decisión de MLOps:** Para que el modelo predictivo sea verdaderamente generalizable en producción, debe ser entrenado **únicamente con las mediciones dinámicas del laboratorio** (`measurements` y `loading`), eliminando las variables estáticas que atan al modelo a un lote de productos específico.

---

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python
* **Manipulación de Datos:** Pandas, NumPy
* **Visualización:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, LightGBM