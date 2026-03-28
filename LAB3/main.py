"""
app-iris-ct: Continuous Training extension for ML-FastAPI-Docker
================================================================
Extends app-iris with:
  - POST /train      → reentrenamiento incremental con nuevas muestras
  - GET  /model/info → versión activa, métricas, historial
  - POST /predict    → inferencia (igual que app-iris, con versión activa)
  - GET  /health     → estado del servicio
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "model_active.joblib"
HISTORY_PATH = MODELS_DIR / "training_history.json"

# ---------------------------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------------------------

class IrisSample(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Longitud del sépalo (cm)")
    sepal_width: float  = Field(..., example=3.5, description="Anchura del sépalo (cm)")
    petal_length: float = Field(..., example=1.4, description="Longitud del pétalo (cm)")
    petal_width: float  = Field(..., example=0.2, description="Anchura del pétalo (cm)")


class LabeledSample(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float  = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float  = Field(..., example=0.2)
    label: int = Field(..., ge=0, le=2, example=0,
                       description="0=setosa, 1=versicolor, 2=virginica")


class TrainRequest(BaseModel):
    samples: List[LabeledSample] = Field(
        ..., min_items=5,
        description="Nuevas muestras etiquetadas para reentrenamiento (mínimo 5)"
    )
    retrain_from_scratch: bool = Field(
        False,
        description="Si True, ignora datos anteriores y entrena solo con las muestras enviadas"
    )

    # --- NUEVOS CAMPOS ---
    policy: str = Field(
        "any_improvement", 
        description="Política de activación: any_improvement, min_delta, per_class_f1"
    )
    min_delta: float = Field(0.0, description="Margen de mejora para min_delta (ej. 0.02)")
    target_class: str = Field("", description="Nombre de la clase a evaluar en per_class_f1 (setosa, versicolor, virginica)")

class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    model_version: str


class TrainResponse(BaseModel):
    status: str
    model_version: str
    accuracy_new: float
    accuracy_previous: Optional[float]
    model_updated: bool
    message: str


class ModelInfo(BaseModel):
    active_version: str
    trained_at: str
    accuracy: float
    n_training_samples: int
    algorithm: str
    history: List[dict]



class ActivationPolicy:
    def __init__(self, policy_type: str, min_delta: float = 0.0, target_class: str = ""):
        self.policy_type = policy_type
        self.min_delta = min_delta
        self.target_class = target_class

    def evaluate_model(self, current_metrics: dict, new_metrics: dict) -> tuple[bool, str]:
        acc_current = current_metrics.get("accuracy", 0.0)
        acc_new = new_metrics.get("accuracy", 0.0)

        # Política 1: Any Improvement
        if self.policy_type == "any_improvement":
            if acc_new >= acc_current:
                return True, f"Activado: El accuracy nuevo ({acc_new:.4f}) iguala o supera al actual ({acc_current:.4f})."
            else:
                return False, f"Rechazado: El accuracy nuevo ({acc_new:.4f}) es menor al actual ({acc_current:.4f})."

        # Política 2: Min Delta (Mejora relativa)
        elif self.policy_type == "min_delta":
            required_limit = acc_current + (acc_current * self.min_delta)
            if acc_new >= required_limit:
                return True, f"Activado: El accuracy nuevo ({acc_new:.4f}) supera el límite requerido ({required_limit:.4f})."
            else:
                return False, f"Rechazado: El accuracy nuevo ({acc_new:.4f}) no alcanza el límite requerido ({required_limit:.4f})."

        # Política 3: Per Class F1-Score
        elif self.policy_type == "per_class_f1":
            f1_current = current_metrics.get("f1_scores", {}).get(self.target_class, 0.0)
            f1_new = new_metrics.get("f1_scores", {}).get(self.target_class, 0.0)
            if f1_new > f1_current:
                return True, f"Activado: F1-score de la clase '{self.target_class}' mejoró de {f1_current:.4f} a {f1_new:.4f}."
            else:
                return False, f"Rechazado: F1-score de '{self.target_class}' no mejoró. Pasó de {f1_current:.4f} a {f1_new:.4f}."
        
        return False, f"Rechazado: Política '{self.policy_type}' desconocida."
# ---------------------------------------------------------------------------
# Utilidades de persistencia
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


def load_history() -> List[dict]:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def save_history(history: List[dict]):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def get_active_model_meta() -> Optional[dict]:
    history = load_history()
    return history[-1] if history else None


# ---------------------------------------------------------------------------
# Bootstrap: si no existe modelo, lo entrenamos con el dataset original
# ---------------------------------------------------------------------------

def bootstrap_model():
    """Entrena un modelo base con el dataset Iris completo al arrancar."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = float(accuracy_score(y_test, clf.predict(X_test)))

    version = "v1.0-base"
    joblib.dump(clf, MODEL_PATH)

    history = [{
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": round(accuracy, 4),
        "n_training_samples": len(X_train),
        "algorithm": "LogisticRegression",
        "source": "bootstrap (iris dataset completo)"
    }]
    save_history(history)
    print(f"[bootstrap] Modelo base creado → versión={version}, accuracy={accuracy:.4f}")


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Iris Continuous Training API",
    description=(
        "Extensión MLOps de app-iris. Sirve predicciones y permite reentrenar "
        "el modelo con nuevas muestras etiquetadas, registrando el historial de versiones."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    if not MODEL_PATH.exists():
        bootstrap_model()
    else:
        meta = get_active_model_meta()
        if meta:
            print(f"[startup] Modelo activo cargado → versión={meta['version']}, "
                  f"accuracy={meta['accuracy']}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Sistema"])
def health():
    meta = get_active_model_meta()
    return {
        "status": "ok",
        "active_model_version": meta["version"] if meta else "none",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inferencia"])
def predict(sample: IrisSample):
    """
    Realiza una predicción con el modelo activo.
    Devuelve la clase predicha, su nombre y la versión del modelo usado.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Modelo no disponible. Llama primero a /train.")

    clf = joblib.load(MODEL_PATH)
    X = np.array([[
        sample.sepal_length,
        sample.sepal_width,
        sample.petal_length,
        sample.petal_width
    ]])
    pred = int(clf.predict(X)[0])
    meta = get_active_model_meta()

    return PredictResponse(
        prediction=pred,
        class_name=CLASS_NAMES[pred],
        model_version=meta["version"] if meta else "unknown"
    )


@app.post("/train", response_model=TrainResponse, tags=["Entrenamiento"])
def train(request: TrainRequest):
    """
    Reentrena el modelo con las nuevas muestras enviadas.
    """
    # 1. Preparar nuevas muestras
    new_X = np.array([[s.sepal_length, s.sepal_width, s.petal_length, s.petal_width]
                       for s in request.samples])
    new_y = np.array([s.label for s in request.samples])

    # 2. Recuperar accuracy del modelo activo
    history = load_history()
    previous_accuracy = history[-1]["accuracy"] if history else None

    # 3. Construir dataset de entrenamiento
    data_file = MODELS_DIR / "accumulated_data.joblib"

    if not request.retrain_from_scratch and data_file.exists():
        saved = joblib.load(data_file)
        X_train = np.vstack([saved["X"], new_X])
        y_train = np.concatenate([saved["y"], new_y])
        source = f"incremental (+{len(new_X)} muestras nuevas, {len(saved['X'])} anteriores)"
    else:
        X_train, y_train = new_X, new_y
        source = f"desde cero ({len(new_X)} muestras)"

    # 4. Necesitamos al menos 2 clases para entrenar
    if len(np.unique(y_train)) < 2:
        raise HTTPException(
            status_code=422,
            detail="El dataset de entrenamiento debe contener al menos 2 clases distintas."
        )

    # 5. Entrenar nuevo modelo y definir dataset de prueba
    clf_new = LogisticRegression(max_iter=300, random_state=42)

    if len(X_train) >= 20:
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        clf_new.fit(X_tr, y_tr)
        X_eval, y_eval = X_val, y_val
        eval_note = f"validación con {len(X_val)} muestras"
    else:
        clf_new.fit(X_train, y_train)
        X_eval, y_eval = X_train, y_train
        eval_note = "evaluación en train (dataset pequeño, < 20 muestras)"

    # --- CALCULAR MÉTRICAS DEL NUEVO MODELO ---
    y_pred_new = clf_new.predict(X_eval)
    accuracy_new = float(accuracy_score(y_eval, y_pred_new))
    
    # Calculamos F1-score para todas las clases (0, 1, 2)
    f1_new_arr = f1_score(y_eval, y_pred_new, average=None, labels=[0, 1, 2], zero_division=0)
    f1_new_dict = {CLASS_NAMES[i]: float(f1) for i, f1 in enumerate(f1_new_arr)}
    new_metrics = {"accuracy": accuracy_new, "f1_scores": f1_new_dict}

    # --- CALCULAR MÉTRICAS DEL MODELO ACTUAL (sobre el mismo dataset de evaluación) ---
    if previous_accuracy is not None and MODEL_PATH.exists():
        clf_current = joblib.load(MODEL_PATH)
        y_pred_current = clf_current.predict(X_eval)
        acc_current = float(accuracy_score(y_eval, y_pred_current))
        f1_curr_arr = f1_score(y_eval, y_pred_current, average=None, labels=[0, 1, 2], zero_division=0)
        f1_curr_dict = {CLASS_NAMES[i]: float(f1) for i, f1 in enumerate(f1_curr_arr)}
        current_metrics = {"accuracy": acc_current, "f1_scores": f1_curr_dict}
    else:
        # Primer modelo o no hay historial
        current_metrics = {"accuracy": 0.0, "f1_scores": {c: 0.0 for c in CLASS_NAMES.values()}}

    # 6. Decidir si activar el nuevo modelo usando ActivationPolicy
    policy = ActivationPolicy(
        policy_type=request.policy,
        min_delta=request.min_delta,
        target_class=request.target_class
    )
    
    model_updated, message = policy.evaluate_model(current_metrics, new_metrics)

    version = f"v{len(history) + 1}.0-{uuid.uuid4().hex[:6]}"
    status = "activado" if model_updated else "rechazado"

    if model_updated:
        joblib.dump(clf_new, MODEL_PATH)
        joblib.dump({"X": X_train, "y": y_train}, data_file)

    # 7. Registrar en historial
    history.append({
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": round(accuracy_new, 4),
        "n_training_samples": len(X_train),
        "algorithm": "LogisticRegression",
        "source": source,
        "eval_note": eval_note,
        "policy_used": request.policy,
        "activation_msg": message,
        "status": status,
        "activated": model_updated
    })
    
    # GUARDAMOS Y DEVOLVEMOS RESPUESTA (lo que te faltaba)
    save_history(history)

    return TrainResponse(
        status=status,
        model_version=version,
        accuracy_new=accuracy_new,
        accuracy_previous=previous_accuracy,
        model_updated=model_updated,
        message=message
    )