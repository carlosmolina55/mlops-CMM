import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn



# 1. Configuración de MLflow
mlflow.set_tracking_uri("sqlite:///mlflow2.db")
mlflow.set_experiment("diabetes_grid_search")


# 2. Carga y división de datos
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis = 1)
y = df["Outcome"]

# División a test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)

# División a train valid
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size= 0.2222, random_state=42)

print(f"Tamaño Train: {len(X_train)}, Validación: {len(X_val)}, Test: {len(X_test)}")


# 3. Definir la rejilla de parámetros (Grid Search)

parametros = {

    "C": [0.01,0.1, 1, 10, 100],            # Fuerza de regularización
    "solver": ["liblinear", "lbfgs"],      # Algoritmo de optimización
    "max_iter": [100, 500, 1000]
}

print("Iniciando experimentos en MLflow...")

# 4. Bucle de entrenamiento y Traxking

for c in parametros["C"]:
    for solver in parametros["solver"]:
        for max_iter in parametros["max_iter"]:

            with mlflow.start_run():
                # -- ENTRENAMIENTO --
                model = LogisticRegression(C=c, solver = solver, max_iter=max_iter)
                model.fit(X_train, y_train)

                # -- VALIDACION --
                y_pred_val = model.predict(X_val)
                
                # Métricas

                acc = accuracy_score(y_val, y_pred_val)
                prec = precision_score(y_val, y_pred_val, zero_division=0)
                rec = recall_score(y_val, y_pred_val)
                f1 = f1_score(y_val, y_pred_val)

                # -- REGISTRO EN MLFLOW --
                # Registro de parámetros
                mlflow.log_param("C", c)
                mlflow.log_param("solver", solver)
                mlflow.log_param("max_iter", max_iter)
                
                # Registramos métricas
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)

                # Guardamos el modelo en el tracking
                mlflow.sklearn.log_model(model, artifact_path="modelo")

print("¡Todos los experimentos han sido registrados con éxito!")
            