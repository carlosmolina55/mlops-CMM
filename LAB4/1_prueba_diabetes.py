import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn


# 1. Configuración de MLflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("diabetes_prueba")

# 2. Carga de datos

df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis = 1)
y = df["Outcome"]

# División simple train test

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size =0.8, random_state = 42)

# 3. Iniciar el run de MLflow

with mlflow.start_run(run_name = "Prueba_Inicial"):

    # Entrenamiento básico
    model = LogisticRegression(max_iter = 1000)
    model.fit(X_train, y_train)

    # Predicción y Métricas
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)

    # Registrar la métrica
    mlflow.log_metric("accuracy", accuracy)

    # Registrar el modelo 

    mlflow.sklearn.log_model(
        sk_model = model,
        artifact_path = "modelo_prueba",
        registered_model_name = "Diabetes Modelo Prueba",
        input_example = X_test.iloc[:2]
    )

print(f"Entrenamiento de prueba finalizado con Accuracy: {accuracy:.4f}")





