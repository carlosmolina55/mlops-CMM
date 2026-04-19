import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

# Usamos tu base de datos correcta
mlflow.set_tracking_uri("sqlite:///mlflow2.db")
mlflow.set_experiment("diabetes_modelo_final")

# 1. Cargamos el dataset original y quitamos el 10% de Test (intocable)
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

with mlflow.start_run(run_name="Modelo_Ganador_C01_lbfgs"):
    
    # 2. Entrenamos con TUS nuevos mejores parámetros
    model = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
    
    # Entrenamos con el 90% de los datos (X_temp, y_temp)
    model.fit(X_temp, y_temp) 
    
    # 3. Lo registramos en el Model Registry Oficial
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="modelo_definitivo",
        registered_model_name="Diabetes_Modelo_Final",
        input_example=X_test.iloc[:2]
    )
    
print("¡Modelo ganador 'Diabetes_Modelo_Final' registrado con éxito!")