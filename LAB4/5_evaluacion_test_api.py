import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Recreamos el dataset de Test usando la misma semilla (random_state=42)
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Al usar random_state=42, separamos EXACTAMENTE el mismo 10% que no vio el modelo al entrenar
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# 2. Preparamos el payload en formato split para la API
payload = {
    "dataframe_split": X_test.to_dict(orient="split")
}

# 3. Hacemos la petición POST a la API (que tienes corriendo en el puerto 1234)
url = "http://localhost:1234/invocations"
print("Enviando el dataset de test (10%) a la API para evaluación final...")

try:
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("¡Predicciones recibidas con éxito!\n")
        
        # Extraemos las predicciones del JSON devuelto por la API
        y_pred = response.json()["predictions"]
        
        # 4. Calculamos las métricas finales (Métricas Reales sobre Test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("=== MÉTRICAS REALES SOBRE EL DATASET DE TEST ===")
        print(f"Accuracy  (Exactitud) : {acc:.4f}")
        print(f"Precision (Precisión) : {prec:.4f}")
        print(f"Recall    (Sensibilidad): {rec:.4f}")
        print(f"F1-Score  (Equilibrio)  : {f1:.4f}")
        print("================================================")
        print("\n¡Práctica terminada a nivel de código!")
        
    else:
        print(f"Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"Error al conectar con la API: {e}. ¿Seguro que la dejaste encendida en la otra terminal?")