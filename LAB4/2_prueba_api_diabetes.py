import requests
import pandas as pd

# 1. Cargamos un par de filas reales de nuestro dataset para hacer la prueba
df = pd.read_csv("diabetes.csv")
X_prueba = df.drop("Outcome", axis=1).iloc[:2] # Cogemos los dos primeros pacientes

# 2. Preparamos el paquete de datos en el formato JSON que MLflow espera
payload = {
    "inputs": X_prueba.values.tolist()
}

# 3. Hacemos la petición POST a la API local
url = "http://localhost:1234/invocations"
print(f"Enviando datos a la API: {payload['inputs']}")

response = requests.post(url, json=payload)

# 4. Mostramos el resultado
if response.status_code == 200:
    print("\n¡Éxito! Predicciones recibidas:")
    print(response.json())
else:
    print(f"\nError {response.status_code}: {response.text}")