from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("alzheimers_disease_data.csv")

## Tratamento de daados
## Exclui colunas desnecessárias
df = df.drop(columns=["PatientID", "DoctorInCharge"])

# Transforma de Int para Float
df[["Age", "SystolicBP", "DiastolicBP"]] = df[["Age", "SystolicBP", "DiastolicBP"]].astype(float)

# Ajusta as casas decimais de todos float para 6 depois da virgula
colunas_float = df.select_dtypes(include=["float64", "float32"]).columns
df[colunas_float] = df[colunas_float].round(6)

# Pré processamento
X = df.drop("Diagnosis", axis=1)  # todos dados SEM rótulo
y = df["Diagnosis"]  # rótulo

# Divisão em conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Treinamento do modelo
# Max 95.12
model = DecisionTreeClassifier(
  max_depth=5, 
  min_samples_leaf=6,
  criterion="entropy",
  random_state=42
  )

model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f'Árvore treinada! Acurácia de teste {acc * 100:.2f}%' )

artifact = {
  "model": model,
  "feature_columns": list(X.columns),
}

model_path = Path(__file__).resolve().parent / "alzheimers_model.pkl"
joblib.dump(artifact, model_path)
print(f"Modelo salvo com sucesso em: '{model_path}'!")
