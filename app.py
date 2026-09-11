import logging
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

app = FastAPI(
  title="Alzheimer Prediction Service",
  description="Microsserviço de predição de Alzheimer baseado em Machine Learning",
  version="1.0.0"
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5500,http://127.0.0.1:5500,null"
  ).split(","),
  allow_credentials=True,
  allow_methods=["POST", "GET"],
  allow_headers=["Content-Type"]
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "alzheimers_model.pkl"

try:
  artifact = joblib.load(MODEL_PATH)
  model = artifact["model"]
  FEATURE_COLUMNS = artifact["feature_columns"]
except Exception as exc: 
  raise RuntimeError(f"Erro ao carregar o modelo: {exc}") from exc

class AnamnesisData(BaseModel):
  model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

  Age: float = Field(..., ge=60, le=90, example=73, description="Idade do paciente")
  Gender: Literal[0, 1] = Field(..., example=0, description="Gênero do paciente")
  Ethnicity: int = Field(..., ge=0, le=3, example=0, description="Etnia do paciente")
  EducationLevel: int = Field(..., ge=0, le=3, example=0, description="Nível de escolaridade")
  BMI: float = Field(..., ge=15, le=40, example=23.4, description="Índice de massa corporal")
  Smoking: Literal[0, 1] = Field(..., example=0, description="Indicador de tabagismo")
  AlcoholConsumption: float = Field(..., ge=0, le=20, example=2.3, description="Consumo de álcool")
  PhysicalActivity: float = Field(..., ge=0, le=10, example=3.2, description="Nível de atividade física")
  DietQuality: float = Field(..., ge=0, le=10, example=7.0, description="Qualidade da dieta")
  SleepQuality: float = Field(..., ge=0, le=10, example=6.5, description="Qualidade do sono")
  FamilyHistoryAlzheimers: Literal[0, 1] = Field(..., example=0, description="Histórico familiar de Alzheimer")
  CardiovascularDisease: Literal[0, 1] = Field(..., example=0, description="Presença de doença cardiovascular")
  Diabetes: Literal[0, 1] = Field(..., example=0, description="Presença de diabetes")
  Depression: Literal[0, 1] = Field(..., example=0, description="Presença de depressão")
  HeadInjury: Literal[0, 1] = Field(..., example=0, description="Histórico de lesão cerebral")
  Hypertension: Literal[0, 1] = Field(..., example=0, description="Presença de hipertensão")
  SystolicBP: int = Field(..., ge=90, le=180, example=130, description="Pressão arterial sistólica")
  DiastolicBP: int = Field(..., ge=60, le=120, example=80, description="Pressão arterial diastólica")
  CholesterolTotal: float = Field(..., ge=150, le=300, example=200.0, description="Colesterol total")
  CholesterolLDL: float = Field(..., ge=50, le=200, example=120.0, description="Colesterol LDL")
  CholesterolHDL: float = Field(..., ge=20, le=100, example=50.0, description="Colesterol HDL")
  CholesterolTriglycerides: float = Field(..., ge=50, le=400, example=150.0, description="Triglicerídeos")
  MMSE: float = Field(..., ge=0, le=30, example=21.0, description="Pontuação no MMSE")
  FunctionalAssessment: float = Field(..., ge=0, le=10, example=6.0, description="Avaliação funcional")
  MemoryComplaints: Literal[0, 1] = Field(..., example=1, description="Queixas de memória")
  BehavioralProblems: Literal[0, 1] = Field(..., example=0, description="Problemas comportamentais")
  ADL: float = Field(..., ge=0, le=10, example=5.0, description="Atividades de vida diária")
  Confusion: Literal[0, 1] = Field(..., example=0, description="Presença de confusão")
  Disorientation: Literal[0, 1] = Field(..., example=0, description="Presença de desorientação")
  PersonalityChanges: Literal[0, 1] = Field(..., example=0, description="Alterações de personalidade")
  DifficultyCompletingTasks: Literal[0, 1] = Field(..., example=0, description="Dificuldade para completar tarefas")
  Forgetfulness: Literal[0, 1] = Field(..., example=1, description="Esquecimento")

  @model_validator(mode="after")
  def validate_blood_pressure(self):
    if self.DiastolicBP >= self.SystolicBP:
      raise ValueError("A pressão sistólica deve ser maior que a diastólica")
    return self

class AlzheimerPredictionResponse(BaseModel):
  risk_level: Literal["BAIXO", "MODERADO", "ALTO", "CRÍTICO"]
  probability: float = Field(..., ge=0, le=1)
  recommendation: str

@app.get("/health", tags=["health"])
def health_check():
  return {"status": "ok", "model_loaded": True}

@app.post(
  "/api/v1/predict-alzheimer",
  response_model=AlzheimerPredictionResponse,
  tags=["prediction"]
)
def evaluate_patient_risk(data: AnamnesisData):
  feature = pd.DataFrame([data.model_dump()], columns=FEATURE_COLUMNS)
  try:
    probabilities = model.predict_proba(feature)
    prob_alzheimers = float(probabilities[0, 1])  # Probabilidade de ter Alzheimer

    if prob_alzheimers >= 0.60:
      risk_level = "CRÍTICO"
      action = "Recomenda-se avaliação médica imediata."
    elif prob_alzheimers >= 0.40:
      risk_level = "ALTO"
      action = "Recomenda-se acompanhamento médico próximo."
    elif prob_alzheimers >= 0.20:
      risk_level = "MODERADO"
      action = "Recomenda-se monitoramento regular e hábitos saudáveis."
    else:
      risk_level = "BAIXO"
      action = "Recomenda-se manter hábitos saudáveis e monitoramento ocasional."

    return AlzheimerPredictionResponse(
      risk_level=risk_level,
      probability=prob_alzheimers, 
      recommendation=action
      )
  except Exception as exc:
    logger.exception("Falha ao processar a predição")
    raise HTTPException(
      status_code=500,
      detail="Não foi possível processar a predição."
    ) from exc


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="127.0.0.1", port=8000)