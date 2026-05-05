from pathlib import Path
import sys
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logger import get_logger


logger = get_logger("api")

MODEL_PATH = Path("models/best_model.pkl")

app = FastAPI(
    title="Marketing ROI Prediction API",
    description="API de prédiction des ventes à partir de budgets marketing.",
    version="1.0.0"
)


class MarketingInput(BaseModel):
    TV: float = Field(..., ge=0, description="Budget TV en million")
    Radio: float = Field(..., ge=0, description="Budget Radio en million")
    Social_Media: float = Field(..., ge=0, description="Budget Social Media en million")
    Influencer: str = Field(..., description="Type d'influenceur : Mega, Macro, Micro, Nano")


def load_model():
    if not MODEL_PATH.exists():
        logger.error(f"Modèle introuvable : {MODEL_PATH}")
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    logger.info("Modèle chargé avec succès dans l'API")
    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/health")
def health():
    logger.info("Health check appelé")

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.get("/model-info")
def model_info():
    logger.info("Endpoint model-info appelé")

    return {
        "model_path": str(MODEL_PATH),
        "task": "regression",
        "target": "Sales",
        "features": ["TV", "Radio", "Social Media", "Influencer"]
    }


@app.post("/predict")
def predict(input_data: MarketingInput):
    logger.info(f"Requête reçue : {input_data}")

    try:
        df = pd.DataFrame([{
            "TV": input_data.TV,
            "Radio": input_data.Radio,
            "Social Media": input_data.Social_Media,
            "Influencer": input_data.Influencer
        }])

        prediction = model.predict(df)[0]

        total_budget = input_data.TV + input_data.Radio + input_data.Social_Media
        roi = prediction / total_budget if total_budget > 0 else 0

        logger.info(
            f"Prédiction API réussie | "
            f"prediction={prediction:.4f}, roi={roi:.4f}"
        )

        return {
            "predicted_sales": round(float(prediction), 4),
            "total_budget": round(float(total_budget), 4),
            "estimated_roi": round(float(roi), 4)
        }

    except Exception as error:
        logger.exception("Erreur pendant la prédiction API")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {error}"
        )