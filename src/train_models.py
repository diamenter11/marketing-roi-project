import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import build_preprocessor
from src.logger import get_logger


logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def load_train_test_data():
    logger.info("Chargement des données train/test...")

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    logger.info(f"X_train : {X_train.shape}")
    logger.info(f"X_test : {X_test.shape}")

    return X_train, X_test, y_train, y_test


def get_models():
    logger.info("Initialisation des modèles...")

    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),
        "MLP Regressor": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=2000,
            random_state=42
        )
    }


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2


def train_and_compare_models():
    X_train, X_test, y_train, y_test = load_train_test_data()

    models = get_models()
    results = []
    trained_models = {}

    for model_name, model in models.items():
        logger.info(f"Entraînement du modèle : {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        mae, rmse, r2 = evaluate_model(pipeline, X_test, y_test)

        logger.info(
            f"{model_name} | MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}"
        )

        results.append({
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        trained_models[model_name] = pipeline

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="RMSE", ascending=True)

    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    logger.info("Comparaison sauvegardée : reports/model_comparison.csv")
    logger.info(f"Meilleur modèle : {best_model_name}")
    logger.info("Meilleur modèle sauvegardé : models/best_model.pkl")

    return results_df


def main():
    results_df = train_and_compare_models()

    logger.info("Résultats finaux :")
    logger.info(f"\n{results_df}")

    logger.info("Étape 3 terminée avec succès")


if __name__ == "__main__":
    main()