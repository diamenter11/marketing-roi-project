import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from src.logger import get_logger


logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def load_data_and_model():
    logger.info("Chargement des données de test et du meilleur modèle...")

    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    model = joblib.load(MODELS_DIR / "best_model.pkl")

    logger.info(f"X_test shape : {X_test.shape}")
    logger.info("Modèle chargé avec succès")

    return X_test, y_test, model


def evaluate_best_model(model, X_test, y_test):
    logger.info("Évaluation du meilleur modèle...")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"MAE : {mae:.4f}")
    logger.info(f"RMSE : {rmse:.4f}")
    logger.info(f"R2 : {r2:.4f}")

    results = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "residual": y_test - y_pred
    })

    results.to_csv(REPORTS_DIR / "best_model_predictions.csv", index=False)

    logger.info("Prédictions sauvegardées : reports/best_model_predictions.csv")

    return results


def plot_predictions(results: pd.DataFrame):
    logger.info("Création du graphique valeurs réelles vs valeurs prédites...")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(results["y_true"], results["y_pred"])
    plt.xlabel("Ventes réelles")
    plt.ylabel("Ventes prédites")
    plt.title("Ventes réelles vs ventes prédites")

    min_value = min(results["y_true"].min(), results["y_pred"].min())
    max_value = max(results["y_true"].max(), results["y_pred"].max())
    plt.plot([min_value, max_value], [min_value, max_value])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "real_vs_predicted.png")
    plt.close()

    logger.info("Graphique sauvegardé : reports/figures/real_vs_predicted.png")


def plot_residuals(results: pd.DataFrame):
    logger.info("Création du graphique des résidus...")

    plt.figure(figsize=(8, 6))
    plt.scatter(results["y_pred"], results["residual"])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Ventes prédites")
    plt.ylabel("Résidus")
    plt.title("Analyse des résidus")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals.png")
    plt.close()

    logger.info("Graphique sauvegardé : reports/figures/residuals.png")


def compute_permutation_importance(model, X_test, y_test):
    logger.info("Calcul de la permutation importance...")

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="neg_root_mean_squared_error"
    )

    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    feature_importance.to_csv(
        REPORTS_DIR / "feature_importance.csv",
        index=False
    )

    logger.info("Importance des variables sauvegardée : reports/feature_importance.csv")
    logger.info(f"\n{feature_importance}")

    return feature_importance


def plot_feature_importance(feature_importance: pd.DataFrame):
    logger.info("Création du graphique d'importance des variables...")

    plt.figure(figsize=(8, 6))
    plt.barh(
        feature_importance["feature"],
        feature_importance["importance_mean"]
    )
    plt.xlabel("Importance moyenne")
    plt.ylabel("Variable")
    plt.title("Permutation importance")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png")
    plt.close()

    logger.info("Graphique sauvegardé : reports/figures/feature_importance.png")


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test, model = load_data_and_model()

    results = evaluate_best_model(model, X_test, y_test)

    plot_predictions(results)
    plot_residuals(results)

    feature_importance = compute_permutation_importance(model, X_test, y_test)
    plot_feature_importance(feature_importance)

    logger.info("Étape 4 terminée avec succès")


if __name__ == "__main__":
    main()