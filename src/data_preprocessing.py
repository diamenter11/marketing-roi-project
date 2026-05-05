import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from src.logger import get_logger


logger = get_logger(__name__)

DATA_PATH = Path("data/marketing_and_sales.csv")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

TARGET = "Sales"
NUMERIC_FEATURES = ["TV", "Radio", "Social Media"]
CATEGORICAL_FEATURES = ["Influencer"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    logger.info("Chargement du dataset...")

    if not path.exists():
        logger.error(f"Fichier introuvable : {path}")
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)

    logger.info("Dataset chargé avec succès")
    logger.info(f"Shape du dataset : {df.shape}")
    logger.info(f"Colonnes détectées : {list(df.columns)}")

    return df


def validate_data(df: pd.DataFrame) -> None:
    expected_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        logger.error(f"Colonnes manquantes : {missing_columns}")
        raise ValueError(f"Colonnes manquantes : {missing_columns}")

    logger.info("Toutes les colonnes attendues sont présentes")

    missing_values = df.isna().sum()
    logger.info(f"Valeurs manquantes par colonne :\n{missing_values}")

    duplicated_rows = df.duplicated().sum()
    logger.info(f"Nombre de doublons : {duplicated_rows}")

    logger.info(f"Types détectés :\n{df.dtypes}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Nettoyage des données...")

    df = df.copy()

    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]

    logger.info(f"Doublons supprimés : {before - after}")

    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)

    missing_before = df.isna().sum().sum()
    logger.info(f"Valeurs manquantes avant suppression : {missing_before}")

    df = df.dropna()

    missing_after = df.isna().sum().sum()
    logger.info(f"Valeurs manquantes après nettoyage : {missing_after}")
    logger.info(f"Shape après nettoyage : {df.shape}")

    return df


def split_data(df: pd.DataFrame):
    logger.info("Séparation features / target...")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    logger.info("Split train/test en cours...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    logger.info(f"X_train shape : {X_train.shape}")
    logger.info(f"X_test shape : {X_test.shape}")
    logger.info(f"y_train shape : {y_train.shape}")
    logger.info(f"y_test shape : {y_test.shape}")

    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    logger.info("Construction du pipeline de preprocessing...")

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    logger.info("Pipeline de preprocessing construit avec succès")

    return preprocessor


def save_artifacts(X_train, X_test, y_train, y_test, preprocessor):
    logger.info("Sauvegarde des fichiers intermédiaires...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

    logger.info("Fichiers sauvegardés dans data/processed/")
    logger.info("Preprocessor sauvegardé dans models/preprocessor.pkl")


def main():
    df = load_data()
    validate_data(df)

    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    preprocessor = build_preprocessor()

    logger.info("Test du preprocessing sur X_train...")
    X_train_transformed = preprocessor.fit_transform(X_train)

    logger.info(f"Shape après preprocessing : {X_train_transformed.shape}")
    logger.info("Preprocessing testé avec succès")

    save_artifacts(X_train, X_test, y_train, y_test, preprocessor)

    logger.info("Étape 2 terminée avec succès")


if __name__ == "__main__":
    main()