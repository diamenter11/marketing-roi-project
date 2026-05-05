from src.data_preprocessing import main as preprocess_main
from src.train_models import main as train_main
from src.evaluate import main as evaluate_main

from src.logger import get_logger

logger = get_logger("main")


def run_pipeline():
    logger.info("=== DÉMARRAGE DU PIPELINE GLOBAL ===")

    try:
        logger.info("Étape 1 : Préprocessing")
        preprocess_main()

        logger.info("Étape 2 : Entraînement modèles")
        train_main()

        logger.info("Étape 3 : Évaluation modèle")
        evaluate_main()

        logger.info("=== PIPELINE TERMINÉ AVEC SUCCÈS ===")

    except Exception as e:
        logger.exception("Erreur dans le pipeline")
        raise e


if __name__ == "__main__":
    run_pipeline()