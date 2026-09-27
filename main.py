import sys
import os
sys.path.extend(["src", "."])

from chestCancerClassifier import logger
from chestCancerClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from chestCancerClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from chestCancerClassifier.pipeline.stage_03_model_trainer import ModelTrainerPipeline
from chestCancerClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

STAGE_01_NAME = "Data Ingestion Stage"
STAGE_02_NAME = "Prepare Base Model Stage"
STAGE_03_NAME = "Model Training Stage"
STAGE_04_NAME = "Model Evaluation Stage"

def run_pipeline():
    try:
        logger.info(f">>>>>> Stage {STAGE_01_NAME} started <<<<<<")
        stage_01 = DataIngestionTrainingPipeline()
        stage_01.main()
        logger.info(f">>>>>> Stage {STAGE_01_NAME} completed <<<<<<\n\nx==========x")

        logger.info(f">>>>>> Stage {STAGE_02_NAME} started <<<<<<")
        stage_02 = PrepareBaseModelTrainingPipeline()
        stage_02.main()
        logger.info(f">>>>>> Stage {STAGE_02_NAME} completed <<<<<<\n\nx==========x")

        logger.info(f">>>>>> Stage {STAGE_03_NAME} started <<<<<<")
        stage_03 = ModelTrainerPipeline()
        stage_03.main()
        logger.info(f">>>>>> Stage {STAGE_03_NAME} completed <<<<<<\n\nx==========x")

        logger.info(f">>>>>> Stage {STAGE_04_NAME} started <<<<<<")
        stage_04 = EvaluationPipeline()
        stage_04.main()
        logger.info(f">>>>>> Stage {STAGE_04_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == '__main__':
    run_pipeline()
