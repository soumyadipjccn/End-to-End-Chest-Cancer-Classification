import sys
import os
sys.path.extend(["src", "."])

from chestCancerClassifier.config.configuration import ConfigurationManager
from chestCancerClassifier.components.model_trainer import ModelTrainer
from chestCancerClassifier import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        model_trainer = ModelTrainer(config=training_config)
        model_trainer.get_base_model()
        model_trainer.train_valid_generator()
        model_trainer.train()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
