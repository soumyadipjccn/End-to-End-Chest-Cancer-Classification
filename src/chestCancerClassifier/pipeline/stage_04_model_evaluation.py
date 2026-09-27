import sys
import os
sys.path.extend(["src", "."])

from chestCancerClassifier.config.configuration import ConfigurationManager
from chestCancerClassifier.components.model_evaluation import ModelEvaluation
from chestCancerClassifier import logger

STAGE_NAME = "Model Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = ModelEvaluation(eval_config)
        evaluation.evaluate()
        evaluation.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
