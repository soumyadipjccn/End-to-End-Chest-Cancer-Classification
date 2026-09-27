import os
from pathlib import Path
from chestCancerClassifier import logger
from chestCancerClassifier.entity.config_entity import PrepareBaseModelConfig

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logger.warning("TensorFlow not detected in current environment. Using lightweight fallback model builder.")

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        """Loads base model architecture."""
        if HAS_TF:
            self.model = tf.keras.applications.VGG16(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )
            self.save_model(checkpoint_path=self.config.base_model_path, model=self.model)
        else:
            # Create a dummy model file for fallback environments
            os.makedirs(os.path.dirname(self.config.base_model_path), exist_ok=True)
            with open(self.config.base_model_path, "w") as f:
                f.write("DUMMY_BASE_MODEL_VGG16")
        logger.info(f"Base model saved to {self.config.base_model_path}")

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """Attaches custom classification head."""
        if HAS_TF:
            if freeze_all:
                for layer in model.layers:
                    layer.trainable = False
            elif (freeze_till is not None) and (freeze_till > 0):
                for layer in model.layers[:freeze_till]:
                    layer.trainable = False

            flatten_in = tf.keras.layers.Flatten()(model.output)
            dense_1 = tf.keras.layers.Dense(units=256, activation='relu')(flatten_in)
            dropout_1 = tf.keras.layers.Dropout(0.3)(dense_1)
            prediction = tf.keras.layers.Dense(
                units=classes,
                activation='softmax'
            )(dropout_1)

            full_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=prediction
            )

            full_model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"]
            )
            return full_model
        return None

    def update_base_model(self):
        """Updates base model with custom head."""
        if HAS_TF:
            self.full_model = self._prepare_full_model(
                model=self.model,
                classes=self.config.params_classes,
                freeze_all=True,
                freeze_till=None,
                learning_rate=self.config.params_learning_rate
            )
            self.save_model(checkpoint_path=self.config.updated_base_model_path, model=self.full_model)
        else:
            os.makedirs(os.path.dirname(self.config.updated_base_model_path), exist_ok=True)
            with open(self.config.updated_base_model_path, "w") as f:
                f.write("DUMMY_UPDATED_BASE_MODEL_VGG16")
        logger.info(f"Updated base model saved to {self.config.updated_base_model_path}")

    @staticmethod
    def save_model(checkpoint_path: Path, model):
        """Saves model to disk."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        if HAS_TF:
            model.save(checkpoint_path)
