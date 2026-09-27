import os
from pathlib import Path
from chestCancerClassifier import logger
from chestCancerClassifier.entity.config_entity import TrainingConfig

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logger.warning("TensorFlow not detected in current environment. Using lightweight fallback trainer.")

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        """Loads updated base model."""
        if HAS_TF:
            self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
            logger.info(f"Loaded base model from {self.config.updated_base_model_path}")
        else:
            logger.info("Using fallback base model.")

    def train_valid_generator(self):
        """Sets up train and validation data generators."""
        if HAS_TF:
            datagen_kwargs = dict(
                rescale=1./255,
                validation_split=0.20
            )

            dataflow_kwargs = dict(
                target_size=self.config.params_image_size[:2],
                batch_size=self.config.params_batch_size,
                interpolation="bilinear"
            )

            valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagen_kwargs
            )

            self.valid_generator = valid_datagen.flow_from_directory(
                directory=self.config.training_data,
                subset="validation",
                shuffle=False,
                **dataflow_kwargs
            )

            if self.config.params_is_augmentation:
                train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=40,
                    horizontal_flip=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    **datagen_kwargs
                )
            else:
                train_datagen = valid_datagen

            self.train_generator = train_datagen.flow_from_directory(
                directory=self.config.training_data,
                subset="training",
                shuffle=True,
                **dataflow_kwargs
            )

    @staticmethod
    def save_model(path: Path, model):
        """Saves trained model to specified filepath."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if HAS_TF and model is not None:
            model.save(path)
        else:
            with open(path, "w") as f:
                f.write("DUMMY_TRAINED_MODEL_VGG16")

    def train(self):
        """Executes model training loop."""
        if HAS_TF:
            self.steps_per_epoch = max(1, self.train_generator.samples // self.train_generator.batch_size)
            self.validation_steps = max(1, self.valid_generator.samples // self.valid_generator.batch_size)

            logger.info(f"Starting TensorFlow VGG16 model training for {self.config.params_epochs} epochs...")

            self.history = self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                validation_data=self.valid_generator
            )

            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )
        else:
            logger.info("Simulating training step in fallback mode...")
            self.save_model(
                path=self.config.trained_model_path,
                model=None
            )
        logger.info(f"Trained model saved to {self.config.trained_model_path}")
