import os
import json
import joblib
import yaml
import base64
from pathlib import Path
from typing import Any, List
from chestCancerClassifier import logger

# Fallback for python-box
try:
    from box import ConfigBox
except ImportError:
    class ConfigBox(dict):
        """Lightweight ConfigBox fallback supporting dot notation access."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for key, val in self.items():
                if isinstance(val, dict):
                    self[key] = ConfigBox(val)
                elif isinstance(val, list):
                    self[key] = [ConfigBox(v) if isinstance(v, dict) else v for v in val]

        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError:
                raise AttributeError(f"'ConfigBox' object has no attribute '{item}'")

        def __setattr__(self, key, value):
            self[key] = value

        def __delattr__(self, item):
            try:
                del self[item]
            except KeyError:
                raise AttributeError(f"'ConfigBox' object has no attribute '{item}'")

# Fallback for ensure_annotations
try:
    from ensure import ensure_annotations
except ImportError:
    def ensure_annotations(func):
        return func

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns ConfigBox."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content if content is not None else {})
    except Exception as e:
        logger.error(f"Error loading YAML file {path_to_yaml}: {e}")
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves dictionary data to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads JSON file data."""
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves binary file using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads binary data using joblib."""
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """Gets size of file in KB."""
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"

def decodeImage(imgstring: str, fileName: str):
    """Decodes base64 string to image file."""
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)

def encodeImageIntoBase64(croppedImagePath: str) -> bytes:
    """Encodes image file to base64 string."""
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
