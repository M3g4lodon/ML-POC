"""Machine Learning Proof of Concept package."""

from .data import DatasetConfig, load_dataset
from .model import ModelConfig, create_model, train_model
from .evaluate import EvaluationResult, evaluate_model
from .main import main

__version__ = "0.1.0" 