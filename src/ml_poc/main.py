from pathlib import Path
import logging

from .data import DatasetConfig, load_dataset
from .model import ModelConfig, create_model, train_model
from .evaluate import evaluate_model


def main() -> None:
    """Run the ML POC workflow."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Configure dataset
    dataset_config = DatasetConfig(
        file_path="data/train.csv",
        target_column="target"
    )
    
    # Load and split data
    logger.info("Loading and splitting dataset")
    X_train, X_test, y_train, y_test = load_dataset(config=dataset_config)
    
    # Configure and create model
    model_config = ModelConfig()
    model = create_model(config=model_config)
    
    # Train model
    logger.info("Training model")
    trained_model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train
    )
    
    # Evaluate model
    logger.info("Evaluating model")
    results = evaluate_model(
        model=trained_model,
        X_test=X_test,
        y_test=y_test
    )
    
    # Print results
    logger.info(f"Model accuracy: {results.accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(results.report)
    
    if results.feature_importance is not None:
        logger.info("\nFeature Importance:")
        logger.info(results.feature_importance)


if __name__ == "__main__":
    main() 