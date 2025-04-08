from typing import Any
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


class ModelConfig(BaseModel):
    """Configuration for model training."""
    
    model_type: str = Field(
        default="random_forest",
        title="Model Type",
        description="Type of model to train"
    )
    random_state: int = Field(
        default=42,
        title="Random State",
        description="Random seed for reproducibility"
    )
    n_estimators: int = Field(
        default=100,
        title="Number of Estimators",
        description="Number of trees in the forest"
    )


def create_model(*, config: ModelConfig) -> BaseEstimator:
    """Create a machine learning model based on the configuration.
    
    Args:
        config: Configuration for model creation
        
    Returns:
        Initialized scikit-learn model
    """
    if config.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            random_state=config.random_state
        )
    raise ValueError(f"Unknown model type: {config.model_type}")


def train_model(
    *,
    model: BaseEstimator,
    X_train: Any,
    y_train: Any
) -> BaseEstimator:
    """Train the model on the provided data.
    
    Args:
        model: Initialized scikit-learn model
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    return model.fit(X_train, y_train) 