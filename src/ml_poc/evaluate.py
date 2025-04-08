from typing import Any
import pandas as pd
from sklearn.metrics import classification_report
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Results from model evaluation."""
    
    accuracy: float = Field(
        title="Accuracy",
        description="Model accuracy on the test set"
    )
    report: str = Field(
        title="Classification Report",
        description="Detailed classification metrics"
    )
    feature_importance: pd.Series | None = Field(
        default=None,
        title="Feature Importance",
        description="Feature importance scores if available"
    )


def evaluate_model(
    *,
    model: Any,
    X_test: Any,
    y_test: Any
) -> EvaluationResult:
    """Evaluate the model's performance on the test set.
    
    Args:
        model: Trained scikit-learn model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Evaluation results including accuracy and classification report
    """
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    report = classification_report(y_test, y_pred)
    
    feature_importance = None
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X_test.columns
        ).sort_values(ascending=False)
    
    return EvaluationResult(
        accuracy=accuracy,
        report=report,
        feature_importance=feature_importance
    ) 