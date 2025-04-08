from typing import Any
import pandas as pd
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Configuration for loading and processing the dataset."""
    
    file_path: str = Field(
        title="File Path",
        description="Path to the dataset file"
    )
    target_column: str = Field(
        title="Target Column",
        description="Name of the target column in the dataset"
    )
    test_size: float = Field(
        default=0.2,
        title="Test Size",
        description="Proportion of the dataset to include in the test split"
    )
    random_state: int = Field(
        default=42,
        title="Random State",
        description="Random seed for reproducibility"
    )


def load_dataset(*, config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and split the dataset into training and testing sets.
    
    Args:
        config: Configuration for loading and processing the dataset
        
    Returns:
        Tuple containing (X_train, X_test, y_train, y_test)
    """
    data = pd.read_csv(config.file_path)
    
    X = data.drop(columns=[config.target_column])
    y = data[config.target_column]
    
    from sklearn.model_selection import train_test_split
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state
    ) 

if __name__ == "__main__":
    config = DatasetConfig(file_path="data/train.csv", target_column="target")
    X_train, X_test, y_train, y_test = load_dataset(config=config)
    print(X_train.head())
    print(y_train.head())
    print(X_test.head())
    print(y_test.head())