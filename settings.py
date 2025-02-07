from pydantic import AnyHttpUrl, Field, field_validator, validator
from pydantic_settings import BaseSettings
from typing import Optional
import json

class MLflowSettings(BaseSettings):
    
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000",
        description="MLFlow tracking URI"
    )
    CLASSIFICATION_MODEL_NAME: str = Field(
        default="RandomForestClassifier",
        description="Name of the classification model"
    )
    REGRESSION_MODEL_NAME: str = Field(
        default="RandomForestRegressor",
        description="Name of the regressor model"
    )
    CLASSIFICATION_MODEL_VERSION: str = Field(
        default = "10",
        description = "Version of classification model"
    )
    REGRESSION_MODEL_VERSION: str = Field(
        default = "10",
        description = "Version of regression model"
    )
    KAFKA_SERVER_URL: str = Field(
        default="localhost:9092",
        description="Kafka url"
    )
    MIN_REGRESSION_TIME: int = Field(
        default=500,
        description="Minimum regression time"
    )
    MAX_REGRESSION_TIME: int = Field(
        default=5000,
        description="Maximim regression time"
    )
    FILTER: bool = Field(
        default = False,
        description = "Filter results undert threshold"
    )
    THRESHOLD: float = Field(
        default = 0.7,
        description = "Threshold to filter out score"
    )
    CLASSIFICATION_WEIGHT: float = Field(
        default = 0.75,
        description = "Classification weight"
    )

    class Config:
        env_file = ".env"  # Set variables from env files
        env_file_encoding = "utf-8"

# Function to load the settings
def load_mlflow_settings() -> MLflowSettings:
    return MLflowSettings()
