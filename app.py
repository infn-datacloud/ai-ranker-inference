from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np
import os
import pandas as pd

# MLflow configuration
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://131.154.99.212:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
# Create the MLflow client
client = mlflow.tracking.MlflowClient()

# Create the FastAPI app
app = FastAPI()

# Model for the inference request
class InferenceRequest(BaseModel):
    inputs: list

DEFAULT_MODEL_NAME = "RandomForestClassifier"
DEFAULT_MODEL_VERSION = "9"
FEATURE_NAMES = ["cpu_diff","ram_diff","storage_diff","instances_diff","floatingips_diff","gpu","sla_failure_percentage","overbooking_ram","avg_deployment_time","failure_percentage","complexity"]
# Inference endpoint
@app.post("/predict/{model_name}/{model_version}")
@app.post("/predict/")
def predict(model_name: str = DEFAULT_MODEL_NAME, model_version: str = DEFAULT_MODEL_VERSION, request: InferenceRequest = None):
    # Load the model
    model_uri = f"models:/{model_name}/{model_version}"
    if request is None:
        raise HTTPException(status_code=400, detail="Input data is required")
    try:
    # Get the model type and load it with the proper function
        model = mlflow.pyfunc.load_model(model_uri)
        model_type = model.metadata.flavors.keys()
        if 'sklearn' in model_type:
            model = mlflow.sklearn.load_model(model_uri)
            feature_names = model.feature_names_in_
        else:
            raise HTTPException(status_code=400, detail="Model type not supported")
        if len(request.inputs[0]) != len(feature_names):
        raise HTTPException(status_code=400, detail=f"Input data has {len(request.inputs[0])} features, but the model expects {num_features} features. Please provide input data with {num_features} features.")
    
        X_new = pd.DataFrame(request.inputs, columns = feature_names)
        y_pred_new = model.predict_proba(X_new)
        return {"predictions": y_pred_new.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    X_new = np.array(request.inputs)
    y_pred_new = model.predict(X_new)
    return {"predictions": y_pred_new.tolist()}

# Endpoint to list the available models
@app.get("/list-models")
def list_models():
    models = client.search_registered_models()

    # Create a list with models' information
    model_list = []
    for model in models:
        model_info = {
            "name": model.name,
            "latest_versions": [
                {"version": mv.version, "stage": mv.current_stage}
                for mv in model.latest_versions
            ],
        }
        model_list.append(model_info)

    return {"models": model_list}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
