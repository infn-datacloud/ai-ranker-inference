from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np
import os

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

# Inference endpoint
@app.post("/predict")
def predict(request: InferenceRequest):
    # Load the model
    model_name = "sk-learn-random-forest-reg-model"
    model_version = "latest"
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    # Get the model type and load it with the proper function
    model = mlflow.pyfunc.load_model(model_uri)
    model_type = model.metadata.flavors.keys()

    if 'sklearn' in model_type:
        model = mlflow.sklearn.load_model(model_uri)
    else:
        return {"error": "Model type not supported"}

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
