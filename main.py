import time
import json
from settings import load_mlflow_settings
import os
import mlflow
import mlflow.sklearn
from kafka import KafkaConsumer
import pandas as pd


# FEATURE_NAMES = ["cpu_diff","ram_diff","storage_diff","instances_diff","floatingips_diff","gpu","sla_failure_percentage","overbooking_ram","avg_deployment_time","failure_percentage","complexity"]
# Inference endpointi
def classification_predict(inputData:dict, model_name: str,  model_version: str ):

    model_uri = f"models:/{model_name}/{model_version}"
    classification_values={}
    try:
    # Get the model type and load it with the proper function
        model = mlflow.pyfunc.load_model(model_uri)
        model_type = model.metadata.flavors.keys()
        if 'sklearn' in model_type:
            model = mlflow.sklearn.load_model(model_uri)
            feature_names = model.feature_names_in_
            print(feature_names)
        else:
            raise ValueError("Model type not supported")
        for key , data in inputData.items():
            X_new = pd.DataFrame([data], columns = feature_names)
            y_pred_new = model.predict_proba(X_new)
            classification_response=y_pred_new.tolist()
            success_prob = classification_response[0][0]
            classification_values[key]=success_prob
        return classification_values

    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")


def regression_predict(inputData:dict, model_name: str,  model_version: str):

    model_uri = f"models:/{model_name}/{model_version}"
    regression_values={}
    try:
    # Get the model type and load it with the proper function
        model = mlflow.pyfunc.load_model(model_uri)
        model_type = model.metadata.flavors.keys()
        if 'sklearn' in model_type:
            model = mlflow.sklearn.load_model(model_uri)
            #feature_names = model.feature_names_in_
        
        else:
            raise ValueError("Model type not supported")
        for key , data in inputData.items():
            X_new = pd.DataFrame([data]) # columns = feature_names)
            y_pred_new = model.predict(X_new)
            regression_response=y_pred_new.tolist()
            regression_value_raw = regression_response[0]
            regression_value = 1 - (regression_value_raw - min_regression_time) / (max_regression_time - min_regression_time)
            regression_values[key]=regression_value
        return regression_values

    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")


def process_inference(input_inference: dict, classification_model_name :str, classification_model_version:str, regression_model_name:str, regression_model_version:str, min_regression_time:int, max_regression_time:int, classification_weight: float =0.75, threshold: float = 0.7, filter_on: bool=False):
    results = {}
    ##### DEFAULT VALUE
    default_value = 0.1

    try:
        # Send requests to classification and regression endpoints
        classification_response = classification_predict( inputData=input_inference, model_name=classification_model_name,model_version=classification_model_version)
        regression_response = regression_predict(inputData=input_inference, model_name=regression_model_name, model_version=regression_model_version)

        # Check for HTTP errors
        #classification_response.raise_for_status()
        #regression_response.raise_for_status()

        for key, value in classification_response.items():
        # Compute the result
            results[key] = value * classification_weight + regression_response[key] * (1-classification_weight)

    # except requests.exceptions.RequestException as e:
    #     # Handle HTTP-related errors
    #     print(f"Request to classification or regression endpoint failed for {key}: {e}")
    #     results[key] = default_value  # Default value for failed requests

    except (KeyError, IndexError, TypeError) as e:
        # Handle unexpected response structure
        print(f"Unexpected response structure for {key}: {e}")
        ##### DEFAULT VALUE
        results[key] = default_value  # Default value for unexpected responses

    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    # Filter results based on the threshold
    if filter_on:
        sorted_results = {key: value for key, value in results.items() if value >= threshold}

    return sorted_results


if __name__ == "__main__":

    settings = load_mlflow_settings()
    classification_model_name = settings.CLASSIFICATION_MODEL_NAME
    classification_model_version = settings.CLASSIFICATION_MODEL_VERSION
    regression_model_name = settings.REGRESSION_MODEL_NAME
    regression_model_version = settings.REGRESSION_MODEL_VERSION
    min_regression_time= settings.MIN_REGRESSION_TIME
    max_regression_time= settings.MAX_REGRESSION_TIME
    filter_mode = settings.FILTER
    threshold = settings.THRESHOLD
    classification_weight = settings.CLASSIFICATION_WEIGHT

    #### MLFLOW SETUP

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    # Create the MLflow client
    client = mlflow.tracking.MlflowClient()

    #### KAFKA SETUP

    kafka_server_url = os.environ.get("KAFKA_HOSTNAME", settings.KAFKA_SERVER_URL)
    topic = os.environ.get("KAFKA_TOPIC", "test")

    consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_server_url,
            #auto_offset_reset='earliest'
            )


if consumer.bootstrap_connected():
    print("Connected")
    print(f"Subscribed topics: {consumer.subscription()}")

    for message in consumer:
        # print(message)
        try:
            message = message.value.decode('utf-8')  # Decode bytes to a string
            input_dict = json.loads(message)
            start_time = time.time()
            sorted_results = process_inference(input_dict["test"], classification_model_name, classification_model_version, regression_model_name, regression_model_version ,min_regression_time=min_regression_time,max_regression_time=max_regression_time, classification_weight=classification_weight, threshold=threshold, filter_on=filter_mode)
            end_time = time.time()
            print(sorted_results)
            # Calculate elapsed time
            elapsed_time = end_time - start_time
            print(f"Inference Time: {elapsed_time:.2f} seconds")
        except Exception as e:
            # Handle exceptions and log them
            print(f"Error processing message: {e}")
