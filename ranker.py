import requests
import time
import json
from app import run_prediction
# Define the endpoint URL
#model_name = "RandomForestClassifier"
#model_version = "9"

# Define the URL
url = f"http://127.0.0.1:8000/predict/" #{model_name}/{model_version}"

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
        return None

# Iterate through each key in inputInference
def process_inference(input_inference: dict, url: str, threshold: float = 0.7, min_provider_number: int = 3):
    results = {}

    for key, data in input_inference.items():
        payload = {"inputs": data}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            # this line could
            success_prob = response.json().get("predictions", [[0]])[0][0]
            results[key] = success_prob
        except requests.exceptions.RequestException as e:
            print(f"Request to {url} failed for {key}: {e}")
            results[key] = 0.5  # Default value for failed requests
        except (KeyError, IndexError) as e:
            print(f"Unexpected response structure for {key}: {e}")
            results[key] = 0.5  # Default value for unexpected responses

    # Filter results based on the threshold
    filtered_results = {key: value for key, value in results.items() if value >= threshold}

    # Decide whether to use filtered results or original results based on the count
    results_to_sort = filtered_results if len(filtered_results) >= min_provider_number else results

    # Sort the results in descending order and take the top `min_provider_number`
    sorted_results = dict(sorted(results_to_sort.items(), key=lambda item: item[1], reverse=True)[:min_provider_number])

    return sorted_results


input_inference = read_json_file("inference_data.json")

sorted_results = process_inference(input_inference, url, 0.7,5)

print(sorted_results)
