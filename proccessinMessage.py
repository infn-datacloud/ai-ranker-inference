import json

def processMessage(message, input_list):
    input_message={}
    message_providers=message["providers"]
    complexity=0
    if message["template_name"]!="Single VM":
        complexity=1
    for el in message_providers:
        calculated_values = { 'cpu_diff':(el["vcpus_quota"] - el["vcpus_requ"] - el["vcpus_usage"]), 
                'ram_diff':(el["ram_gb_quota"] - el["ram_gb_requ"] - el["ram_gb_usage"]),
                'storage_diff':el["storage_gb_quota"] - el["storage_gb_requ"] - el["storage_gb_usage"],
                'instances_diff':el["n_instances_quota"] - el["n_instances_requ"] - el["n_instances_usage"],
                'floatingips_diff':el["floating_ips_quota"] - el["floating_ips_requ"] - el["floating_ips_usage"],
                'gpu':float(bool(el["gpus_requ"])),
                'sla_failure_percentage':el["test_failure_perc"],
                'complexity':complexity}
        # overbooking_ram = ?
        # avg_deployment_time = ?
        # failure_percentage = ?
        input_message[el["provider_name"]+"-"+el["region_name"]]=[calculated_values[key] for key in input_list if key in calculated_values]
    return input_message


#file_path = 'message.json'

# Open and read the JSON file
#with open(file_path, 'r') as file:
#    data = json.load(file)
# Print the loaded data

