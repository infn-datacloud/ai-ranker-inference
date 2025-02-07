import json
import os
from kafka import KafkaProducer

kafka_server_url = os.environ.get("KAFKA_HOSTNAME", "localhost:9092")
topic = os.environ.get("KAFKA_TOPIC", "test")

producer = KafkaProducer(
    bootstrap_servers=kafka_server_url,
    client_id="example",
    value_serializer=lambda x: json.dumps(x, sort_keys=True).encode("utf-8"),
)

if producer.bootstrap_connected():
    print("Connected")

def sender(message):
    #message = {"test": "value"}
    producer.send(topic, message)
    producer.flush()
