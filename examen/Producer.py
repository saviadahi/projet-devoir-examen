# edge/kafka_producer.py
from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(
    bootstrap_servers=['fog_server:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_reading(village_id, voltage, current, power):
    message = {
        'village_id': village_id,
        'voltage': voltage,
        'current': current,
        'power': power,
        'timestamp': time.time()
    }
    
    producer.send(f'{village_id}_data', message)
    print(f"✅ Sent: {message}")