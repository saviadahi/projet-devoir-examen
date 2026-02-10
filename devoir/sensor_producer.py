"""
Producteur de données de capteurs pour Kafka
Simule des capteurs industriels (température, vibration) avec anomalies
"""

import json
import time
import random
import numpy as np
from kafka import KafkaProducer
from datetime import datetime
import argparse


class SensorSimulator:
    """Simule un capteur industriel avec génération d'anomalies"""
    
    def __init__(self, sensor_id, anomaly_rate=0.05):
        """
        Args:
            sensor_id: Identifiant du capteur
            anomaly_rate: Probabilité de générer une anomalie (0-1)
        """
        self.sensor_id = sensor_id
        self.anomaly_rate = anomaly_rate
        
        # Paramètres normaux
        self.normal_temp_mean = 25.0
        self.normal_temp_std = 2.0
        self.normal_vib_mean = 5.0
        self.normal_vib_std = 1.0
        
        # Paramètres d'anomalie
        self.anomaly_temp_mean = 45.0
        self.anomaly_temp_std = 5.0
        self.anomaly_vib_mean = 15.0
        self.anomaly_vib_std = 3.0
    
    def generate_data(self):
        """Génère une mesure de capteur (normale ou anomalie)"""
        
        # Décider si c'est une anomalie
        is_anomaly = random.random() < self.anomaly_rate
        
        if is_anomaly:
            temperature = np.random.normal(self.anomaly_temp_mean, self.anomaly_temp_std)
            vibration = np.random.normal(self.anomaly_vib_mean, self.anomaly_vib_std)
            label = 1  # Anomalie
        else:
            temperature = np.random.normal(self.normal_temp_mean, self.normal_temp_std)
            vibration = np.random.normal(self.normal_vib_mean, self.normal_vib_std)
            label = 0  # Normal
        
        # Créer le message
        data = {
            'sensor_id': self.sensor_id,
            'temperature': float(temperature),
            'vibration': float(vibration),
            'timestamp': time.time(),
            'label': label  # CETTE LIGNE DOIT ÊTRE LÀ
        }
        
        return data
        

class KafkaSensorProducer:
    """Producteur Kafka pour envoyer les données de capteurs"""
    
    def __init__(self, bootstrap_servers='localhost:9092'):
        """
        Args:
            bootstrap_servers: Adresse du serveur Kafka
        """
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        print(f"✅ Connecté à Kafka: {bootstrap_servers}")
    
    def send_data(self, topic, data):
        """Envoie des données vers un topic Kafka"""
        try:
            future = self.producer.send(
                topic, 
                key=data['sensor_id'],
                value=data
            )
            # Attendre la confirmation
            record_metadata = future.get(timeout=10)
            return True
        except Exception as e:
            print(f"❌ Erreur d'envoi: {e}")
            return False
    
    def close(self):
        """Ferme le producteur"""
        self.producer.flush()
        self.producer.close()


def run_sensor_producer(node_id, interval=1.0, anomaly_rate=0.05):
    """
    Exécute le producteur de capteur
    
    Args:
        node_id: ID du nœud (1, 2, 3...)
        interval: Intervalle entre les mesures (secondes)
        anomaly_rate: Taux d'anomalies
    """
    sensor_id = f"node-{node_id}"
    topic = f"sensor-data-node-{node_id}"
    
    print(f"\n{'='*60}")
    print(f"🚀 Démarrage du capteur: {sensor_id}")
    print(f"📊 Topic Kafka: {topic}")
    print(f"⚠️  Taux d'anomalies: {anomaly_rate*100}%")
    print(f"⏱️  Intervalle: {interval}s")
    print(f"{'='*60}\n")
    
    # Créer le simulateur et le producteur
    simulator = SensorSimulator(sensor_id, anomaly_rate)
    producer = KafkaSensorProducer()
    
    # Statistiques
    total_sent = 0
    anomalies_sent = 0
    
    try:
        while True:
            # Générer les données
            data = simulator.generate_data()
            
            # Envoyer vers Kafka
            if producer.send_data(topic, data):
                total_sent += 1
                if data['label'] == 1:
                    anomalies_sent += 1
                
                # Afficher un résumé toutes les 10 mesures
                if total_sent % 10 == 0:
                    print(f"📈 [{sensor_id}] Envoyé: {total_sent} | "
                          f"Anomalies: {anomalies_sent} ({anomalies_sent/total_sent*100:.1f}%) | "
                          f"Dernier: T={data['temperature']:.2f}°C, V={data['vibration']:.2f}")
            
            # Attendre avant la prochaine mesure
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Arrêt du capteur {sensor_id}")
        print(f"📊 Total envoyé: {total_sent}")
        print(f"⚠️  Anomalies: {anomalies_sent} ({anomalies_sent/total_sent*100:.1f}%)")
    
    finally:
        producer.close()
        print("✅ Producteur fermé")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulateur de capteur pour Kafka')
    parser.add_argument('--node-id', type=int, default=1, 
                        help='ID du nœud (1, 2, 3...)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Intervalle entre mesures (secondes)')
    parser.add_argument('--anomaly-rate', type=float, default=0.05,
                        help='Taux d\'anomalies (0.0-1.0)')
    
    args = parser.parse_args()
    
    run_sensor_producer(
        node_id=args.node_id,
        interval=args.interval,
        anomaly_rate=args.anomaly_rate
    )