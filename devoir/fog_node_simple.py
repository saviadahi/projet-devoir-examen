
import json
import time
import argparse
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kafka import KafkaConsumer, KafkaProducer
import numpy as np

from fog_nodes.anomaly_model import LogisticRegressionSGD, normalize_features


class SimpleFogNode:
    """Nœud Fog simplifié pour apprentissage distribué"""
    
    def __init__(self, node_id, kafka_bootstrap='localhost:9092', 
                 learning_rate=0.01, update_interval=10):
        """
        Args:
            node_id: ID du nœud (1, 2, 3...)
            kafka_bootstrap: Serveur Kafka
            learning_rate: Taux d'apprentissage
            update_interval: Intervalle de publication des poids (secondes)
        """
        self.node_id = node_id
        self.kafka_bootstrap = kafka_bootstrap
        self.learning_rate = learning_rate
        self.update_interval = update_interval
        
        # Topics
        self.input_topic = f'sensor-data-node-{node_id}'
        self.weights_topic = 'model-weights'
        self.global_model_topic = 'global-model'
        
        # Modèle local
        self.model = LogisticRegressionSGD(learning_rate=learning_rate)
        
        # Statistiques
        self.processed_count = 0
        self.last_update_time = time.time()
        self.batch_data = []
        
        # Consumer Kafka
        self.consumer = KafkaConsumer(
            self.input_topic,
            bootstrap_servers=kafka_bootstrap,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        # Producteur Kafka pour publier les poids
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        print(f"\n{'='*70}")
        print(f"🌫️  Initialisation du Nœud Fog {node_id} (Version Simple)")
        print(f"{'='*70}")
        print(f"📥 Input topic: {self.input_topic}")
        print(f"📤 Weights topic: {self.weights_topic}")
        print(f"📡 Global model topic: {self.global_model_topic}")
        print(f"🎓 Learning rate: {learning_rate}")
        print(f"⏱️  Update interval: {update_interval}s")
        print(f"{'='*70}\n")
    
    def publish_weights(self):
        """Publie les poids du modèle local vers Kafka"""
        weights_data = {
            'node_id': self.node_id,
            'timestamp': time.time(),
            'weights': self.model.get_weights(),
            'stats': self.model.get_stats(),
            'samples_processed': self.processed_count
        }
        
        try:
            self.producer.send(self.weights_topic, value=weights_data)
            self.producer.flush()
            
            stats = self.model.get_stats()
            print(f"📤 [{datetime.now().strftime('%H:%M:%S')}] "
                  f"Poids publiés | Loss: {stats['current_loss']:.4f} | "
                  f"Samples: {self.processed_count}")
            
            return True
        except Exception as e:
            print(f"❌ Erreur publication: {e}")
            return False
    
    def train_on_batch(self):
        """Entraîne le modèle sur le batch accumulé"""
        if len(self.batch_data) == 0:
            return
        
        # Convertir en arrays numpy
        X = np.array([d['features'] for d in self.batch_data])
        y = np.array([d['label'] for d in self.batch_data])
        
        # Entraîner avec mini-batch SGD
        loss = self.model.mini_batch_sgd(X, y, batch_size=min(32, len(X)))
        
        # Afficher progression
        if self.processed_count % 50 == 0:
            stats = self.model.get_stats()
            print(f"🎓 [{datetime.now().strftime('%H:%M:%S')}] "
                  f"Entraînement | Samples: {self.processed_count} | "
                  f"Loss: {loss:.4f} | Avg Loss (10): {stats['avg_loss_10']:.4f}")
        
        # Vider le batch
        self.batch_data = []
    
    def start(self):
        """Démarre le nœud Fog"""
        
        print(f"🚀 Nœud Fog {self.node_id} démarré - En attente de données...\n")
        
        try:
            for message in self.consumer:
                data = message.value
                
                # Normaliser les features
                features = normalize_features(data['temperature'], data['vibration'])
                
                # Ajouter au batch
                self.batch_data.append({
                    'features': features,
                    'label': data['label']
                })
                
                self.processed_count += 1
                
                # Entraîner tous les 32 exemples
                if len(self.batch_data) >= 32:
                    self.train_on_batch()
                
                # Vérifier si on doit publier les poids
                current_time = time.time()
                if current_time - self.last_update_time >= self.update_interval:
                    # Entraîner sur les données restantes
                    if len(self.batch_data) > 0:
                        self.train_on_batch()
                    
                    # Publier les poids
                    self.publish_weights()
                    self.last_update_time = current_time
        
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Arrêt du Nœud Fog {self.node_id}")
            
            # Sauvegarder le modèle final
            model_path = f"models/fog_node_{self.node_id}_final.json"
            os.makedirs("models", exist_ok=True)
            self.model.save_model(model_path)
            
            # Publier les poids finaux
            if len(self.batch_data) > 0:
                self.train_on_batch()
            self.publish_weights()
            
            # Statistiques finales
            stats = self.model.get_stats()
            print(f"\n{'='*70}")
            print(f"📊 Statistiques finales - Nœud {self.node_id}")
            print(f"{'='*70}")
            print(f"Samples traités: {self.processed_count}")
            print(f"Itérations: {stats['iterations']}")
            print(f"Loss finale: {stats['current_loss']:.4f}")
            print(f"Norme des poids: {stats['weights_norm']:.4f}")
            print(f"{'='*70}\n")
            
            self.consumer.close()
            self.producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Nœud Fog Simple (sans Spark)')
    parser.add_argument('--node-id', type=int, required=True,
                        help='ID du nœud (1, 2, 3...)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Taux d\'apprentissage')
    parser.add_argument('--update-interval', type=int, default=10,
                        help='Intervalle de publication des poids (secondes)')
    parser.add_argument('--kafka', type=str, default='localhost:9092',
                        help='Serveur Kafka')
    
    args = parser.parse_args()
    
    # Créer et démarrer le nœud
    node = SimpleFogNode(
        node_id=args.node_id,
        kafka_bootstrap=args.kafka,
        learning_rate=args.learning_rate,
        update_interval=args.update_interval
    )
    
    node.start()