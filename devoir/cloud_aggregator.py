import json
import time
import argparse
from datetime import datetime
from collections import defaultdict
import os

from kafka import KafkaConsumer, KafkaProducer
import numpy as np


class FederatedAggregator:
    """Agrégateur FedAvg pour apprentissage fédéré"""
    
    def __init__(self, num_nodes=3, kafka_bootstrap='localhost:9092', 
                 aggregation_interval=30):
        """
        Args:
            num_nodes: Nombre de nœuds Fog attendus
            kafka_bootstrap: Serveur Kafka
            aggregation_interval: Intervalle d'agrégation (secondes)
        """
        self.num_nodes = num_nodes
        self.kafka_bootstrap = kafka_bootstrap
        self.aggregation_interval = aggregation_interval
        
        # Topics
        self.weights_topic = 'model-weights'
        self.global_model_topic = 'global-model'
        
        # Stockage des poids reçus
        self.node_weights = {}  # {node_id: weights_data}
        self.aggregation_count = 0
        self.global_model_history = []
        
        # Consumer Kafka
        self.consumer = KafkaConsumer(
            self.weights_topic,
            bootstrap_servers=kafka_bootstrap,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        # Producteur Kafka
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        print(f"\n{'='*70}")
        print(f"☁️  Initialisation de l'Agrégateur Cloud (FedAvg)")
        print(f"{'='*70}")
        print(f"📥 Weights topic: {self.weights_topic}")
        print(f"📤 Global model topic: {self.global_model_topic}")
        print(f"🔢 Nombre de nœuds: {num_nodes}")
        print(f"⏱️  Intervalle d'agrégation: {aggregation_interval}s")
        print(f"{'='*70}\n")
    
    def federated_averaging(self):
        """
        Applique l'algorithme FedAvg
        
        Formule: w_global = Σ (n_k / n_total) * w_k
        où n_k = nombre de samples du nœud k
        """
        if len(self.node_weights) == 0:
            return None
        
        print(f"\n{'='*70}")
        print(f"🔄 Agrégation FedAvg - Round {self.aggregation_count + 1}")
        print(f"{'='*70}")
        
        # Calculer le nombre total de samples
        total_samples = sum(data['samples_processed'] 
                          for data in self.node_weights.values())
        
        print(f"📊 Nœuds participants: {len(self.node_weights)}/{self.num_nodes}")
        print(f"📈 Total samples: {total_samples}\n")
        
        # Afficher les détails de chaque nœud
        for node_id, data in sorted(self.node_weights.items()):
            weight_ratio = data['samples_processed'] / total_samples
            stats = data['stats']
            print(f"  Nœud {node_id}:")
            print(f"    ├─ Samples: {data['samples_processed']} ({weight_ratio*100:.1f}%)")
            print(f"    ├─ Loss: {stats['current_loss']:.6f}")
            print(f"    └─ Poids dans l'agrégation: {weight_ratio:.4f}")
        
        # Initialiser les poids globaux
        global_weights = None
        global_bias = 0.0
        
        # Agréger avec pondération
        for node_id, data in self.node_weights.items():
            weights_dict = data['weights']
            node_samples = data['samples_processed']
            weight_ratio = node_samples / total_samples
            
            node_weights = np.array(weights_dict['weights'])
            node_bias = weights_dict['bias']
            
            if global_weights is None:
                global_weights = weight_ratio * node_weights
                global_bias = weight_ratio * node_bias
            else:
                global_weights += weight_ratio * node_weights
                global_bias += weight_ratio * node_bias
        
        # Créer le modèle global
        global_model = {
            'weights': global_weights.tolist(),
            'bias': float(global_bias),
            'aggregation_round': self.aggregation_count + 1,
            'num_nodes': len(self.node_weights),
            'total_samples': total_samples,
            'timestamp': time.time()
        }
        
        # Calculer la loss moyenne pondérée
        avg_loss = sum(data['stats']['current_loss'] * data['samples_processed'] 
                      for data in self.node_weights.values()) / total_samples
        
        print(f"\n✅ Modèle global créé:")
        print(f"   ├─ w[0]: {global_weights[0]:+.6f}")
        print(f"   ├─ w[1]: {global_weights[1]:+.6f}")
        print(f"   ├─ bias: {global_bias:+.6f}")
        print(f"   └─ Loss moyenne pondérée: {avg_loss:.6f}")
        print(f"{'='*70}\n")
        
        return global_model, avg_loss
    
    def publish_global_model(self, global_model):
        """Publie le modèle global vers Kafka"""
        try:
            self.producer.send(self.global_model_topic, value=global_model)
            self.producer.flush()
            
            print(f"📤 Modèle global publié (Round {global_model['aggregation_round']})")
            
            return True
        except Exception as e:
            print(f"❌ Erreur publication: {e}")
            return False
    
    def save_global_model(self, global_model, avg_loss):
        """Sauvegarde le modèle global"""
        os.makedirs("models", exist_ok=True)
        
        # Sauvegarder le modèle
        round_num = global_model['aggregation_round']
        model_path = f"models/global_model_round_{round_num}.json"
        
        with open(model_path, 'w') as f:
            json.dump(global_model, f, indent=2)
        
        # Mettre à jour l'historique
        self.global_model_history.append({
            'round': round_num,
            'timestamp': global_model['timestamp'],
            'avg_loss': avg_loss,
            'num_nodes': global_model['num_nodes'],
            'total_samples': global_model['total_samples']
        })
        
        # Sauvegarder l'historique
        history_path = "models/aggregation_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.global_model_history, f, indent=2)
        
        print(f"💾 Modèle sauvegardé: {model_path}")
    
    def start(self):
        """Démarre l'agrégateur"""
        
        print(f"🚀 Agrégateur démarré - En attente de poids...\n")
        
        last_aggregation_time = time.time()
        
        try:
            for message in self.consumer:
                data = message.value
                node_id = data['node_id']
                
                # Stocker les poids reçus
                self.node_weights[node_id] = data
                
                timestamp = datetime.fromtimestamp(data['timestamp'])
                stats = data['stats']
                
                print(f"📥 [{timestamp.strftime('%H:%M:%S')}] "
                      f"Poids reçus du Nœud {node_id} | "
                      f"Samples: {data['samples_processed']} | "
                      f"Loss: {stats['current_loss']:.4f}")
                
                # Vérifier si on doit agréger
                current_time = time.time()
                time_since_last = current_time - last_aggregation_time
                
                # Conditions d'agrégation:
                # 1. On a reçu des poids de tous les nœuds
                # 2. OU l'intervalle est dépassé et on a au moins 1 nœud
                should_aggregate = (
                    len(self.node_weights) >= self.num_nodes or
                    (time_since_last >= self.aggregation_interval and 
                     len(self.node_weights) > 0)
                )
                
                if should_aggregate:
                    # Agréger
                    result = self.federated_averaging()
                    
                    if result:
                        global_model, avg_loss = result
                        
                        # Publier
                        self.publish_global_model(global_model)
                        
                        # Sauvegarder
                        self.save_global_model(global_model, avg_loss)
                        
                        # Réinitialiser
                        self.aggregation_count += 1
                        self.node_weights.clear()
                        last_aggregation_time = current_time
        
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Arrêt de l'agrégateur")
            
            # Agrégation finale si des poids sont en attente
            if len(self.node_weights) > 0:
                print("\n🔄 Agrégation finale...")
                result = self.federated_averaging()
                if result:
                    global_model, avg_loss = result
                    self.publish_global_model(global_model)
                    self.save_global_model(global_model, avg_loss)
            
            # Afficher les statistiques finales
            self.display_final_stats()
            
            self.consumer.close()
            self.producer.close()
    
    def display_final_stats(self):
        """Affiche les statistiques finales"""
        print(f"\n{'='*70}")
        print(f"📊 Statistiques finales de l'agrégation")
        print(f"{'='*70}")
        print(f"Nombre total d'agrégations: {self.aggregation_count}")
        
        if self.global_model_history:
            print(f"\n📈 Évolution de la loss moyenne:")
            for entry in self.global_model_history:
                timestamp = datetime.fromtimestamp(entry['timestamp'])
                print(f"  Round {entry['round']:2d} | "
                      f"{timestamp.strftime('%H:%M:%S')} | "
                      f"Loss: {entry['avg_loss']:.6f} | "
                      f"Nodes: {entry['num_nodes']} | "
                      f"Samples: {entry['total_samples']}")
            
            # Amélioration
            if len(self.global_model_history) > 1:
                first_loss = self.global_model_history[0]['avg_loss']
                last_loss = self.global_model_history[-1]['avg_loss']
                improvement = (first_loss - last_loss) / first_loss * 100
                
                print(f"\n✅ Amélioration totale: {improvement:.2f}%")
                print(f"   Loss initiale: {first_loss:.6f}")
                print(f"   Loss finale: {last_loss:.6f}")
        
        print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agrégateur Cloud FedAvg')
    parser.add_argument('--num-nodes', type=int, default=3,
                        help='Nombre de nœuds Fog')
    parser.add_argument('--interval', type=int, default=30,
                        help='Intervalle d\'agrégation (secondes)')
    parser.add_argument('--kafka', type=str, default='localhost:9092',
                        help='Serveur Kafka')
    
    args = parser.parse_args()
    
    # Créer et démarrer l'agrégateur
    aggregator = FederatedAggregator(
        num_nodes=args.num_nodes,
        kafka_bootstrap=args.kafka,
        aggregation_interval=args.interval
    )
    
    aggregator.start()