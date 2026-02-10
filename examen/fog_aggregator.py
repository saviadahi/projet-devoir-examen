"""
Agrégateur Fog - Niveau Régional
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
import os

class FogAggregator:
    def __init__(self, region_name, village_ids):
        self.region_name = region_name
        self.village_ids = village_ids
        self.edge_weights = []
        self.aggregated_weights = None
        
    def load_edge_weights(self, weights_path='models/edge_weights.json'):
        """Charge les poids des modèles Edge"""
        if not os.path.exists(weights_path):
            print(f"❌ Fichier non trouvé: {weights_path}")
            return []
            
        with open(weights_path, 'r') as f:
            all_weights = json.load(f)
        
        # Filtrer les poids pour cette région
        self.edge_weights = [
            w for w in all_weights 
            if w['village_id'] in self.village_ids
        ]
        
        print(f"📡 Région {self.region_name}: {len(self.edge_weights)} villages chargés")
        return self.edge_weights
    
    def aggregate_weights(self):
        """Agrège les poids des modèles Edge"""
        if not self.edge_weights:
            print(f"⚠️  Aucun poids pour {self.region_name}")
            return None
        
        print(f"\n🔄 Agrégation pour {self.region_name}...")
        
        # Calculer la pondération
        total_samples = sum(w['training_stats']['samples_trained'] 
                          for w in self.edge_weights)
        
        # Initialiser
        scaler_mean_sum = np.zeros(3)
        scaler_scale_sum = np.zeros(3)
        
        # Moyenne pondérée
        for weight in self.edge_weights:
            n_samples = weight['training_stats']['samples_trained']
            weight_factor = n_samples / total_samples
            
            scaler_mean_sum += np.array(weight['scaler_mean']) * weight_factor
            scaler_scale_sum += np.array(weight['scaler_scale']) * weight_factor
        
        # Statistiques
        total_anomalies = sum(w['training_stats']['anomalies_detected'] 
                             for w in self.edge_weights)
        avg_contamination = np.mean([w['contamination'] 
                                    for w in self.edge_weights])
        
        self.aggregated_weights = {
            'region_name': self.region_name,
            'num_villages': len(self.edge_weights),
            'scaler_mean': scaler_mean_sum.tolist(),
            'scaler_scale': scaler_scale_sum.tolist(),
            'contamination': float(avg_contamination),
            'total_samples': int(total_samples),
            'total_anomalies': int(total_anomalies),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Agrégation OK:")
        print(f"   Villages: {len(self.edge_weights)}")
        print(f"   Échantillons: {total_samples}")
        print(f"   Anomalies: {total_anomalies}")
        
        return self.aggregated_weights
    
    def save_aggregated_weights(self, output_path='models/fog_weights.json'):
        """Sauvegarde les poids agrégés"""
        if self.aggregated_weights is None:
            return None
        
        # Charger existants
        try:
            with open(output_path, 'r') as f:
                all_fog_weights = json.load(f)
        except FileNotFoundError:
            all_fog_weights = []
        
        # Mettre à jour
        all_fog_weights = [w for w in all_fog_weights 
                          if w['region_name'] != self.region_name]
        all_fog_weights.append(self.aggregated_weights)
        
        # Sauvegarder
        with open(output_path, 'w') as f:
            json.dump(all_fog_weights, f, indent=2)
        
        print(f"💾 Sauvegardé: {output_path}")
        return output_path
    
    def process_alerts(self, threshold=0.12):
        """Détecte les alertes urgentes"""
        print(f"\n🚨 Analyse alertes {self.region_name}...")
        
        alerts = []
        for weight in self.edge_weights:
            rate = weight['training_stats']['anomaly_rate']
            
            if rate > threshold:
                severity = 'HIGH' if rate > 0.25 else 'MEDIUM'
                alert = {
                    'village_id': weight['village_id'],
                    'anomaly_rate': rate,
                    'severity': severity,
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
                print(f"⚠️  ALERTE {severity}: {alert['village_id']} ({rate*100:.1f}%)")
        
        if not alerts:
            print(f"✅ Pas d'alerte pour {self.region_name}")
        
        return alerts


def simulate_fog_layer():
    """Simule la couche Fog"""
    print("\n" + "="*60)
    print("🌫️  SIMULATION COUCHE FOG (Niveau Régional)")
    print("="*60)
    
    # Régions de Mauritanie
    regions = {
        'Trarza': ['Village_1', 'Village_2'],
        'Gorgol': ['Village_3', 'Village_4'],
        'Brakna': ['Village_5']
    }
    
    print(f"\n📍 Régions configurées:")
    for region, villages in regions.items():
        print(f"   {region}: {', '.join(villages)}")
    
    # Traiter chaque région
    fog_aggregators = {}
    all_alerts = []
    
    for region_name, village_ids in regions.items():
        print(f"\n{'='*60}")
        print(f"📡 Traitement: {region_name}")
        print(f"{'='*60}")
        
        fog = FogAggregator(region_name, village_ids)
        fog.load_edge_weights()
        fog.aggregate_weights()
        alerts = fog.process_alerts(threshold=0.12)
        all_alerts.extend(alerts)
        fog.save_aggregated_weights()
        
        fog_aggregators[region_name] = fog
    
    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ COUCHE FOG")
    print("="*60)
    print(f"Régions: {len(regions)}")
    print(f"Alertes: {len(all_alerts)}")
    
    if all_alerts:
        print("\n🚨 Alertes:")
        for alert in all_alerts:
            print(f"   • {alert['village_id']}: {alert['severity']} "
                  f"({alert['anomaly_rate']*100:.1f}%)")
    
    return fog_aggregators, all_alerts


def generate_fog_statistics():
    """Statistiques Fog"""
    try:
        with open('models/fog_weights.json', 'r') as f:
            fog_weights = json.load(f)
        
        print("\n" + "="*60)
        print("📈 STATISTIQUES FOG")
        print("="*60)
        
        stats = []
        for region in fog_weights:
            rate = region['total_anomalies'] / region['total_samples'] * 100
            stats.append({
                'Région': region['region_name'],
                'Villages': region['num_villages'],
                'Échantillons': region['total_samples'],
                'Anomalies': region['total_anomalies'],
                'Taux': f"{rate:.2f}%"
            })
        
        df = pd.DataFrame(stats)
        print(df.to_string(index=False))
        
        return df
        
    except Exception as e:
        print(f"⚠️  Erreur: {e}")
        return None


if __name__ == "__main__":
    print("🚀 Démarrage de la couche Fog...\n")
    
    # Vérifier que les fichiers existent
    if not os.path.exists('models/edge_weights.json'):
        print("❌ ERREUR: Lancez d'abord edge_detector.py !")
        print("   Commande: python edge\\edge_detector.py")
    else:
        fog_aggregators, alerts = simulate_fog_layer()
        stats = generate_fog_statistics()
        
        print("\n✅ Phase Fog terminée!")
        print("📤 Prêt pour le Cloud (Federated Learning)")