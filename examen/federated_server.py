"""
Serveur Cloud - Federated Learning (FedAvg)
Fusionne les modèles régionaux en un modèle global
Sans jamais accéder aux données brutes des villages
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
import os

class FederatedServer:
    def __init__(self):
        self.global_model = None
        self.federation_history = []
        self.current_round = 0
        
    def load_fog_weights(self, fog_weights_path='models/fog_weights.json'):
        """Charge les poids agrégés de la couche Fog"""
        if not os.path.exists(fog_weights_path):
            print(f"❌ Fichier non trouvé: {fog_weights_path}")
            return []
        
        with open(fog_weights_path, 'r') as f:
            self.fog_weights = json.load(f)
        
        print(f"☁️  Cloud: {len(self.fog_weights)} régions chargées")
        for region in self.fog_weights:
            print(f"   • {region['region_name']}: {region['num_villages']} villages, "
                  f"{region['total_samples']} échantillons")
        
        return self.fog_weights
    
    def federated_averaging(self):
        """
        Algorithme FedAvg - Federated Averaging
        Fusionne les modèles régionaux avec pondération par nombre d'échantillons
        """
        if not self.fog_weights:
            print("⚠️  Aucun poids Fog à fusionner")
            return None
        
        print("\n" + "="*60)
        print("🤖 FEDERATED AVERAGING (FedAvg)")
        print("="*60)
        
        self.current_round += 1
        print(f"🔄 Round {self.current_round}")
        
        # Calculer le poids total pour la pondération
        total_samples = sum(r['total_samples'] for r in self.fog_weights)
        print(f"📊 Total échantillons globaux: {total_samples}")
        
        # Initialiser les moyennes globales
        global_scaler_mean = np.zeros(3)
        global_scaler_scale = np.zeros(3)
        global_contamination = 0
        total_anomalies = 0
        
        print("\n📡 Fusion des modèles régionaux:")
        
        # FedAvg: Moyenne pondérée par le nombre d'échantillons
        for region in self.fog_weights:
            weight_factor = region['total_samples'] / total_samples
            
            global_scaler_mean += np.array(region['scaler_mean']) * weight_factor
            global_scaler_scale += np.array(region['scaler_scale']) * weight_factor
            global_contamination += region['contamination'] * weight_factor
            total_anomalies += region['total_anomalies']
            
            print(f"   • {region['region_name']}: "
                  f"poids={weight_factor*100:.1f}% "
                  f"({region['total_samples']} échantillons)")
        
        # Créer le modèle global
        self.global_model = {
            'round': self.current_round,
            'num_regions': len(self.fog_weights),
            'total_villages': sum(r['num_villages'] for r in self.fog_weights),
            'total_samples': int(total_samples),
            'total_anomalies': int(total_anomalies),
            'global_anomaly_rate': float(total_anomalies / total_samples),
            'scaler_mean': global_scaler_mean.tolist(),
            'scaler_scale': global_scaler_scale.tolist(),
            'contamination': float(global_contamination),
            'timestamp': datetime.now().isoformat()
        }
        
        # Historique
        self.federation_history.append(self.global_model.copy())
        
        print("\n✅ Modèle Global Créé:")
        print(f"   Régions: {self.global_model['num_regions']}")
        print(f"   Villages: {self.global_model['total_villages']}")
        print(f"   Échantillons: {self.global_model['total_samples']}")
        print(f"   Anomalies: {self.global_model['total_anomalies']}")
        print(f"   Taux global: {self.global_model['global_anomaly_rate']*100:.2f}%")
        
        return self.global_model
    
    def save_global_model(self, output_path='models/global_model.json'):
        """Sauvegarde le modèle global"""
        if self.global_model is None:
            print("⚠️  Aucun modèle global à sauvegarder")
            return None
        
        with open(output_path, 'w') as f:
            json.dump(self.global_model, f, indent=2)
        
        print(f"\n💾 Modèle global sauvegardé: {output_path}")
        return output_path
    
    def save_federation_history(self, output_path='models/federation_history.json'):
        """Sauvegarde l'historique des rounds"""
        with open(output_path, 'w') as f:
            json.dump(self.federation_history, f, indent=2)
        
        print(f"📜 Historique sauvegardé: {output_path}")
        return output_path
    
    def compare_models(self):
        """Compare les performances régionales vs globales"""
        print("\n" + "="*60)
        print("📊 COMPARAISON DES MODÈLES")
        print("="*60)
        
        comparison = []
        
        # Modèles régionaux
        print("\n🏘️  Modèles Régionaux (Fog):")
        for region in self.fog_weights:
            rate = region['total_anomalies'] / region['total_samples'] * 100
            comparison.append({
                'Niveau': f"Fog - {region['region_name']}",
                'Villages': region['num_villages'],
                'Échantillons': region['total_samples'],
                'Anomalies': region['total_anomalies'],
                'Taux': f"{rate:.2f}%"
            })
            print(f"   • {region['region_name']}: {rate:.2f}% anomalies")
        
        # Modèle global
        print("\n☁️  Modèle Global (Cloud):")
        global_rate = self.global_model['global_anomaly_rate'] * 100
        comparison.append({
            'Niveau': 'Cloud - Global',
            'Villages': self.global_model['total_villages'],
            'Échantillons': self.global_model['total_samples'],
            'Anomalies': self.global_model['total_anomalies'],
            'Taux': f"{global_rate:.2f}%"
        })
        print(f"   • Modèle fusionné: {global_rate:.2f}% anomalies")
        
        df = pd.DataFrame(comparison)
        print("\n" + "="*60)
        print(df.to_string(index=False))
        
        return df
    
    def generate_insights(self):
        """Génère des insights pour la SOMELEC"""
        print("\n" + "="*60)
        print("💡 INSIGHTS POUR LA SOMELEC")
        print("="*60)
        
        insights = []
        
        # Analyse par région
        for region in self.fog_weights:
            rate = region['total_anomalies'] / region['total_samples']
            
            if rate > 0.15:
                severity = "🔴 CRITIQUE"
                action = "Intervention urgente requise"
            elif rate > 0.10:
                severity = "🟠 ATTENTION"
                action = "Surveillance renforcée recommandée"
            else:
                severity = "🟢 NORMAL"
                action = "Maintenance préventive standard"
            
            insight = {
                'région': region['region_name'],
                'villages': region['num_villages'],
                'taux_anomalie': f"{rate*100:.2f}%",
                'statut': severity,
                'recommandation': action
            }
            insights.append(insight)
            
            print(f"\n{severity} {region['region_name']}")
            print(f"   Taux d'anomalie: {rate*100:.2f}%")
            print(f"   Villages concernés: {region['num_villages']}")
            print(f"   ➜ {action}")
        
        # Statistiques globales
        global_rate = self.global_model['global_anomaly_rate']
        print(f"\n📈 Vue d'ensemble nationale:")
        print(f"   Taux d'anomalie moyen: {global_rate*100:.2f}%")
        print(f"   Villages surveillés: {self.global_model['total_villages']}")
        print(f"   Total anomalies: {self.global_model['total_anomalies']}")
        
        # Estimation de l'impact
        print(f"\n💰 Estimation d'impact:")
        cost_per_anomaly = 50000  # MRU (exemple)
        potential_savings = self.global_model['total_anomalies'] * cost_per_anomaly
        print(f"   Coût estimé des pannes: {potential_savings:,.0f} MRU")
        print(f"   Économies potentielles avec détection précoce: {potential_savings*0.7:,.0f} MRU")
        
        return insights


def run_federated_learning():
    """Exécute le processus complet de Federated Learning"""
    print("\n" + "="*70)
    print("☁️  SERVEUR CLOUD - FEDERATED LEARNING")
    print("="*70)
    
    # Initialiser le serveur
    server = FederatedServer()
    
    # Charger les poids Fog
    server.load_fog_weights()
    
    # Exécuter FedAvg
    global_model = server.federated_averaging()
    
    # Sauvegarder
    server.save_global_model()
    server.save_federation_history()
    
    # Analyses
    comparison = server.compare_models()
    insights = server.generate_insights()
    
    return server, global_model


def simulate_multiple_rounds(num_rounds=3):
    """Simule plusieurs rounds de Federated Learning"""
    print("\n" + "="*70)
    print(f"🔄 SIMULATION DE {num_rounds} ROUNDS DE FEDERATED LEARNING")
    print("="*70)
    
    server = FederatedServer()
    
    for round_num in range(num_rounds):
        print(f"\n{'='*70}")
        print(f"📍 ROUND {round_num + 1}/{num_rounds}")
        print(f"{'='*70}")
        
        server.load_fog_weights()
        server.federated_averaging()
        
        # Simuler une amélioration progressive
        if round_num < num_rounds - 1:
            print("\n⏳ Attente du prochain round...")
    
    server.save_global_model()
    server.save_federation_history()
    
    print("\n" + "="*70)
    print("📈 ÉVOLUTION SUR LES ROUNDS")
    print("="*70)
    
    for i, hist in enumerate(server.federation_history):
        print(f"Round {i+1}: {hist['global_anomaly_rate']*100:.2f}% anomalies")
    
    return server


if __name__ == "__main__":
    # Option 1: Run simple
    server, global_model = run_federated_learning()
    
    # Option 2: Simuler plusieurs rounds (décommentez si souhaité)
    # server = simulate_multiple_rounds(num_rounds=3)
    
    print("\n" + "="*70)
    print("✅ FEDERATED LEARNING TERMINÉ AVEC SUCCÈS!")
    print("="*70)
    print("\n🎯 Avantages du Federated Learning:")
    print("   ✅ Confidentialité: Aucune donnée brute partagée")
    print("   ✅ Efficacité: Réduction de la bande passante")
    print("   ✅ Robustesse: Modèle global plus performant")
    print("   ✅ Scalabilité: Ajout facile de nouveaux villages")
    
    print("\n📂 Fichiers générés:")
    print("   • models/global_model.json")
    print("   • models/federation_history.json")
    
    print("\n🚀 Prochaine étape: Dashboard de visualisation!")