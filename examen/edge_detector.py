"""
Détecteur d'anomalies Edge pour chaque village
Utilise Isolation Forest pour détecter les anomalies dans le réseau électrique
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle
import json
from datetime import datetime

class EdgeAnomalyDetector:
    def __init__(self, village_id, contamination=0.1):
        self.village_id = village_id
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.is_trained = False
        self.training_stats = {}
        
    def prepare_features(self, df):
        features = df[['voltage', 'current', 'power']].values
        return features
    
    def train(self, df):
        print(f"\n🔧 Entraînement du modèle Edge pour {self.village_id}")
        
        village_data = df[df['village_id'] == self.village_id].copy()
        
        if len(village_data) < 10:
            print(f"⚠️  Pas assez de données pour {self.village_id}")
            return False
        
        X = self.prepare_features(village_data)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        
        predictions = self.model.predict(X_scaled)
        anomalies_detected = (predictions == -1).sum()
        
        self.training_stats = {
            'village_id': self.village_id,
            'samples_trained': len(village_data),
            'anomalies_detected': int(anomalies_detected),
            'anomaly_rate': float(anomalies_detected / len(village_data)),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Entraînement terminé: {anomalies_detected} anomalies détectées")
        print(f"📈 Taux: {self.training_stats['anomaly_rate']*100:.2f}%")
        
        return True
    
    def predict(self, df):
        if not self.is_trained:
            return df
        
        village_data = df[df['village_id'] == self.village_id].copy()
        if len(village_data) == 0:
            return df
        
        X = self.prepare_features(village_data)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        
        village_data['predicted_anomaly'] = (predictions == -1).astype(int)
        village_data['anomaly_score'] = scores
        
        return village_data
    
    def get_model_weights(self):
        if not self.is_trained:
            return None
        
        return {
            'village_id': self.village_id,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'contamination': self.contamination,
            'training_stats': self.training_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_model(self, filepath=None):
        if filepath is None:
            filepath = f"models/edge_{self.village_id}.pkl"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'village_id': self.village_id,
            'training_stats': self.training_stats,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Modèle sauvegardé: {filepath}")
        return filepath
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.village_id = model_data['village_id']
        self.training_stats = model_data['training_stats']
        self.is_trained = model_data['is_trained']
        
        return True


def train_all_edge_models(data_path='data/electrical_data.csv'):
    print("\n" + "="*60)
    print("🏭 ENTRAÎNEMENT DES MODÈLES EDGE (Niveau Village)")
    print("="*60)
    
    df = pd.read_csv(data_path)
    print(f"\n📊 Données chargées: {len(df)} lectures")
    
    villages = df['village_id'].unique()
    print(f"🏘️  Nombre de villages: {len(villages)}")
    
    edge_models = {}
    all_weights = []
    
    for village in villages:
        detector = EdgeAnomalyDetector(village, contamination=0.1)
        success = detector.train(df)
        
        if success:
            edge_models[village] = detector
            detector.save_model()
            weights = detector.get_model_weights()
            all_weights.append(weights)
    
    weights_path = 'models/edge_weights.json'
    with open(weights_path, 'w') as f:
        json.dump(all_weights, f, indent=2)
    
    print(f"\n💾 Poids Edge sauvegardés: {weights_path}")
    print(f"✅ {len(edge_models)} modèles entraînés!\n")
    
    return edge_models, all_weights


def test_edge_detection(data_path='data/electrical_data.csv'):
    print("\n" + "="*60)
    print("🧪 TEST DE DÉTECTION D'ANOMALIES")
    print("="*60)
    
    df = pd.read_csv(data_path)
    villages = df['village_id'].unique()
    results = []
    
    for village in villages:
        try:
            detector = EdgeAnomalyDetector(village)
            detector.load_model(f'models/edge_{village}.pkl')
            
            predictions = detector.predict(df)
            
            true_anomalies = int(predictions['anomaly'].sum())
            predicted_anomalies = int(predictions['predicted_anomaly'].sum())
            
            correct = (predictions['anomaly'] == predictions['predicted_anomaly']).sum()
            accuracy = round(correct / len(predictions) * 100, 2)
            
            results.append({
                'village': village,
                'vraies_anomalies': true_anomalies,
                'predites': predicted_anomalies,
                'precision': accuracy
            })
            
            print(f"\n📍 {village}:")
            print(f"   Vraies: {true_anomalies} | Prédites: {predicted_anomalies}")
            print(f"   Précision: {accuracy}%")
            
        except Exception as e:
            print(f"⚠️  Erreur {village}: {e}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    edge_models, weights = train_all_edge_models()
    results = test_edge_detection()
    
    print("\n" + "="*60)
    print("📊 RÉSUMÉ GLOBAL")
    print("="*60)
    print(results.to_string(index=False))
    print("\n✅ Phase Edge terminée!")