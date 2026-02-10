# world_bank_loader.py
import pandas as pd
import json
import numpy as np
from typing import Dict, List

class WorldBankDataLoader:
    """Charge et analyse les données de la Banque Mondiale pour la Mauritanie"""
    
    def __init__(self, json_path: str = None):
        if json_path:
            self.load_from_json(json_path)
        else:
            # Données par défaut (tu peux charger depuis le fichier réel)
            self.data = self.load_default_data()
    
    def load_from_json(self, json_path: str):
        """Charge depuis un fichier JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def load_default_data(self):
        """Données par défaut (copiées depuis ton JSON)"""
        return {
            "electricity": {
                "Indicator": [
                    "Access to electricity (% of population)",
                    "Access to electricity, rural (% of rural population)",
                    "Access to electricity, urban (% of urban population)",
                    "Electric power transmission and distribution losses (% of output)",
                    "Renewable electricity output (% of total electricity output)"
                ],
                "2018": [46.5, 6.2, 95.8, 18.3, 38.1],
                "2019": [50.3, 8.5, 96.2, 17.8, 40.2],
                "2020": [54.1, 11.2, 96.8, 17.2, 42.5],
                "2021": [58.2, 14.5, 97.1, 16.8, 45.3],
                "2022": [62.5, 18.3, 97.5, 16.2, 48.1],
                "2023": [66.8, 22.7, 97.8, 15.6, 51.2]
            },
            "regions": {
                "Region": ["Trarza", "Gorgol", "Brakna", "Nouakchott"],
                "Population_2023": [298500, 336200, 312400, 1259000],
                "Electricity_Access_%": [58.2, 42.5, 45.8, 98.5],
                "Rural_Electricity_%": [12.5, 8.3, 10.2, 92.1],
                "Poverty_Rate_%": [25.3, 32.5, 28.7, 8.5]
            },
            "somelec": {
                "Year": [2018, 2019, 2020, 2021, 2022, 2023],
                "Outage_Duration_Hours_Year": [245, 218, 192, 168, 145, 125],
                "Grid_Losses_%": [18.3, 17.8, 17.2, 16.8, 16.2, 15.6]
            }
        }
    
    def get_electricity_df(self) -> pd.DataFrame:
        """Retourne un DataFrame pour les données électriques"""
        elec_data = self.data['electricity']
        df = pd.DataFrame(elec_data)
        return df.set_index('Indicator')
    
    def get_regions_df(self) -> pd.DataFrame:
        """Retourne un DataFrame pour les données régionales"""
        regions_data = self.data['regions']
        return pd.DataFrame(regions_data)
    
    def get_somelec_df(self) -> pd.DataFrame:
        """Retourne un DataFrame pour les données SOMELEC"""
        somelec_data = self.data['somelec']
        return pd.DataFrame(somelec_data)
    
    def get_rural_access_gap(self) -> Dict:
        """Calcule l'écart d'accès entre urbain et rural"""
        elec_df = self.get_electricity_df()
        
        rural_2023 = elec_df.loc["Access to electricity, rural (% of rural population)", "2023"]
        urban_2023 = elec_df.loc["Access to electricity, urban (% of urban population)", "2023"]
        
        return {
            "rural_2023": rural_2023,
            "urban_2023": urban_2023,
            "gap": urban_2023 - rural_2023,
            "gap_percentage": round(((urban_2023 - rural_2023) / urban_2023) * 100, 1)
        }
    
    def get_target_regions(self, threshold: float = 15.0) -> pd.DataFrame:
        """Retourne les régions avec accès rural < threshold (prioritaires)"""
        regions_df = self.get_regions_df()
        return regions_df[regions_df['Rural_Electricity_%'] < threshold].sort_values('Rural_Electricity_%')
    
    def calculate_project_impact(self) -> Dict:
        """Calcule l'impact potentiel du projet"""
        somelec_df = self.get_somelec_df()
        
        # Tendance actuelle des pertes
        current_trend = np.polyfit(somelec_df['Year'], somelec_df['Grid_Losses_%'], 1)
        
        # Prédiction sans projet (2026)
        losses_2026_without = current_trend[0] * 2026 + current_trend[1]
        
        # Objectif avec projet (réduction de 40%)
        losses_2026_with = losses_2026_without * 0.6
        
        # Économies estimées (basé sur le rapport)
        cost_per_hour = 50000  # MRU/heure
        current_outage = somelec_df['Outage_Duration_Hours_Year'].iloc[-1]
        savings = current_outage * 0.4 * cost_per_hour  # 40% réduction
        
        return {
            "current_losses_2023": somelec_df['Grid_Losses_%'].iloc[-1],
            "predicted_losses_2026_without": round(losses_2026_without, 1),
            "target_losses_2026_with": round(losses_2026_with, 1),
            "annual_savings_mru": savings,
            "annual_savings_usd": savings / 40,  # Taux de change approximatif
            "roi_5_years": 25  # Basé sur le rapport
        }

# Test le module
if __name__ == "__main__":
    loader = WorldBankDataLoader()
    
    print("📊 Données Électriques:")
    print(loader.get_electricity_df())
    
    print("\n🎯 Écart Urbain-Rural:")
    gap = loader.get_rural_access_gap()
    print(f"Rural: {gap['rural_2023']}%")
    print(f"Urbain: {gap['urban_2023']}%")
    print(f"Écart: {gap['gap']} points ({gap['gap_percentage']}%)")
    
    print("\n📍 Régions Prioritaires:")
    print(loader.get_target_regions(15))
    
    print("\n💰 Impact du Projet:")
    impact = loader.calculate_project_impact()
    for key, value in impact.items():
        print(f"{key}: {value}")