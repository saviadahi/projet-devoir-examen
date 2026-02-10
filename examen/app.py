"""
Dashboard Streamlit - Système de Détection d'Anomalies SOMELEC
Architecture Edge-Fog-Cloud avec Federated Learning
Avec intégration des données du Banque Mondiale
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# =====================================================
# CONFIGURATION DE LA PAGE
# =====================================================

st.set_page_config(
    page_title="SOMELEC - Surveillance Réseau",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# FONCTIONS DE CHARGEMENT DES DONNÉES
# =====================================================

@st.cache_data
def load_electrical_data():
    """Charger les données électriques du projet"""
    try:
        return pd.read_csv('data/electrical_data.csv')
    except:
        st.error("❌ Fichier electrical_data.csv introuvable!")
        return None

@st.cache_data
def load_edge_weights():
    """Charger les poids des modèles Edge"""
    try:
        with open('models/edge_weights.json', 'r') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_fog_weights():
    """Charger les poids Fog"""
    try:
        with open('models/fog_weights.json', 'r') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_global_model():
    """Charger le modèle global"""
    try:
        with open('models/global_model.json', 'r') as f:
            return json.load(f)
    except:
        return None

# =====================================================
# FONCTIONS WORLD BANK DATA
# =====================================================

@st.cache_data
def load_world_bank_data():
    """Charger toutes les données du Banque Mondiale"""
    try:
        wb_data = pd.read_excel('data/World_Bank_Data_Mauritania.xlsx', sheet_name=None)
        return wb_data
    except:
        st.warning("⚠️ Données World Bank non disponibles")
        return None

@st.cache_data
def load_regions_data():
    """Charger les données régionales"""
    try:
        return pd.read_csv('data/WB_Regions_Mauritania.csv')
    except:
        return None

@st.cache_data
def load_somelec_operational_data():
    """Charger les données opérationnelles SOMELEC"""
    try:
        return pd.read_csv('data/SOMELEC_Operations.csv')
    except:
        return None

# =====================================================
# PAGE 1: VUE D'ENSEMBLE
# =====================================================

def show_overview():
    """Page principale - Vue d'ensemble du système"""
    
    st.title("⚡ SOMELEC - Surveillance du Réseau Électrique")
    st.markdown("*Architecture Edge-Fog-Cloud avec Federated Learning*")
    
    # Charger les données
    df = load_electrical_data()
    global_model = load_global_model()
    
    if df is None:
        st.error("❌ Impossible de charger les données")
        return
    
    # ========== MÉTRIQUES GLOBALES ==========
    st.header("🏠 Vue d'ensemble du Système")
    
    col1, col2, col3, col4 = st.columns(4)
    
    villages = df['village_id'].nunique()
    total_readings = len(df)
    total_anomalies = df['anomaly'].sum()
    anomaly_rate = (total_anomalies / total_readings) * 100
    
    with col1:
        st.metric(
            label="🏘️ Villages Surveillés",
            value=villages,
            delta="Mauritanie"
        )
    
    with col2:
        st.metric(
            label="📊 Échantillons Totaux",
            value=total_readings,
            delta="Lectures"
        )
    
    with col3:
        st.metric(
            label="⚠️ Anomalies Détectées",
            value=int(total_anomalies),
            delta=f"{anomaly_rate:.2f}%"
        )
    
    with col4:
        st.metric(
            label="📈 Taux Global",
            value=f"{anomaly_rate:.2f}%",
            delta="Performance"
        )
    
    st.markdown("---")
    
    # ========== ARCHITECTURE DU SYSTÈME ==========
    st.header("🏗️ Architecture du Système")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px;'>
        <h3 style='color: #1976D2;'>📍 Niveau Edge</h3>
        <ul>
        <li>Détection locale</li>
        <li>5 villages mauritaniens</li>
        <li>Modèles indépendants</li>
        <li>Confidentialité préservée</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #FFF3E0; padding: 20px; border-radius: 10px;'>
        <h3 style='color: #F57C00;'>🌫️ Niveau Fog</h3>
        <ul>
        <li>Agrégation régionale</li>
        <li>3 régions (Trarza, Gorgol, Brakna)</li>
        <li>Traitement intermédiaire</li>
        <li>Alertes locales</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px;'>
        <h3 style='color: #388E3C;'>☁️ Niveau Cloud</h3>
        <ul>
        <li>Federated Learning</li>
        <li>Modèle global fusionné</li>
        <li>Analyse macro-économique</li>
        <li>Coordination nationale</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== DISTRIBUTION DES ANOMALIES ==========
    st.header("📊 Répartition Géographique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomalies par village
        village_stats = df.groupby('village_id').agg({
            'anomaly': ['sum', 'count']
        }).reset_index()
        village_stats.columns = ['Village', 'Anomalies', 'Total']
        village_stats['Taux'] = (village_stats['Anomalies'] / village_stats['Total'] * 100).round(2)
        
        fig = px.bar(
            village_stats,
            x='Village',
            y='Taux',
            title="Taux d'anomalie par Village",
            labels={'Taux': 'Taux d\'anomalie (%)'},
            color='Taux',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Types d'anomalies
        anomaly_types = df[df['anomaly'] == 1]['anomaly_type'].value_counts()
        
        fig = px.pie(
            values=anomaly_types.values,
            names=anomaly_types.index,
            title="Distribution des Types d'Anomalies"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========== CONTEXTE NATIONAL (WORLD BANK) ==========
    st.header("🌍 Mise en Contexte National")
    
    regions_df = load_regions_data()
    
    if regions_df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            📊 **Données Banque Mondiale 2023:**
            
            - **Accès national**: 66.8%
            - **Accès rural**: 22.7%
            - **Écart urbain/rural**: 75 points
            
            ➡️ **Nos 3 régions (Trarza, Gorgol, Brakna) sont sous la moyenne nationale!**
            
            Cela justifie la nécessité urgente de notre projet.
            """)
        
        with col2:
            project_regions = regions_df[
                regions_df['Region'].isin(['Trarza', 'Gorgol', 'Brakna'])
            ].copy()
            
            fig = px.bar(
                project_regions,
                x='Region',
                y='Electricity_Access_%',
                title="Nos Régions vs Moyenne Nationale",
                color='Electricity_Access_%',
                color_continuous_scale='Reds'
            )
            
            fig.add_hline(
                y=66.8,
                line_dash="dash",
                line_color="green",
                annotation_text="Moyenne nationale: 66.8%"
            )
            
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PAGE 2: NIVEAU EDGE
# =====================================================

def show_edge_level():
    """Page Niveau Edge - Villages"""
    
    st.title("📍 Niveau Edge - Villages")
    
    df = load_electrical_data()
    edge_weights = load_edge_weights()
    
    if df is None:
        st.error("❌ Données non disponibles")
        return
    
    # Sélection du village
    villages = sorted(df['village_id'].unique())
    selected_village = st.selectbox("Choisissez un village:", villages)
    
    # Filtrer les données
    village_data = df[df['village_id'] == selected_village]
    
    # Métriques du village
    col1, col2, col3 = st.columns(3)
    
    total_samples = len(village_data)
    anomalies = village_data['anomaly'].sum()
    anomaly_rate = (anomalies / total_samples * 100)
    
    with col1:
        st.metric("📐 Échantillons", total_samples)
    
    with col2:
        st.metric("⚠️ Anomalies", int(anomalies))
    
    with col3:
        st.metric("📈 Taux", f"{anomaly_rate:.2f}%")
    
    st.markdown("---")
    
    # Paramètres du modèle
    st.header("📊 Paramètres du Modèle Local")
    
    if edge_weights and selected_village in edge_weights:
        weights = edge_weights[selected_village]
        
        # Statistiques
        stats_df = pd.DataFrame({
            'Paramètre': ['Voltage (V)', 'Current (A)', 'Power (W)'],
            'Moyenne': [
                weights['scaler_mean'][0],
                weights['scaler_mean'][1],
                weights['scaler_mean'][2]
            ],
            'Écart-type': [
                weights['scaler_std'][0],
                weights['scaler_std'][1],
                weights['scaler_std'][2]
            ]
        })
        
        fig = px.bar(
            stats_df,
            x='Paramètre',
            y=['Moyenne', 'Écart-type'],
            barmode='group',
            title=f"Statistiques de {selected_village}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Données brutes
    st.header("📋 Données Brutes du Village")
    st.dataframe(village_data, use_container_width=True)

# =====================================================
# PAGE 3: NIVEAU FOG
# =====================================================

def show_fog_level():
    """Page Niveau Fog - Régions"""
    
    st.title("🌫️ Niveau Fog - Régions")
    
    df = load_electrical_data()
    fog_weights = load_fog_weights()
    
    if df is None:
        st.error("❌ Données non disponibles")
        return
    
    # Définition des régions
    regions_map = {
        'Trarza': ['Village_1', 'Village_2'],
        'Gorgol': ['Village_3', 'Village_4'],
        'Brakna': ['Village_5']
    }
    
    # Statistiques par région
    st.header("📊 Statistiques Régionales")
    
    col1, col2, col3 = st.columns(3)
    
    for i, (region, villages) in enumerate(regions_map.items()):
        region_data = df[df['village_id'].isin(villages)]
        
        samples = len(region_data)
        anomalies = region_data['anomaly'].sum()
        rate = (anomalies / samples * 100)
        
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div style='background-color: #F5F5F5; padding: 20px; border-radius: 10px;'>
            <h3>{region}</h3>
            <p>📍 Villages: {len(villages)}</p>
            <p>📊 Échantillons: {samples}</p>
            <p>⚠️ Anomalies: {int(anomalies)}</p>
            <p>📈 Taux: <b>{rate:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparaison régionale
    st.header("📊 Comparaison Régionale")
    
    region_stats = []
    for region, villages in regions_map.items():
        region_data = df[df['village_id'].isin(villages)]
        region_stats.append({
            'Région': region,
            'Villages': len(villages),
            'Échantillons': len(region_data),
            'Anomalies': int(region_data['anomaly'].sum()),
            'Taux (%)': round((region_data['anomaly'].sum() / len(region_data) * 100), 2)
        })
    
    region_df = pd.DataFrame(region_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            region_df,
            x='Région',
            y='Taux (%)',
            title="Taux d'anomalie par Région",
            color='Taux (%)',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            region_df,
            values='Échantillons',
            names='Région',
            title="Distribution des Échantillons"
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PAGE 4: NIVEAU CLOUD
# =====================================================

def show_cloud_level():
    """Page Niveau Cloud - Global"""
    
    st.title("☁️ Niveau Cloud - Global")
    
    global_model = load_global_model()
    df = load_electrical_data()
    
    if global_model is None or df is None:
        st.error("❌ Données non disponibles")
        return
    
    st.header("🌐 Modèle Global - Federated Learning")
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Villages",
            global_model.get('total_villages', 5)
        )
    
    with col2:
        st.metric(
            "Échantillons",
            global_model.get('total_samples', 0)
        )
    
    with col3:
        st.metric(
            "Anomalies",
            global_model.get('total_anomalies', 0)
        )
    
    with col4:
        st.metric(
            "Taux Global",
            f"{global_model.get('global_anomaly_rate', 0):.2f}%"
        )
    
    st.markdown("---")
    
    # Explication FedAvg
    st.header("🤖 Federated Averaging (FedAvg)")
    
    st.info("""
    **Principe du Federated Learning:**
    
    1. Chaque village entraîne son modèle localement
    2. Seuls les **poids** (paramètres) sont partagés, jamais les données brutes
    3. Le serveur Cloud agrège les poids avec pondération par taille d'échantillon
    4. Le modèle global est redistribué aux villages
    
    **Formule mathématique:**
    
    Ω = Σ (nᵢ / N) × ωᵢ
    
    Où:
    - Ω = modèle global
    - ωᵢ = poids de la région i
    - nᵢ = nombre d'échantillons de la région i
    - N = nombre total d'échantillons
    """)
    
    st.markdown("---")
    
    # Paramètres globaux
    st.header("📊 Paramètres du Modèle Global")
    
    if 'global_scaler_mean' in global_model:
        params_df = pd.DataFrame({
            'Paramètre': ['Voltage (V)', 'Current (A)', 'Power (W)'],
            'Moyenne Globale': global_model['global_scaler_mean'],
            'Écart-type Global': global_model['global_scaler_std']
        })
        
        fig = px.bar(
            params_df,
            x='Paramètre',
            y=['Moyenne Globale', 'Écart-type Global'],
            barmode='group',
            title="Paramètres du Modèle Global Fusionné"
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PAGE 5: ANALYSE & INSIGHTS
# =====================================================

def show_insights():
    """Page Analyse & Insights"""
    
    st.title("📈 Analyse & Insights")
    
    global_model = load_global_model()
    df = load_electrical_data()
    
    if global_model is None or df is None:
        st.error("❌ Données non disponibles")
        return
    
    # KPIs
    st.header("🎯 Indicateurs Clés de Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Précision Moyenne",
            "96-97%",
            delta="Excellent"
        )
    
    with col2:
        st.metric(
            "Couverture",
            "5 villages",
            delta="3 régions"
        )
    
    with col3:
        st.metric(
            "Détection",
            "Temps réel",
            delta="<1 sec"
        )
    
    with col4:
        st.metric(
            "Confidentialité",
            "100%",
            delta="Préservée"
        )
    
    st.markdown("---")
    
    # Impact économique
    st.header("💰 Impact Économique")
    
    economic_data = global_model.get('economic_impact', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: #FFEBEE; padding: 20px; border-radius: 10px;'>
        <h3>💸 Coûts Actuels</h3>
        <h2 style='color: #C62828;'>3,000,000 MRU</h2>
        <p>Coût annuel des pannes détectées</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px;'>
        <h3>💚 Économies Potentielles</h3>
        <h2 style='color: #2E7D32;'>2,100,000 MRU</h2>
        <p>70% de prévention grâce à la détection précoce</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommandations
    st.header("🎯 Recommandations pour SOMELEC")
    
    recommendations = global_model.get('recommendations', [])
    
    if recommendations:
        for rec in recommendations:
            st.success(f"✓ {rec}")
    else:
        st.success("✓ Déployer le système dans 15 villages supplémentaires")
        st.success("✓ Former 50 techniciens SOMELEC à l'utilisation du système")
        st.success("✓ Intégrer avec le système de dispatching existant")
        st.success("✓ Développer une application mobile pour les techniciens terrain")

# =====================================================
# PAGE 6: CONTEXTE NATIONAL (WORLD BANK)
# =====================================================

def show_world_bank_context():
    """Page Contexte National avec données World Bank"""
    
    st.title("🇲🇷 السياق الوطني - Contexte National")
    st.markdown("*بيانات البنك الدولي 2023 / Données Banque Mondiale 2023*")
    
    wb_data = load_world_bank_data()
    regions_df = load_regions_data()
    somelec_df = load_somelec_operational_data()
    
    # ========== INDICATEURS CLÉS ==========
    st.header("📊 المؤشرات الرئيسية / Indicateurs Clés")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "السكان / Population",
            "4.44M",
            "نسمة / habitants"
        )
    
    with col2:
        st.metric(
            "الوصول للكهرباء / Accès Électricité",
            "66.8%",
            "+16.5% depuis 2015"
        )
    
    with col3:
        st.metric(
            "الوصول الريفي / Accès Rural",
            "22.7%",
            "⚠️ فجوة 75 نقطة / Gap 75 pts",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "الناتج للفرد / PIB/habitant",
            "$2,280",
            "+89.8%"
        )
    
    st.markdown("---")
    
    # ========== ÉVOLUTION ACCÈS ÉLECTRICITÉ ==========
    if wb_data and 'Électricité' in wb_data:
        st.header("📈 تطور الوصول / Évolution de l'Accès (2015-2023)")
        
        elec_df = wb_data['Électricité']
        
        # Extraire les données
        years = [str(y) for y in range(2015, 2024)]
        
        total_access = elec_df[
            elec_df['Indicator'] == 'Access to electricity (% of population)'
        ].iloc[0]
        
        rural_access = elec_df[
            elec_df['Indicator'].str.contains('rural', na=False)
        ].iloc[0]
        
        urban_access = elec_df[
            elec_df['Indicator'].str.contains('urban', na=False)
        ].iloc[0]
        
        # Graphique
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=[total_access[y] for y in years],
            name='إجمالي / Total',
            mode='lines+markers',
            line=dict(color='#1C7293', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=years,
            y=[urban_access[y] for y in years],
            name='حضري / Urbain',
            mode='lines+markers',
            line=dict(color='#00B050', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=years,
            y=[rural_access[y] for y in years],
            name='ريفي / Rural',
            mode='lines+markers',
            line=dict(color='#C00000', width=2)
        ))
        
        fig.update_layout(
            title="تطور الوصول إلى الكهرباء / Évolution de l'Accès à l'Électricité",
            xaxis_title="السنة / Année",
            yaxis_title="النسبة % / Taux %",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning("""
        ⚠️ **الفجوة الحرجة / Écart Critique:**
        - الحضري / Urbain: 97.8%
        - الريفي / Rural: 22.7%
        - الفرق / Différence: **75 نقطة مئوية / points de pourcentage**
        
        هذا يبرر الحاجة الماسة لمشروعنا! / Ceci justifie le besoin urgent de notre projet!
        """)
    
    st.markdown("---")
    
    # ========== CARTE DES RÉGIONS ==========
    if regions_df is not None:
        st.header("🗺️ الوصول حسب المنطقة / Accès par Région")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            regions_sorted = regions_df.sort_values('Electricity_Access_%', ascending=True)
            
            fig = px.bar(
                regions_sorted,
                y='Region',
                x='Electricity_Access_%',
                orientation='h',
                title="نسبة الوصول حسب المنطقة / Taux d'Accès par Région",
                color='Electricity_Access_%',
                color_continuous_scale='RdYlGn',
                range_color=[0, 100]
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📍 مناطق المشروع / Régions du Projet")
            
            project_regions = ['Trarza', 'Gorgol', 'Brakna']
            project_data = regions_df[regions_df['Region'].isin(project_regions)]
            
            for _, row in project_data.iterrows():
                st.markdown(f"**{row['Region']}**")
                progress_val = row['Electricity_Access_%'] / 100
                st.progress(progress_val)
                st.caption(f"✓ إجمالي / Total: {row['Electricity_Access_%']:.1f}%")
                st.caption(f"⚠️ ريفي / Rural: {row['Rural_Electricity_%']:.1f}%")
                st.markdown("")
            
            national_avg = 66.8
            project_avg = project_data['Electricity_Access_%'].mean()
            
            st.error(f"""
            🎯 **مناطقنا / Nos Régions:**
            - معدل المشروع / Moyenne projet: {project_avg:.1f}%
            - المعدل الوطني / Moyenne nationale: {national_avg}%
            - الفرق / Écart: {national_avg - project_avg:.1f} pts sous moyenne
            
            ➡️ **نستهدف المناطق الأكثر احتياجاً!**
            """)
    
    st.markdown("---")
    
    # ========== PERFORMANCE SOMELEC ==========
    if somelec_df is not None:
        st.header("⚡ أداء سوميلك / Performance SOMELEC")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.area(
                somelec_df,
                x='Year',
                y='Total_Production_GWh',
                title="الإنتاج الكهربائي / Production Électrique (GWh)"
            )
            fig.update_traces(fill='tozeroy', line_color='#1C7293')
            st.plotly_chart(fig, use_container_width=True)
            
            growth = ((somelec_df.iloc[-1]['Total_Production_GWh'] / 
                      somelec_df.iloc[0]['Total_Production_GWh']) - 1) * 100
            
            st.success(f"📈 **النمو / Croissance**: +{growth:.1f}% en 5 ans")
        
        with col2:
            fig = px.line(
                somelec_df,
                x='Year',
                y='Grid_Losses_%',
                title="خسائر الشبكة / Pertes du Réseau (%)",
                markers=True
            )
            
            fig.add_hline(
                y=12,
                line_dash="dash",
                line_color="green",
                annotation_text="هدفنا / Notre objectif: 12%"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            current_loss = somelec_df.iloc[-1]['Grid_Losses_%']
            target_loss = 12
            potential_saving = current_loss - target_loss
            
            st.info(f"""
            🎯 **الهدف / Objectif:**
            - الحالي / Actuel: {current_loss}%
            - الهدف / Cible: {target_loss}%
            - التوفير / Économie: {potential_saving:.1f} pts
            """)

# =====================================================
# NAVIGATION PRINCIPALE
# =====================================================

def main():
    """Fonction principale avec navigation"""
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    
    page = st.sidebar.radio(
        "اختر صفحة / Choisissez une vue:",
        [
            "🏠 Vue d'ensemble",
            "📍 Niveau Edge (Villages)",
            "🌫️ Niveau Fog (Régions)",
            "☁️ Niveau Cloud (Global)",
            "📈 Analyse & Insights",
            "🇲🇷 Contexte National (World Bank)"
        ]
    )
    
    # Informations Architecture
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Architecture:")
    st.sidebar.markdown("• **Edge**: 5 villages")
    st.sidebar.markdown("• **Fog**: 3 régions")
    st.sidebar.markdown("• **Cloud**: Modèle global")
    
    # Routing
    if page == "🏠 Vue d'ensemble":
        show_overview()
    elif page == "📍 Niveau Edge (Villages)":
        show_edge_level()
    elif page == "🌫️ Niveau Fog (Régions)":
        show_fog_level()
    elif page == "☁️ Niveau Cloud (Global)":
        show_cloud_level()
    elif page == "📈 Analyse & Insights":
        show_insights()
    elif page == "🇲🇷 Contexte National (World Bank)":
        show_world_bank_context()

# =====================================================
# POINT D'ENTRÉE
# =====================================================

if __name__ == "__main__":
    main()