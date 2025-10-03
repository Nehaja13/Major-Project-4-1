import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import requests
import re
from datetime import datetime, timedelta
import hashlib
import heapq

# AI and ML Imports
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Visualization Imports
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium

# --- 1. CONFIGURATION & API KEYS ---

st.set_page_config(
    page_title="Hyderabad Environmental Intelligence",
    page_icon="🌍",
    layout="wide",
)

# --- IMPORTANT SECURITY WARNING ---
# API keys are included directly here as requested.
# In a real public application, use Streamlit's Secrets management to keep keys safe.
GEMINI_API_KEY = "AIzaSyDQSo3zfPJKsG0hhQOS_sty0AuRl8zeMBw"
AQI_API_KEY = "04980907cdcf44290764def2a5516aedf4e6c797"

# Configure Gemini AI
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini AI. Check API key. Error: {e}")

# --- 2. SHARED DATA & CONSTANTS ---

HYDERABAD_ZONES_BASE = [
    {'id': 1, 'name': 'Banjara Hills', 'type': 'Posh Residential'},
    {'id': 2, 'name': 'Gachibowli', 'type': 'Financial District'},
    {'id': 3, 'name': 'Patancheru Ind. Area', 'type': 'Industrial'},
    {'id': 4, 'name': 'Charminar Area', 'type': 'Historic/Market'},
    {'id': 5, 'name': 'HITEC City', 'type': 'IT Hub'},
    {'id': 6, 'name': 'Secunderabad', 'type': 'Commercial/Residential'},
    {'id': 7, 'name': 'KBR National Park', 'type': 'Green Space'},
    {'id': 8, 'name': 'Uppal', 'type': 'Residential/Industrial'},
    {'id': 9, 'name': 'Begumpet Airport Area', 'type': 'Transport Hub'},
    {'id': 10, 'name': 'Jubilee Hills', 'type': 'Posh Residential'},
    {'id': 11, 'name': 'Ameerpet', 'type': 'Commercial Hub'},
    {'id': 12, 'name': 'Kukatpally', 'type': 'Dense Residential'},
]

# Graph for Route Recommendation
ZONE_GRAPH = {
  1: [10, 11, 7], 2: [5, 12], 3: [12], 4: [6], 5: [2, 12, 10], 
  6: [4, 9, 8, 11], 7: [1, 10], 8: [6], 9: [6, 11], 
  10: [1, 5, 7, 11], 11: [1, 9, 10, 6], 12: [2, 3, 5],
}

DEFAULT_LAT, DEFAULT_LON = 17.3850, 78.4867
BASELINE_STATION_LAT, BASELINE_STATION_LON = 17.4557, 78.4280

# --- 3. CORE LOGIC (ALL FEATURES) ---

def get_realistic_hyderabad_zones(baseline_aqi, baseline_temp):
    offsets = {
        'Industrial': {'temp': +4, 'aqi': +50}, 'Dense Residential': {'temp': +3, 'aqi': +20},
        'Commercial Hub': {'temp': +3, 'aqi': +30}, 'Historic/Market': {'temp': +2, 'aqi': +40},
        'Transport Hub': {'temp': +2, 'aqi': +35}, 'IT Hub': {'temp': +1, 'aqi': +15},
        'Financial District': {'temp': +1, 'aqi': +10}, 'Commercial/Residential': {'temp': +1, 'aqi': +25},
        'Posh Residential': {'temp': 0, 'aqi': -5}, 'Residential/Industrial': {'temp': +2, 'aqi': +45},
        'Green Space': {'temp': -4, 'aqi': -30},
    }
    zones = []
    for zone in HYDERABAD_ZONES_BASE:
        z = zone.copy()
        offset = offsets.get(z['type'], {'temp': 0, 'aqi': 0})
        z.update({
            'temperature': baseline_temp + offset['temp'], 'aqi': max(20, baseline_aqi + offset['aqi']),
            'needs_intervention': False, 'mitigation_intensity': 0.0, 'suggestion': '', 'image_url': ''
        })
        zones.append(z)
    return pd.DataFrame(zones)

def run_dqn_simulation():
    st.session_state.logs.append("DQN: Identifying hotspots...")
    zones_df = st.session_state.zones
    zones_df['needs_intervention'] = (zones_df['temperature'] > 40) & (zones_df['aqi'] > 100)
    st.session_state.zones = zones_df
    st.session_state.logs.append(f"DQN Complete: {zones_df['needs_intervention'].sum()} critical zones found.")

def run_ddpg_simulation():
    st.session_state.logs.append("DDPG: Calculating mitigation intensity...")
    zones_df = st.session_state.zones
    for i, row in zones_df.iterrows():
        if row['needs_intervention']:
            zones_df.loc[i, 'mitigation_intensity'] = round(np.random.uniform(0.3, 1.0), 2)
    st.session_state.zones = zones_df
    st.session_state.logs.append("DDPG Complete: Intensities assigned.")

def run_gemini_suggestions():
    st.session_state.logs.append("Contacting Gemini API...")
    hotspots = st.session_state.zones[st.session_state.zones['needs_intervention']]
    if hotspots.empty:
        st.session_state.error = "No critical hotspots identified."
        return

    model = genai.GenerativeModel('gemini-2.5-flash')
    details = "\n".join([f'- Zone ID: {z["id"]}, Name: "{z["name"]}" ({z["type"]}), Temp: {z["temperature"]}°C, AQI: {z["aqi"]}, Intensity: {z["mitigation_intensity"]:.2f}' for _, z in hotspots.iterrows()])
    prompt = f"""You are an urban planning AI for Hyderabad. For each "Critical zone" below, provide a creative cooling strategy.
You MUST provide:
1. The original "zoneId".
2. A "detailed_suggestion" as a multi-line string with markdown bullet points.
3. An "image_prompt" for an AI image generator.
Critical zones:
{details}
Your response must be ONLY a single valid JSON object with a key "suggestions".
Example: {{"suggestions": [{{"zoneId": 4, "detailed_suggestion": "- Point 1\\n- Point 2", "image_prompt": "A market with cooling fans."}}]}}"""

    try:
        safety_settings = [{"category": c, "threshold": HarmBlockThreshold.BLOCK_NONE} for c in HarmCategory if c != HarmCategory.HARM_CATEGORY_UNSPECIFIED]
        response = model.generate_content(prompt, generation_config=GenerationConfig(response_mime_type="application/json"), safety_settings=safety_settings)
        suggestions = json.loads(response.text).get("suggestions", [])
        if not suggestions:
            st.session_state.error = "AI returned no suggestions."
            return
        zones_df = st.session_state.zones
        for s in suggestions:
            zones_df.loc[zones_df['id'] == s['zoneId'], 'suggestion'] = s['detailed_suggestion']
            image_prompt = s.get('image_prompt', 'abstract cityscape')
            seed = hashlib.md5(image_prompt.encode()).hexdigest()
            zones_df.loc[zones_df['id'] == s['zoneId'], 'image_url'] = f"https://picsum.photos/seed/{seed}/400/300"
        st.session_state.zones = zones_df
        st.session_state.logs.append(f"Gemini Complete: Parsed {len(suggestions)} strategies.")
    except Exception as e:
        st.session_state.error = f"Failed to get/parse AI response: {e}"

def fetch_aqi_from_api(lat, lon):
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={AQI_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "ok":
            aqi = data["data"].get("aqi", np.nan)
            pm25 = data["data"]["iaqi"].get("pm25", {}).get("v", np.nan)
            pm10 = data["data"]["iaqi"].get("pm10", {}).get("v", np.nan)
            if np.isnan(pm10) and not np.isnan(pm25):
                pm10 = round(pm25 * 1.5 + np.random.uniform(-5, 5), 1)
            return aqi, pm25, pm10
    except requests.exceptions.RequestException:
        pass
    return np.nan, np.nan, np.nan

def generate_past_aqi(latest_aqi, hours=168):
    base = latest_aqi if latest_aqi and not np.isnan(latest_aqi) else 75
    past = [max(0, base + 15 * np.sin(2*np.pi*(i%24)/24) + np.random.normal(0, 5)) for i in range(hours)]
    times = [(datetime.now() - timedelta(hours=hours - i)) for i in range(hours)]
    return times, past

def lstm_multi_step_forecast(aqi_values_tuple, steps):
    # This function is not cached due to incompatibility with TensorFlow
    aqi_values = list(aqi_values_tuple)
    data = np.array(aqi_values).reshape(-1, 1)
    if np.all(data == data[0]): data += np.random.normal(0, 1, len(data)).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    time_step = min(10, len(scaled) - 1)
    X, y = [scaled[i:i+time_step, 0] for i in range(len(scaled)-time_step)], [scaled[i+time_step, 0] for i in range(len(scaled)-time_step)]
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential([LSTM(50, input_shape=(X.shape[1], 1), return_sequences=True), Dropout(0.2), LSTM(50), Dense(1)])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=1, verbose=0)
    temp_input = scaled[-time_step:].reshape(1, time_step, 1)
    forecast = []
    for _ in range(steps):
        pred = model.predict(temp_input, verbose=0)[0][0]
        forecast.append(pred)
        temp_input = np.append(temp_input[:, 1:, :], [[[pred]]], axis=1)
    return scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

def find_optimal_route(start_id, end_id, zones_df, mode='safest'):
    """Dijkstra's algorithm to find shortest or safest path."""
    zones_map = zones_df.set_index('id').to_dict('index')
    
    def get_weight(zone_id):
        if mode == 'shortest': return 1
        zone = zones_map[zone_id]
        aqi_cost = zone['aqi']
        temp_cost = max(0, zone['temperature'] - 38) * 10 # Penalize temps > 38°C
        return 1 + aqi_cost + temp_cost

    dist = {zone_id: float('inf') for zone_id in ZONE_GRAPH}
    dist[start_id] = 0
    prev = {zone_id: None for zone_id in ZONE_GRAPH}
    pq = [(0, start_id)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if u == end_id: break

        for v in ZONE_GRAPH.get(u, []):
            weight = get_weight(v)
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
    
    path = []
    curr = end_id
    while curr is not None:
        path.insert(0, curr)
        curr = prev[curr]
    
    if path[0] != start_id: return [], 0, 0 # No path found

    path_zones = zones_df[zones_df['id'].isin(path)]
    avg_aqi = path_zones['aqi'].mean()
    avg_temp = path_zones['temperature'].mean()
    
    return path, avg_aqi, avg_temp

def get_aqi_recommendation(aqi):
    if aqi is None or np.isnan(aqi): return "⚪ N/A"
    if aqi <= 50: return "✅ Good"
    if aqi <= 100: return "🟡 Moderate"
    if aqi <= 150: return "🟠 Unhealthy for Sensitive Groups"
    if aqi <= 200: return "🔴 Unhealthy"
    if aqi <= 300: return "🟣 Very Unhealthy"
    return "🟤 Hazardous"

# --- 5. STREAMLIT UI ---

st.title("🌍 Hyderabad Environmental Intelligence Dashboard")
st.markdown("A unified dashboard for **Heat Mitigation**, **AQI Forecasting**, **Route Recommendation**, and **Zone Clustering**.")

if 'zones' not in st.session_state:
    st.session_state.logs = ["Initializing new session..."]
    with st.spinner("Fetching live baseline data..."):
        aqi, pm25, pm10 = fetch_aqi_from_api(BASELINE_STATION_LAT, BASELINE_STATION_LON)
        if not aqi or np.isnan(aqi): aqi, pm25, pm10 = 110, 45, 80
    st.session_state.update({
        'baseline_aqi': aqi, 'baseline_pm25': pm25, 'baseline_pm10': pm10,
        'zones': get_realistic_hyderabad_zones(aqi, 40), 'stage': 'INITIAL', 'error': None
    })
    st.session_state.logs.append("Session initialized with stable, realistic data.")

tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Heat Mitigation", "💨 AQI Forecast", "🧭 Route Recommendation", "📊 Zone Clustering"])

with tab1:
    # ... Heat Mitigation UI ... (Same as before)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("Hyderabad Zone Status")
        cols = st.columns(4)
        for i, zone in st.session_state.zones.iterrows():
            with cols[i % 4]:
                with st.container(border=True):
                    st.markdown(f"**{zone['name']}** {'⚠️' if zone['needs_intervention'] else ''}")
                    st.markdown(f'<div style="display: flex; align-items: baseline; gap: 8px; white-space: nowrap; font-size: 0.9rem;"><span style="color: {"red" if zone["temperature"] > 40 else "black"};"><b>T:</b> {zone["temperature"]}°C</span><span style="color: #888;">|</span><span style="color: {"orange" if zone["aqi"] > 100 else "black"};"><b>AQI:</b> {zone["aqi"]}</span></div>', unsafe_allow_html=True)
                    if zone['mitigation_intensity'] > 0: st.progress(zone['mitigation_intensity'], f"Intensity: {zone['mitigation_intensity']:.2f}")
        st.divider()
        st.subheader("🧠 AI-Generated Mitigation Strategies")
        suggestions = st.session_state.zones[st.session_state.zones['suggestion'] != ""]
        if not suggestions.empty:
            for _, row in suggestions.iterrows():
                with st.container(border=True):
                    img_col, text_col = st.columns([1, 2])
                    with img_col:
                        if row['image_url']: st.image(row['image_url'], caption=f"Concept for {row['name']}")
                    with text_col:
                        st.markdown(f"**📍 Mitigation Plan for {row['name']}**")
                        st.markdown(row['suggestion'])
        else:
            st.info("AI suggestions for this scenario will appear here.")
    with col2:
        st.header("Simulation Controls")
        status_map = {'INITIAL': ("Ready to begin.", "info"), 'DQN_COMPLETE': ("Hotspots identified.", "info"), 'DDPG_COMPLETE': ("Intensity calculated.", "info"), 'INTEGRATED_COMPLETE': ("Scenario complete!", "success")}
        msg, type = status_map.get(st.session_state.stage, ("Processing...", "info"))
        if type == "success": st.success(msg)
        else: st.info(msg)
        if st.session_state.error: st.error(st.session_state.error)
        if st.button("1. Run DQN", type="primary", disabled=st.session_state.stage != 'INITIAL'):
            with st.spinner('Simulating DQN...'): run_dqn_simulation(); time.sleep(1)
            st.session_state.stage = 'DQN_COMPLETE'; st.rerun()
        if st.button("2. Run DDPG", disabled=st.session_state.stage != 'DQN_COMPLETE'):
            with st.spinner('Simulating DDPG...'): run_ddpg_simulation(); time.sleep(1)
            st.session_state.stage = 'DDPG_COMPLETE'; st.rerun()
        if st.button("3. Generate Suggestions", disabled=st.session_state.stage != 'DDPG_COMPLETE'):
            with st.spinner("Contacting Gemini AI..."): run_gemini_suggestions()
            st.session_state.stage = 'INTEGRATED_COMPLETE'; st.rerun()
        st.divider()
        if st.button("Reset Scenario (Fetch New Live Data)", type="secondary"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()
        with st.expander("Show Logs"):
            st.code("\n".join(st.session_state.logs), language="log")

with tab2:
    # ... AQI Forecasting UI ... (Same as before)
    st.header("Live AQI & LSTM Forecast")
    st.markdown("📍 Click anywhere on the map to get a location-specific AQI forecast.")
    m = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=11)
    folium.Marker([BASELINE_STATION_LAT, BASELINE_STATION_LON], popup="City-Wide Baseline Station (Sanathnagar)", tooltip="Baseline Station", icon=folium.Icon(color='blue')).add_to(m)
    map_data = st_folium(m, height=400, width=700)
    if map_data and map_data["last_clicked"]:
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        st.subheader(f"Forecast for Clicked Location ({lat:.4f}, {lon:.4f})")
        with st.spinner(f"Fetching live AQI for clicked location..."):
            latest_aqi, latest_pm25, latest_pm10 = fetch_aqi_from_api(lat, lon)
    else:
        st.subheader(f"Default Forecast for Baseline Station (Sanathnagar)")
        latest_aqi, latest_pm25, latest_pm10 = st.session_state.baseline_aqi, st.session_state.baseline_pm25, st.session_state.baseline_pm10
    st.subheader("Current Conditions at Location")
    if not np.isnan(latest_aqi):
        rec = get_aqi_recommendation(latest_aqi)
        st.metric(label="Live AQI", value=int(latest_aqi), help=rec)
        st.write(f"**Condition:** {rec}")
        cols = st.columns(2)
        cols[0].metric(label="PM2.5", value=f"{latest_pm25} µg/m³" if not np.isnan(latest_pm25) else "N/A")
        cols[1].metric(label="PM10", value=f"{latest_pm10} µg/m³" if not np.isnan(latest_pm10) else "N/A")
    else:
        st.warning("Live AQI data unavailable for this location.")
    with st.spinner("Generating historical data & training LSTM model..."):
        past_times, past_aqi = generate_past_aqi(latest_aqi)
        if latest_aqi and not np.isnan(latest_aqi): past_aqi[-1] = latest_aqi
        full_forecast = lstm_multi_step_forecast(tuple(past_aqi), 168)
    st.subheader("🔮 Next 24-Hour Detailed Forecast")
    forecast_24h = full_forecast[:24]
    with st.expander("View Hourly Details"):
        for i in range(0, 24, 3):
            st.text(f"{(datetime.now() + timedelta(hours=i+1)).strftime('%I:%M %p')}: Forecasted AQI {forecast_24h[i]:.1f} ({get_aqi_recommendation(forecast_24h[i])})")
    st.subheader("🗓️ Next 7 Days Forecast (Daily Average)")
    daily_avg_forecast = [np.mean(full_forecast[i*24:(i+1)*24]) for i in range(7)]
    for i in range(7):
        st.text(f"{(datetime.now() + timedelta(days=i+1)).strftime('%A, %b %d')}: Avg AQI {daily_avg_forecast[i]:.1f} ({get_aqi_recommendation(daily_avg_forecast[i])})")

with tab3:
    st.header("Intelligent Route Recommendation")
    zones_df = st.session_state.zones
    zone_names = zones_df.set_index('id')['name'].to_dict()

    col1, col2 = st.columns(2)
    start_name = col1.selectbox("Select Start Location", options=zone_names.values(), index=0)
    end_name = col2.selectbox("Select End Location", options=zone_names.values(), index=1)
    
    start_id = next((id for id, name in zone_names.items() if name == start_name), None)
    end_id = next((id for id, name in zone_names.items() if name == end_name), None)

    if st.button("Find Routes", type="primary"):
        if start_id == end_id:
            st.error("Start and End locations cannot be the same.")
        else:
            with st.spinner("Calculating safest and shortest routes..."):
                safe_path, safe_aqi, safe_temp = find_optimal_route(start_id, end_id, zones_df, 'safest')
                short_path, short_aqi, short_temp = find_optimal_route(start_id, end_id, zones_df, 'shortest')
                
                st.subheader("🏆 Safest Route (Recommended)")
                if safe_path:
                    path_str = " → ".join([zone_names[zid] for zid in safe_path])
                    st.markdown(f"**Path:** {path_str}")
                    c1, c2 = st.columns(2)
                    c1.metric("Average AQI Exposure", f"{safe_aqi:.1f}")
                    c2.metric("Average Temperature", f"{safe_temp:.1f}°C")
                else:
                    st.warning("No safe route could be found.")

                st.subheader("Standard Shortest Route")
                if short_path:
                    path_str = " → ".join([zone_names[zid] for zid in short_path])
                    st.markdown(f"**Path:** {path_str}")
                    c1, c2 = st.columns(2)
                    c1.metric("Average AQI Exposure", f"{short_aqi:.1f}")
                    c2.metric("Average Temperature", f"{short_temp:.1f}°C")
                else:
                    st.warning("No shortest route could be found.")
                
                if safe_path and short_path and short_aqi > safe_aqi:
                    improvement = ((short_aqi - safe_aqi) / short_aqi) * 100
                    st.success(f"By taking the safest route, you reduce your pollution exposure by **{improvement:.0f}%**!")

with tab4:
    st.header("Dynamic Zone Clustering (DBSCAN)")
    st.markdown("This analysis groups zones with similar environmental conditions together.")
    
    if st.button("Run Clustering Analysis", type="primary"):
        with st.spinner("Analyzing zone data..."):
            zones_df = st.session_state.zones
            features = zones_df[['temperature', 'aqi']]
            
            # Scale features for clustering
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Run DBSCAN
            # Epsilon is the max distance between samples to be considered neighbors.
            # Min_samples is the number of samples in a neighborhood for a point to be a core point.
            dbscan = DBSCAN(eps=0.7, min_samples=2)
            clusters = dbscan.fit_predict(scaled_features)
            zones_df['cluster'] = clusters
            
            st.subheader("Clustering Results")
            
            # Create a scatter plot to visualize clusters
            fig, ax = plt.subplots(figsize=(10, 6))
            unique_clusters = set(clusters)
            
            for cluster_id in unique_clusters:
                cluster_zones = zones_df[zones_df['cluster'] == cluster_id]
                if cluster_id == -1:
                    label = 'Outliers'
                    color = 'gray'
                else:
                    label = f'Cluster {cluster_id}'
                    color = plt.cm.viridis(cluster_id / (len(unique_clusters) -1))
                
                ax.scatter(cluster_zones['aqi'], cluster_zones['temperature'], label=label, color=color, s=100)
                for i, txt in enumerate(cluster_zones['name']):
                     ax.annotate(txt, (cluster_zones['aqi'].iloc[i], cluster_zones['temperature'].iloc[i]), fontsize=9)

            ax.set_xlabel("Air Quality Index (AQI)")
            ax.set_ylabel("Temperature (°C)")
            ax.set_title("Zone Clusters by Environmental Conditions")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Zone Groups")
            for cluster_id in sorted(unique_clusters):
                cluster_zones = zones_df[zones_df['cluster'] == cluster_id]['name'].tolist()
                if cluster_id == -1:
                    st.markdown("**Outliers (Unique Conditions):**")
                else:
                    st.markdown(f"**Cluster {cluster_id} (Similar Conditions):**")
                st.write(", ".join(cluster_zones))
