import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import requests
from datetime import datetime, timedelta
import hashlib
import heapq
from streamlit_geolocation import streamlit_geolocation

# AI and ML Imports
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Visualization Imports
from streamlit_folium import st_folium
import folium
import plotly.express as px

# --- 1. CONFIGURATION & PAGE SETUP ---

st.set_page_config(
    page_title="Hyderabad Environmental Intelligence",
    page_icon="🌍",
    layout="wide",
)

# --- SECRETS MANAGEMENT ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    AQI_API_KEY = st.secrets["AQI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (FileNotFoundError, KeyError):
    st.error("🚨 API Keys not found! Please configure your Streamlit secrets.")
    st.info("To run this app, create a file at `.streamlit/secrets.toml` with your API keys.")
    st.code("""
# .streamlit/secrets.toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
AQI_API_KEY = "YOUR_AQI_API_KEY_HERE"
    """)
    st.stop()


# --- 2. SHARED DATA & CONSTANTS ---

HYDERABAD_ZONES_BASE = [
    {'id': 1, 'name': 'Banjara Hills', 'type': 'Posh Residential', 'lat': 17.414, 'lon': 78.435},
    {'id': 2, 'name': 'Gachibowli', 'type': 'Financial District', 'lat': 17.44, 'lon': 78.34},
    {'id': 3, 'name': 'Patancheru Ind. Area', 'type': 'Industrial', 'lat': 17.53, 'lon': 78.26},
    {'id': 4, 'name': 'Charminar Area', 'type': 'Historic/Market', 'lat': 17.361, 'lon': 78.474},
    {'id': 5, 'name': 'HITEC City', 'type': 'IT Hub', 'lat': 17.444, 'lon': 78.377},
    {'id': 6, 'name': 'Secunderabad', 'type': 'Commercial/Residential', 'lat': 17.439, 'lon': 78.5},
    {'id': 7, 'name': 'KBR National Park', 'type': 'Green Space', 'lat': 17.42, 'lon': 78.42},
    {'id': 8, 'name': 'Uppal', 'type': 'Residential/Industrial', 'lat': 17.398, 'lon': 78.558},
    {'id': 9, 'name': 'Begumpet Airport Area', 'type': 'Transport Hub', 'lat': 17.447, 'lon': 78.465},
    {'id': 10, 'name': 'Jubilee Hills', 'type': 'Posh Residential', 'lat': 17.43, 'lon': 78.4},
    {'id': 11, 'name': 'Ameerpet', 'type': 'Commercial Hub', 'lat': 17.437, 'lon': 78.448},
    {'id': 12, 'name': 'Kukatpally', 'type': 'Dense Residential', 'lat': 17.485, 'lon': 78.4},
]

ZONE_GRAPH = {
  1: [10, 11, 7], 2: [5, 12], 3: [12], 4: [6], 5: [2, 12, 10],
  6: [4, 9, 8, 11], 7: [1, 10], 8: [6], 9: [6, 11],
  10: [1, 5, 7, 11], 11: [1, 9, 10, 6], 12: [2, 3, 5],
}

DEFAULT_LAT, DEFAULT_LON = 17.43, 78.45
BASELINE_STATION_LAT, BASELINE_STATION_LON = 17.4557, 78.4280

# --- 3. CORE LOGIC & HELPER FUNCTIONS ---

@st.cache_data
def get_realistic_hyderabad_zones(baseline_aqi, baseline_temp):
    """Generates simulated environmental data for zones based on their type."""
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
            'temperature': baseline_temp + offset['temp'] + np.random.uniform(-1, 1),
            'aqi': max(20, baseline_aqi + offset['aqi'] + np.random.randint(-5, 5)),
            'needs_intervention': False, 'mitigation_intensity': 0.0, 'suggestion': '', 'image_url': ''
        })
        zones.append(z)
    return pd.DataFrame(zones)

def run_dqn_simulation():
    """Simulates identifying critical hotspots based on temperature and AQI thresholds."""
    st.session_state.logs.append("DQN: Identifying hotspots...")
    zones_df = st.session_state.zones.copy()
    zones_df['needs_intervention'] = (zones_df['temperature'] > 40) & (zones_df['aqi'] > 100)
    st.session_state.zones = zones_df
    st.session_state.logs.append(f"DQN Complete: {zones_df['needs_intervention'].sum()} critical zones found.")

def run_ddpg_simulation():
    """Simulates calculating the required intensity for mitigation actions."""
    st.session_state.logs.append("DDPG: Calculating mitigation intensity...")
    zones_df = st.session_state.zones.copy()
    for i, row in zones_df.iterrows():
        if row['needs_intervention']:
            zones_df.loc[i, 'mitigation_intensity'] = round(np.random.uniform(0.3, 1.0), 2)
    st.session_state.zones = zones_df
    st.session_state.logs.append("DDPG Complete: Intensities assigned.")

def run_gemini_suggestions():
    """Contacts Gemini API to get mitigation suggestions for critical zones."""
    st.session_state.logs.append("Contacting Gemini API...")
    hotspots = st.session_state.zones[st.session_state.zones['needs_intervention']]
    if hotspots.empty:
        st.session_state.error = "No critical hotspots identified to generate suggestions."
        return

    model = genai.GenerativeModel('gemini-2.5-flash')
    details = "\n".join([f'- Zone ID: {z["id"]}, Name: "{z["name"]}" ({z["type"]}), Temp: {z["temperature"]:.1f}°C, AQI: {z["aqi"]}, Intensity: {z["mitigation_intensity"]:.2f}' for _, z in hotspots.iterrows()])
    
    prompt = f"""You are an expert urban planning AI for Hyderabad, India. You specialize in creative, practical, and culturally relevant strategies for heat and pollution mitigation. For each "Critical Zone" listed below, provide a tailored strategy. Your response MUST be a single, valid JSON object with one key: "suggestions", containing a list of objects, each with "zoneId", "detailed_suggestion", and "image_prompt". Use '\\n' for newlines in the suggestion. Critical Zones:\n{details}"""

    try:
        generation_config = GenerationConfig(response_mime_type="application/json")
        safety_settings = [{"category": c, "threshold": HarmBlockThreshold.BLOCK_NONE} for c in HarmCategory if c != HarmCategory.HARM_CATEGORY_UNSPECIFIED]
        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
        suggestions = json.loads(response.text).get("suggestions", [])

        zones_df = st.session_state.zones.copy()
        for s in suggestions:
            suggestion_text = s.get('detailed_suggestion', '').replace('\\n', '<br>')
            seed = hashlib.md5(s.get('image_prompt', '').encode()).hexdigest()
            zones_df.loc[zones_df['id'] == s['zoneId'], ['suggestion', 'image_url']] = [suggestion_text, f"https://picsum.photos/seed/{seed}/400/300"]
        st.session_state.zones = zones_df
        st.session_state.logs.append(f"Gemini Complete: Parsed {len(suggestions)} strategies.")
    except Exception as e:
        st.session_state.error = f"Failed to get or parse AI response: {e}"
        st.session_state.logs.append(f"Gemini Error: {e}")

@st.cache_data(ttl=3600)
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
            return aqi, pm25, pm10
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch live AQI data: {e}")
    return np.nan, np.nan, np.nan
    
def generate_past_aqi(latest_aqi, hours=168):
    """Generates a week of simulated historical AQI data for forecasting."""
    base = latest_aqi if latest_aqi and not np.isnan(latest_aqi) else 75
    past = [max(10, base + 15 * np.sin(2*np.pi*(i%24)/24) + np.random.normal(0, 5)) for i in range(hours)]
    times = [(datetime.now() - timedelta(hours=hours - i)) for i in range(hours)]
    return times, past

def mock_lstm_forecast(past_aqi, steps=168):
    """A mock forecasting function to simulate a more complex model without heavy dependencies."""
    if not past_aqi: return [75] * steps
    last_value = past_aqi[-1]
    forecast = []
    for i in range(steps):
        cyclical_trend = 15 * np.sin(2 * np.pi * ((len(past_aqi) + i) % 24) / 24)
        random_noise = np.random.normal(0, 3)
        next_value = last_value * 0.95 + (75 * 0.05) + cyclical_trend + random_noise
        forecast.append(max(10, next_value))
        last_value = next_value
    return forecast

def find_optimal_route(start_id, end_id, zones_df, mode='safest'):
    zones_map = zones_df.set_index('id').to_dict('index')
    
    def get_weight(zone_id):
        zone = zones_map.get(zone_id, {})
        if mode == 'shortest': return 1
        return 1 + zone.get('aqi', 1000) + max(0, zone.get('temperature', 50) - 38) * 10

    dist = {zone['id']: float('inf') for _, zone in zones_df.iterrows()}
    prev = {zone['id']: None for _, zone in zones_df.iterrows()}
    pq = [(0, start_id)]
    dist[start_id] = 0

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')): continue
        if u == end_id: break
        for v in ZONE_GRAPH.get(u, []):
            weight = get_weight(v)
            if dist.get(u, float('inf')) + weight < dist.get(v, float('inf')):
                dist[v] = dist[u] + weight
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
    path, curr = [], end_id
    while curr is not None:
        path.insert(0, curr)
        curr = prev.get(curr)

    if not path or path[0] != start_id: return [], 0, 0
    path_zones = zones_df[zones_df['id'].isin(path)]
    if path_zones.empty: return [], 0, 0
    return path, path_zones['aqi'].mean(), path_zones['temperature'].mean()

def get_aqi_recommendation(aqi):
    if aqi is None or np.isnan(aqi): return "⚪ N/A", "#808080"
    if aqi <= 50: return "✅ Good", "#28a745"
    if aqi <= 100: return "🟡 Moderate", "#ffc107"
    if aqi <= 150: return "🟠 Unhealthy for Sensitive Groups", "#fd7e14"
    if aqi <= 200: return "🔴 Unhealthy", "#dc3545"
    if aqi <= 300: return "🟣 Very Unhealthy", "#6f42c1"
    return "🟤 Hazardous", "#795548"

# --- INITIALIZE SESSION STATE ---
if 'zones' not in st.session_state:
    st.session_state.logs = ["Initializing new session..."]
    with st.spinner("Fetching live baseline data..."):
        aqi, pm25, pm10 = fetch_aqi_from_api(BASELINE_STATION_LAT, BASELINE_STATION_LON)
        if not aqi or np.isnan(aqi):
            st.session_state.logs.append("Using fallback baseline data.")
            st.session_state.baseline_aqi = 110
            st.session_state.baseline_pm25 = 45
            st.session_state.baseline_pm10 = 80
        else:
            st.session_state.logs.append("Live baseline data fetched successfully.")
            st.session_state.baseline_aqi = aqi
            st.session_state.baseline_pm25 = pm25
            st.session_state.baseline_pm10 = pm10
    
    st.session_state.zones = get_realistic_hyderabad_zones(st.session_state.baseline_aqi, 40)
    st.session_state.stage = 'INITIAL'
    st.session_state.error = None
    st.session_state.route_result = None
    st.session_state.cluster_result = None
    st.session_state.is_navigating = False
    st.session_state.user_location = None

# --- UI LAYOUT ---
st.title("🌍 Hyderabad Environmental Intelligence Dashboard")
st.markdown("A unified dashboard for AI-powered **Heat Mitigation**, **AQI Forecasting**, **Route Recommendation**, and **Zone Clustering**.")

tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Heat Mitigation", "💨 AQI Forecast", "🧭 Route Planner", "📊 Zone Clustering"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("Hyderabad Zone Status")
        st.markdown("Simulated real-time environmental status across key city zones.")
        
        cols = st.columns(4)
        for i, zone in st.session_state.zones.iterrows():
            with cols[i % 4]:
                with st.container(border=True):
                    temp_color = '#ff4b4b' if zone["temperature"] > 40 else 'inherit'
                    aqi_color = '#ff8c00' if zone["aqi"] > 100 else 'inherit'
                    st.markdown(f"**{zone['name']}** {'🔥' if zone['needs_intervention'] else ''}")
                    st.markdown(f'<span style="color: {temp_color};"><b>Temp:</b> {zone["temperature"]:.1f}°C</span> | <span style="color: {aqi_color};"><b>AQI:</b> {zone["aqi"]}</span>', unsafe_allow_html=True)
                    if zone['mitigation_intensity'] > 0:
                        st.progress(zone['mitigation_intensity'], f"Intensity: {zone['mitigation_intensity']:.2f}")

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
                        st.markdown(f"##### 📍 Mitigation Plan for {row['name']}")
                        st.markdown(row['suggestion'], unsafe_allow_html=True)
        else:
            st.info("Run the full simulation to generate AI-powered mitigation strategies for critical zones.")

    with col2:
        st.header("Simulation Controls")
        if st.session_state.error: st.error(st.session_state.error)

        if st.button("1. Identify Hotspots (DQN)", type="primary", use_container_width=True, disabled=st.session_state.stage != 'INITIAL'):
            with st.spinner('Simulating DQN...'): run_dqn_simulation(); time.sleep(1)
            st.session_state.stage = 'DQN_COMPLETE'; st.rerun()
        if st.button("2. Calculate Intensity (DDPG)", use_container_width=True, disabled=st.session_state.stage != 'DQN_COMPLETE'):
            with st.spinner('Simulating DDPG...'): run_ddpg_simulation(); time.sleep(1)
            st.session_state.stage = 'DDPG_COMPLETE'; st.rerun()
        if st.button("3. Generate AI Suggestions", use_container_width=True, disabled=st.session_state.stage != 'DDPG_COMPLETE'):
            with st.spinner("🤖 Contacting Gemini AI..."): run_gemini_suggestions()
            st.session_state.stage = 'INTEGRATED_COMPLETE'; st.rerun()
        
        st.divider()
        if st.button("Reset Scenario", type="secondary", use_container_width=True):
            st.session_state.zones = get_realistic_hyderabad_zones(st.session_state.baseline_aqi, 40)
            st.session_state.stage = 'INITIAL'
            st.session_state.error = None
            st.session_state.route_result = None
            st.session_state.cluster_result = None
            st.rerun()
        
        with st.expander("Show Simulation Logs"):
            st.code("\n".join(st.session_state.logs), language="log")

with tab2:
    st.header("Live AQI & 7-Day Forecast")
    st.markdown("📍 Click anywhere on the map to get a location-specific AQI forecast, or use the default baseline station.")
    
    m = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=11)
    folium.Marker(
        [BASELINE_STATION_LAT, BASELINE_STATION_LON],
        popup="City-Wide Baseline Station (Sanathnagar)",
        tooltip="Baseline Station",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)
    
    map_data = st_folium(m, height=400, width=700)

    if map_data and map_data["last_clicked"]:
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        st.subheader(f"Forecast for Clicked Location ({lat:.4f}, {lon:.4f})")
        with st.spinner(f"Fetching live AQI at clicked location..."):
            latest_aqi, latest_pm25, latest_pm10 = fetch_aqi_from_api(lat, lon)
    else:
        st.subheader(f"Forecast for Baseline Station (Sanathnagar)")
        latest_aqi, latest_pm25, latest_pm10 = st.session_state.baseline_aqi, st.session_state.baseline_pm25, st.session_state.baseline_pm10

    st.markdown("#### Current Conditions")
    if not np.isnan(latest_aqi):
        rec, color = get_aqi_recommendation(latest_aqi)
        st.metric(label="Live Air Quality Index (AQI)", value=int(latest_aqi))
        st.markdown(f"**Condition:** <span style='color:{color};'>{rec}</span>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric(label="PM2.5", value=f"{latest_pm25} µg/m³" if not np.isnan(latest_pm25) else "N/A")
        c2.metric(label="PM10", value=f"{latest_pm10} µg/m³" if not np.isnan(latest_pm10) else "N/A")
    else:
        st.warning("Live AQI data unavailable for this location. Using historical average for forecast.")

    with st.spinner("Generating historical data & creating forecast..."):
        _, past_aqi = generate_past_aqi(latest_aqi)
        if latest_aqi and not np.isnan(latest_aqi): past_aqi[-1] = latest_aqi
        full_forecast = mock_lstm_forecast(past_aqi, steps=168)

    st.markdown("#### 🔮 Next 7 Days Forecast (Daily Average)")
    daily_avg_forecast = [np.mean(full_forecast[i*24:(i+1)*24]) for i in range(7)]
    forecast_df = pd.DataFrame({
        'Day': [(datetime.now() + timedelta(days=i+1)).strftime('%a, %b %d') for i in range(7)],
        'Average AQI': daily_avg_forecast
    })
    
    fig = px.bar(forecast_df, x='Day', y='Average AQI', title="7-Day Average AQI Forecast",
                 labels={'Average AQI': 'Forecasted AQI Value'}, text_auto='.2s')
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Intelligent Route Recommendation")
    st.markdown("Find the safest (lowest pollution & heat) or the shortest route. Now with live navigation!")
    
    zones_df = st.session_state.zones
    zone_names = zones_df.set_index('id')['name'].to_dict()

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        start_name = st.selectbox("Select Start Location", options=zone_names.values(), index=0, key="start_loc")
    with col2:
        end_name = st.selectbox("Select End Location", options=zone_names.values(), index=len(zone_names)-1, key="end_loc")
    with col3:
        st.write("") # Spacer
        if st.button("Use My Location", use_container_width=True):
            location = streamlit_geolocation()
            if location and location.get('latitude') and location.get('longitude'):
                user_lat, user_lon = location['latitude'], location['longitude']
                distances = zones_df.apply(lambda row: np.sqrt((row['lat']-user_lat)**2 + (row['lon']-user_lon)**2), axis=1)
                closest_zone_id = zones_df.loc[distances.idxmin()]['id']
                # Update the session state for the selectbox
                st.session_state.start_loc = zone_names[closest_zone_id]
                st.success(f"Start set to nearest zone: {zone_names[closest_zone_id]}")
                st.rerun()

    if st.button("Find Routes", type="primary", use_container_width=True):
        if start_name == end_name:
            st.error("Start and End locations cannot be the same.")
        else:
            with st.spinner("Calculating safest and shortest routes..."):
                start_id = next((id for id, name in zone_names.items() if name == start_name), None)
                end_id = next((id for id, name in zone_names.items() if name == end_name), None)
                if start_id is not None and end_id is not None:
                    safe_path, safe_aqi, safe_temp = find_optimal_route(start_id, end_id, zones_df, 'safest')
                    short_path, short_aqi, short_temp = find_optimal_route(start_id, end_id, zones_df, 'shortest')
                    st.session_state.route_result = {
                        "safe_path": safe_path, "safe_aqi": safe_aqi, "safe_temp": safe_temp,
                        "short_path": short_path, "short_aqi": short_aqi, "short_temp": short_temp,
                        "start_id": start_id, "end_id": end_id
                    }

    if st.session_state.get('route_result'):
        res = st.session_state.route_result
        start_zone = zones_df.loc[zones_df['id'] == res['start_id']].iloc[0]
        map_center = [start_zone['lat'], start_zone['lon']]
        
        route_map = folium.Map(location=map_center, zoom_start=12)
        
        def get_coords(path):
            return [(zones_df.loc[zones_df['id'] == zid, 'lat'].iloc[0], zones_df.loc[zones_df['id'] == zid, 'lon'].iloc[0]) for zid in path if zid in zones_df['id'].values]

        safe_coords = get_coords(res["safe_path"])
        short_coords = get_coords(res["short_path"])

        if safe_coords:
            folium.PolyLine(safe_coords, color='green', weight=6, opacity=0.8, tooltip="Safest Route").add_to(route_map)
        if short_coords:
            folium.PolyLine(short_coords, color='red', weight=3, opacity=0.8, dash_array='5, 5', tooltip="Shortest Route").add_to(route_map)
        
        end_zone = zones_df.loc[zones_df['id'] == res["end_id"]].iloc[0]
        folium.Marker([start_zone['lat'], start_zone['lon']], popup=f"START: {start_zone['name']}", icon=folium.Icon(color='green', icon='play')).add_to(route_map)
        folium.Marker([end_zone['lat'], end_zone['lon']], popup=f"END: {end_zone['name']}", icon=folium.Icon(color='red', icon='stop')).add_to(route_map)

        if st.session_state.is_navigating and st.session_state.user_location:
            user_loc = st.session_state.user_location
            folium.Marker([user_loc['latitude'], user_loc['longitude']], popup="Your Location", icon=folium.Icon(color='blue', icon='user')).add_to(route_map)

        st_folium(route_map, height=450, width=700)

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("##### 🏆 Safest Route (Recommended)")
            st.markdown(f"**Path:** {' → '.join([zone_names.get(zid, str(zid)) for zid in res['safe_path']])}")
            st.metric("Avg AQI / Temp", f"{res['safe_aqi']:.1f} / {res['safe_temp']:.1f}°C")
        with res_col2:
            st.markdown("##### 📏 Standard Shortest Route")
            st.markdown(f"**Path:** {' → '.join([zone_names.get(zid, str(zid)) for zid in res['short_path']])}")
            st.metric("Avg AQI / Temp", f"{res['short_aqi']:.1f} / {res['short_temp']:.1f}°C")

        st.divider()
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if not st.session_state.is_navigating:
                if st.button("▶️ Start Navigation", use_container_width=True, type="primary"):
                    st.session_state.is_navigating = True
                    st.rerun()
            else:
                if st.button("⏹️ Stop Navigation", use_container_width=True):
                    st.session_state.is_navigating = False
                    st.session_state.user_location = None
                    st.rerun()
        with nav_col2:
             if st.session_state.is_navigating:
                st.info("Navigation mode is active. Your position will be shown on the map.")
                current_pos = streamlit_geolocation()
                if current_pos and current_pos.get('latitude') and current_pos.get('longitude'):
                    st.session_state.user_location = current_pos

with tab4:
    st.header("Dynamic Zone Clustering (DBSCAN)")
    st.markdown("This analysis uses unsupervised learning to group zones with similar environmental conditions.")
    
    if st.button("Run Clustering Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing and clustering zone data..."):
            zones_df_copy = st.session_state.zones.copy()
            features = zones_df_copy[['temperature', 'aqi']]
            scaled_features = StandardScaler().fit_transform(features)
            dbscan = DBSCAN(eps=0.7, min_samples=2)
            zones_df_copy['cluster'] = dbscan.fit_predict(scaled_features)
            st.session_state.cluster_result = zones_df_copy
    
    if st.session_state.get('cluster_result') is not None:
        clustered_df = st.session_state.cluster_result
        st.subheader("Interactive Cluster Plot")
        fig = px.scatter(
            clustered_df, x='aqi', y='temperature', color='cluster',
            color_continuous_scale=px.colors.qualitative.Vivid,
            hover_name='name', labels={'cluster': 'Cluster ID'},
            title="Zone Clusters by Temperature and AQI"
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Geospatial Cluster Visualization")
        cluster_map = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=11)
        color_map = {-1: 'gray', 0: 'blue', 1: 'green', 2: 'purple', 3: 'orange', 4: 'red'}
        for _, zone in clustered_df.iterrows():
            cluster_id = zone['cluster']
            color = color_map.get(cluster_id, 'black')
            folium.Marker(
                location=[zone['lat'], zone['lon']],
                popup=f"<strong>{zone['name']}</strong><br>Cluster: {'Outlier' if cluster_id == -1 else cluster_id}",
                tooltip=zone['name'],
                icon=folium.Icon(color=color)
            ).add_to(cluster_map)
        st_folium(cluster_map, height=450, width=700)

        st.subheader("Cluster Groups")
        for cluster_id in sorted(clustered_df['cluster'].unique()):
            cluster_zones = clustered_df[clustered_df['cluster'] == cluster_id]['name'].tolist()
            if cluster_id == -1:
                st.markdown(f"**⚪ Outliers (Unique Conditions):** {', '.join(cluster_zones)}")
            else:
                st.markdown(f"**🔵 Cluster {cluster_id} (Similar Conditions):** {', '.join(cluster_zones)}")
