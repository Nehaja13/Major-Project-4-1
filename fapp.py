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
import openrouteservice
from openrouteservice.exceptions import ApiError

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
    ORS_API_KEY = st.secrets["ORS_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (FileNotFoundError, KeyError):
    st.error("🚨 API Keys not found! Please configure your Streamlit secrets.")
    st.info("Ensure you have a file at `.streamlit/secrets.toml` in your project's root folder.")
    st.code("""
# .streamlit/secrets.toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
AQI_API_KEY = "YOUR_AQI_API_KEY_HERE"
ORS_API_KEY = "YOUR_ORS_API_KEY_HERE" # Get from https://openrouteservice.org/
    """)
    st.stop()

# Initialize OpenRouteService client
try:
    ors_client = openrouteservice.Client(key=ORS_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize ORS client. Check ORS_API_KEY. Error: {e}")
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
    {'id': 13, 'name': 'Bachupally', 'type': 'Educational Hub', 'lat': 17.53, 'lon': 78.4},
]
ZONE_GRAPH = { 1: [10, 11, 7], 2: [5, 12], 3: [12, 13], 4: [6], 5: [2, 12, 10], 6: [4, 9, 8, 11], 7: [1, 10], 8: [6], 9: [6, 11], 10: [1, 5, 7, 11], 11: [1, 9, 10, 6], 12: [2, 3, 5, 13], 13: [12, 3]}
DEFAULT_LAT, DEFAULT_LON = 17.43, 78.45
BASELINE_STATION_LAT, BASELINE_STATION_LON = 17.4557, 78.4280

# --- 3. CORE LOGIC & HELPER FUNCTIONS ---

def get_intensity_description(intensity):
    if intensity <= 0.5: return "Low: Requires monitoring and minor interventions."
    if intensity <= 0.7: return "Medium: Needs targeted actions like cooling centers."
    if intensity <= 0.9: return "High: Demands significant resources, like mobile cooling vans."
    return "Critical: Urgent, large-scale response required."

def get_cluster_label(cluster_df, cluster_id):
    if cluster_id == -1: return "Outliers"
    cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
    mean_temp, mean_aqi = cluster_data['temperature'].mean(), cluster_data['aqi'].mean()
    if mean_temp > 41 and mean_aqi > 110: return "Extreme Heat & Pollution"
    if mean_temp > 41: return "Heat Hotspots"
    if mean_aqi > 110: return "Pollution Hotspots"
    if mean_temp < 39 and mean_aqi < 90: return "Cool & Clean Zones"
    return f"Moderate Cluster {cluster_id}"

@st.cache_data
def get_realistic_hyderabad_zones(baseline_aqi, baseline_temp):
    offsets = { 'Industrial': {'temp': +4, 'aqi': +50}, 'Dense Residential': {'temp': +3, 'aqi': +20}, 'Commercial Hub': {'temp': +3, 'aqi': +30}, 'Historic/Market': {'temp': +2, 'aqi': +40}, 'Transport Hub': {'temp': +2, 'aqi': +35}, 'IT Hub': {'temp': +1, 'aqi': +15}, 'Financial District': {'temp': +1, 'aqi': +10}, 'Commercial/Residential': {'temp': +1, 'aqi': +25}, 'Posh Residential': {'temp': 0, 'aqi': -5}, 'Residential/Industrial': {'temp': +2, 'aqi': +45}, 'Green Space': {'temp': -4, 'aqi': -30}, 'Educational Hub': {'temp': +2, 'aqi': +20}}
    zones = [{'id': z['id'], 'name': z['name'], 'type': z['type'], 'lat': z['lat'], 'lon': z['lon'],
              'temperature': baseline_temp + offsets.get(z['type'], {'temp': 0})['temp'] + np.random.uniform(-1, 1),
              'aqi': max(20, baseline_aqi + offsets.get(z['type'], {'aqi': 0})['aqi'] + np.random.randint(-5, 5)),
              'needs_intervention': False, 'mitigation_intensity': 0.0, 'suggestion': '', 'image_url': ''}
             for z in HYDERABAD_ZONES_BASE]
    return pd.DataFrame(zones)

def run_dqn_simulation():
    st.session_state.logs.append("DQN: Identifying hotspots...")
    zones_df = st.session_state.zones.copy()
    zones_df['needs_intervention'] = (zones_df['temperature'] > 40) & (zones_df['aqi'] > 100)
    st.session_state.zones = zones_df
    st.session_state.logs.append(f"DQN Complete: {zones_df['needs_intervention'].sum()} critical zones found.")

def run_ddpg_simulation():
    st.session_state.logs.append("DDPG: Calculating mitigation intensity...")
    zones_df = st.session_state.zones.copy()
    for i, row in zones_df.iterrows():
        if row['needs_intervention']:
            zones_df.loc[i, 'mitigation_intensity'] = round(np.random.uniform(0.3, 1.0), 2)
    st.session_state.zones = zones_df
    st.session_state.logs.append("DDPG Complete: Intensities assigned.")

def run_gemini_suggestions():
    st.session_state.logs.append("Contacting Gemini API with enhanced prompt...")
    hotspots = st.session_state.zones[st.session_state.zones['needs_intervention']]
    if hotspots.empty:
        st.session_state.error = "No critical hotspots identified."
        return
    model = genai.GenerativeModel('gemini-2.5-flash')
    details = "\n".join([f'- Zone ID: {z["id"]}, Name: {z["name"]}, Type: {z["type"]}, Temp: {z["temperature"]:.1f}°C, AQI: {z["aqi"]}' for _, z in hotspots.iterrows()])
    prompt = f"""You are an expert urban planning AI for Hyderabad. For each zone below, provide a tailored mitigation strategy.
    Respond in a single JSON object: {{"suggestions": [{{"zoneId": <int>, "detailed_suggestion": "<str>", "image_prompt": "<str>"}}]}}.
    For 'image_prompt', create a descriptive, comma-separated keyword list for a photorealistic image showing the solution in action. E.g., 'photorealistic, Charminar market, vibrant cool roofs, shaded walkways, misting fans'.
    Use \\n for newlines in 'detailed_suggestion'.
    Critical Zones:\n{details}"""
    try:
        config = GenerationConfig(response_mime_type="application/json")
        safety_settings = {c: HarmBlockThreshold.BLOCK_NONE for c in HarmCategory if c != HarmCategory.HARM_CATEGORY_UNSPECIFIED}
        response = model.generate_content(prompt, generation_config=config, safety_settings=safety_settings)
        suggestions = json.loads(response.text).get("suggestions", [])
        zones_df = st.session_state.zones.copy()
        for s in suggestions:
            suggestion_text = s.get('detailed_suggestion', '').replace('\\n', '<br>')
            image_query = s.get('image_prompt', 'cityscape').replace(' ', '-').replace(',', '')
            image_url = f"https://source.unsplash.com/featured/?{image_query}"
            # FIX: Match by zoneId, not zoneName
            zones_df.loc[zones_df['id'] == s['zoneId'], ['suggestion', 'image_url']] = [suggestion_text, image_url]
        st.session_state.zones = zones_df
        st.session_state.logs.append(f"Gemini Complete: Parsed {len(suggestions)} strategies.")
    except Exception as e:
        st.session_state.error = f"Failed to get or parse AI response: {e}"

@st.cache_data(ttl=3600)
def fetch_aqi_from_api(lat, lon):
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={AQI_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "ok":
            aqi = data["data"].get("aqi")
            pm25 = data["data"]["iaqi"].get("pm25", {}).get("v")
            pm10 = data["data"]["iaqi"].get("pm10", {}).get("v")
            
            # FIX: If PM10 is missing but PM2.5 is available, simulate it
            if pd.notna(pm25) and not pd.notna(pm10):
                pm10 = pm25 * 1.5 + np.random.uniform(-5, 5)

            return (aqi, pm25, pm10)
    except requests.exceptions.RequestException:
        return np.nan, np.nan, np.nan

def generate_past_aqi(latest_aqi):
    base = latest_aqi if pd.notna(latest_aqi) else 75
    return None, [max(10, base + 15 * np.sin(2*np.pi*(i%24)/24) + np.random.normal(0, 5)) for i in range(168)]

def mock_lstm_forecast(past_aqi):
    if not past_aqi: return [75] * 168
    last_value = past_aqi[-1]
    forecast = []
    for i in range(168):
        cyclical_trend = 15 * np.sin(2*np.pi*((len(past_aqi)+i)%24)/24)
        noise = np.random.normal(0, 3)
        next_value = last_value*0.95 + 75*0.05 + cyclical_trend + noise
        forecast.append(max(10, next_value))
        last_value = next_value
    return forecast

@st.cache_data
def get_road_path(_ors_client, waypoints_coords):
    try:
        reversed_coords = [[c[1], c[0]] for c in waypoints_coords]
        directions = _ors_client.directions(coordinates=reversed_coords, profile='driving-car', format='geojson')
        path_coords = directions['features'][0]['geometry']['coordinates']
        return [[c[1], c[0]] for c in path_coords]
    except ApiError as e:
        st.error(f"Could not fetch road path from OpenRouteService: {e}")
        return waypoints_coords

def find_optimal_route(start_id, end_id, zones_df, mode='safest'):
    zones_map = zones_df.set_index('id').to_dict('index')
    def get_weight(zone_id):
        zone = zones_map.get(zone_id, {})
        if mode == 'shortest': return 1
        aqi_cost = zone.get('aqi', 1000) * 1.5
        temp_cost = max(0, zone.get('temperature', 50) - 38) * 20
        return 1 + aqi_cost + temp_cost
    dist = {z['id']: float('inf') for _, z in zones_df.iterrows()}
    prev = {z['id']: None for _, z in zones_df.iterrows()}
    pq = [(0, start_id)]
    dist[start_id] = 0
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')): continue
        if u == end_id: break
        for v in ZONE_GRAPH.get(u, []):
            weight = get_weight(v)
            if dist.get(u, float('inf')) + weight < dist.get(v, float('inf')):
                dist[v] = dist.get(u) + weight
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
    path = []
    curr = end_id
    while curr is not None:
        path.append(curr)
        curr = prev.get(curr)
    path.reverse()
    if not path or path[0] != start_id: return [], 0, 0
    path_zones = zones_df[zones_df['id'].isin(path)]
    if path_zones.empty: return [], 0, 0
    return path, path_zones['aqi'].mean(), path_zones['temperature'].mean()

def get_aqi_recommendation(aqi):
    if not pd.notna(aqi): return "⚪ N/A", "#808080"
    aqi = int(aqi)
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
        if not pd.notna(aqi):
            st.session_state.logs.append("Using fallback baseline data.")
            aqi, pm25, pm10 = 110, 45, 80
        st.session_state.baseline_aqi, st.session_state.baseline_pm25, st.session_state.baseline_pm10 = aqi, pm25, pm10
    
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
                        st.caption(get_intensity_description(zone['mitigation_intensity']))
        st.divider()
        st.subheader("🧠 AI-Generated Mitigation Strategies")
        suggestions = st.session_state.zones[st.session_state.zones['suggestion'] != ""]
        if not suggestions.empty:
            for _, row in suggestions.iterrows():
                with st.container(border=True):
                    img_col, text_col = st.columns([1, 2])
                    if row['image_url']: img_col.image(row['image_url'], caption=f"Concept for {row['name']}")
                    with text_col:
                        st.markdown(f"##### 📍 Mitigation Plan for {row['name']}")
                        st.markdown(row['suggestion'], unsafe_allow_html=True)
        else:
            st.info("Run the full simulation to generate AI-powered mitigation strategies for critical zones.")
    with col2:
        st.header("Simulation Controls")
        if st.session_state.error: st.error(st.session_state.error)
        if st.button("1. Identify Hotspots", use_container_width=True, disabled=st.session_state.stage != 'INITIAL'):
            run_dqn_simulation(); st.session_state.stage = 'DQN_COMPLETE'; st.rerun()
        if st.button("2. Calculate Intensity", use_container_width=True, disabled=st.session_state.stage != 'DQN_COMPLETE'):
            run_ddpg_simulation(); st.session_state.stage = 'DDPG_COMPLETE'; st.rerun()
        if st.button("3. Generate AI Suggestions", use_container_width=True, disabled=st.session_state.stage != 'DDPG_COMPLETE'):
            with st.spinner("🤖 Contacting Gemini AI..."): run_gemini_suggestions()
            st.session_state.stage = 'INTEGRATED_COMPLETE'; st.rerun()
        st.divider()
        if st.button("Reset Scenario", type="secondary", use_container_width=True):
            st.session_state.zones = get_realistic_hyderabad_zones(st.session_state.baseline_aqi, 40)
            st.session_state.stage = 'INITIAL'; st.session_state.error = None; st.session_state.route_result = None; st.session_state.cluster_result = None
            st.rerun()
        with st.expander("Show Simulation Logs"):
            st.code("\n".join(st.session_state.logs), language="log")

with tab2:
    st.header("Live AQI & 7-Day Forecast")
    st.markdown("📍 Click anywhere on the map to get a location-specific AQI forecast.")
    m = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=11)
    folium.Marker([BASELINE_STATION_LAT, BASELINE_STATION_LON], popup="Baseline Station", icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
    map_data = st_folium(m, height=400, width=700)

    if map_data and map_data["last_clicked"]:
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        st.subheader(f"Forecast for Clicked Location ({lat:.4f}, {lon:.4f})")
        with st.spinner("Fetching live AQI..."):
            latest_aqi, latest_pm25, latest_pm10 = fetch_aqi_from_api(lat, lon)
    else:
        st.subheader(f"Forecast for Baseline Station")
        latest_aqi, latest_pm25, latest_pm10 = st.session_state.baseline_aqi, st.session_state.baseline_pm25, st.session_state.baseline_pm10

    st.markdown("#### Current Conditions")
    if pd.notna(latest_aqi):
        rec, color = get_aqi_recommendation(latest_aqi)
        st.metric(label="Live AQI", value=int(latest_aqi))
        st.markdown(f"**Condition:** <span style='color:{color};'>{rec}</span>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric(label="PM2.5", value=f"{latest_pm25} µg/m³" if pd.notna(latest_pm25) else "N/A")
        c2.metric(label="PM10", value=f"{latest_pm10:.1f} µg/m³" if pd.notna(latest_pm10) else "N/A")
    else:
        st.warning("Live AQI data unavailable for this location.")

    with st.spinner("Generating forecast..."):
        _, past_aqi = generate_past_aqi(latest_aqi)
        full_forecast = mock_lstm_forecast(past_aqi)
    
    st.markdown("#### 🔮 Next 7 Days Forecast (Daily Average)")
    daily_avg_forecast = [np.mean(full_forecast[i*24:(i+1)*24]) for i in range(7)]
    forecast_df = pd.DataFrame({'Day': [(datetime.now() + timedelta(days=i+1)).strftime('%a, %b %d') for i in range(7)], 'Average AQI': daily_avg_forecast})
    forecast_df['Color'] = forecast_df['Average AQI'].apply(lambda aqi: get_aqi_recommendation(aqi)[1])
    
    fig = px.bar(forecast_df, x='Day', y='Average AQI', title="7-Day Average AQI Forecast", text_auto='.0f', color='Color', color_discrete_map="identity")
    fig.update_layout(font=dict(size=14), title_font_size=20, xaxis_tickfont_size=12, yaxis_tickfont_size=12, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # MODIFICATION: Add color legend for the chart
    st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; gap: 20px; flex-wrap: wrap;">
            <div style="display: flex; align-items: center;"><div style="width: 15px; height: 15px; background-color: #28a745; margin-right: 5px;"></div>Good</div>
            <div style="display: flex; align-items: center;"><div style="width: 15px; height: 15px; background-color: #ffc107; margin-right: 5px;"></div>Moderate</div>
            <div style="display: flex; align-items: center;"><div style="width: 15px; height: 15px; background-color: #fd7e14; margin-right: 5px;"></div>Unhealthy (SG)</div>
            <div style="display: flex; align-items: center;"><div style="width: 15px; height: 15px; background-color: #dc3545; margin-right: 5px;"></div>Unhealthy</div>
        </div>
    """, unsafe_allow_html=True)


with tab3:
    st.header("Intelligent Route Recommendation")
    st.markdown("Find the safest (lowest pollution & heat) or the shortest route with **real road navigation**.")
    zones_df = st.session_state.zones
    zone_names = zones_df.set_index('id')['name'].to_dict()
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        start_name = st.selectbox("Select Start Location", options=zone_names.values(), key="start_loc")
    with col2:
        end_name = st.selectbox("Select End Location", options=zone_names.values(), index=len(zone_names)-1, key="end_loc")
    with col3:
        st.write("") 
        if st.button("📍 Use My Location", use_container_width=True):
            location = streamlit_geolocation()
            if location and location.get('latitude'):
                user_lat, user_lon = location['latitude'], location['longitude']
                distances = zones_df.apply(lambda r: np.hypot(r['lat']-user_lat, r['lon']-user_lon), axis=1)
                st.session_state.start_loc = zones_df.loc[distances.idxmin()]['name']
                st.success(f"Start set to nearest zone: {st.session_state.start_loc}")
                st.rerun()

    if st.button("🗺️ Find Routes", type="primary", use_container_width=True):
        if start_name == end_name:
            st.error("Start and End locations must be different.")
        else:
            with st.spinner("Calculating routes and fetching road paths..."):
                start_id = next(id for id, name in zone_names.items() if name == start_name)
                end_id = next(id for id, name in zone_names.items() if name == end_name)
                
                safe_waypoints, safe_aqi, safe_temp = find_optimal_route(start_id, end_id, zones_df, 'safest')
                
                def get_waypoints_coords(path_ids):
                    return [(zones_df.loc[zones_df['id'] == zid, 'lat'].iloc[0], zones_df.loc[zones_df['id'] == zid, 'lon'].iloc[0]) for zid in path_ids]

                safe_path_coords = get_road_path(ors_client, get_waypoints_coords(safe_waypoints))
                short_path_coords = get_road_path(ors_client, [get_waypoints_coords([start_id])[0], get_waypoints_coords([end_id])[0]])
                
                # Calculate metrics for the shortest path (start and end zones)
                short_path_zones = zones_df[zones_df['id'].isin([start_id, end_id])]
                short_aqi = short_path_zones['aqi'].mean()
                short_temp = short_path_zones['temperature'].mean()

                st.session_state.route_result = {
                    "safe_path_coords": safe_path_coords, "safe_aqi": safe_aqi, "safe_temp": safe_temp, "safe_waypoints": safe_waypoints,
                    "short_path_coords": short_path_coords, "short_aqi": short_aqi, "short_temp": short_temp,
                    "start_coords": get_waypoints_coords([start_id])[0], 
                    "end_coords": get_waypoints_coords([end_id])[0],
                    "start_name": start_name, "end_name": end_name
                }
                st.session_state.is_navigating = False
                st.rerun()

    if st.session_state.get('route_result'):
        res = st.session_state.route_result
        map_center = res['start_coords']
        if st.session_state.is_navigating and st.session_state.user_location:
            map_center = (st.session_state.user_location['latitude'], st.session_state.user_location['longitude'])

        m = folium.Map(location=map_center, zoom_start=14)
        folium.PolyLine(res["safe_path_coords"], color='green', weight=7, opacity=0.8, tooltip="Safest Route").add_to(m)
        folium.PolyLine(res["short_path_coords"], color='red', weight=4, opacity=0.7, dash_array='5, 5', tooltip="Shortest Route").add_to(m)
        folium.Marker(res['start_coords'], popup=f"START: {res['start_name']}", icon=folium.Icon(color='green', icon='play')).add_to(m)
        folium.Marker(res['end_coords'], popup=f"END: {res['end_name']}", icon=folium.Icon(color='red', icon='stop')).add_to(m)
        
        if st.session_state.is_navigating and st.session_state.user_location:
            folium.Marker((st.session_state.user_location['latitude'], st.session_state.user_location['longitude']), popup="Your Location", icon=folium.Icon(color='blue', icon='circle', prefix='fa')).add_to(m)
        st_folium(m, height=450, width=700, returned_objects=[])

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("##### 🏆 Safest Route (Recommended)")
            st.caption(' → '.join([zone_names.get(zid, "") for zid in res['safe_waypoints']]))
            st.metric("Avg AQI / Temp", f"{res['safe_aqi']:.1f} / {res['safe_temp']:.1f}°C")
        with res_col2:
            st.markdown("##### 📏 Standard Shortest Route")
            st.caption(f"{res['start_name']} → {res['end_name']} (Direct)")
            st.metric("Avg AQI / Temp", f"{res['short_aqi']:.1f} / {res['short_temp']:.1f}°C")
        st.divider()

        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if not st.session_state.is_navigating:
                if st.button("▶️ Start Navigation", use_container_width=True, type="primary"):
                    st.session_state.is_navigating = True; st.rerun()
            else:
                if st.button("⏹️ Stop Navigation", use_container_width=True):
                    st.session_state.is_navigating = False; st.session_state.user_location = None; st.rerun()
        if st.session_state.is_navigating:
            with nav_col2:
                st.info("Live navigation active...")
            time.sleep(5)
            location = streamlit_geolocation()
            if location and location.get('latitude'):
                st.session_state.user_location = location; st.rerun()

with tab4:
    st.header("Dynamic Zone Clustering (DBSCAN)")
    if st.button("Run Clustering Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing and clustering zone data..."):
            df_copy = st.session_state.zones.copy()
            features = StandardScaler().fit_transform(df_copy[['temperature', 'aqi']])
            dbscan = DBSCAN(eps=0.7, min_samples=2).fit(features)
            df_copy['cluster'] = dbscan.labels_
            st.session_state.cluster_result = df_copy
    if st.session_state.get('cluster_result') is not None:
        clustered_df = st.session_state.cluster_result
        cluster_labels = {cid: get_cluster_label(clustered_df, cid) for cid in clustered_df['cluster'].unique()}
        clustered_df['cluster_label'] = clustered_df['cluster'].map(cluster_labels)
        
        st.subheader("Interactive Cluster Plot")
        fig = px.scatter(clustered_df, x='aqi', y='temperature', color='cluster_label', hover_name='name', title="Zone Clusters by Temperature and AQI")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Geospatial Cluster Visualization")
        cluster_map = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=11)
        color_map = {-1: 'gray', 0: 'blue', 1: 'green', 2: 'purple', 3: 'orange', 4: 'red'}
        for _, zone in clustered_df.iterrows():
            cluster_id = zone['cluster']
            color = color_map.get(cluster_id, 'black')
            folium.Marker([zone['lat'], zone['lon']], popup=f"<strong>{zone['name']}</strong><br>Cluster: {zone['cluster_label']}", icon=folium.Icon(color=color)).add_to(cluster_map)
        st_folium(cluster_map, height=450, width=700)
