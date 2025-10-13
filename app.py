import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import requests
from datetime import datetime, timedelta
import heapq
from streamlit_geolocation import streamlit_geolocation
import openrouteservice
from openrouteservice.exceptions import ApiError
import urllib.parse
from folium.features import DivIcon
# Install once (if not installed already):
#pip install streamlit-js-eval
from streamlit_js_eval import streamlit_js_eval
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



@st.cache_data(ttl=1800)
def fetch_aqi_for_zone(lat, lon):
    """Fetch live AQI data from WAQI API for a given zone."""
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={AQI_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("status") == "ok":
            aqi = data["data"].get("aqi", np.nan)
            iaqi = data["data"].get("iaqi", {})
            pm25 = iaqi.get("pm25", {}).get("v", np.nan)
            pm10 = iaqi.get("pm10", {}).get("v", np.nan)
            return aqi, pm25, pm10
    except Exception:
        return np.nan, np.nan, np.nan
    return np.nan, np.nan, np.nan


@st.cache_data(ttl=1800)
def fetch_temperature_for_zone(lat, lon):
    """Fetch live temperature (°C) using Open-Meteo API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data["current"]["temperature_2m"]
    except Exception:
        return np.nan





@st.cache_data
def get_realistic_hyderabad_zones(baseline_aqi, baseline_temp):
    # smaller per-zone temperature offsets and less random noise for realism
    offsets = {
        'Industrial': {'temp': +3, 'aqi': +50},
        'Dense Residential': {'temp': +1, 'aqi': +20},
        'Commercial Hub': {'temp': +1, 'aqi': +30},
        'Historic/Market': {'temp': +1, 'aqi': +40},
        'Transport Hub': {'temp': +1, 'aqi': +35},
        'IT Hub': {'temp': 0, 'aqi': +15},
        'Financial District': {'temp': 0, 'aqi': +10},
        'Commercial/Residential': {'temp': 0, 'aqi': +25},
        'Posh Residential': {'temp': -1, 'aqi': -5},
        'Residential/Industrial': {'temp': +1, 'aqi': +45},
        'Green Space': {'temp': -3, 'aqi': -30},
        'Educational Hub': {'temp': 0, 'aqi': +20}
    }
    zones = []
    for z in HYDERABAD_ZONES_BASE:
        offs = offsets.get(z['type'], {'temp': 0, 'aqi': 0})
        temp = baseline_temp + offs['temp'] + np.random.uniform(-0.7, 0.7)  # smaller jitter
        aqi = max(10, int(baseline_aqi + offs['aqi'] + np.random.randint(-5, 6)))
        zones.append({
            'id': z['id'], 'name': z['name'], 'type': z['type'], 'lat': z['lat'], 'lon': z['lon'],
            'temperature': round(temp, 1), 'aqi': aqi,
            'needs_intervention': False, 'mitigation_intensity': 0.0, 'suggestion': '', 'image_url': ''
        })
    return pd.DataFrame(zones)
# def run_dqn_simulation():
#     st.session_state.logs.append("DQN: Identifying hotspots...")
#     zones_df = st.session_state.zones.copy()
#     zones_df['needs_intervention'] = (zones_df['temperature'] > 40) & (zones_df['aqi'] > 100)
#     st.session_state.zones = zones_df
#     st.session_state.logs.append(f"DQN Complete: {zones_df['needs_intervention'].sum()} critical zones found.")



def run_dqn_simulation():
    st.session_state.logs.append("DQN: Identifying hotspots...")
    zones_df = st.session_state.zones.copy()
    zones_df['needs_intervention'] = (zones_df['temperature'] > 40) & (zones_df['aqi'] > 100)
    st.session_state.zones = zones_df
    st.session_state.logs.append(f"DQN Complete: {zones_df['needs_intervention'].sum()} critical zones found.")
    st.session_state.last_heat_mitigation_time = datetime.now()  # <-- Add this

# def run_ddpg_simulation():
#     st.session_state.logs.append("DDPG: Calculating mitigation intensity...")
#     zones_df = st.session_state.zones.copy()
#     for i, row in zones_df.iterrows():
#         if row['needs_intervention']:
#             zones_df.loc[i, 'mitigation_intensity'] = round(np.random.uniform(0.3, 1.0), 2)
#     st.session_state.zones = zones_df
#     st.session_state.logs.append("DDPG Complete: Intensities assigned.")




def run_ddpg_simulation():
    st.session_state.logs.append("DDPG: Calculating mitigation intensity...")
    zones_df = st.session_state.zones.copy()
    for i, row in zones_df.iterrows():
        if row['needs_intervention']:
            zones_df.loc[i, 'mitigation_intensity'] = round(np.random.uniform(0.3, 1.0), 2)
    st.session_state.zones = zones_df
    st.session_state.logs.append("DDPG Complete: Intensities assigned.")
    st.session_state.last_heat_mitigation_time = datetime.now()  # <-- Add this


def _build_image_prompt_for_zone(zone):
    prompts = [
        "photorealistic",
        f"{zone.get('name', 'Hyderabad')}",
        f"{zone.get('type', '')}",
        "urban mitigation",
        "shaded walkways",
        "cool roofs",
        "vegetation",
        "community engagement"
    ]
    return ", ".join([p for p in prompts if p])
def run_gemini_suggestions():
    st.session_state.logs.append("Contacting Gemini API with enhanced prompt...")
    hotspots = st.session_state.zones[st.session_state.zones['needs_intervention']]
    if hotspots.empty:
        st.session_state.error = "No critical hotspots identified."
        return
    model = genai.GenerativeModel('gemini-2.5-flash')
    details = "\n".join(
        f"- Zone ID: {row['id']}, Name: {row['name']}, Type: {row['type']}, "
        f"Temp: {row['temperature']:.1f}°C, AQI: {row['aqi']}"
        for _, row in hotspots.iterrows()
    )
    # Create the structured prompt
    prompt = f"""You are an expert urban planning AI for Hyderabad. For each zone below, provide a tailored mitigation strategy.
    Respond in a single JSON object: {{"suggestions": [{{"zoneId": <int>, "detailed_suggestion": "<str>", "image_prompt": "<str>"}}]}}.
    For 'image_prompt', create a descriptive, comma-separated keyword list for a photorealistic image showing the solution in action.
    Use \\n for newlines in 'detailed_suggestion'.
    Critical Zones:
    {details}
    """
    try:
        config = GenerationConfig(response_mime_type="application/json")
        safety_settings = {
            c: HarmBlockThreshold.BLOCK_NONE
            for c in HarmCategory
            if c != HarmCategory.HARM_CATEGORY_UNSPECIFIED
        }

        response = model.generate_content(
            prompt,
            generation_config=config,
            safety_settings=safety_settings
        )
        suggestions = []
        parsed = None
        response_text = None
        # Try parsing JSON safely
        try:
            response_text = getattr(response, 'text', None) or str(response)
            parsed = json.loads(response_text)
        except Exception:
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    cand_text = response.candidates[0].get('content', response.candidates[0].get('message', ''))
                    parsed = json.loads(cand_text)
            except Exception:
                parsed = None

        if parsed and isinstance(parsed, dict) and parsed.get('suggestions'):
            suggestions = parsed.get('suggestions')
        # Fallback suggestions if Gemini response fails
        if not suggestions:
            st.session_state.logs.append('Gemini response parsing failed or returned no suggestions; using fallback generator.')
            for _, z in hotspots.iterrows():
                detailed = (
                    "Immediate measures: deploy mobile misting fans and temporary shaded canopies.\n"
                    "Mid-term: install cool roofs and increase tree canopy along main walkways.\n"
                    "Community: run heat awareness and distribution of hydration kits."
                )
                img_prompt = _build_image_prompt_for_zone(z)
                suggestions.append({
                    "zoneId": int(z['id']),
                    "detailed_suggestion": detailed,
                    "image_prompt": img_prompt
                })

        zones_df = st.session_state.zones.copy()
        # Process AI suggestions
        for s in suggestions:
            try:
                zid = int(s.get('zoneId') or s.get('zone_id') or s.get('id'))
            except Exception:
                zid = None
            detailed_suggestion = s.get('detailed_suggestion', '')
            if detailed_suggestion:
                detailed_suggestion = detailed_suggestion.replace('\\n', '\n')
            image_prompt = s.get('image_prompt', '')
            if not image_prompt:
                zone_row = zones_df[zones_df['id'] == zid] if zid is not None else None
                if zone_row is not None and not zone_row.empty:
                    image_prompt = _build_image_prompt_for_zone(zone_row.iloc[0])
                else:
                    image_prompt = 'photorealistic, urban mitigation, shaded walkway, cool roofs'
            safe_query = urllib.parse.quote_plus(image_prompt.replace(',', ''))
            image_url = f"https://source.unsplash.com/featured/?{safe_query}"
            if zid is not None and zid in zones_df['id'].values:
                zones_df.loc[zones_df['id'] == zid, ['suggestion', 'image_url']] = [detailed_suggestion, image_url]
            else:
                for _, z in zones_df.iterrows():
                    if z['name'].lower() in detailed_suggestion.lower():
                        zones_df.loc[zones_df['id'] == z['id'], ['suggestion', 'image_url']] = [detailed_suggestion, image_url]

        st.session_state.zones = zones_df
        st.session_state.logs.append(f"Gemini Complete: Parsed {len(suggestions)} strategies.")
        st.session_state.last_heat_mitigation_time = datetime.now()  # <-- Add this

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
            pm25 = data["data"].get("iaqi", {}).get("pm25", {}).get("v")
            pm10 = data["data"].get("iaqi", {}).get("pm10", {}).get("v")

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
        # ORS expects [lon, lat]
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
    # sensible default baseline temperature (user can adjust via slider)
    st.session_state.baseline_temp = 29
    st.session_state.zones = get_realistic_hyderabad_zones(st.session_state.baseline_aqi, st.session_state.baseline_temp)
    st.session_state.stage = 'INITIAL'
    st.session_state.error = None
    st.session_state.route_result = None
    st.session_state.cluster_result = None
    st.session_state.is_navigating = False
    st.session_state.user_location = None
    st.session_state.user_start_coords = None

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
                with st.container():
                    temp_color = '#ff4b4b' if zone["temperature"] > 40 else 'inherit'
                    aqi_color = '#ff8c00' if zone["aqi"] > 100 else 'inherit'
                    st.markdown(f"**{zone['name']}** {'🔥' if zone['needs_intervention'] else ''}")
                    st.markdown(f'<span style="color: {temp_color};"><b>Temp:</b> {zone["temperature"]:.1f}°C</span> | <span style="color: {aqi_color};"><b>AQI:</b> {zone["aqi"]}</span>', unsafe_allow_html=True)
                    if zone['mitigation_intensity'] > 0:
                        st.progress(zone['mitigation_intensity'])
                        st.caption(f"Intensity: {zone['mitigation_intensity']:.2f} — {get_intensity_description(zone['mitigation_intensity'])}")
        st.divider()
        st.subheader("🧠 AI-Generated Mitigation Strategies")
        suggestions = st.session_state.zones[st.session_state.zones['suggestion'] != ""]
        if not suggestions.empty:
            for _, row in suggestions.iterrows():
                with st.container():
                    img_col, text_col = st.columns([1, 2])
                    if row['image_url']:
                        try:
                            img_col.image(row['image_url'], caption=f"Concept for {row['name']}")
                        except Exception:
                            img_col.info('Image not available')
                    with text_col:
                        st.markdown(f"##### 📍 Mitigation Plan for {row['name']}")
                        st.markdown(row['suggestion'], unsafe_allow_html=True)
        else:
            st.info("Run the full simulation to generate AI-powered mitigation strategies for critical zones.")
    with col2:
        st.header("Simulation Controls")
        # Baseline temperature control to keep reported temps realistic
        new_temp = st.number_input("Baseline Temperature (°C)", min_value=10.0, max_value=45.0, value=float(st.session_state.baseline_temp), step=0.5)
        if st.button("Apply Baseline Temperature", use_container_width=True):
            st.session_state.baseline_temp = float(new_temp)
            st.session_state.zones = get_realistic_hyderabad_zones(st.session_state.baseline_aqi, st.session_state.baseline_temp)
            st.success(f"Applied baseline temperature: {st.session_state.baseline_temp}°C")
            st.rerun()
        if st.session_state.error: st.error(st.session_state.error)
        if st.button("1. Identify Hotspots", use_container_width=True, disabled=st.session_state.stage != 'INITIAL'):
            run_dqn_simulation(); st.session_state.stage = 'DQN_COMPLETE'; st.rerun()
        if st.button("2. Calculate Intensity", use_container_width=True, disabled=st.session_state.stage != 'DQN_COMPLETE'):
            run_ddpg_simulation(); st.session_state.stage = 'DDPG_COMPLETE'; st.rerun()
        if st.button("3. Generate AI Suggestions", use_container_width=True, disabled=st.session_state.stage != 'DDPG_COMPLETE'):
            with st.spinner("🤖 Contacting Gemini AI..."):
                run_gemini_suggestions()
            st.session_state.stage = 'INTEGRATED_COMPLETE'; st.rerun()
        st.divider()
        if st.button("Reset Scenario", type="secondary", use_container_width=True):
            st.session_state.zones = get_realistic_hyderabad_zones(st.session_state.baseline_aqi, st.session_state.baseline_temp)
            st.session_state.stage = 'INITIAL'; st.session_state.error = None; st.session_state.route_result = None; st.session_state.cluster_result = None
            st.session_state.user_start_coords = None; st.session_state.user_location = None
            st.rerun()
        with st.expander("Show Simulation Logs"):
    # Example dummy code display
            st.code("""
        def example():
            print("Hello Hyderabad")
        """, language="python")

            # Display actual logs
            st.code("\n".join(st.session_state.logs), language="log")
    if 'last_heat_mitigation_time' in st.session_state:
        st.markdown(
            f"**Last Simulation Run:** {st.session_state.last_heat_mitigation_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    else:
        st.markdown("**Last Simulation Run:** Not yet executed.")


with tab2:
    st.header("Live AQI & 7-Day Forecast")
    st.markdown("📍 Click anywhere on the map to get a location-specific AQI forecast.")
    m = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=11)
    folium.Marker([BASELINE_STATION_LAT, BASELINE_STATION_LON], popup="Baseline Station", icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
    map_data = st_folium(m, height=400, width=700)

    if map_data and map_data.get("last_clicked"):
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
    fig.update_traces(textfont=dict(size=14))
    fig.update_layout(font=dict(size=16), title_font_size=22, xaxis_tickfont_size=14, yaxis_tickfont_size=14, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; gap: 20px; flex-wrap: wrap;">
            <div style="display: flex; align-items: center;"><div style="width: 15px; height: 15px; background-color: #28a745; margin-right: 5px;"></div>Good</div>
            <div style="display: flex; align-items: center;"><div style="width: 15px; height: 15px; background-color: #ffc107; margin-right: 5px;"></div>Moderate</div>
            <div style="display: flex; align-items: center;"><div style="width: 15px; height: 15px; background-color: #fd7e14; margin-right: 5px;"></div>Unhealthy (SG)</div>
            <div style="display: flex; align-items: center;"><div style="width: 15px; height: 15px; background-color: #dc3545; margin-right: 5px;"></div>Unhealthy</div>
        </div>
    """, unsafe_allow_html=True)
from streamlit_js_eval import get_geolocation
import numpy as np
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
# TAB 3: Intelligent Route Planner (complete, with live tracking)
from streamlit_geolocation import streamlit_geolocation
from streamlit_folium import st_folium
import folium
import openrouteservice
from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
# ---- TAB 3: Intelligent Route Planner (Google-Maps-like navigation) ----
from streamlit_geolocation import streamlit_geolocation
from streamlit_folium import st_folium
import folium
import openrouteservice
from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
import time







with tab3:
    st.header("🧭 Intelligent Route Planner")
    st.markdown("Find the safest (lowest pollution & heat) or the shortest route with **real road navigation** and live updates (route refreshes every 60s during navigation).")

    zones_df = st.session_state.zones
    zone_names = zones_df.set_index('id')['name'].to_dict()

    # --- UI: Start / End / Use My Location ---
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        start_options = ["📍 My Current Location"] + list(zone_names.values())
        current_start = st.session_state.get("start_loc", list(zone_names.values())[0])
        if current_start not in start_options:
            current_start = list(zone_names.values())[0]
        start_idx = start_options.index(current_start)
        start_name = st.selectbox("Select Start Location", options=start_options, index=start_idx, key="start_loc")
    with col2:
        end_name = st.selectbox("Select End Location", options=list(zone_names.values()), index=len(zone_names)-1, key="end_loc")
    with col3:
        st.write("")
        if st.button("📍 Use My Location", use_container_width=True):
            loc = streamlit_geolocation()
            if loc and loc.get("latitude") and loc.get("longitude"):
                user_lat, user_lon = loc["latitude"], loc["longitude"]
                st.session_state.user_start_coords = (user_lat, user_lon)
                st.session_state.start_loc = "📍 My Current Location"
                st.success(f"✅ Using your current location as start: ({user_lat:.4f}, {user_lon:.4f})")
                st.session_state.route_result = None
                st.session_state.is_navigating = False
                st.session_state.user_location = {"latitude": user_lat, "longitude": user_lon}
                st.rerun()
            else:
                st.warning("⚠️ Unable to access location. Please allow location permission in your browser (and ensure app is served on localhost or HTTPS).")

    # --- Route mode toggle ---
    route_mode = st.radio("Select Route Mode", ["Safest Route", "Shortest Route", "Show Both"], horizontal=True, index=0)

    # --- Helper function: reduce waypoints to ≤ 70 ---
    def reduce_waypoints(coords, max_points=70):
        if len(coords) <= max_points:
            return coords
        factor = len(coords) // max_points + 1
        return coords[::factor]

    # --- Find Route button ---
    if st.button("🗺️ Find Route", type="primary", use_container_width=True):
        if start_name == end_name:
            st.error("Start and End locations must be different.")
        else:
            with st.spinner("Calculating routes..."):
                user_coords = st.session_state.get("user_start_coords", None)
                if start_name == "📍 My Current Location" and user_coords:
                    user_lat, user_lon = user_coords
                    distances = zones_df.apply(lambda r: np.hypot(r["lat"] - user_lat, r["lon"] - user_lon), axis=1)
                    start_id = zones_df.loc[distances.idxmin()]["id"]
                    start_coords_for_route = user_coords
                else:
                    start_coords_for_route = None
                    start_id = next(id for id, name in zone_names.items() if name == start_name)
                end_id = next(id for id, name in zone_names.items() if name == end_name)

                # SAFEST ROUTE
                safe_waypoints, safe_aqi, safe_temp = find_optimal_route(start_id, end_id, zones_df, "safest")
                safe_path_coords = get_road_path(
                    ors_client,
                    [(zones_df.loc[zones_df["id"] == w, "lat"].iloc[0], zones_df.loc[zones_df["id"] == w, "lon"].iloc[0]) for w in safe_waypoints]
                ) if safe_waypoints else []

                safe_path_coords_reduced = reduce_waypoints(safe_path_coords, max_points=70)
                safe_path_coords_lonlat = [(lon, lat) for lat, lon in safe_path_coords_reduced]

                if safe_path_coords_lonlat:
                    try:
                        safe_route_data = ors_client.directions(
                            coordinates=safe_path_coords_lonlat,
                            profile='driving-car',
                            format='geojson'
                        )
                        safe_duration = safe_route_data['features'][0]['properties']['summary']['duration'] / 60
                        safe_distance = safe_route_data['features'][0]['properties']['summary']['distance'] / 1000
                    except Exception:
                        safe_duration, safe_distance = None, None
                else:
                    safe_duration, safe_distance = None, None

                # SHORTEST ROUTE
                if start_coords_for_route:
                    short_path_coords = get_road_path(ors_client, [
                        start_coords_for_route,
                        (zones_df.loc[zones_df["id"] == end_id, "lat"].iloc[0], zones_df.loc[zones_df["id"] == end_id, "lon"].iloc[0])
                    ])
                else:
                    short_path_coords = get_road_path(ors_client, [
                        (zones_df.loc[zones_df["id"] == start_id, "lat"].iloc[0], zones_df.loc[zones_df["id"] == start_id, "lon"].iloc[0]),
                        (zones_df.loc[zones_df["id"] == end_id, "lat"].iloc[0], zones_df.loc[zones_df["id"] == end_id, "lon"].iloc[0])
                    ])

                short_path_coords_reduced = reduce_waypoints(short_path_coords, max_points=70)
                short_path_coords_lonlat = [(lon, lat) for lat, lon in short_path_coords_reduced]

                try:
                    short_route_data = ors_client.directions(
                        coordinates=short_path_coords_lonlat,
                        profile='driving-car',
                        format='geojson'
                    )
                    short_duration = short_route_data['features'][0]['properties']['summary']['duration'] / 60
                    short_distance = short_route_data['features'][0]['properties']['summary']['distance'] / 1000
                except Exception:
                    short_duration, short_distance = None, None

                short_zones = zones_df[zones_df["id"].isin([start_id, end_id])]
                short_aqi = short_zones["aqi"].mean()
                short_temp = short_zones["temperature"].mean()

                # --- Save route result ---
                st.session_state.route_result = {
                    "mode": route_mode,
                    "safe_path_coords": safe_path_coords,
                    "safe_aqi": safe_aqi,
                    "safe_temp": safe_temp,
                    "safe_duration": safe_duration,
                    "safe_distance": safe_distance,
                    "short_path_coords": short_path_coords,
                    "short_aqi": short_aqi,
                    "short_temp": short_temp,
                    "short_duration": short_duration,
                    "short_distance": short_distance,
                    "start_coords": start_coords_for_route if start_coords_for_route else (
                        zones_df.loc[zones_df["id"] == start_id, "lat"].iloc[0],
                        zones_df.loc[zones_df["id"] == start_id, "lon"].iloc[0]
                    ),
                    "end_coords": (
                        zones_df.loc[zones_df["id"] == end_id, "lat"].iloc[0],
                        zones_df.loc[zones_df["id"] == end_id, "lon"].iloc[0]
                    ),
                    "start_name": start_name,
                    "end_name": end_name
                }
                st.session_state.is_navigating = False
                st.session_state.last_nav_update = time.time()
                st.rerun()

    # --- Map rendering + metrics ---
    if st.session_state.get("route_result"):
        res = st.session_state.route_result
        is_navigating = st.session_state.get("is_navigating", False)

        # --- Route preview map ---
        preview_map = folium.Map(location=res["start_coords"], zoom_start=13)
        if res["mode"] in ["Safest Route", "Show Both"] and res.get("safe_path_coords"):
            folium.PolyLine(res["safe_path_coords"], color="green", weight=7, opacity=0.8, tooltip="Safest Route").add_to(preview_map)
        if res["mode"] in ["Shortest Route", "Show Both"] and res.get("short_path_coords"):
            folium.PolyLine(res["short_path_coords"], color="blue", weight=4, opacity=0.7, dash_array="5,5", tooltip="Shortest Route").add_to(preview_map)
        folium.Marker(res["start_coords"], popup=f"START: {res['start_name']}", icon=folium.Icon(color="green", icon="play")).add_to(preview_map)
        folium.Marker(res["end_coords"], popup=f"END: {res['end_name']}", icon=folium.Icon(color="red", icon="flag")).add_to(preview_map)

        # --- Legend only when showing both routes ---
        if res["mode"] == "Show Both":
            legend_html = """
            <div style="
                position: fixed; 
                bottom: 75px; left: 25px; width: 150px; height: 70px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                box-shadow: 3px 3px 6px rgba(0,0,0,0.2);
            ">
            <b>Route Legend</b><br>
            <span style="color:green; font-weight:bold;">&#9632;</span> Safest Route<br>
            <span style="color:blue; font-weight:bold;">&#9632;</span> Shortest Route
            </div>
            """
            preview_map.get_root().html.add_child(folium.Element(legend_html))

        st_folium(preview_map, height=480, width=920)

        # --- Metrics for both routes ---
        st.divider()
        c1, c2 = st.columns(2)
        if res["mode"] in ["Safest Route", "Show Both"]:
            with c1:
                st.markdown("### 🏆 Safest Route (Recommended)")
                st.metric("Avg AQI / Temp", f"{res['safe_aqi']:.1f} / {res['safe_temp']:.1f}°C")
                if res.get("safe_duration") and res.get("safe_distance"):
                    st.metric("Distance / Duration", f"{res['safe_distance']:.1f} km / {res['safe_duration']:.1f} min")
        if res["mode"] in ["Shortest Route", "Show Both"]:
            with c2:
                st.markdown("### 📏 Shortest Route")
                st.metric("Avg AQI / Temp", f"{res['short_aqi']:.1f} / {res['short_temp']:.1f}°C")
                if res.get("short_duration") and res.get("short_distance"):
                    st.metric("Distance / Duration", f"{res['short_distance']:.1f} km / {res['short_duration']:.1f} min")

        # --- Navigation control + live tracking ---
        st.markdown("### 🧭 Navigation (embedded, live updates every 60s)")
        nav_col1, nav_col2 = st.columns([1, 1])
        with nav_col1:
            if not is_navigating:
                if st.button("▶ Start Navigation", use_container_width=True):
                    st.session_state.is_navigating = True
                    fresh = streamlit_geolocation()
                    if fresh and fresh.get("latitude") and fresh.get("longitude"):
                        st.session_state.user_location = {"latitude": fresh["latitude"], "longitude": fresh["longitude"]}
                        st.session_state.user_start_coords = (fresh["latitude"], fresh["longitude"])
                    else:
                        st.session_state.user_location = st.session_state.get("user_location", None)
                    st.session_state.last_nav_update = time.time()
                    st.rerun()
            else:
                if st.button("⏹ Stop Navigation", use_container_width=True):
                    st.session_state.is_navigating = False
                    st.session_state.user_location = None
                    st.rerun()
        with nav_col2:
            live_tracking = st.checkbox("Enable live tracking (every 60s)", value=True, key="nav_live_checkbox")

        # --- Live navigation map ---
        if is_navigating:
            if live_tracking:
                st_autorefresh(interval=60000, key="nav_refresh")
            latest = streamlit_geolocation()
            if latest and latest.get("latitude") and latest.get("longitude"):
                st.session_state.user_location = {"latitude": latest["latitude"], "longitude": latest["longitude"]}
                st.session_state.user_start_coords = (latest["latitude"], latest["longitude"])

            center = (st.session_state["user_location"]["latitude"], st.session_state["user_location"]["longitude"]) \
                     if st.session_state.get("user_location") else res["start_coords"]
            nav_map = folium.Map(location=center, zoom_start=15)

            try:
                start_lonlat = (st.session_state["user_location"]["longitude"], st.session_state["user_location"]["latitude"]) \
                               if st.session_state.get("user_location") else (res["start_coords"][1], res["start_coords"][0])
                end_lonlat = (res["end_coords"][1], res["end_coords"][0])
                route_geo = ors_client.directions(coordinates=[start_lonlat, end_lonlat],
                                                 profile='driving-car', format='geojson')
                folium.GeoJson(route_geo, name="Route").add_to(nav_map)
            except Exception:
                if res.get("safe_path_coords"):
                    folium.PolyLine(res["safe_path_coords"], color="green", weight=6, opacity=0.8).add_to(nav_map)
                if res.get("short_path_coords"):
                    folium.PolyLine(res["short_path_coords"], color="blue", weight=4, opacity=0.7, dash_array="5,5").add_to(nav_map)

            folium.Marker([res["end_coords"][0], res["end_coords"][1]], popup="Destination", icon=folium.Icon(color="red")).add_to(nav_map)
            if st.session_state.get("user_location"):
                u = st.session_state["user_location"]
                folium.CircleMarker(location=[u["latitude"], u["longitude"]], radius=7, color="blue", fill=True, fill_color="blue", popup="You (live)").add_to(nav_map)
                folium.Circle(location=[u["latitude"], u["longitude"]], radius=12, color="blue", fill=False, opacity=0.2).add_to(nav_map)
            else:
                folium.Marker(res["start_coords"], popup="Start", icon=folium.Icon(color="green")).add_to(nav_map)

            st_folium(nav_map, height=600, width=920)
            st.success("Live navigation active — route updates every 60 seconds. Click ⏹ Stop Navigation to end.")
















# with tab3:
#     st.header("🧭 Intelligent Route Planner")
#     st.markdown("Find the safest (lowest pollution & heat) or the shortest route with **real road navigation** and live updates (route refreshes every 60s during navigation).")

#     zones_df = st.session_state.zones
#     zone_names = zones_df.set_index('id')['name'].to_dict()

#     # --- UI: Start / End / Use My Location ---
#     col1, col2, col3 = st.columns([2, 2, 1])
#     with col1:
#         start_options = ["📍 My Current Location"] + list(zone_names.values())
#         current_start = st.session_state.get("start_loc", list(zone_names.values())[0])
#         if current_start not in start_options:
#             current_start = list(zone_names.values())[0]
#         start_idx = start_options.index(current_start)
#         start_name = st.selectbox("Select Start Location", options=start_options, index=start_idx, key="start_loc")
#     with col2:
#         end_name = st.selectbox("Select End Location", options=list(zone_names.values()), index=len(zone_names)-1, key="end_loc")
#     with col3:
#         st.write("")
#         if st.button("📍 Use My Location", use_container_width=True):
#             loc = streamlit_geolocation()
#             if loc and loc.get("latitude") and loc.get("longitude"):
#                 user_lat, user_lon = loc["latitude"], loc["longitude"]
#                 st.session_state.user_start_coords = (user_lat, user_lon)
#                 st.session_state.start_loc = "📍 My Current Location"
#                 st.success(f"✅ Using your current location as start: ({user_lat:.4f}, {user_lon:.4f})")
#                 st.session_state.route_result = None
#                 st.session_state.is_navigating = False
#                 st.session_state.user_location = {"latitude": user_lat, "longitude": user_lon}
#                 st.rerun()
#             else:
#                 st.warning("⚠️ Unable to access location. Please allow location permission in your browser (and ensure app is served on localhost or HTTPS).")

#     # --- Route mode toggle ---
#     route_mode = st.radio("Select Route Mode", ["Safest Route", "Shortest Route", "Show Both"], horizontal=True, index=0)

#     # --- Helper function: reduce waypoints to ≤ 70 ---
#     def reduce_waypoints(coords, max_points=70):
#         if len(coords) <= max_points:
#             return coords
#         factor = len(coords) // max_points + 1
#         return coords[::factor]

#     # --- Find Route button ---
#     if st.button("🗺️ Find Route", type="primary", use_container_width=True):
#         if start_name == end_name:
#             st.error("Start and End locations must be different.")
#         else:
#             with st.spinner("Calculating routes..."):
#                 user_coords = st.session_state.get("user_start_coords", None)
#                 if start_name == "📍 My Current Location" and user_coords:
#                     user_lat, user_lon = user_coords
#                     distances = zones_df.apply(lambda r: np.hypot(r["lat"] - user_lat, r["lon"] - user_lon), axis=1)
#                     start_id = zones_df.loc[distances.idxmin()]["id"]
#                     start_coords_for_route = user_coords
#                 else:
#                     start_coords_for_route = None
#                     start_id = next(id for id, name in zone_names.items() if name == start_name)
#                 end_id = next(id for id, name in zone_names.items() if name == end_name)

#                 # SAFEST ROUTE (graph-based)
#                 safe_waypoints, safe_aqi, safe_temp = find_optimal_route(start_id, end_id, zones_df, "safest")
#                 safe_path_coords = get_road_path(
#                     ors_client,
#                     [(zones_df.loc[zones_df["id"] == w, "lat"].iloc[0], zones_df.loc[zones_df["id"] == w, "lon"].iloc[0]) for w in safe_waypoints]
#                 ) if safe_waypoints else []

#                 # Reduce to ≤ 70 waypoints for ORS
#                 safe_path_coords_reduced = reduce_waypoints(safe_path_coords, max_points=70)
#                 safe_path_coords_lonlat = [(lon, lat) for lat, lon in safe_path_coords_reduced]  # ORS expects (lon, lat)

#                 # Get duration/distance safely
#                 if safe_path_coords_lonlat:
#                     try:
#                         safe_route_data = ors_client.directions(
#                             coordinates=safe_path_coords_lonlat,
#                             profile='driving-car',
#                             format='geojson'
#                         )
#                         safe_duration = safe_route_data['features'][0]['properties']['summary']['duration'] / 60  # minutes
#                         safe_distance = safe_route_data['features'][0]['properties']['summary']['distance'] / 1000  # km
#                     except Exception:
#                         safe_duration, safe_distance = None, None
#                 else:
#                     safe_duration, safe_distance = None, None

#                 # SHORTEST ROUTE
#                 if start_coords_for_route:
#                     short_path_coords = get_road_path(ors_client, [
#                         start_coords_for_route,
#                         (zones_df.loc[zones_df["id"] == end_id, "lat"].iloc[0], zones_df.loc[zones_df["id"] == end_id, "lon"].iloc[0])
#                     ])
#                 else:
#                     short_path_coords = get_road_path(ors_client, [
#                         (zones_df.loc[zones_df["id"] == start_id, "lat"].iloc[0], zones_df.loc[zones_df["id"] == start_id, "lon"].iloc[0]),
#                         (zones_df.loc[zones_df["id"] == end_id, "lat"].iloc[0], zones_df.loc[zones_df["id"] == end_id, "lon"].iloc[0])
#                     ])

#                 # Reduce short path for ORS
#                 short_path_coords_reduced = reduce_waypoints(short_path_coords, max_points=70)
#                 short_path_coords_lonlat = [(lon, lat) for lat, lon in short_path_coords_reduced]

#                 # Duration/distance shortest
#                 try:
#                     short_route_data = ors_client.directions(
#                         coordinates=short_path_coords_lonlat,
#                         profile='driving-car',
#                         format='geojson'
#                     )
#                     short_duration = short_route_data['features'][0]['properties']['summary']['duration'] / 60
#                     short_distance = short_route_data['features'][0]['properties']['summary']['distance'] / 1000
#                 except Exception:
#                     short_duration, short_distance = None, None

#                 short_zones = zones_df[zones_df["id"].isin([start_id, end_id])]
#                 short_aqi = short_zones["aqi"].mean()
#                 short_temp = short_zones["temperature"].mean()

#                 # Save route result
#                 st.session_state.route_result = {
#                     "mode": route_mode,
#                     "safe_path_coords": safe_path_coords,
#                     "safe_aqi": safe_aqi,
#                     "safe_temp": safe_temp,
#                     "safe_duration": safe_duration,
#                     "safe_distance": safe_distance,
#                     "short_path_coords": short_path_coords,
#                     "short_aqi": short_aqi,
#                     "short_temp": short_temp,
#                     "short_duration": short_duration,
#                     "short_distance": short_distance,
#                     "start_coords": start_coords_for_route if start_coords_for_route else (
#                         zones_df.loc[zones_df["id"] == start_id, "lat"].iloc[0],
#                         zones_df.loc[zones_df["id"] == start_id, "lon"].iloc[0]
#                     ),
#                     "end_coords": (
#                         zones_df.loc[zones_df["id"] == end_id, "lat"].iloc[0],
#                         zones_df.loc[zones_df["id"] == end_id, "lon"].iloc[0]
#                     ),
#                     "start_name": start_name,
#                     "end_name": end_name
#                 }
#                 st.session_state.is_navigating = False
#                 st.session_state.last_nav_update = time.time()
#                 st.rerun()

#     # --- Route preview map + metrics ---
#     if st.session_state.get("route_result"):
#         res = st.session_state.route_result
#         preview_map = folium.Map(location=res["start_coords"], zoom_start=13)
#         if res["mode"] in ["Safest Route", "Show Both"] and res.get("safe_path_coords"):
#             folium.PolyLine(res["safe_path_coords"], color="green", weight=7, opacity=0.8, tooltip="Safest Route").add_to(preview_map)
#         if res["mode"] in ["Shortest Route", "Show Both"] and res.get("short_path_coords"):
#             folium.PolyLine(res["short_path_coords"], color="blue", weight=4, opacity=0.7, dash_array="5,5", tooltip="Shortest Route").add_to(preview_map)
#         folium.Marker(res["start_coords"], popup=f"START: {res['start_name']}", icon=folium.Icon(color="green", icon="play")).add_to(preview_map)
#         folium.Marker(res["end_coords"], popup=f"END: {res['end_name']}", icon=folium.Icon(color="red", icon="flag")).add_to(preview_map)

#         st_folium(preview_map, height=480, width=920)
#         # --- Legend for route colors ---
#         # --- Legend for route colors ---
#         legend_html = """
#         <div style="
#             position: fixed; 
#             bottom: 75px; left: 25px; width: 150px; height: 70px; 
#             border:2px solid grey; z-index:9999; font-size:14px;
#             background-color:white;
#             padding: 10px;
#             box-shadow: 3px 3px 6px rgba(0,0,0,0.2);
#         ">
#         <b>Route Legend</b><br>
#         <span style="color:green; font-weight:bold;">&#9632;</span> Safest Route<br>
#         <span style="color:blue; font-weight:bold;">&#9632;</span> Shortest Route
#         </div>
#         """
#         # MUST add legend before calling st_folium
#         preview_map.get_root().html.add_child(folium.Element(legend_html))

#         # Now render the map
#         if st.session_state.get("is_navigating", False):
#     # Live navigation active → show nav_map
#             st_folium(nav_map, height=600, width=920)
#         else:
#             # Navigation inactive → show route preview map
#             preview_map.get_root().html.add_child(folium.Element(legend_html))  # add legend
#             st_folium(preview_map, height=480, width=920)

        

#         st.divider()
#         c1, c2 = st.columns(2)
#         if res["mode"] in ["Safest Route", "Show Both"]:
#             with c1:
#                 st.markdown("### 🏆 Safest Route (Recommended)")
#                 st.metric("Avg AQI / Temp", f"{res['safe_aqi']:.1f} / {res['safe_temp']:.1f}°C")
#                 if res.get("safe_duration") and res.get("safe_distance"):
#                     st.metric("Distance / Duration", f"{res['safe_distance']:.1f} km / {res['safe_duration']:.1f} min")
#         if res["mode"] in ["Shortest Route", "Show Both"]:
#             with c2:
#                 st.markdown("### 📏 Shortest Route")
#                 st.metric("Avg AQI / Temp", f"{res['short_aqi']:.1f} / {res['short_temp']:.1f}°C")
#                 if res.get("short_duration") and res.get("short_distance"):
#                     st.metric("Distance / Duration", f"{res['short_distance']:.1f} km / {res['short_duration']:.1f} min")

#         # --- Navigation control and live tracking ---
#         st.markdown("### 🧭 Navigation (embedded, live updates every 60s)")
#         nav_col1, nav_col2 = st.columns([1, 1])
#         with nav_col1:
#             if not st.session_state.get("is_navigating", False):
#                 if st.button("▶ Start Navigation", use_container_width=True):
#                     st.session_state.is_navigating = True
#                     fresh = streamlit_geolocation()
#                     if fresh and fresh.get("latitude") and fresh.get("longitude"):
#                         st.session_state.user_location = {"latitude": fresh["latitude"], "longitude": fresh["longitude"]}
#                         st.session_state.user_start_coords = (fresh["latitude"], fresh["longitude"])
#                     else:
#                         st.session_state.user_location = st.session_state.get("user_location", None)
#                     st.session_state.last_nav_update = time.time()
#                     st.rerun()
#             else:
#                 if st.button("⏹ Stop Navigation", use_container_width=True):
#                     st.session_state.is_navigating = False
#                     st.session_state.user_location = None
#                     st.rerun()
#         with nav_col2:
#             live_tracking = st.checkbox("Enable live tracking (every 60s)", value=True, key="nav_live_checkbox")

#         if st.session_state.get("is_navigating", False):
#             if live_tracking:
#                 st_autorefresh(interval=60000, key="nav_refresh")
#             latest = streamlit_geolocation()
#             if latest and latest.get("latitude") and latest.get("longitude"):
#                 st.session_state.user_location = {"latitude": latest["latitude"], "longitude": latest["longitude"]}
#                 st.session_state.user_start_coords = (latest["latitude"], latest["longitude"])
#             center = (st.session_state["user_location"]["latitude"], st.session_state["user_location"]["longitude"]) if st.session_state.get("user_location") else res["start_coords"]
#             nav_map = folium.Map(location=center, zoom_start=15)
#             try:
#                 start_lonlat = (st.session_state["user_location"]["longitude"], st.session_state["user_location"]["latitude"]) if st.session_state.get("user_location") else (res["start_coords"][1], res["start_coords"][0])
#                 end_lonlat = (res["end_coords"][1], res["end_coords"][0])
#                 route_geo = ors_client.directions(coordinates=[start_lonlat, end_lonlat], profile='driving-car', format='geojson')
#                 folium.GeoJson(route_geo, name="Route").add_to(nav_map)
#             except Exception:
#                 if res.get("safe_path_coords"):
#                     folium.PolyLine(res["safe_path_coords"], color="green", weight=6, opacity=0.8).add_to(nav_map)
#                 if res.get("short_path_coords"):
#                     folium.PolyLine(res["short_path_coords"], color="blue", weight=4, opacity=0.7, dash_array="5,5").add_to(nav_map)
#             folium.Marker([res["end_coords"][0], res["end_coords"][1]], popup="Destination", icon=folium.Icon(color="red")).add_to(nav_map)
#             if st.session_state.get("user_location"):
#                 u = st.session_state["user_location"]
#                 folium.CircleMarker(location=[u["latitude"], u["longitude"]], radius=7, color="blue", fill=True, fill_color="blue", popup="You (live)").add_to(nav_map)
#                 folium.Circle(location=[u["latitude"], u["longitude"]], radius=12, color="blue", fill=False, opacity=0.2).add_to(nav_map)
#             else:
#                 folium.Marker(res["start_coords"], popup="Start", icon=folium.Icon(color="green")).add_to(nav_map)
#             st_folium(nav_map, height=600, width=920)
#             st.success("Live navigation active — route updates every 60 seconds. Click ⏹ Stop Navigation to end.")













# with tab3:
#     st.header("🧭 Intelligent Route Planner")
#     st.markdown("Find the safest (lowest pollution & heat) or the shortest route with **real road navigation** and live updates (route refreshes every 60s during navigation).")

#     zones_df = st.session_state.zones
#     zone_names = zones_df.set_index('id')['name'].to_dict()

#     # --- UI: Start / End / Use My Location ---
#     col1, col2, col3 = st.columns([2, 2, 1])
#     with col1:
#         # Ensure "📍 My Current Location" is the first option so we can set it easily
#         start_options = ["📍 My Current Location"] + list(zone_names.values())
#         # determine index to show (fall back to first zone)
#         current_start = st.session_state.get("start_loc", list(zone_names.values())[0])
#         if current_start not in start_options:
#             current_start = list(zone_names.values())[0]
#         start_idx = start_options.index(current_start)
#         start_name = st.selectbox("Select Start Location", options=start_options, index=start_idx, key="start_loc")
#     with col2:
#         end_name = st.selectbox("Select End Location", options=list(zone_names.values()), index=len(zone_names)-1, key="end_loc")
#     with col3:
#         st.write("")  # spacer
#         if st.button("📍 Use My Location", use_container_width=True):
#             loc = streamlit_geolocation()
#             if loc and loc.get("latitude") and loc.get("longitude"):
#                 user_lat, user_lon = loc["latitude"], loc["longitude"]
#                 # store the raw GPS coords
#                 st.session_state.user_start_coords = (user_lat, user_lon)
#                 # set the dropdown label to the special label so it is obvious
#                 st.session_state.start_loc = "📍 My Current Location"
#                 st.success(f"✅ Using your current location as start: ({user_lat:.4f}, {user_lon:.4f})")
#                 # clear any previous route and navigation state
#                 st.session_state.route_result = None
#                 st.session_state.is_navigating = False
#                 st.session_state.user_location = {"latitude": user_lat, "longitude": user_lon}
#                 st.rerun()
#             else:
#                 st.warning("⚠️ Unable to access location. Please allow location permission in your browser (and ensure app is served on localhost or HTTPS).")
#     # --- Route mode toggle ---
#     route_mode = st.radio("Select Route Mode", ["Safest Route", "Shortest Route", "Show Both"], horizontal=True, index=0)
#     # --- Find Route button ---
#     if st.button("🗺️ Find Route", type="primary", use_container_width=True):
#         if start_name == end_name:
#             st.error("Start and End locations must be different.")
#         else:
#             with st.spinner("Calculating routes..."):
#                 # If user picked "My Current Location" use stored coords; else use selected zone
#                 user_coords = st.session_state.get("user_start_coords", None)
#                 if start_name == "📍 My Current Location" and user_coords:
#                     # determine nearest zone id to user's coords (for graph-based routing)
#                     user_lat, user_lon = user_coords
#                     distances = zones_df.apply(lambda r: np.hypot(r["lat"] - user_lat, r["lon"] - user_lon), axis=1)
#                     start_id = zones_df.loc[distances.idxmin()]["id"]
#                     start_coords_for_route = user_coords
#                 else:
#                     start_coords_for_route = None
#                     start_id = next(id for id, name in zone_names.items() if name == start_name)

#                 end_id = next(id for id, name in zone_names.items() if name == end_name)

#                 # SAFEST ROUTE (graph-based) -> translate to road coordinates via get_road_path if waypoints exist
#                 safe_waypoints, safe_aqi, safe_temp = find_optimal_route(start_id, end_id, zones_df, "safest")
#                 safe_path_coords = get_road_path(
#                     ors_client,
#                     [(zones_df.loc[zones_df["id"] == w, "lat"].iloc[0], zones_df.loc[zones_df["id"] == w, "lon"].iloc[0]) for w in safe_waypoints]
#                 ) if safe_waypoints else []

#                 # SHORTEST ROUTE (direct)
#                 if start_coords_for_route:
#                     short_path_coords = get_road_path(ors_client, [
#                         start_coords_for_route,
#                         (zones_df.loc[zones_df["id"] == end_id, "lat"].iloc[0], zones_df.loc[zones_df["id"] == end_id, "lon"].iloc[0])
#                     ])
#                 else:
#                     short_path_coords = get_road_path(ors_client, [
#                         (zones_df.loc[zones_df["id"] == start_id, "lat"].iloc[0], zones_df.loc[zones_df["id"] == start_id, "lon"].iloc[0]),
#                         (zones_df.loc[zones_df["id"] == end_id, "lat"].iloc[0], zones_df.loc[zones_df["id"] == end_id, "lon"].iloc[0])
#                     ])

#                 short_zones = zones_df[zones_df["id"].isin([start_id, end_id])]
#                 short_aqi = short_zones["aqi"].mean()
#                 short_temp = short_zones["temperature"].mean()
#                 # Save route result (we store start_coords as actual lat/lon if available)
#                 st.session_state.route_result = {
#                     "mode": route_mode,
#                     "safe_path_coords": safe_path_coords,
#                     "safe_aqi": safe_aqi,
#                     "safe_temp": safe_temp,
#                     "short_path_coords": short_path_coords,
#                     "short_aqi": short_aqi,
#                     "short_temp": short_temp,
#                     "start_coords": start_coords_for_route if start_coords_for_route else (
#                         zones_df.loc[zones_df["id"] == start_id, "lat"].iloc[0],
#                         zones_df.loc[zones_df["id"] == start_id, "lon"].iloc[0]
#                     ),
#                     "end_coords": (
#                         zones_df.loc[zones_df["id"] == end_id, "lat"].iloc[0],
#                         zones_df.loc[zones_df["id"] == end_id, "lon"].iloc[0]
#                     ),
#                     "start_name": start_name,
#                     "end_name": end_name
#                 }
#                 st.session_state.is_navigating = False
#                 # record last update time for 60s checks
#                 st.session_state.last_nav_update = time.time()
#                 st.rerun()

#     # --- Route preview map + metrics ---
#     if st.session_state.get("route_result"):
#         res = st.session_state.route_result
#         preview_map = folium.Map(location=res["start_coords"], zoom_start=13)
#         # draw routes
#         if res["mode"] in ["Safest Route", "Show Both"] and res.get("safe_path_coords"):
#             folium.PolyLine(res["safe_path_coords"], color="green", weight=7, opacity=0.8, tooltip="Safest Route").add_to(preview_map)
#         if res["mode"] in ["Shortest Route", "Show Both"] and res.get("short_path_coords"):
#             folium.PolyLine(res["short_path_coords"], color="blue", weight=4, opacity=0.7, dash_array="5,5", tooltip="Shortest Route").add_to(preview_map)

#         # start/end markers
#         folium.Marker(res["start_coords"], popup=f"START: {res['start_name']}", icon=folium.Icon(color="green", icon="play")).add_to(preview_map)
#         folium.Marker(res["end_coords"], popup=f"END: {res['end_name']}", icon=folium.Icon(color="red", icon="flag")).add_to(preview_map)

#         st_folium(preview_map, height=480, width=920, returned_objects=[])

#         st.divider()
#         c1, c2 = st.columns(2)
#         if res["mode"] in ["Safest Route", "Show Both"]:
#             with c1:
#                 st.markdown("### 🏆 Safest Route (Recommended)")
#                 st.metric("Avg AQI / Temp", f"{res['safe_aqi']:.1f} / {res['safe_temp']:.1f}°C")
#         if res["mode"] in ["Shortest Route", "Show Both"]:
#             with c2:
#                 st.markdown("### 📏 Shortest Route")
#                 st.metric("Avg AQI / Temp", f"{res['short_aqi']:.1f} / {res['short_temp']:.1f}°C")

#         # --- Navigation control and live tracking ---
#         st.markdown("### 🧭 Navigation (embedded, live updates every 60s)")
#         nav_col1, nav_col2 = st.columns([1, 1])
#         with nav_col1:
#             if not st.session_state.get("is_navigating", False):
#                 if st.button("▶ Start Navigation", use_container_width=True):
#                     # enter navigation mode
#                     st.session_state.is_navigating = True
#                     # try to refresh immediate user location
#                     fresh = streamlit_geolocation()
#                     if fresh and fresh.get("latitude") and fresh.get("longitude"):
#                         st.session_state.user_location = {"latitude": fresh["latitude"], "longitude": fresh["longitude"]}
#                         st.session_state.user_start_coords = (fresh["latitude"], fresh["longitude"])
#                     else:
#                         # if we don't get a fresh location, ensure we still have something in session
#                         st.session_state.user_location = st.session_state.get("user_location", None)
#                     st.session_state.last_nav_update = time.time()
#                     st.rerun()
#             else:
#                 if st.button("⏹ Stop Navigation", use_container_width=True):
#                     st.session_state.is_navigating = False
#                     st.session_state.user_location = None
#                     st.rerun()

#         with nav_col2:
#             live_tracking = st.checkbox("Enable live tracking (every 60s)", value=True, key="nav_live_checkbox")

#         # If navigation active, perform live updates (every 60s)
#         if st.session_state.get("is_navigating", False):
#             # trigger periodic rerun every 60 seconds (only when enabled)
#             if live_tracking:
#                 # this will cause the script to re-run every 60_000 ms
#                 st_autorefresh(interval=60000, key="nav_refresh")

#             # re-acquire latest user location (each run) — will prompt permission if not allowed
#             latest = streamlit_geolocation()
#             if latest and latest.get("latitude") and latest.get("longitude"):
#                 st.session_state.user_location = {"latitude": latest["latitude"], "longitude": latest["longitude"]}
#                 st.session_state.user_start_coords = (latest["latitude"], latest["longitude"])

#             # build navigation map centered on user's latest location if available, else route start
#             center = (st.session_state["user_location"]["latitude"], st.session_state["user_location"]["longitude"]) if st.session_state.get("user_location") else res["start_coords"]
#             nav_map = folium.Map(location=center, zoom_start=15)

#             # compute fresh route from current position to destination using ORS
#             try:
#                 # ORS expects (lon, lat)
#                 if st.session_state.get("user_location"):
#                     start_lonlat = (st.session_state["user_location"]["longitude"], st.session_state["user_location"]["latitude"])
#                 else:
#                     start_lonlat = (res["start_coords"][1], res["start_coords"][0])
#                 end_lonlat = (res["end_coords"][1], res["end_coords"][0])

#                 route_geo = ors_client.directions(coordinates=[start_lonlat, end_lonlat], profile='driving-car', format='geojson')
#                 folium.GeoJson(route_geo, name="Route").add_to(nav_map)
#             except Exception as e:
#                 # fallback to previously computed polylines
#                 if res.get("safe_path_coords"):
#                     folium.PolyLine(res["safe_path_coords"], color="green", weight=6, opacity=0.8).add_to(nav_map)
#                 if res.get("short_path_coords"):
#                     folium.PolyLine(res["short_path_coords"], color="blue", weight=4, opacity=0.7, dash_array="5,5").add_to(nav_map)

#             # add start / destination markers
#             folium.Marker([res["end_coords"][0], res["end_coords"][1]], popup="Destination", icon=folium.Icon(color="red")).add_to(nav_map)
#             # add moving user marker (blue)
#             if st.session_state.get("user_location"):
#                 u = st.session_state["user_location"]
#                 folium.CircleMarker(location=[u["latitude"], u["longitude"]], radius=7, color="blue", fill=True, fill_color="blue", popup="You (live)").add_to(nav_map)
#                 folium.Circle(location=[u["latitude"], u["longitude"]], radius=12, color="blue", fill=False, opacity=0.2).add_to(nav_map)
#             else:
#                 # show start point if no live user location
#                 folium.Marker(res["start_coords"], popup="Start", icon=folium.Icon(color="green")).add_to(nav_map)

#             # show map
#             st_folium(nav_map, height=600, width=920)
#             st.success("Live navigation active — route updates every 60 seconds. Click ⏹ Stop Navigation to end.")

with tab4:
    st.header("🌐 Dynamic Zone Clustering (DBSCAN)")
    if st.button("Run Clustering Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing and clustering zone data..."):
            df_copy = st.session_state.zones.copy()
            # Normalize features for DBSCAN
            scaler = StandardScaler()
            features = scaler.fit_transform(df_copy[['temperature', 'aqi']])
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=0.75, min_samples=2).fit(features)
            df_copy['cluster'] = dbscan.labels_
            # Compute descriptive cluster labels
            def get_cluster_label(df, cid):
                if cid == -1:
                    return "Noise / Outlier Zone"
                sub = df[df['cluster'] == cid]
                avg_temp = sub['temperature'].mean()
                avg_aqi = sub['aqi'].mean()
                if avg_temp > 35 and avg_aqi > 150:
                    return "🔥 High Heat & High Pollution"
                elif avg_temp > 35 and avg_aqi <= 150:
                    return "☀️ Hot but Cleaner Air"
                elif avg_temp <= 35 and avg_aqi > 150:
                    return "🌫️ Cool but Polluted"
                else:
                    return "🌿 Cool & Clean Zone"
            cluster_labels = {
                cid: get_cluster_label(df_copy, cid)
                for cid in df_copy['cluster'].unique()
            }
            df_copy['cluster_label'] = df_copy['cluster'].map(cluster_labels)
            # Save results + labels persistently
            st.session_state.cluster_result = df_copy
            st.session_state.cluster_labels = cluster_labels
    # --- Display if clustering has been run ---
    if st.session_state.get('cluster_result') is not None:
        clustered_df = st.session_state.cluster_result
        cluster_labels = st.session_state.get('cluster_labels', {})
        st.subheader("📊 Cluster Summary")
        summary_df = (
            clustered_df.groupby('cluster_label')
            .agg({'temperature': 'mean', 'aqi': 'mean', 'id': 'count'})
            .rename(columns={'id': 'zone_count', 'temperature': 'avg_temp', 'aqi': 'avg_aqi'})
            .reset_index()
        )
        st.dataframe(summary_df.style.background_gradient(cmap='coolwarm', subset=['avg_aqi', 'avg_temp']))
        # --- Plotly Visualization ---
        st.subheader("🧩 Cluster Distribution (AQI vs Temperature)")
        fig = px.scatter(
            clustered_df,
            x='aqi',
            y='temperature',
            color='cluster_label',
            hover_name='name',
            hover_data={'lat': False, 'lon': False},
            title="Zone Clusters by Temperature and AQI",
            labels={'aqi': 'Air Quality Index', 'temperature': 'Temperature (°C)'},
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(legend_title_text='Cluster Label', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        # --- Folium Map Visualization ---
        st.subheader("🗺️ Geospatial Cluster Visualization")
        cluster_map = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=11, tiles="cartodb positron")
        marker_cluster = folium.plugins.MarkerCluster().add_to(cluster_map)
        color_palette = [
            "red", "green", "blue", "purple", "orange", "darkred",
            "cadetblue", "darkgreen", "lightgray", "black"
        ]
        color_map = {
            cid: color_palette[i % len(color_palette)]
            for i, cid in enumerate(sorted(clustered_df['cluster'].unique()))
        }
        for _, zone in clustered_df.iterrows():
            cluster_id = zone['cluster']
            color = color_map.get(cluster_id, 'gray')
            popup_html = f"""
                <strong>{zone['name']}</strong><br>
                <b>Cluster:</b> {zone['cluster_label']}<br>
                🌡️ <b>Temp:</b> {zone['temperature']:.1f}°C<br>
                🏭 <b>AQI:</b> {zone['aqi']}
            """
            folium.CircleMarker(
                location=[zone['lat'], zone['lon']],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(marker_cluster)
        # --- Legend ---
        if cluster_labels:
            legend_html = """
            <div style="position: fixed; bottom: 50px; left: 50px; width: 240px;
                        background-color: white; border-radius: 8px; padding: 10px; 
                        box-shadow: 0 0 8px rgba(0,0,0,0.3); font-size: 13px;">
                <b>🗂️ Cluster Legend</b><br>
            """
            for cid, label in cluster_labels.items():
                color = color_map.get(cid, 'gray')
                legend_html += f'<div><i style="background:{color};width:12px;height:12px;display:inline-block;border-radius:50%;margin-right:6px;"></i>{label}</div>'
            legend_html += "</div>"
            cluster_map.get_root().html.add_child(folium.Element(legend_html))
        st_folium(cluster_map, height=500, width=750)
#pip install streamlit-geolocation


