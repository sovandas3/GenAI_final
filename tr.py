###############################################################
# AI Travel Advisor v4 — Amadeus Flight Offers + LLM + Live Map
###############################################################

import os
import requests
import folium
import httpx
from datetime import date

import streamlit as st
from openai import OpenAI
import urllib3

# Disable SSL warnings for enterprise network
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


###############################################################
# ---------------- CONFIG: REPLACE THESE ----------------------
###############################################################

OPENAI_API_KEY = "sk-wwXoGekBcGsk52Y3lWZv1g"
GENAI_ENDPOINT = "https://genailab.tcs.in"
MODEL_NAME = "azure/genailab-maas-gpt-4o"

AMADEUS_KEY = "mqg4GsHmKpkBOUdSAUWJMNFHT9LKICgI"
AMADEUS_SECRET = "rE9OrhGSAxaWEvlu"
AMADEUS_BASE = "https://test.api.amadeus.com"


###############################################################
# ---------------- Airport List ------------------------------
###############################################################

AIRPORTS = {
    "DEL": {"name": "Delhi IGI (DEL)", "lat": 28.5562, "lon": 77.1000},
    "BLR": {"name": "Bengaluru (BLR)", "lat": 13.1979, "lon": 77.7063},
    "BOM": {"name": "Mumbai (BOM)", "lat": 19.0896, "lon": 72.8656},
    "CCU": {"name": "Kolkata (CCU)", "lat": 22.6547, "lon": 88.4467},
    "MAA": {"name": "Chennai (MAA)", "lat": 12.9900, "lon": 80.1693},
    "BBI": {"name": "Bhubaneswar (BBI)", "lat": 20.2444, "lon": 85.8178},
}

AIRPORT_OPTIONS = list(AIRPORTS.keys())
extract_iata = lambda x: x.strip()


###############################################################
# ---------------- Amadeus API Layer -------------------------
###############################################################

def get_amadeus_token():
    url = f"{AMADEUS_BASE}/v1/security/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_KEY,
        "client_secret": AMADEUS_SECRET,
    }

    r = requests.post(url, data=payload, verify=False)
    return r.json().get("access_token")


def get_amadeus_flights(src, dst, date_str):
    token = get_amadeus_token()
    if not token:
        return {"error": True, "reason": "Authentication failed"}

    url = f"{AMADEUS_BASE}/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "originLocationCode": src,
        "destinationLocationCode": dst,
        "departureDate": date_str,
        "adults": 1
    }

    r = requests.get(url, headers=headers, params=params, verify=False)
    return r.json() if r.status_code == 200 else {"error": True, "response": r.text}


def summarize_offers(data):
    if "data" not in data:
        return {"error": True, "reason": "No valid flight data"}

    flights = data["data"]

    result = {
        "count": len(flights),
        "sample": []
    }

    for f in flights[:6]:
        seg = f["itineraries"][0]["segments"][0]
        result["sample"].append({
            "flight": f"{seg['carrierCode']}{seg['number']}",
            "dep": seg["departure"]["iataCode"],
            "arr": seg["arrival"]["iataCode"],
            "time": seg["departure"]["at"]
        })

    return result


###############################################################
# ---------------- OpenSky Live Map ---------------------------
###############################################################

INDIA_BOX = {"lat_min": 6.5, "lat_max": 37.2, "lon_min": 68.1, "lon_max": 97.5}

def opensky_feed():
    url = "https://opensky-network.org/api/states/all"
    r = requests.get(url, verify=False)
    data = r.json().get("states", [])

    flights = []
    for s in data:
        lat, lon = s[6], s[5]
        if lat and lon and INDIA_BOX["lat_min"] < lat < INDIA_BOX["lat_max"] and INDIA_BOX["lon_min"] < lon < INDIA_BOX["lon_max"]:
            flights.append({"callsign": s[1], "lat": lat, "lon": lon})
    return flights


def generate_map(flights):
    m = folium.Map(location=[22.9, 78.9], zoom_start=5)
    for f in flights:
        folium.Marker([f["lat"], f["lon"]], tooltip=f["callsign"], icon=folium.Icon(color="red", icon="plane")).add_to(m)
    return m


###############################################################
# ---------------- Streamlit UI -------------------------------
###############################################################

st.set_page_config("AI Flight Assistant", layout="wide")

if "chat" not in st.session_state:
    st.session_state.chat = []

st.sidebar.title("✈ Travel Search")

src = st.sidebar.selectbox("Source Airport", AIRPORT_OPTIONS)
dst = st.sidebar.selectbox("Destination Airport", AIRPORT_OPTIONS)
travel_date = st.sidebar.date_input("Date", value=date.today())
date_str = travel_date.strftime("%Y-%m-%d")

st.title("🧠 AI Travel Advisor with Real Aviation Data")


# Chat History
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


user_message = st.chat_input("Ask: 'How many flights?' or 'Any delays?'")

if user_message:

    st.session_state.chat.append({"role": "user", "content": user_message})

    # Fetch Data
    amadeus_data = summarize_offers(get_amadeus_flights(src, dst, date_str))
    live_map_data = opensky_feed()
    map_obj = generate_map(live_map_data)

    # Final Prompt
    system_context = f"""
User Question: {user_message}

Route: {src} → {dst}
Date: {date_str}

Real Route Flight Data:
{amadeus_data}

Live Indian Flights Tracked: {len(live_map_data)}

Answer clearly:
- Exact number if available
- Airline + Flight numbers
- Expected delays/risks
- Confidence %
"""

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=GENAI_ENDPOINT.rstrip("/"),
        http_client=httpx.Client(verify=False),
    )

    reply = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an aviation expert. Use ONLY the data provided."},
            {"role": "user", "content": system_context}
        ],
        temperature=0.25
    ).choices[0].message.content

    with st.chat_message("assistant"):
        st.write(reply)
        st.components.v1.html(map_obj._repr_html_(), height=420)

    st.session_state.chat.append({"role": "assistant", "content": reply})
    st.rerun()
