###############################################################
# TCS AI Fridays Hackathon
# Travel & Logistics Real-Time Disruption Advisor
# LangGraph + Streamlit version with rich output template
#
# Multi-"agent" design using different LLM models:
#   Planner       -> azure_ai/genailab-maas-Phi-4-reasoning
#   FlightAgent   -> azure/genailab-maas-gpt-35-turbo
#   WeatherAgent  -> azure_ai/genailab-maas-DeepSeek-R1
#   MapAgent      -> azure/genailab-maas-gpt-4o-mini
#   Responder     -> azure/genailab-maas-gpt-4o
###############################################################

from __future__ import annotations

import json
from datetime import date
from typing import TypedDict, Dict, List, Any

import requests
import streamlit as st
import folium
import urllib3
import httpx
from openai import OpenAI

from langgraph.graph import StateGraph, END

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

###############################################################
# ---------------- CONFIG: FILL THESE -------------------------
###############################################################
OPENAI_API_KEY = "sk-wwXoGekBcGsk52Y3lWZv1g"
GENAI_ENDPOINT = "https://genailab.tcs.in"

AMADEUS_KEY = "mqg4GsHmKpkBOUdSAUWJMNFHT9LKICgI"
AMADEUS_SECRET = "rE9OrhGSAxaWEvlu"
AMADEUS_BASE = "https://test.api.amadeus.com"

TAVILY_API_KEY = "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"

# Model mapping for "agents"
PLANNER_MODEL   = "azure_ai/genailab-maas-Phi-4-reasoning"
FLIGHT_MODEL    = "azure/genailab-maas-gpt-35-turbo"
WEATHER_MODEL   = "azure_ai/genailab-maas-DeepSeek-R1"
MAP_MODEL       = "azure/genailab-maas-gpt-4o-mini"
RESPONDER_MODEL = "azure/genailab-maas-gpt-4o"

###############################################################
# ---------------- AIRPORTS (INDIA, W/ ICAO) ------------------
###############################################################

# IATA: (Full name, ICAO, lat, lon)
AIRPORTS = {
    "DEL": ("Delhi – Indira Gandhi International Airport", "VIDP", 28.5562, 77.1000),
    "BLR": ("Bengaluru – Kempegowda International Airport", "VOBL", 13.1979, 77.7063),
    "BOM": ("Mumbai – Chhatrapati Shivaji Maharaj Intl", "VABB", 19.0896, 72.8656),
    "CCU": ("Kolkata – Netaji Subhas Chandra Bose Intl", "VECC", 22.6547, 88.4467),
    "MAA": ("Chennai – Anna International Airport", "VOMM", 12.9900, 80.1693),
    "BBI": ("Bhubaneswar – Biju Patnaik Airport", "VEBS", 20.2444, 85.8178),
    "HYD": ("Hyderabad – Rajiv Gandhi International Airport", "VOHS", 17.2400, 78.4300),
    "GOI": ("Goa International Airport", "VOGO", 15.3800, 73.8300),
}

airport_options = [f"{v[0]} ({k})" for k, v in AIRPORTS.items()]
extract_iata = lambda s: s.split("(")[-1].replace(")", "").strip()

###############################################################
# -------------------- LLM CLIENT + HELPERS -------------------
###############################################################

http_client = httpx.Client(verify=False)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=GENAI_ENDPOINT.rstrip("/"),
    http_client=http_client,
)

def call_llm(model: str, system_prompt: str, user_prompt: str, temperature: float = 0.25) -> str:
    """Generic helper to call one of your models."""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[LLM ERROR] model={model}: {e}")
        return f"LLM error: {e}"

###############################################################
# -------------------- EXTERNAL APIs --------------------------
###############################################################

# ---------- Amadeus ----------
def get_amadeus_token():
    url = f"{AMADEUS_BASE}/v1/security/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_KEY,
        "client_secret": AMADEUS_SECRET,
    }
    try:
        r = requests.post(url, data=payload, verify=False, timeout=10)
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        print("Amadeus token error:", e)
        return None


def fetch_flights(src: str, dst: str, date_str: str) -> Dict[str, Any]:
    token = get_amadeus_token()
    if not token:
        return {"count": 0, "flights": [], "error": "Amadeus token failed"}

    url = f"{AMADEUS_BASE}/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "originLocationCode": src,
        "destinationLocationCode": dst,
        "departureDate": date_str,
        "adults": 1,
    }

    try:
        r = requests.get(url, headers=headers, params=params, verify=False, timeout=15)
        if r.status_code != 200:
            return {"count": 0, "flights": [], "error": f"Amadeus {r.status_code}: {r.text}"}
        data = r.json()
    except Exception as e:
        print("Amadeus flights error:", e)
        return {"count": 0, "flights": [], "error": str(e)}

    if "data" not in data:
        return {"count": 0, "flights": []}

    flights = []
    for f in data["data"][:10]:
        seg = f["itineraries"][0]["segments"][0]
        flights.append({
            "airline": seg["carrierCode"],
            "flight": f"{seg['carrierCode']}{seg['number']}",
            "dep": seg["departure"]["at"],
            "arr": seg["arrival"]["at"],
        })

    return {"count": len(flights), "flights": flights}


# ---------- METAR / TAF ----------
def get_metar(icao: str) -> str:
    url = f"https://aviationweather.gov/api/data/metar?ids={icao}&format=json"
    try:
        r = requests.get(url, verify=False, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        js = r.json()
        if isinstance(js, list) and js:
            return js[0].get("rawOb", "No METAR data.")
        return "No METAR data."
    except Exception as e:
        print("METAR error:", e)
        return "METAR unavailable."


def get_taf(icao: str) -> str:
    url = f"https://aviationweather.gov/api/data/taf?ids={icao}&format=json"
    try:
        r = requests.get(url, verify=False, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        js = r.json()
        if isinstance(js, list) and js:
            return js[0].get("rawTaf", "No TAF data.")
        return "No TAF data."
    except Exception as e:
        print("TAF error:", e)
        return "TAF unavailable."


# ---------- Tavily (news / disruption) ----------
def tavily_search_disruptions(city: str, src: str, dst: str) -> str:
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": f"airport delays, weather disruption, strikes, congestion at {city} for route {src} to {dst} India",
        "search_depth": "basic",
        "max_results": 5,
    }
    try:
        r = requests.post(url, json=payload, verify=False, timeout=10)
        if r.status_code != 200:
            return f"Tavily error {r.status_code}"
        js = r.json()
        res = js.get("results", [])
        if not res:
            return "No disruption-related news found."
        snippets = [x.get("content", "") for x in res[:2]]
        return "\n".join(snippets)
    except Exception as e:
        print("Tavily error:", e)
        return "Tavily unavailable."


# ---------- OpenSky live airspace ----------
INDIA_ZONE = {"lat_min": 0, "lat_max": 45, "lon_min": 60, "lon_max": 105}

def opensky_live() -> List[Dict[str, Any]]:
    url = "https://opensky-network.org/api/states/all"
    flights: List[Dict[str, Any]] = []
    try:
        r = requests.get(url, verify=False, timeout=10)
        js = r.json()
        for s in js.get("states", []):
            lat, lon = s[6], s[5]
            if (
                lat and lon
                and INDIA_ZONE["lat_min"] < lat < INDIA_ZONE["lat_max"]
                and INDIA_ZONE["lon_min"] < lon < INDIA_ZONE["lon_max"]
            ):
                flights.append({"callsign": s[1], "lat": lat, "lon": lon})
    except Exception as e:
        print("OpenSky error:", e)
    return flights


def build_map(src: str, dst: str, live_flights: List[Dict[str, Any]]) -> folium.Map:
    m = folium.Map(location=[22.9, 78.9], zoom_start=5)
    for f in live_flights:
        folium.Marker(
            [f["lat"], f["lon"]],
            tooltip=f"✈ {f['callsign']}",
            icon=folium.Icon(color="red", icon="plane"),
        ).add_to(m)

    src_name, _, src_lat, src_lon = AIRPORTS[src]
    dst_name, _, dst_lat, dst_lon = AIRPORTS[dst]

    folium.Marker(
        [src_lat, src_lon],
        tooltip=f"Origin: {src_name} ({src})",
        icon=folium.Icon(color="green"),
    ).add_to(m)

    folium.Marker(
        [dst_lat, dst_lon],
        tooltip=f"Destination: {dst_name} ({dst})",
        icon=folium.Icon(color="blue"),
    ).add_to(m)

    return m

###############################################################
# ---------------- "AGENT" LOGIC FUNCTIONS --------------------
###############################################################

def planner_decide(user_q: str, src: str, dst: str, date_str: str) -> Dict[str, bool]:
    system = """
You are the Planner for a Travel & Logistics Real-Time Disruption Advisor.

Given a user query and basic route info, decide which data sources to use.

Respond with STRICT JSON ONLY (no extra text), using this schema:
{
  "call_flight_data": true/false,
  "call_weather": true/false,
  "call_disruption_news": true/false,
  "call_live_airspace": true/false
}
"""
    user = f"User question: {user_q}\nRoute: {src} -> {dst}\nDate: {date_str}"
    text = call_llm(PLANNER_MODEL, system, user, temperature=0.1)
    try:
        return json.loads(text)
    except Exception:
        return {
            "call_flight_data": True,
            "call_weather": True,
            "call_disruption_news": True,
            "call_live_airspace": True,
        }


def summarize_flights(flight_json: Dict[str, Any]) -> str:
    system = """
You summarize raw flight schedule JSON.

- Mention total number of flights.
- List key airlines.
- Indicate rough spread across the day (morning / afternoon / evening), if possible.
- If count=0, explain that schedule data is unavailable.

Return a short markdown section titled **Flights Summary**.
"""
    user = f"Raw flight JSON:\n{json.dumps(flight_json)}"
    return call_llm(FLIGHT_MODEL, system, user, temperature=0.2)


def analyze_weather_and_disruption(
    src: str,
    dst: str,
    date_str: str,
    src_metar: str,
    src_taf: str,
    dst_metar: str,
    dst_taf: str,
    disrupt_src: str,
    disrupt_dst: str,
) -> str:
    system = """
You are an aviation weather & disruption analyst.

You receive:
- METAR/TAF for origin and destination
- Text snippets about disruptions from web sources (Tavily)

You must:
- Explain operational weather at origin & destination in simple English.
- Highlight risks (fog, storms, low visibility, crosswinds, etc.).
- Incorporate disruption text (strikes, ATC issues, congestion).
- Rate disruption risk as Low / Moderate / High.

Return a nicely formatted markdown section with bullet points.
"""
    user = f"""
Route {src} -> {dst} on {date_str}

Origin {src} METAR: {src_metar}
Origin {src} TAF: {src_taf}

Destination {dst} METAR: {dst_metar}
Destination {dst} TAF: {dst_taf}

Origin disruption news:
{disrupt_src}

Destination disruption news:
{disrupt_dst}
"""
    return call_llm(WEATHER_MODEL, system, user, temperature=0.2)


def decide_map_inclusion(user_q: str, live_count: int) -> bool:
    keyword_flag = any(
        kw in user_q.lower()
        for kw in ["live", "map", "track", "where is", "airspace", "visual"]
    )
    system = """
You decide whether a live airspace map should be shown.

Reply ONLY with a single token: YES or NO.

Show map (YES) if:
- user mentions "map", "live tracking", "where is my flight", "airspace", or similar
- OR if they clearly care about visualizing traffic

Otherwise say NO.
"""
    user = f"User question: {user_q}\nLive flights over India: {live_count}"
    resp = call_llm(MAP_MODEL, system, user, temperature=0.1)
    return keyword_flag or ("yes" in resp.lower())


def build_final_report(
    user_q: str,
    src: str,
    dst: str,
    date_str: str,
    flight_json: Dict[str, Any],
    flight_summary: str,
    weather_block: str,
    live_count: int,
    map_included: bool,
) -> str:
    src_full = AIRPORTS[src][0].split("–")[0].strip()
    dst_full = AIRPORTS[dst][0].split("–")[0].strip()

    system = """
You are the final AI Disruption Advisor.

You must ALWAYS format the answer using THIS TEMPLATE (markdown):

## ✈️ Flight Intelligence Report

### 🧭 Route Summary  
- **From:** <Source City> (<SRC>)  
- **To:** <Destination City> (<DST>)  
- **Date:** <DATE>  
- **Query:** "<User Intent Summary>"

---

### 📋 Flight Availability  
🛫 **Total Flights Identified:** <COUNT>  

- If flights exist (flight_json['count'] > 0):  
  Create a markdown table:

  | Airline | Flight No. | Departure | Arrival | Duration |
  |--------|------------|-----------|---------|----------|
  | ...    | ...        | ...       | ...     | ...      |

  Then under the table, include or adapt the provided "Flights Summary".

- If no flights:  
  Write a short warning:  
  "⚠️ No valid flight schedule was found for this route/date from the flight API."

---

### 🌦 Weather & Airport Conditions  

Describe weather using the provided weather block, but structure it as:

📍 **Origin: <SRC>**  
- Conditions: <summary>  
- Impact: <Low / Moderate / High>  

📍 **Destination: <DST>**  
- Conditions: <summary>  
- Impact: <Low / Moderate / High>  

Then state:  
🧭 **Weather Risk Level:** 🟢 Low | 🟡 Moderate | 🔴 High  

---

### 📰 Operational Advisory  

Summarize disruptions / strikes / ATC / congestion based on the weather/disruption analysis and Tavily content.  
If nothing significant: say  
"No major operational alerts detected at the moment. Normal vigilance recommended."

---

### 📡 Real-Time Airspace Activity  

- **Tracked flights over India:** <live_count>  
- Traffic density estimate:  
  - 🟢 Light (<100)  
  - 🟡 Normal (100–300)  
  - 🔴 Heavy (>300)  

If map_included is true, add:  
"🗺 Live airspace visualization is displayed below."

Otherwise:  
"🗺 Map is not displayed. Ask for a 'live map' or 'flight tracking' to see it."

---

### 🎯 Recommendation  

Provide 2–4 bullet points, such as:  
- Overall disruption risk: Low / Medium / High  
- Suggested action (monitor only / add arrival buffer / consider rerouting / rebooking)  
- Best reliability time window, if applicable  

---

### 📌 Confidence Level  

End with a line like:  
**Confidence:** 75%  

---

✈ **Status Summary:**  
One single concise sentence that directly answers the user's question.

---

Use concise, professional language. Use emojis exactly as in the template, no extra decoration.  
DO NOT dump raw JSON anywhere. Only summarized content.
"""
    user = f"""
User question: {user_q}
Route: {src} -> {dst}
Source city name: {src_full}
Destination city name: {dst_full}
Date: {date_str}

Flight JSON:
{json.dumps(flight_json)}

Flights summary (markdown from another agent):
{flight_summary}

Weather + disruption (markdown from another agent):
{weather_block}

Live airspace count: {live_count}
Map_included: {map_included}
"""
    return call_llm(RESPONDER_MODEL, system, user, temperature=0.28)

###############################################################
# ---------------- LANGGRAPH STATE & NODES --------------------
###############################################################

class AgentState(TypedDict, total=False):
    user_question: str
    src: str
    dst: str
    date_str: str
    plan: Dict[str, bool]

    flight_json: Dict[str, Any]
    flight_summary: str

    weather_block: str

    live_flights: List[Dict[str, Any]]
    live_count: int

    map_included: bool

    final_answer: str


def planner_node(state: AgentState) -> AgentState:
    plan = planner_decide(state["user_question"], state["src"], state["dst"], state["date_str"])
    return {"plan": plan}


def flights_node(state: AgentState) -> AgentState:
    plan = state.get("plan", {})
    if not plan.get("call_flight_data", False):
        return {}
    fj = fetch_flights(state["src"], state["dst"], state["date_str"])
    fs = summarize_flights(fj)
    return {"flight_json": fj, "flight_summary": fs}


def weather_node(state: AgentState) -> AgentState:
    plan = state.get("plan", {})
    if not (plan.get("call_weather", False) or plan.get("call_disruption_news", False)):
        return {}
    src_name, src_icao, _, _ = AIRPORTS[state["src"]]
    dst_name, dst_icao, _, _ = AIRPORTS[state["dst"]]
    src_metar = get_metar(src_icao)
    src_taf = get_taf(src_icao)
    dst_metar = get_metar(dst_icao)
    dst_taf = get_taf(dst_icao)

    src_city = src_name.split("–")[0].strip()
    dst_city = dst_name.split("–")[0].strip()

    disrupt_src = tavily_search_disruptions(src_city, state["src"], state["dst"])
    disrupt_dst = tavily_search_disruptions(dst_city, state["src"], state["dst"])

    wb = analyze_weather_and_disruption(
        state["src"],
        state["dst"],
        state["date_str"],
        src_metar, src_taf,
        dst_metar, dst_taf,
        disrupt_src, disrupt_dst,
    )
    return {"weather_block": wb}


def live_node(state: AgentState) -> AgentState:
    plan = state.get("plan", {})
    if not plan.get("call_live_airspace", False):
        return {"live_flights": [], "live_count": 0}
    flights = opensky_live()
    return {"live_flights": flights, "live_count": len(flights)}


def map_node(state: AgentState) -> AgentState:
    mi = decide_map_inclusion(state["user_question"], state.get("live_count", 0))
    return {"map_included": mi}


def responder_node(state: AgentState) -> AgentState:
    fj = state.get("flight_json", {"count": 0, "flights": []})
    fs = state.get("flight_summary", "No flight summary.")
    wb = state.get("weather_block", "No weather data.")
    lc = state.get("live_count", 0)
    mi = state.get("map_included", False)

    ans = build_final_report(
        user_q=state["user_question"],
        src=state["src"],
        dst=state["dst"],
        date_str=state["date_str"],
        flight_json=fj,
        flight_summary=fs,
        weather_block=wb,
        live_count=lc,
        map_included=mi,
    )
    return {"final_answer": ans}


# Build LangGraph
graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("flights", flights_node)
graph.add_node("weather", weather_node)
graph.add_node("live", live_node)
graph.add_node("map", map_node)
graph.add_node("responder", responder_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "flights")
graph.add_edge("flights", "weather")
graph.add_edge("weather", "live")
graph.add_edge("live", "map")
graph.add_edge("map", "responder")
graph.add_edge("responder", END)

app = graph.compile()

###############################################################
# -------------------- STREAMLIT UI ---------------------------
###############################################################

st.set_page_config("TCS – Real-Time Disruption Advisor (LangGraph)", layout="wide")
st.title("🧠✈ Travel & Logistics Real-Time Disruption Advisor")

st.markdown(
    "> LangGraph-powered multi-agent pipeline for monitoring flight disruptions, "
    "weather, live airspace, and recommending rerouting / rebooking."
)

src_choice = st.sidebar.selectbox("Source airport", airport_options)
dst_choice = st.sidebar.selectbox("Destination airport", airport_options)
src = extract_iata(src_choice)
dst = extract_iata(dst_choice)

travel_date = st.sidebar.date_input("Travel date", value=date.today())
date_str = travel_date.strftime("%Y-%m-%d")

if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_map" not in st.session_state:
    st.session_state.last_map = None

# Show previous chat
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Ask about disruptions, delays, rerouting options, or live status...")

if user_q:
    st.session_state.chat.append({"role": "user", "content": user_q})

    initial_state: AgentState = {
        "user_question": user_q,
        "src": src,
        "dst": dst,
        "date_str": date_str,
        "plan": {},
        "flight_json": {"count": 0, "flights": []},
        "flight_summary": "",
        "weather_block": "",
        "live_flights": [],
        "live_count": 0,
        "map_included": False,
        "final_answer": "",
    }

    final_state = app.invoke(initial_state)

    answer = final_state.get("final_answer", "No answer generated.")
    live_flights = final_state.get("live_flights", [])
    live_count = final_state.get("live_count", 0)
    map_included = final_state.get("map_included", False)

    map_html = None
    if map_included and live_count > 0:
        fmap = build_map(src, dst, live_flights)
        map_html = fmap._repr_html_()
        st.session_state.last_map = map_html

    with st.chat_message("assistant"):
        st.markdown(answer)
        if map_html:
            st.markdown("### 🗺 Live Airspace Map")
            st.components.v1.html(map_html, height=480)

    st.session_state.chat.append({"role": "assistant", "content": answer})
    st.rerun()

# Show last map snapshot
if st.session_state.last_map:
    st.markdown("### 🗺 Last Live Airspace Snapshot")
    st.components.v1.html(st.session_state.last_map, height=480)
