#!/usr/bin/env python3
# app_full_enhanced_v4.py
"""
Travel & Logistics Real-Time Advisor — Enhanced v4

Changes from v3:
- Full-screen Login page (email + password) before dashboard
- flightapi.io integration functions (replace key / small param tweaks as needed)
- One-way and Round-trip flight search + IATA filtering
- Results in tabular form with status labels: 'Airborne', 'Scheduled', 'In Ground'
- Pydeck map for flight paths + nearby aircraft (OpenSky)
- Chat (Q&A) with LLM for suggestions & itinerary creation + guardrail checks
- LLM evaluation metrics (approximate BLEU/ROUGE/METEOR-like scores) shown in a table
- SMS sending via Twilio included
- Functions are documented with comments before each function
- Agentic flow: modular functions to allow future orchestration

Config expected (via env or .env):
- OPENAI_MODEL, OPENAI_API_KEY, AZURE_OPENAI_BASE, AZURE_OPENAI_API_VERSION (as before)
- FLIGHTAPI_KEY (for flightapi.io) - set to "" for demo/mock mode
- TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM (for SMS)
- GMAIL_SENDER, GMAIL_APP_PASSWORD (for email)
- AUTH_USERS (json mapping "email":"sha256(salt+password)") or leave empty for demo-mode
- AUTH_SALT
"""

import os
import json
import time
import math
import traceback
import requests
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
st.set_page_config(page_title="Travel Disruption Advisor — Enhanced v4", layout="wide")

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import pydeck as pdk

# Optional
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

# OpenAI compatibility (as prior)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AZURE_OPENAI_BASE = os.getenv("AZURE_OPENAI_BASE", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01")

FLIGHTAPI_KEY = os.getenv("FLIGHTAPI_KEY", "")  
GMAIL_SENDER = os.getenv("GMAIL_SENDER", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

TWILIO_SID = os.getenv("TWILIO_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM", "")
DEFAULT_NOTIFY_PHONE = os.getenv("DEFAULT_NOTIFY_PHONE", "")

AUTH_USERS_JSON = os.getenv("AUTH_USERS", "")  # '{"user@example.com":"<sha256hex>"}'
AUTH_SALT = os.getenv("AUTH_SALT", "")

# Simple metrics collector (same as v3)
from collections import defaultdict
from time import perf_counter
class MetricsCollector:
    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = {}
        self.histograms = defaultdict(list)
        self._timers = {}
    def increment(self, name, amount=1): self.counters[name] += amount
    def gauge(self, name, value): self.gauges[name] = value
    def record_latency(self, name, seconds):
        if seconds is None: return
        self.histograms[name].append(float(seconds))
    def start_timer(self, name): self._timers[name] = perf_counter()
    def stop_timer(self, name):
        start = self._timers.pop(name, None)
        if start is None: return None
        elapsed = perf_counter() - start
        self.record_latency(name, elapsed)
        return elapsed
    def get_metrics(self):
        hist_stats = {}
        for k, arr in self.histograms.items():
            if len(arr) == 0:
                hist_stats[k] = {"count": 0, "min": None, "max": None, "avg": None}
            else:
                hist_stats[k] = {"count": len(arr), "min": min(arr), "max": max(arr), "avg": sum(arr) / len(arr)}
        return {"counters": dict(self.counters), "gauges": dict(self.gauges), "histograms": hist_stats}

metrics = MetricsCollector()

# Small IATA fallback
IATA_TO_COORDS = {
    "DEL": (28.5562, 77.1000),
    "BLR": (13.1986, 77.7066),
    "BOM": (19.0896, 72.8656),
    "MAA": (12.9959, 80.1690),
    "HYD": (17.2403, 78.4294),
    "AMD": (23.0776, 72.6347),
    "COK": (10.1520, 76.4019),
}

# ---------------- Utility functions ----------------

def now_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        try:
            return {'text': resp.text}
        except Exception:
            return {'error': 'unknown response'}

def sha256_hash(password: str, salt: str = AUTH_SALT) -> str:
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()

def parse_latlon_input(s: str) -> Optional[Tuple[float,float]]:
    """Parse a 'lat,lon' input string into a tuple (lat,lon)."""
    try:
        if "," in s:
            a,b = s.split(",",1); a=a.strip(); b=b.strip()
            if all(c.replace(".","",1).replace("-","",1).isdigit() for c in [a,b]):
                return float(a), float(b)
    except Exception:
        pass
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.asin(min(1, math.sqrt(a)))

# ---------------- Auth ----------------

def load_auth_users()->Dict[str,str]:
    if not AUTH_USERS_JSON:
        return {}
    try:
        return json.loads(AUTH_USERS_JSON)
    except Exception:
        return {}

AUTH_USERS = load_auth_users()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["user_email"] = None
    st.session_state["history"] = []
    st.session_state["chat_history"] = []

def authenticate_user(email: str, password: str) -> bool:
    """
    Validate credentials provided against AUTH_USERS mapping.
    If AUTH_USERS is empty, demo-mode will accept any credentials and create session in demo mode.
    """
    if not AUTH_USERS:
        # demo mode, accept any login
        st.session_state["authenticated"] = True
        st.session_state["user_email"] = email or "demo@example.com"
        return True
    if email in AUTH_USERS and sha256_hash(password) == AUTH_USERS[email]:
        st.session_state["authenticated"] = True
        st.session_state["user_email"] = email
        return True
    return False

# def logout_user():
#     """Reset session state to logout."""
#     st.session_state["authenticated"] = False
#     st.session_state["user_email"] = None
#     st.experimental_rerun()

def logout_user():
    """Reset session state to logout."""
    st.session_state["authenticated"] = False
    st.session_state["user_email"] = None
    try:
        st.experimental_rerun()
    except Exception:
        return

# ---------------- Flight API integration ----------------

def flightapi_search_oneway(origin_iata: str, dest_iata: str, date_iso: Optional[str] = None, mock: bool = False) -> Dict[str, Any]:
    """
    Query flightapi.io (one-way).
    - origin_iata, dest_iata: IATA codes or city identifiers
    - date_iso: departure date in 'YYYY-MM-DD' (optional)
    - Returns a dict with 'data': list of flight dicts
    NOTE: flightapi.io endpoint patterns may differ. This function attempts a typical REST call:
      GET https://www.flightapi.io/onewaytrip/<APIKEY>/<ORIGIN>/<DEST>/<YYYY-MM-DD>/...
    Adapt the URL/params if flightapi.io has a different contract.
    If FLIGHTAPI_KEY is empty, returns mocked sample data.
    """
    if mock or not FLIGHTAPI_KEY:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        sample = []
        # create a few sample flights with varying statuses and durations
        for i in range(4):
            depart = now + timedelta(hours=1 + i*2)
            arrive = depart + timedelta(hours=2 + i)
            sample.append({
                "flight_iata": f"AI{201 + i}",
                "airline": "DemoAir",
                "departure": {"iata": origin_iata, "scheduled": depart.isoformat()},
                "arrival": {"iata": dest_iata, "scheduled": arrive.isoformat()},
                "departure_time": depart.isoformat(),
                "arrival_time": arrive.isoformat(),
                "duration_min": int((arrive-depart).total_seconds()//60),
                "status": "scheduled" if i%3 else "en-route" if i%2 else "landed"
            })
        return {"data": sample}
    try:
        # Example: GET https://www.flightapi.io/onewaytrip/{APIKEY}/{ORIGIN}/{DEST}/{YYYY-MM-DD}
        base = "https://www.flightapi.io"
        if date_iso:
            url = f"{base}/onewaytrip/{FLIGHTAPI_KEY}/{origin_iata}/{dest_iata}/{date_iso}"
        else:
            url = f"{base}/onewaytrip/{FLIGHTAPI_KEY}/{origin_iata}/{dest_iata}"
        r = requests.get(url, timeout=12)
        return safe_json(r)
    except Exception as e:
        return {"error": str(e)}

def flightapi_search_roundtrip(origin_iata: str, dest_iata: str, dep_date_iso: str, ret_date_iso: str, mock: bool = False) -> Dict[str, Any]:
    """
    Query flightapi.io for a round trip.
    If no FLIGHTAPI_KEY, returns mock.
    """
    if mock or not FLIGHTAPI_KEY:
        # For mock, just call oneway twice (depart and return) and label them as roundtrip segments
        out1 = flightapi_search_oneway(origin_iata, dest_iata, dep_date_iso, mock=True)
        out2 = flightapi_search_oneway(dest_iata, origin_iata, ret_date_iso, mock=True)
        # tag segments
        for f in out1["data"]:
            f["segment"] = "outbound"
        for f in out2["data"]:
            f["segment"] = "return"
        return {"data": out1["data"] + out2["data"]}
    try:
        base = "https://www.flightapi.io"
        # Example hypothetical endpoint:
        url = f"{base}/roundtrip/{FLIGHTAPI_KEY}/{origin_iata}/{dest_iata}/{dep_date_iso}/{ret_date_iso}"
        r = requests.get(url, timeout=14)
        return safe_json(r)
    except Exception as e:
        return {"error": str(e)}

def classify_flight_status(flight_record: Dict[str, Any]) -> str:
    """
    Map API flight state fields to 'Airborne' / 'Scheduled' / 'In Ground'
    Uses common field names: 'status', 'flight_status', presence of 'airborne' boolean, or comparing scheduled/actual times.
    """
    status_raw = str(flight_record.get("status") or flight_record.get("flight_status") or "").lower()
    # common heuristics
    if any(k in status_raw for k in ["en-route", "airborne", "airborn", "airborne", "airbourne"]):
        return "Airborne"
    if any(k in status_raw for k in ["scheduled", "booked", "scheduled to depart"]):
        return "Scheduled"
    if any(k in status_raw for k in ["landed", "in_ground", "land", "arrived", "on ground", "arrived"]):
        return "In Ground"
    # fallback: if arrival_time < now -> In Ground
    try:
        arr = flight_record.get("arrival_time") or (flight_record.get("arrival") or {}).get("scheduled") or (flight_record.get("arrival") or {}).get("estimated")
        if arr:
            arr_ts = datetime.fromisoformat(arr).replace(tzinfo=timezone.utc).timestamp()
            if arr_ts < datetime.utcnow().replace(tzinfo=timezone.utc).timestamp():
                return "In Ground"
    except Exception:
        pass
    # default to scheduled
    return "Scheduled"

def flights_to_dataframe(payload: Dict[str, Any], filter_iata: Optional[str] = None) -> pd.DataFrame:
    """
    Defensive parser: produce columns Flight, Airline, Departure, Arrival, Source, Destination, Duration, Duration_min, Status.
    Works even if payload is empty or has different field names.
    """
    rows = []
    for f in payload.get("data", []):
        flight_iata = f.get("flight_iata") or f.get("flight") or f.get("flight_number") or "N/A"
        airline = f.get("airline") if isinstance(f.get("airline"), str) else (f.get("airline", {}).get("name") if isinstance(f.get("airline"), dict) else "N/A")
        dep = f.get("departure") or {}
        arr = f.get("arrival") or {}
        # sometimes APIs use 'departure_time' or nested 'departure' dict with 'scheduled'
        dep_time = f.get("departure_time") or dep.get("scheduled") or dep.get("estimated") or dep.get("actual") or ""
        arr_time = f.get("arrival_time") or arr.get("scheduled") or arr.get("estimated") or arr.get("actual") or ""
        # source/dest fallback to strings if present at top-level
        src = (dep.get("iata") or dep.get("icao") or dep.get("city") or f.get("source") or f.get("from") or "") if isinstance(dep, dict) else (f.get("source") or "")
        dst = (arr.get("iata") or arr.get("icao") or arr.get("city") or f.get("destination") or f.get("to") or "") if isinstance(arr, dict) else (f.get("destination") or "")
        duration_min = f.get("duration_min") or f.get("flight_time_minutes") or f.get("duration") or None
        if duration_min:
            try:
                duration_min_int = int(duration_min)
                duration_text = f"{duration_min_int//60}h {duration_min_int%60}m"
            except Exception:
                duration_text = str(duration_min)
        else:
            duration_text = "unknown"
        status_label = classify_flight_status(f)
        rows.append({
            "Flight": flight_iata,
            "Airline": airline,
            "Departure": dep_time,
            "Arrival": arr_time,
            "Source": src,
            "Destination": dst,
            "Duration": duration_text,
            "Duration_min": duration_min,
            "Status": status_label,
            "raw": f
        })
    cols = ["Flight","Airline","Departure","Arrival","Source","Destination","Duration","Duration_min","Status","raw"]
    if rows:
        df = pd.DataFrame(rows)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]
    else:
        df = pd.DataFrame([], columns=cols)
    if filter_iata:
        fi = filter_iata.strip().upper()
        if "Source" in df.columns and "Destination" in df.columns:
            srcs = df["Source"].fillna("").astype(str).str.upper()
            dsts = df["Destination"].fillna("").astype(str).str.upper()
            df = df[(srcs == fi) | (dsts == fi)].reset_index(drop=True)
        else:
            df = pd.DataFrame([], columns=cols)
    return df



# ---------------- OpenSky nearby aircraft ----------------

def get_nearby_aircraft(lat: float, lon: float, radius_km: float = 200.0) -> List[Dict[str, Any]]:
    """
    Uses OpenSky to get nearby aircraft. Returns a list of dicts with callsign, lat, lon, distance_km, velocity_m_s
    """
    try:
        url = "https://opensky-network.org/api/states/all"
        r = requests.get(url, timeout=12)
        j = safe_json(r)
        if "states" not in j:
            return []
        out = []
        for s in j["states"]:
            callsign = s[1].strip() if s[1] else ""
            lon2 = s[5]; lat2 = s[6]; vel = s[9] if len(s) > 9 else None
            if lat2 is None or lon2 is None: continue
            d = haversine_km(lat, lon, lat2, lon2)
            if d <= radius_km:
                out.append({"callsign": callsign or "N/A", "lat": lat2, "lon": lon2, "distance_km": round(d, 1), "velocity_m_s": vel or 0.0})
        out = sorted(out, key=lambda x: x["distance_km"])
        return out[:200]
    except Exception:
        return []

# ---------------- LLM Chat & Recommendation (light wrapper) ----------------

# We provide a simple wrapper that uses whichever OpenAI client is available.
# If you are using Azure OpenAI or the new OpenAI Python SDK, adapt the calls below.
# For now, we'll attempt to use the legacy `openai` if available, else fall back to no-ops.

OPENAI_CLIENT = None
OPENAI_CLIENT_INSTANCE = None
try:
    from openai import OpenAI as OpenAIClass
    OPENAI_CLIENT_INSTANCE = OpenAIClass(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    try:
        import openai as openai_legacy
        OPENAI_CLIENT = openai_legacy
        if OPENAI_API_KEY:
            openai_legacy.api_key = OPENAI_API_KEY
    except Exception:
        OPENAI_CLIENT = None

def guardrail_check(user_text: str) -> Tuple[bool, str]:
    """
    Very simple guardrail: deny requests containing prohibited categories like violence, threats, illicit.
    Returns (allowed: bool, reason: str)
    """
    banned_keywords = ["kill", "bomb", "attack", "murder", "suicide", "harm", "weapon", "explosive", "terror", "threat"]
    low = user_text.lower()
    for kw in banned_keywords:
        if kw in low:
            return False, f"User input contains banned keyword: {kw}"
    return True, "ok"

def llm_chat_reply(prompt: str, max_tokens:int=400, temperature:float=0.2) -> str:
    """
    Send prompt to LLM and return textual reply. Handles multiple client variants (best-effort).
    If no OpenAI client available, returns a deterministic fallback.
    """
    start = time.perf_counter()
    # guardrail: ensure prompt length small on client
    try:
        if OPENAI_CLIENT_INSTANCE:
            # New SDK style
            resp = OPENAI_CLIENT_INSTANCE.chat.completions.create(
                model=OPENAI_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=max_tokens, temperature=temperature
            )
            # attempt to extract content robustly
            try:
                text = resp.choices[0].message.content
            except Exception:
                try:
                    text = resp.choices[0].message["content"]
                except Exception:
                    text = str(resp)
            metrics.record_latency("llm_chat_seconds", time.perf_counter()-start); metrics.increment("llm_chats", 1)
            return str(text).strip()
        if OPENAI_CLIENT and getattr(OPENAI_CLIENT, "__name__", "").lower() == "openai":
            # legacy API
            resp = OPENAI_CLIENT.ChatCompletion.create(model=OPENAI_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=max_tokens, temperature=temperature)
            text = None
            try:
                text = resp.choices[0].message.content
            except Exception:
                try:
                    text = resp.choices[0].text
                except Exception:
                    text = str(resp)
            metrics.record_latency("llm_chat_seconds", time.perf_counter()-start); metrics.increment("llm_chats", 1)
            return str(text).strip()
    except Exception as e:
        # don't crash the app on LLM failure
        metrics.record_latency("llm_chat_seconds", time.perf_counter()-start); metrics.increment("llm_chat_failures", 1)
        return f"[LLM_ERROR] {e}"
    # fallback deterministic reply
    metrics.record_latency("llm_chat_seconds", time.perf_counter()-start); metrics.increment("llm_chat_fallbacks", 1)
    return "LLM not available. This is a fallback reply. Please configure OPENAI_API_KEY."

# ---------------- LLM Metrics (approximate) ----------------

def tokenize_simple(text: str) -> List[str]:
    return [t for t in "".join(c if c.isalnum() else " " for c in (text or "")).lower().split() if t]

def bleu_like(reference: str, candidate: str) -> float:
    """
    Very small BLEU-like unigram precision: (# overlapping unigrams / total candidate unigrams)
    """
    r_tok = tokenize_simple(reference)
    c_tok = tokenize_simple(candidate)
    if not c_tok: return 0.0
    overlap = sum(1 for t in c_tok if t in r_tok)
    return overlap / len(c_tok)

def rouge_like(reference: str, candidate: str) -> float:
    """
    ROUGE-L-ish: longest common subsequence ratio approximated with token overlap / reference length
    """
    r_tok = tokenize_simple(reference)
    c_tok = tokenize_simple(candidate)
    if not r_tok: return 0.0
    overlap = sum(1 for t in r_tok if t in c_tok)
    return overlap / len(r_tok)

def meteor_like(reference: str, candidate: str) -> float:
    """
    Simple METEOR-like average of precision & recall on tokens.
    """
    r_tok = tokenize_simple(reference); c_tok = tokenize_simple(candidate)
    if not c_tok or not r_tok: return 0.0
    precision = sum(1 for t in c_tok if t in r_tok) / len(c_tok)
    recall = sum(1 for t in r_tok if t in c_tok) / len(r_tok)
    if precision + recall == 0: return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_llm_metrics(reference: str, candidate: str) -> Dict[str, float]:
    """
    Returns a dict of approximate BLEU / ROUGE / METEOR and a simple 'self_check' sanity metric.
    """
    b = bleu_like(reference, candidate)
    r = rouge_like(reference, candidate)
    m = meteor_like(reference, candidate)
    # Self-check: if candidate length < 0.2 * reference length, low score
    ref_len = len(tokenize_simple(reference)); cand_len = len(tokenize_simple(candidate))
    self_check = 1.0 if (ref_len == 0 or cand_len / max(ref_len,1) >= 0.4) else 0.0
    return {"BLEU": round(b,3), "ROUGE": round(r,3), "METEOR": round(m,3), "SelfCheck": self_check}

# ---------------- Notifications ----------------

def send_email(to: str, subject: str, body: str) -> Tuple[bool, str]:
    """Simple Gmail SMTP send - use only if GMAIL_APP_PASSWORD & GMAIL_SENDER configured."""
    if not GMAIL_APP_PASSWORD or not GMAIL_SENDER:
        return False, "Email config missing (GMAIL_SENDER/GMAIL_APP_PASSWORD)"
    try:
        import smtplib
        srv = smtplib.SMTP("smtp.gmail.com", 587, timeout=20); srv.starttls()
        srv.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
        msg = f"Subject: {subject}\n\n{body}"
        srv.sendmail(GMAIL_SENDER, [to], msg.encode("utf-8")); srv.quit()
        return True, "sent"
    except Exception as e:
        return False, str(e)

def send_sms_via_twilio(to: str, body: str) -> Tuple[bool, str]:
    """Send SMS with Twilio (if configured)."""
    if not TWILIO_AVAILABLE or not TWILIO_SID or not TWILIO_TOKEN or not TWILIO_FROM:
        return False, "Twilio not configured"
    try:
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(body=body, from_=TWILIO_FROM, to=to)
        return True, f"sid={msg.sid}"
    except Exception as e:
        return False, str(e)

# ---------------- UI: Login Page ----------------

def show_login_page():
    """
    Render a modal-like login page in the main area (not sidebar).
    Requires email & password. On success, sets session_state['authenticated'] True.
    """
    st.markdown("<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True)
    st.title("🔒 Travel & Logistics Advisor — Login")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.write("Please log in to continue. If `AUTH_USERS` is not configured, demo-mode will be used.")
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", value=st.session_state.get("login_email", ""))
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
            if submitted:
                ok = authenticate_user(email.strip(), password)
                if ok:
                    st.success(f"Logged in as {st.session_state.get('user_email')}")
                    # Try to rerun cleanly if the function exists; otherwise return and rely on Streamlit's normal rerun.
                    try:
                        # Some Streamlit builds expose experimental_rerun; call it if present.
                        st.experimental_rerun()
                    except Exception:
                        # If it's not present (AttributeError) or fails, just return so the current run finishes.
                        return
                else:
                    st.error("Invalid credentials. If AUTH_USERS not configured, app runs in demo mode; set AUTH_USERS to enable true auth.")

# ---------------- UI: Dashboard ----------------

def show_dashboard():
    """
    Render the main dashboard (two-column layout) with:
    - Flight search controls
    - Map (pydeck)
    - Flight table
    - Chat Q&A panel
    - Metrics & history
    """
    st.sidebar.header(f"User: {st.session_state.get('user_email','')}")
    if st.sidebar.button("Log out"):
        logout_user()
    st.title("🚨 Travel & Logistics Real-Time Advisor — Dashboard (v4)")
    st.sidebar.markdown("### Flight Search")
    origin = st.sidebar.text_input("Origin (IATA / city / 'lat,lon')", value="DEL")
    destination = st.sidebar.text_input("Destination (IATA / city / 'lat,lon')", value="BLR")
    trip_type = st.sidebar.selectbox("Trip type", ["One-way", "Round-trip"])
    dep_date = st.sidebar.date_input("Departure date (optional)", value=None)
    ret_date = None
    if trip_type == "Round-trip":
        ret_date = st.sidebar.date_input("Return date (optional)", value=None)
    flight_search_btn = st.sidebar.button("Search flights")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Logistics / Routing")
    route_from = st.sidebar.text_input("Pickup (address or lat,lon)", value="Indira Gandhi International Airport, Delhi")
    route_to = st.sidebar.text_input("Dropoff (address or lat,lon)", value="Kempegowda International Airport, Bangalore")
    route_btn = st.sidebar.button("Compute fastest route")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Notifications")
    notify_email = st.sidebar.text_input("Notify email", value=st.session_state.get("user_email", "user@example.com"))
    notify_phone = st.sidebar.text_input("Notify phone (for SMS)", value=DEFAULT_NOTIFY_PHONE)
    send_notify_btn = st.sidebar.button("Send last summary (email+SMS)")

    # layout main
    map_col, info_col = st.columns([2, 1])

    # flight search processing
    selected_df = None
    flights_payload = None
    filter_iata = None

    # Derive IATA: if origin or dest looks like IATA code (3 letters), use that for filtering
    def iata_guess(s: str) -> Optional[str]:
        if not s: return None
        s2 = s.strip().upper()
        if len(s2) == 3 and s2.isalpha(): return s2
        return None

    filter_iata = None
    # --- Replace the existing flight_search_btn handling with this block ---
    if flight_search_btn:
        metrics.start_timer("flight_search_seconds")
        # set mock_mode True if you want guaranteed local data (helpful for debugging)
        mock_mode = False  # keep this logic in production
        # mock_mode = not bool(FLIGHTAPI_KEY)
        # For debugging you can force mock_mode = True, e.g. mock_mode = True
        o_iata = iata_guess(origin) or origin
        d_iata = iata_guess(destination) or destination
        dep_iso = dep_date.isoformat() if dep_date else None

        # Call the API (one-way or roundtrip)
        if trip_type == "One-way":
            flights_payload = flightapi_search_oneway(o_iata, d_iata, date_iso=dep_iso, mock=mock_mode)
        else:
            ret_iso = ret_date.isoformat() if ret_date else None
            flights_payload = flightapi_search_roundtrip(o_iata, d_iata, dep_date_iso=dep_iso, ret_date_iso=ret_iso, mock=mock_mode)

        metrics.stop_timer("flight_search_seconds"); metrics.increment("flight_searches", 1)

        # DEBUG: show the raw payload keys / small sample so we can see what the API returned
        st.sidebar.markdown("**Flight API raw payload (DEBUG)**")
        try:
            st.sidebar.write({k: (str(v)[:1000] + "..." if isinstance(v, (dict, list)) and len(str(v))>1000 else v) for k,v in (flights_payload.items() if isinstance(flights_payload, dict) else {"payload": str(flights_payload)}.items())})
        except Exception:
            st.sidebar.write("Couldn't display raw payload.")

        # If payload missing 'data' or empty, warn and fallback to mock sample (so UI is usable)
        if not flights_payload or (isinstance(flights_payload, dict) and not flights_payload.get("data")):
            st.warning("Flight API returned no data. Showing mock sample flights for debugging (check FLIGHTAPI_KEY / network).")
            # force mock data
            flights_payload = flightapi_search_oneway(o_iata, d_iata, date_iso=dep_iso, mock=True)

        # convert to DataFrame defensively
        df = flights_to_dataframe(flights_payload, filter_iata=(iata_guess(origin) or iata_guess(destination)))
        selected_df = df
        st.session_state["history"].insert(0, {"ts": now_iso(), "type": "flight_search", "origin": origin, "destination": destination, "trip": trip_type, "rows": len(df)})
        st.session_state["history"] = st.session_state["history"][:200]

    # Map: show flights & nearby aircraft
    with map_col:
        st.subheader("Map (routes & flights)")
        markers = []; paths = []
        def geocode_label(label: str):
            xy = parse_latlon_input(label)
            if xy: return xy[0], xy[1], label
            lab = (label or "").strip().upper()
            if lab in IATA_TO_COORDS:
                lat, lon = IATA_TO_COORDS[lab]
                return lat, lon, lab
            return None

        # Try to get coords from selected_df if present, else from inputs
        ocoord = None; dcoord = None
        if selected_df is not None and not selected_df.empty:
            # prefer first row's Source/Destination
            first = selected_df.iloc[0]
            ocoord = geocode_label(first.get("Source")) or geocode_label(origin)
            dcoord = geocode_label(first.get("Destination")) or geocode_label(destination)
        else:
            ocoord = geocode_label(origin)
            dcoord = geocode_label(destination)

        if ocoord:
            markers.append({"name":"Origin", "lat": ocoord[0], "lon": ocoord[1], "color":[0,200,0]})
        if dcoord:
            markers.append({"name":"Destination", "lat": dcoord[0], "lon": dcoord[1], "color":[200,0,0]})
        if ocoord and dcoord:
            paths.append({"path":[[ocoord[1], ocoord[0]],[dcoord[1], dcoord[0]]], "name":"flight_line"})

        # Also add nearby aircraft around origin if available
        if ocoord:
            try:
                ac = get_nearby_aircraft(ocoord[0], ocoord[1], radius_km=200.0)
                for a in ac[:80]:
                    markers.append({"name": f"{a['callsign']} ({a['distance_km']} km)", "lat": a['lat'], "lon": a['lon'], "color":[255,215,0]})
            except Exception:
                pass

        # Render deck; ensure at least a default map shows
        if markers or paths:
            dfm = pd.DataFrame(markers) if markers else pd.DataFrame([], columns=["name","lat","lon","color"])
            layers = []
            if not dfm.empty:
                layers.append(pdk.Layer("ScatterplotLayer", dfm, get_position=["lon","lat"], get_fill_color="color", get_radius=6000, pickable=True))
            if paths:
                dfp = pd.DataFrame(paths)
                layers.append(pdk.Layer("PathLayer", dfp, get_path="path", get_width=4, get_color=[0,128,255], pickable=False))
            center_lat = dfm["lat"].mean() if not dfm.empty else 20.5937
            center_lon = dfm["lon"].mean() if not dfm.empty else 78.9629
            deck = pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6), tooltip={"text":"{name}"})
            st.pydeck_chart(deck)
        else:
            st.info("No map items yet. Search flights or compute a route.")


    # Info column: Flight & Chat
    with info_col:
        st.subheader("Flight Results")
        if selected_df is not None:
            if selected_df.empty:
                st.info("No flights returned.")
            else:
                # show table with columns asked
                requested_cols = ["Flight","Airline","Departure","Arrival","Source","Destination","Duration","Status"]
                show_cols = [c for c in requested_cols if c in selected_df.columns]
                st.dataframe(selected_df[show_cols].reset_index(drop=True))
                csv = selected_df[show_cols].to_csv(index=False)
                st.download_button("Download CSV", data=csv, file_name="flights.csv", mime="text/csv")
                # mark fastest flight by Duration_min if available
                try:
                    if "Duration_min" in selected_df.columns and selected_df["Duration_min"].notnull().any():
                        best = selected_df[selected_df["Duration_min"].notnull()].sort_values("Duration_min").iloc[0]
                        st.success(f"Suggested fastest flight: {best['Flight']} ({best['Duration']}) — Status: {best['Status']}")
                        metrics.increment("recommended_flights", 1)
                    else:
                        st.info("Could not determine fastest flight.")
                except Exception:
                    st.info("Could not determine fastest flight.")
        else:
            st.info("Search for flights to see results here.")

        st.markdown("---")
        st.subheader("Chat / Q&A (Ask for recommendations or itineraries)")
        user_msg = st.text_area("Your question (e.g., 'Which flight is fastest? Make a 2-day itinerary for BLR trip')", key="chat_input", height=120)
        if st.button("Ask LLM"):
            if not user_msg or user_msg.strip() == "":
                st.warning("Please write a question.")
            else:
                allowed, reason = guardrail_check(user_msg)
                if not allowed:
                    st.error(f"Guardrail blocked the request: {reason}")
                else:
                    # build context (simple)
                    context = {
                        "user": st.session_state.get("user_email"),
                        "last_search": {
                            "origin": origin, "destination": destination, "trip_type": trip_type
                        },
                        "note": "Do not provide illegal or violent advice."
                    }
                    prompt = f"You are a travel advisor. Context: {json.dumps(context)}\nUser question: {user_msg}\nIf user asks for itinerary, provide a day-wise plan. Keep answer concise."
                    reply = llm_chat_reply(prompt)
                    st.markdown("**LLM reply:**")
                    st.write(reply)
                    # compute metrics against a naive reference if user asked to summarize something — for demo we'll compare to last selected flight summary (if exists).
                    metrics_table = None
                    if selected_df is not None and not selected_df.empty:
                        # make a reference text: joined top5 flights descriptions
                        ref_text = "\n".join(selected_df.head(5).apply(lambda r: f"{r['Flight']} from {r['Source']} to {r['Destination']} dep {r['Departure']}", axis=1).tolist())
                        m = compute_llm_metrics(ref_text, reply)
                        metrics_table = pd.DataFrame([m])
                        st.markdown("**LLM metrics (approx)**")
                        st.table(metrics_table)
                    # store chat history
                    st.session_state["chat_history"].insert(0, {"ts": now_iso(), "q": user_msg, "a": reply})
                    st.session_state["chat_history"] = st.session_state["chat_history"][:200]

        if st.checkbox("Show recent chat history"):
            for ch in st.session_state["chat_history"][:20]:
                st.write(f"{ch['ts']}: Q: {ch['q']}")
                st.write(f"A: {ch['a']}")
                st.markdown("---")

        st.markdown("---")
        st.subheader("Notifications")
        if send_notify_btn:
            body = "Travel Disruption Advisor — Summary\n\n"
            if selected_df is not None and not selected_df.empty:
                body += "Top flights:\n" + selected_df.head(5).to_string() + "\n\n"
            ok, msg = send_email(notify_email, f"Advisor summary {origin}->{destination}", body)
            if ok:
                st.success("Email sent"); metrics.increment("emails_sent", 1)
            else:
                st.error("Email failed: " + msg)
            if notify_phone:
                ok2, m2 = send_sms_via_twilio(notify_phone, f"Advisor summary {origin}->{destination}")
                if ok2:
                    st.success("SMS sent"); metrics.increment("sms_sent", 1)
                else:
                    st.error("SMS failed: " + m2)

    # bottom: metrics, history
    st.markdown("---")
    st.subheader("App Metrics & History")
    st.json(metrics.get_metrics())
    if st.checkbox("Show history (last 20)"):
        st.subheader("History")
        for h in st.session_state["history"][:20]:
            st.write(h)

# ---------------- Main App Flow ----------------

def main():
    # If not authenticated show login page, else dashboard
    if not st.session_state.get("authenticated", False):
        show_login_page()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()
