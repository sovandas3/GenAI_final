# mcp_server.py

from __future__ import annotations

import os
import requests
import urllib3
from typing import Dict, Any, List

from mcp.server.fastmcp import FastMCP

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================
# MCP INSTANCE
# ============================================================

mcp = FastMCP("Travel-Disruption-MCP")


# ============================================================
# TOOL BLOCK 1 - Tavily Disruption Search
# ============================================================


@mcp.tool()
def tavily_search_disruptions(query: str) -> Dict[str, Any]:
    """
    Search disruption-related aviation news via Tavily.
    """
    # TEMP: hardcode key for testing
    TAVILY_API_KEY = "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"

    print("DEBUG TAVILY KEY PREFIX:", TAVILY_API_KEY[:10])  # just to verify it's used

    if not TAVILY_API_KEY:
        return {"error": "Missing Tavily API Key"}

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
            },
            timeout=10,
            verify=False,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "query": query,
            "results": data.get("results", []),
        }
    except Exception as e:
        return {"error": str(e)}




# ============================================================
# TOOL BLOCK 2 - Amadeus Flights
# ============================================================

def _amadeus_token() -> str | None:
    AMADEUS_KEY = os.environ.get("AMADEUS_KEY", "mqg4GsHmKpkBOUdSAUWJMNFHT9LKICgI")
    AMADEUS_SECRET = os.environ.get("AMADEUS_SECRET", "rE9OrhGSAxaWEvlu")
    AMADEUS_BASE = os.environ.get("AMADEUS_BASE", "https://test.api.amadeus.com")

    try:
        resp = requests.post(
            f"{AMADEUS_BASE}/v1/security/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": AMADEUS_KEY,
                "client_secret": AMADEUS_SECRET,
            },
            timeout=10,
            verify=False,
        )
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception:
        return None


@mcp.tool()
def fetch_flights_amadeus(src: str, dst: str, date_str: str) -> Dict[str, Any]:
    """
    Retrieve flight offers between airports.
    """
    AMADEUS_BASE = os.environ.get("AMADEUS_BASE", "https://test.api.amadeus.com")

    token = _amadeus_token()
    if not token:
        return {"error": "Amadeus Authentication Failed"}

    try:
        resp = requests.get(
            f"{AMADEUS_BASE}/v2/shopping/flight-offers",
            headers={"Authorization": f"Bearer {token}"},
            params={"originLocationCode": src, "destinationLocationCode": dst, "departureDate": date_str, "adults": 1},
            timeout=12, verify=False,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": str(e)}

    if "data" not in data:
        return {"count": 0, "flights": []}

    flights = []
    for f in data["data"][:10]:
        seg = f["itineraries"][0]["segments"][0]
        flights.append({
            "airline": seg["carrierCode"],
            "flight": f"{seg['carrierCode']}{seg['number']}",
            "dep": seg["departure"]["at"],
            "arr": seg["arrival"]["at"]
        })

    return {"count": len(flights), "flights": flights}


# ============================================================
# TOOL BLOCK 3 - METAR Weather
# ============================================================

@mcp.tool()
def get_metar_mcp(icao: str) -> Dict[str, Any]:
    """
    Fetch METAR weather report for an airport.
    """
    try:
        resp = requests.get(
            f"https://aviationweather.gov/api/data/metar?ids={icao}&format=json",
            timeout=8, verify=False
        )
        resp.raise_for_status()
        js = resp.json()
        return {"icao": icao, "raw": js[0].get("rawOb", "No METAR")} if js else {"icao": icao, "raw": "No METAR"}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# TOOL BLOCK 4 - TAF Weather Forecast
# ============================================================

@mcp.tool()
def get_taf_mcp(icao: str) -> Dict[str, Any]:
    """
    Fetch TAF weather forecast for an airport.
    """
    try:
        resp = requests.get(
            f"https://aviationweather.gov/api/data/taf?ids={icao}&format=json",
            timeout=8, verify=False
        )
        resp.raise_for_status()
        js = resp.json()
        return {"icao": icao, "raw": js[0].get("rawTaf", "No TAF")} if js else {"icao": icao, "raw": "No TAF"}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# TOOL BLOCK 5 - OpenSky Live Air Traffic
# ============================================================

@mcp.tool()
def opensky_live_mcp() -> Dict[str, Any]:
    """
    Get live flights over India.
    """
    INDIA_ZONE = {"lat_min": 0, "lat_max": 45, "lon_min": 60, "lon_max": 105}

    try:
        resp = requests.get("https://opensky-network.org/api/states/all", timeout=10, verify=False)
        resp.raise_for_status()
        js = resp.json()
    except Exception as e:
        return {"error": str(e)}

    flights = [
        {"callsign": s[1], "lat": s[6], "lon": s[5]}
        for s in js.get("states", [])
        if s[6] and s[5] and INDIA_ZONE["lat_min"] < s[6] < INDIA_ZONE["lat_max"] and INDIA_ZONE["lon_min"] < s[5] < INDIA_ZONE["lon_max"]
    ]

    return {"count": len(flights), "flights": flights}


# ============================================================
# MCP RUN
# ============================================================

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
