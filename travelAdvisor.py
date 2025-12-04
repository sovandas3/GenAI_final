"""
travel_disruption_app.py

Single-file structure combining:
- LangGraph (multi-agent for Travel & Logistics Disruption Advisor)
- MCP server (FastMCP) with ONLY Tavily search
- Streamlit UI
- Audit log (JSONL) + simple in-session memory

Install (example):
    pip install langgraph langchain-core langchain-openai mcp streamlit requests

Run Streamlit UI:
    streamlit run travel_disruption_app.py

Run MCP server (stdio example):
    python travel_disruption_app.py mcp
"""

import os
import sys
import json
from datetime import datetime
from typing import TypedDict, List, Literal, Dict, Any, Optional

import requests
from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_openai import ChatOpenAI

from mcp.server.fastmcp import FastMCP

import streamlit as st


# ============================================================
# CONFIG / ENV
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-_hlWKUmXssdcqD4O7AI3hQ")

# External APIs: set these as env vars in real use
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "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")

if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-_hlWKUmXssdcqD4O7AI3hQ"):
    print("[WARN] OPENAI_API_KEY is not set or is placeholder. LLM calls will fail.")

llm = ChatOpenAI(
    model="azure/genailab-maas-gpt-4o",
    temperature=0.2,
)

AUDIT_LOG_PATH = "audit_log.jsonl"


# ============================================================
# AUDIT LOG UTIL
# ============================================================

def append_audit_log(event_type: str, payload: dict) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "payload": payload,
    }
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] Failed to write audit log: {e}")


# ============================================================
# DUMMY DATA FETCHERS (REPLACE WITH REAL APIs IF YOU WANT)
# ============================================================

def fetch_flight_data(src: str, dst: str, date_str: str) -> Dict[str, Any]:
    """
    Placeholder: mock flight API response.
    Replace with real Skyscanner / Amadeus / internal API integration.
    """
    return {
        "route": {"src": src, "dst": dst, "date": date_str},
        "count": 3,
        "flights": [
            {
                "airline": "IndiGo",
                "flight_no": "6E123",
                "departure": f"{date_str} 06:30",
                "arrival": f"{date_str} 08:45",
                "duration": "2h 15m",
            },
            {
                "airline": "Air India",
                "flight_no": "AI456",
                "departure": f"{date_str} 11:15",
                "arrival": f"{date_str} 13:40",
                "duration": "2h 25m",
            },
            {
                "airline": "Vistara",
                "flight_no": "UK789",
                "departure": f"{date_str} 18:20",
                "arrival": f"{date_str} 20:30",
                "duration": "2h 10m",
            },
        ],
    }


def fetch_weather_and_disruption(src: str, dst: str, date_str: str) -> Dict[str, str]:
    """
    Placeholder: mock weather & disruption texts.
    You can later swap this to call Tavily via MCP or directly.
    """
    weather_block = (
        f"METAR/TAF-like data for {src} and {dst} on {date_str}: "
        "VMC conditions, scattered clouds, some afternoon convective activity near destination."
    )
    disruption_block = (
        "Recent news mentions minor ATC delays and moderate congestion during evening peak hours."
    )
    return {
        "weather_block": weather_block,
        "disruption_block": disruption_block,
    }


def fetch_live_airspace() -> Dict[str, Any]:
    """
    Placeholder: mock live airspace info.
    """
    return {
        "live_count": 185,
    }


# ============================================================
# LANGGRAPH STATE
# ============================================================

class GraphState(TypedDict, total=False):
    # Core
    messages: List[BaseMessage]
    user_query: str
    src: str
    dst: str
    date_str: str

    # Planner output
    planner_raw_json: str
    call_flight_data: bool
    call_weather: bool
    call_disruption_news: bool
    call_live_airspace: bool

    # Data fetch results
    flight_json: Dict[str, Any]
    weather_block: str
    disruption_block: str
    live_airspace_count: int

    # Agent outputs
    flights_summary_md: str
    weather_disruption_md: str
    map_inclusion: str  # "YES" or "NO"
    final_report_md: str


# ============================================================
# HELPERS
# ============================================================

def build_conversation_context(chat_history: List[Dict[str, str]]) -> str:
    """
    Simple in-session memory summary: concatenate user and assistant messages.
    """
    lines = []
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines[-10:])  # last 10 messages max


# ============================================================
# AGENT 1: PLANNER
# ============================================================

def planner_decide(state: GraphState) -> GraphState:
    system = SystemMessage(
        content="""
You are the Planner for a Travel & Logistics Real-Time Disruption Advisor.

Given a user query and basic route info, decide which data sources to use.

Respond with STRICT JSON ONLY (no extra text), using this schema:
{
  "call_flight_data": true/false,
  "call_weather": true/false,
  "call_disruption_news": true/false,
  "call_live_airspace": true/false
}
""".strip()
    )

    conversation_context = ""
    if "messages" in state:
        parts = []
        for m in state["messages"]:
            role = type(m).__name__.replace("Message", "").lower()
            parts.append(f"{role}: {m.content}")
        conversation_context = "\n".join(parts[-10:])

    planner_input = HumanMessage(
        content=(
            f"Route:\n"
            f"- From: {state.get('src')}\n"
            f"- To: {state.get('dst')}\n"
            f"- Date: {state.get('date_str')}\n\n"
            f"User query:\n{state.get('user_query')}\n\n"
            f"Recent conversation context:\n{conversation_context}"
        )
    )

    response = llm.invoke([system, planner_input])

    raw = response.content.strip()
    call_flight = False
    call_weather = False
    call_disruption = False
    call_live = False

    try:
        parsed = json.loads(raw)
        call_flight = bool(parsed.get("call_flight_data", False))
        call_weather = bool(parsed.get("call_weather", False))
        call_disruption = bool(parsed.get("call_disruption_news", False))
        call_live = bool(parsed.get("call_live_airspace", False))
    except Exception as e:
        append_audit_log(
            "planner_parse_error",
            {"raw_response": raw, "error": str(e)},
        )

    new_state: GraphState = dict(state)
    new_state["planner_raw_json"] = raw
    new_state["call_flight_data"] = call_flight
    new_state["call_weather"] = call_weather
    new_state["call_disruption_news"] = call_disruption
    new_state["call_live_airspace"] = call_live

    append_audit_log(
        "planner_decide",
        {
            "route": {
                "src": state.get("src"),
                "dst": state.get("dst"),
                "date": state.get("date_str"),
            },
            "user_query": state.get("user_query"),
            "decision": {
                "call_flight_data": call_flight,
                "call_weather": call_weather,
                "call_disruption_news": call_disruption,
                "call_live_airspace": call_live,
            },
        },
    )

    return new_state


# ============================================================
# DATA FETCH NODE (NON-LLM)
# ============================================================

def fetch_data_node(state: GraphState) -> GraphState:
    new_state: GraphState = dict(state)

    src = state.get("src", "")
    dst = state.get("dst", "")
    date_str = state.get("date_str", "")

    if state.get("call_flight_data"):
        flight_json = fetch_flight_data(src, dst, date_str)
        new_state["flight_json"] = flight_json
    else:
        new_state["flight_json"] = {"count": 0, "flights": []}

    if state.get("call_weather") or state.get("call_disruption_news"):
        wd = fetch_weather_and_disruption(src, dst, date_str)
        new_state["weather_block"] = wd.get("weather_block", "")
        new_state["disruption_block"] = wd.get("disruption_block", "")
    else:
        new_state["weather_block"] = ""
        new_state["disruption_block"] = ""

    if state.get("call_live_airspace"):
        air = fetch_live_airspace()
        new_state["live_airspace_count"] = int(air.get("live_count", 0))
    else:
        new_state["live_airspace_count"] = 0

    append_audit_log(
        "data_fetched",
        {
            "route": {"src": src, "dst": dst, "date": date_str},
            "flags": {
                "flight": state.get("call_flight_data"),
                "weather": state.get("call_weather"),
                "disruption": state.get("call_disruption_news"),
                "live_airspace": state.get("call_live_airspace"),
            },
            "flight_count": new_state["flight_json"].get("count", 0),
            "live_airspace_count": new_state["live_airspace_count"],
        },
    )

    return new_state


# ============================================================
# AGENT 2: SUMMARIZE FLIGHTS
# ============================================================

def summarize_flights_agent(state: GraphState) -> GraphState:
    system = SystemMessage(
        content="""
You summarize raw flight schedule JSON.

- Mention total number of flights.
- List key airlines.
- Indicate rough spread across the day (morning / afternoon / evening), if possible.
- If count=0, explain that schedule data is unavailable.

Return a short markdown section titled **Flights Summary**.
""".strip()
    )

    flight_json = state.get("flight_json", {"count": 0, "flights": []})
    flights_text = json.dumps(flight_json, ensure_ascii=False)

    user = HumanMessage(
        content=f"Here is the flight JSON for the route. Analyze it:\n{flights_text}"
    )

    response = llm.invoke([system, user])

    new_state: GraphState = dict(state)
    new_state["flights_summary_md"] = response.content

    append_audit_log(
        "summarize_flights",
        {
            "flight_count": flight_json.get("count", 0),
            "summary_preview": response.content[:200],
        },
    )

    return new_state


# ============================================================
# AGENT 3: WEATHER & DISRUPTION ANALYSIS
# ============================================================

def analyze_weather_and_disruption_agent(state: GraphState) -> GraphState:
    system = SystemMessage(
        content="""
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
""".strip()
    )

    src = state.get("src", "")
    dst = state.get("dst", "")
    weather_block = state.get("weather_block", "")
    disruption_block = state.get("disruption_block", "")

    user = HumanMessage(
        content=(
            f"Route: {src} -> {dst}\n\n"
            f"Weather / METAR / TAF-like data:\n{weather_block}\n\n"
            f"Disruption / news snippets:\n{disruption_block}"
        )
    )

    response = llm.invoke([system, user])

    new_state: GraphState = dict(state)
    new_state["weather_disruption_md"] = response.content

    append_audit_log(
        "analyze_weather_and_disruption",
        {
            "src": src,
            "dst": dst,
            "weather_preview": weather_block[:200],
            "disruption_preview": disruption_block[:200],
            "analysis_preview": response.content[:200],
        },
    )

    return new_state


# ============================================================
# AGENT 4: DECIDE MAP INCLUSION
# ============================================================

def decide_map_inclusion_agent(state: GraphState) -> GraphState:
    system = SystemMessage(
        content="""
You decide whether a live airspace map should be shown.

Reply ONLY with a single token: YES or NO.

Show map (YES) if:
- user mentions "map", "live tracking", "where is my flight", "airspace", or similar
- OR if they clearly care about visualizing traffic

Otherwise say NO.
""".strip()
    )

    user_query = state.get("user_query", "")

    user = HumanMessage(
        content=f"User query:\n{user_query}\n\nReply with YES or NO only."
    )

    response = llm.invoke([system, user])
    decision = response.content.strip().upper()

    if decision not in ("YES", "NO"):
        decision = "NO"

    new_state: GraphState = dict(state)
    new_state["map_inclusion"] = decision

    append_audit_log(
        "decide_map_inclusion",
        {
            "user_query": user_query,
            "decision": decision,
        },
    )

    return new_state


# ============================================================
# AGENT 5: BUILD FINAL REPORT
# ============================================================

def build_final_report_agent(state: GraphState) -> GraphState:
    system = SystemMessage(
        content="""
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
""".strip()
    )

    src = state.get("src", "")
    dst = state.get("dst", "")
    date_str = state.get("date_str", "")
    user_query = state.get("user_query", "")

    flight_json = state.get("flight_json", {"count": 0, "flights": []})
    flights_summary_md = state.get("flights_summary_md", "")
    weather_disruption_md = state.get("weather_disruption_md", "")
    live_count = state.get("live_airspace_count", 0)
    map_inclusion = state.get("map_inclusion", "NO")

    user_content = (
        f"Route: {src} -> {dst}\n"
        f"Date: {date_str}\n"
        f"User query: {user_query}\n\n"
        f"Flight JSON (summarize, do NOT dump):\n{json.dumps(flight_json, ensure_ascii=False)}\n\n"
        f"Flights Summary (markdown):\n{flights_summary_md}\n\n"
        f"Weather & Disruption Analysis (markdown):\n{weather_disruption_md}\n\n"
        f"live_count (tracked flights over India): {live_count}\n"
        f"map_included flag: {map_inclusion == 'YES'}\n\n"
        f"Now produce the final report strictly following the given template."
    )

    user = HumanMessage(content=user_content)

    response = llm.invoke([system, user])

    new_state: GraphState = dict(state)
    new_state["final_report_md"] = response.content

    append_audit_log(
        "build_final_report",
        {
            "src": src,
            "dst": dst,
            "date": date_str,
            "live_count": live_count,
            "map_inclusion": map_inclusion,
            "report_preview": response.content[:200],
        },
    )

    return new_state


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_multi_agent_graph():
    graph = StateGraph(GraphState)

    graph.add_node("planner_decide", planner_decide)
    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("summarize_flights", summarize_flights_agent)
    graph.add_node("analyze_weather_disruption", analyze_weather_and_disruption_agent)
    graph.add_node("decide_map", decide_map_inclusion_agent)
    graph.add_node("build_final_report", build_final_report_agent)

    graph.set_entry_point("planner_decide")

    graph.add_edge("planner_decide", "fetch_data")
    graph.add_edge("fetch_data", "summarize_flights")
    graph.add_edge("summarize_flights", "analyze_weather_disruption")
    graph.add_edge("analyze_weather_disruption", "decide_map")
    graph.add_edge("decide_map", "build_final_report")
    graph.add_edge("build_final_report", END)

    return graph.compile()


multi_agent_app = build_multi_agent_graph()


def run_disruption_advisor(
    user_query: str,
    src: str,
    dst: str,
    date_str: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    High-level helper:
    - Creates initial state
    - Runs the LangGraph multi-agent pipeline
    - Returns final markdown report
    """
    if chat_history is None:
        chat_history = []

    conv_context = build_conversation_context(chat_history)

    base_messages: List[BaseMessage] = []
    if conv_context:
        base_messages.append(
            HumanMessage(
                content=f"Conversation context so far:\n{conv_context}"
            )
        )
    base_messages.append(
        HumanMessage(
            content=f"Current user query for disruption advisor:\n{user_query}"
        )
    )

    initial_state: GraphState = {
        "messages": base_messages,
        "user_query": user_query,
        "src": src,
        "dst": dst,
        "date_str": date_str,
    }

    final_state = multi_agent_app.invoke(initial_state)

    report = final_state.get("final_report_md", "No report generated.")

    append_audit_log(
        "chat_turn",
        {
            "user_query": user_query,
            "src": src,
            "dst": dst,
            "date": date_str,
            "report_preview": report[:200],
        },
    )

    return report


# ============================================================
# MCP SERVER (FastMCP) – ONLY TAVILY
# ============================================================

mcp = FastMCP("Travel-Disruption-MCP")


@mcp.tool()
def tavily_search_disruptions(query: str) -> Dict[str, Any]:
    """
    Use Tavily to search for disruption-related content:
    strikes, ATC issues, delays, etc.
    """
    if not TAVILY_API_KEY:
        return {"error": "TAVILY_API_KEY not configured"}

    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "topic": "news",
        "max_results": 5,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        append_audit_log(
            "tavily_search_disruptions",
            {
                "query": query,
                "result_count": len(data.get("results", [])),
            },
        )
        return data
    except Exception as e:
        append_audit_log(
            "tavily_search_error",
            {"error": str(e), "query": query},
        )
        return {"error": str(e)}


def run_mcp_server():
    append_audit_log("server_event", {"server": "mcp", "status": "start"})
    mcp.run(transport="stdio")


# ============================================================
# STREAMLIT UI
# ============================================================

def run_streamlit_ui():
    st.set_page_config(page_title="Travel Disruption Advisor", page_icon="✈️")

    st.title("✈️ Multi-Agent Travel & Logistics Disruption Advisor")

    st.markdown(
        """
        This demo uses:
        - **LangGraph** multi-agent pipeline (planner → data fetch → flights → weather → map → final report)  
        - **Dummy data sources** (you can wire your real APIs)  
        - **MCP server** in this same file with ONLY:
            - `tavily_search_disruptions(query)`  
        - **Audit log**: `audit_log.jsonl`  
        """
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # [{"role": "user"/"assistant", "content": str}]

    with st.sidebar:
        st.header("Route Inputs")
        src = st.text_input("From (IATA or city)", value="DEL")
        dst = st.text_input("To (IATA or city)", value="BOM")
        date_str = st.text_input("Date (YYYY-MM-DD)", value="2025-12-10")
        st.markdown("---")
        st.markdown(
            """
            MCP server (optional, Tavily only):

            ```bash
            python travel_disruption_app.py mcp
            ```
            """
        )

    # Show existing conversation
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask about your route, disruptions, risk, etc...")

    if user_query:
        user_msg = f"**Route:** {src} → {dst} on {date_str}\n\n**Query:** {user_query}"
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing flights, weather, disruptions..."):
                try:
                    report = run_disruption_advisor(
                        user_query=user_query,
                        src=src,
                        dst=dst,
                        date_str=date_str,
                        chat_history=st.session_state.chat_history,
                    )
                except Exception as e:
                    report = f"Error in disruption advisor pipeline: {e}"
                    append_audit_log(
                        "error",
                        {"where": "run_streamlit_ui", "error": str(e)},
                    )

            st.markdown(report)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": report}
            )

    with st.expander("Session metadata", expanded=False):
        st.write(
            {
                "turns": len(
                    [m for m in st.session_state.chat_history if m["role"] == "user"]
                ),
                "total_messages": len(st.session_state.chat_history),
            }
        )
        st.caption("For detailed events, see `audit_log.jsonl`.")


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def main():
    """
    - `python travel_disruption_app.py mcp`  -> MCP server (stdio, Tavily only)
    - `streamlit run travel_disruption_app.py` -> Streamlit UI
    - `python travel_disruption_app.py` -> local UI run (for debug)
    """
    if len(sys.argv) > 1 and sys.argv[1].lower() == "mcp":
        print("[INFO] Starting MCP server over stdio...")
        run_mcp_server()
    else:
        run_streamlit_ui()


if __name__ == "__main__":
    main()
