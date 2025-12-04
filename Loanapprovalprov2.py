# LoanApprovalWorkflow_fixed_funcs.py
import os
import httpx
import traceback
import re
import streamlit as st

from autogen import ConversableAgent, GroupChat, GroupChatManager
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ============================================================
# CONFIG
# ============================================================
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "sk-wwXoGekBcGsk52Y3lWZv1g")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "https://genailab.tcs.in")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-06-01")

LLM_MODEL = os.environ.get("AZURE_LLM_MODEL", "azure_ai/DeepSeek-V3-0324")
EMBED_MODEL = os.environ.get("AZURE_EMBED_MODEL", "azure/genailab-maas-text-embedding-3-large")

POLICY_DOCS_FOLDER = "data/policy_docs"

# ============================================================
# Deepcopy-Safe Client
# ============================================================
class DeepcopySafeClient(httpx.Client):
    def __deepcopy__(self, memo):
        return DeepcopySafeClient(verify=False, timeout=self.timeout)

http_client = DeepcopySafeClient(verify=False, timeout=60.0)

# ============================================================
# Helper functions
# ============================================================
def make_llm_conf():
    return {
        "model": LLM_MODEL,
        "api_key": AZURE_API_KEY,
        "api_type": "azure",
        "api_base": AZURE_ENDPOINT,
        "base_url": AZURE_ENDPOINT,
        "azure_endpoint": AZURE_ENDPOINT,
        "api_version": AZURE_API_VERSION,
        "http_client": http_client,
    }

def load_policy_docs():
    docs = []
    if not os.path.exists(POLICY_DOCS_FOLDER):
        return docs
    for fname in os.listdir(POLICY_DOCS_FOLDER):
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(POLICY_DOCS_FOLDER, fname))
            docs.extend(loader.load())
    return docs

def build_vectordb(docs):
    if not docs:
        return None

    embedder = AzureOpenAIEmbeddings(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        model=EMBED_MODEL,
        http_client=http_client,
    )
    return Chroma.from_documents(docs, embedding=embedder)

# ============================================================
# RunResponse extractor
# ============================================================
def extract_text_from_runresponse(result) -> str:
    # Try common fields in order of priority
    try:
        if hasattr(result, "summary") and result.summary:
            return result.summary
    except:
        pass

    for field in ("output_messages", "chat_history", "messages", "_messages", "_summary"):
        try:
            if hasattr(result, field):
                val = getattr(result, field)
                if isinstance(val, (list, tuple)) and val:
                    last = val[-1]
                    if isinstance(last, dict):
                        return last.get("content", str(last))
                    # try common attributes on message object
                    for attr in ("content", "text", "message"):
                        if hasattr(last, attr):
                            v = getattr(last, attr)
                            if v:
                                return v
                if isinstance(val, str) and val:
                    return val
        except Exception:
            pass

    # fallback
    try:
        return str(result)
    except Exception:
        return "<unreadable RunResponse>"

# ============================================================
# Output parsing & UI helpers
# ============================================================
def parse_metrics(response):
    metrics = {
        "DTI Ratio": None,
        "FOIR": None,
        "Disposable Income": None,
        "Decision": None
    }
    dti_match = re.search(r"DTI.*?(\d+)%", response)
    foir_match = re.search(r"FOIR.*?(\d+)%", response)
    di_match = re.search(r"(inadequate|insufficient) disposable income", response, re.I)
    if dti_match:
        metrics["DTI Ratio"] = f"{dti_match.group(1)}%"
    if foir_match:
        metrics["FOIR"] = f"{foir_match.group(1)}%"
    if di_match:
        metrics["Disposable Income"] = "Insufficient"
    resp = response.lower()
    if "reject" in resp:
        metrics["Decision"] = "Rejected"
    elif "approve" in resp:
        metrics["Decision"] = "Approved"
    else:
        metrics["Decision"] = "Requires Review"
    return metrics

def show_loan_output(response):
    metrics = parse_metrics(response)
    if metrics["Decision"] == "Approved":
        st.markdown("<h2 style='color:#27ae60'>✅ Loan Approved</h2>", unsafe_allow_html=True)
    elif metrics["Decision"] == "Rejected":
        st.markdown("<h2 style='color:#c0392b'>❌ Loan Rejected</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2>⚠️ Requires Human Review</h2>", unsafe_allow_html=True)

    st.markdown("<h3>AI Summary</h3>", unsafe_allow_html=True)
    st.write(response)
    st.markdown("<h3>Risk Metrics</h3>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <table>
            <tr><th>Metric</th><th>Value</th><th>Risk</th></tr>
            <tr>
                <td>DTI Ratio</td>
                <td>{metrics.get("DTI Ratio", "Not found")}</td>
                <td>{'High 🚨' if metrics.get('DTI Ratio') and int(metrics['DTI Ratio'].replace('%','')) > 40 else 'Low'}</td>
            </tr>
            <tr>
                <td>FOIR</td>
                <td>{metrics.get("FOIR", "Not found")}</td>
                <td>{'Critical 🚨' if metrics.get('FOIR') and int(metrics['FOIR'].replace('%','')) > 50 else 'Stable'}</td>
            </tr>
            <tr>
                <td>Disposable Income</td>
                <td>{metrics.get("Disposable Income", "OK")}</td>
                <td>{'Insufficient 🚫' if metrics.get('Disposable Income') == 'Insufficient' else 'OK'}</td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h3>Recommendations</h3>", unsafe_allow_html=True)
    st.markdown(
        "- Try a smaller loan amount\n"
        "- Consider extending the tenure\n"
        "- Seek alternative financial support\n"
        "- Improve income or reduce liabilities"
    )

# ============================================================
# LoanApprovalWorkflow (fixed function registration)
# ============================================================
class LoanApprovalWorkflow:
    def __init__(self, max_rounds: int = 5):
        self.max_rounds = max_rounds
        self.manager = self._build_manager()

    def _build_manager(self):
        docs = load_policy_docs()
        vectordb = build_vectordb(docs)

        # RAG tool (must be annotated)
        def rag_search(query: str) -> str:
            if vectordb is None:
                return "No policy documents available."
            results = vectordb.similarity_search(query, k=3)
            return "\n\n".join([d.page_content for d in results])

        # Create rag_agent WITHOUT functions kwarg
        rag_agent = ConversableAgent(
            name="rag_agent",
            system_message="Use retrieved policy context to answer RAG queries.",
            llm_config=make_llm_conf(),
        )
        # Register the function/tool with the agent via decorator returned by register_for_llm
        # This version handles autogen implementations that require registration rather than 'functions='
        rag_agent.register_for_llm(name="rag_search", description="Search policy docs for query")(rag_search)

        # Other agents (normal)
        intake_agent = ConversableAgent(
            name="intake_agent",
            system_message="Extract applicant details and validate completeness.",
            llm_config=make_llm_conf(),
        )

        docs_agent = ConversableAgent(
            name="docs_agent",
            system_message="Analyze documents like salary slips and bank statements.",
            llm_config=make_llm_conf(),
        )

        credit_agent = ConversableAgent(
            name="credit_agent",
            system_message="Compute DTI, FOIR, EMI capacity and return a numeric risk score + reasons.",
            llm_config=make_llm_conf(),
        )

        decision_agent = ConversableAgent(
            name="decision_agent",
            system_message="Make final loan approval decision: APPROVE / REJECT / MORE INFO with reasons.",
            llm_config=make_llm_conf(),
        )

        group = GroupChat(
            agents=[intake_agent, docs_agent, credit_agent, decision_agent, rag_agent],
            messages=[],
            max_round=self.max_rounds,
        )

        manager = GroupChatManager(groupchat=group, llm_config=make_llm_conf())
        return manager

    def invoke(self, text: str) -> str:
        try:
            resp = self.manager.run(task=text, sender="user")
            return extract_text_from_runresponse(resp) if 'extract_text_from_runresponse' in globals() else extract_text_from_runresponse(resp)
        except Exception:
            return "Workflow error:\n" + traceback.format_exc()

# Provide compatibility if earlier extractor name used
def extract_text_from_runresponse(rr) -> str:
    # fallback extractor for compatibility (wraps existing extractor)
    return extract_text_from_runresponse_alias(rr)

def extract_text_from_runresponse_alias(rr) -> str:
    # reuse the primary extractor above
    return extract_text_from_runresponse_main(rr)

def extract_text_from_runresponse_main(rr) -> str:
    # Primary, robust extractor logic (same as extract_text_from_runresponse at top)
    try:
        if hasattr(rr, "summary") and rr.summary:
            return rr.summary
    except:
        pass
    for field in ("output_messages", "chat_history", "messages", "_messages", "_summary"):
        try:
            if hasattr(rr, field):
                val = getattr(rr, field)
                if isinstance(val, (list, tuple)) and val:
                    last = val[-1]
                    if isinstance(last, dict):
                        return last.get("content", str(last))
                    for attr in ("content", "text", "message"):
                        if hasattr(last, attr):
                            v = getattr(last, attr)
                            if v:
                                return v
                if isinstance(val, str) and val:
                    return val
        except Exception:
            pass
    try:
        return str(rr)
    except Exception:
        return "<unreadable RunResponse>"

# Keep a simple alias mapping so invoke calls the robust extractor
def extract_text_from_runresponse(rr):
    return extract_text_from_runresponse_main(rr)

# ============================================================
# Streamlit UI
# ============================================================
def main():
    st.set_page_config(page_title="Loan Approval AI Workflow", page_icon="💰", layout="wide")
    st.title("Loan Approval Agentic AI System")

    with st.form("loan_input"):
        name = st.text_input("Applicant Name", "Rahul Sharma")
        age = st.number_input("Age", 18, 100, 30)
        salary = st.number_input("Monthly Salary (₹)", 0, 10_000_000, 85000)
        emi = st.number_input("Existing EMI (₹)", 0, 10_000_000, 8500)
        amount = st.number_input("Requested Loan Amount (₹)", 0, 10_000_000, 500000)
        tenure = st.number_input("Loan Tenure (Months)", 1, 360, 36)
        purpose = st.text_input("Loan Purpose", "Medical Emergency")
        submitted = st.form_submit_button("Run Loan Analysis")

    if submitted:
        user_text = f"""
Applicant Name: {name}
Age: {age}
Monthly Salary: {salary}
Existing EMI: {emi}
Requested Loan: {amount}
Tenure: {tenure} months
Purpose: {purpose}
"""
        st.info("Running multi-agent workflow...")
        wf = LoanApprovalWorkflow()
        result = wf.invoke(user_text)
        show_loan_output(result)

if __name__ == "__main__":
    main()
