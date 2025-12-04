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

LLM_MODEL = os.environ.get("AZURE_LLM_MODEL", "azure/genailab-maas-gpt-4o")
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

def extract_text_from_chat_result(result) -> str:
    if hasattr(result, "summary") and result.summary:
        return result.summary
    if hasattr(result, "chat_history"):
        hist = result.chat_history
        if isinstance(hist, list) and hist:
            last = hist[-1]
            if isinstance(last, dict):
                return last.get("content", str(last))
            if hasattr(last, "content"):
                return last.content
    if hasattr(result, "messages"):
        msgs = result.messages
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if isinstance(last, dict):
                return last.get("content", str(last))
            if hasattr(last, "content"):
                return last.content
    return str(result)

# ============================================================
# Output Formatting
# ============================================================

def parse_metrics(response):
    # Basic regular expressions to extract metrics; improve as needed!
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
    metrics["Decision"] = "Rejected" if "reject" in response.lower() else "Approved" if "approve" in response.lower() else "Requires Review"
    return metrics

def show_loan_output(response):
    # Extract metrics and decision
    metrics = parse_metrics(response)
    # Decision header
    if metrics["Decision"] == "Approved":
        st.markdown("<h2 style='color:#27ae60'>✅ Loan Approved</h2>", unsafe_allow_html=True)
    elif metrics["Decision"] == "Rejected":
        st.markdown("<h2 style='color:#c0392b'>❌ Loan Rejected</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2>⚠️ Requires Review</h2>", unsafe_allow_html=True)

    st.markdown("<h3>Applicant Summary</h3>", unsafe_allow_html=True)
    st.write(response)

    st.markdown("<h3>Key Risk Metrics</h3>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <table>
            <tr>
                <th>Metric</th><th>Value</th><th>Risk Level</th>
            </tr>
            <tr>
                <td>DTI Ratio</td><td>{metrics.get("DTI Ratio", "Not found")}</td>
                <td>{'High 🚨' if metrics.get('DTI Ratio') and int(metrics['DTI Ratio'].replace('%','')) > 40 else 'Acceptable'}</td>
            </tr>
            <tr>
                <td>FOIR</td><td>{metrics.get("FOIR", "Not found")}</td>
                <td>{'Very High 🚨' if metrics.get('FOIR') and int(metrics['FOIR'].replace('%','')) > 50 else 'Acceptable'}</td>
            </tr>
            <tr>
                <td>Disposable Income</td><td>{metrics.get("Disposable Income", "OK")}</td>
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
# LoanApprovalWorkflow
# ============================================================

class LoanApprovalWorkflow:
    def __init__(self):
        self.manager, self.intake_agent = self._build()

    def _build(self):
        docs = load_policy_docs()
        vectordb = build_vectordb(docs)

        def rag_search(query: str) -> str:
            if vectordb is None:
                return "No policy documents available."
            results = vectordb.similarity_search(query, k=3)
            return "\n\n".join([d.page_content for d in results])

        rag_agent = ConversableAgent(
            name="rag_agent",
            system_message="Answer loan policy queries using retrieved context.",
            llm_config=make_llm_conf(),
            functions=[rag_search],
            function_map={"rag_search": rag_search},
        )

        intake = ConversableAgent(
            name="intake_agent",
            system_message="Extract all applicant details from the input.",
            llm_config=make_llm_conf(),
        )

        docs_agent = ConversableAgent(
            name="docs_agent",
            system_message="Interpret bank statements, IDs, and income documents.",
            llm_config=make_llm_conf(),
        )

        credit = ConversableAgent(
            name="credit_agent",
            system_message="Calculate DTI, FOIR, risk score, and repayment ability.",
            llm_config=make_llm_conf(),
        )

        decision = ConversableAgent(
            name="decision_agent",
            system_message="Make the final decision: APPROVE, REJECT, or MORE INFO.",
            llm_config=make_llm_conf(),
        )

        group = GroupChat(
            agents=[intake, docs_agent, credit, decision, rag_agent],
            messages=[],
            max_round=8,
        )

        manager = GroupChatManager(
            groupchat=group,
            llm_config=make_llm_conf(),
        )

        return manager, intake

    def invoke(self, text: str) -> str:
        try:
            chat_result = self.intake_agent.initiate_chat(
                self.manager,
                message=text,
                summary_method="reflection_with_llm",
            )
            return extract_text_from_chat_result(chat_result)
        except Exception:
            return "Workflow error:\n" + traceback.format_exc()

# ============================================================
# Streamlit UI
# ============================================================

def main():
    st.set_page_config(page_title="Loan Approval AI Workflow", page_icon="💰")

    st.title("Loan Approval AI Workflow")
    st.markdown(
        "Enter applicant details below and get the loan approval analysis powered by AI."
    )
    with st.form(key="applicant_form"):
        name = st.text_input("Applicant Name", "Rahul Sharma")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0, value=85000)
        existing_emi = st.number_input("Existing EMI (₹)", min_value=0, value=8500)
        requested_loan = st.number_input("Requested Loan Amount (₹)", min_value=0, value=500000)
        tenure_months = st.number_input("Loan Tenure (months)", min_value=1, max_value=360, value=36)
        purpose = st.text_input("Purpose of Loan", "Medical Emergency")

        submit_button = st.form_submit_button("Submit")

    if submit_button:
        input_text = f"""
Applicant Name: {name}
Age: {age}
Monthly Salary: {monthly_salary}
Existing EMI: {existing_emi}
Requested Loan: {requested_loan}
Tenure: {tenure_months} months
Purpose: {purpose}
"""
        st.markdown("### Processing your request...")
        wf = LoanApprovalWorkflow()
        response = wf.invoke(input_text)
        st.markdown("## Loan Approval AI Response")
        show_loan_output(response)

if __name__ == "__main__":
    main()
