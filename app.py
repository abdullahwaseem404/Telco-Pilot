import os
import random
import streamlit as st
from typing import List
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="TelcoPilot Operations Assistant",
    page_icon="📡",
    layout="wide"
)

with st.sidebar:
    st.title("📡 TelcoPilot Control")
    st.caption("Multi-Domain Agentic Assistant for Telecom Operations")
    
    st.subheader("Model Configuration")
    selected_model = st.selectbox(
        "Choose Local LLM",
        ["qwen2.5:1.5b", "nemotron-3-nano:4b", "llama3.2:1b", "llama3.1"],
        index=0
    )


@st.cache_resource(show_spinner="Initializing FAISS Vector Stores & Ollama Embeddings...")
def setup_agent_environment(model_name: str):
    BILLING_DOCS = [
        Document(page_content="Refunds can only be issued if the network outage lasted more than 4 hours."),
        Document(page_content="Late fees are waived automatically if the account has been active for 5+ years."),
    ]
    NETWORK_DOCS = [
        Document(page_content="Line diagnostics error code 'ERR-77' indicates a physical fiber cut in the region."),
        Document(page_content="Node restarts take approximately 15 minutes and will drop all active sessions."),
    ]
    STREAMING_DOCS = [
        Document(page_content="Stream buffering on 4K tiers is usually caused by edge CDN cache misses."),
        Document(page_content="Live event streams require a minimum bitrate of 15Mbps to avoid quality degradation."),
    ]

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    billing_db = FAISS.from_documents(BILLING_DOCS, embeddings)
    network_db = FAISS.from_documents(NETWORK_DOCS, embeddings)
    streaming_db = FAISS.from_documents(STREAMING_DOCS, embeddings)

    @tool
    def search_billing_policy(query: str) -> str:
        """Useful for searching telecom billing policies, refund rules, and fee structures."""
        docs = billing_db.similarity_search(query, k=1)
        return docs[0].page_content if docs else "No relevant billing policy found."

    @tool
    def search_network_policy(query: str) -> str:
        """Useful for searching network troubleshooting policies and error code definitions."""
        docs = network_db.similarity_search(query, k=1)
        return docs[0].page_content if docs else "No relevant network policy found."

    @tool
    def search_streaming_ops(query: str) -> str:
        """Useful for searching streaming operations, CDN issues, and buffering policies."""
        docs = streaming_db.similarity_search(query, k=1)
        return docs[0].page_content if docs else "No relevant streaming policy found."

    @tool
    def lookup_account(account_id: object) -> str:
        """Looks up account details. Input should be an account ID string."""
        account_id_str = str(account_id)
        return f"Account {account_id_str}: Active. Plan: 1Gbps Fiber + 4K Streaming. Tenure: 6 years."

    @tool
    def run_line_diagnostics(line_id: object) -> str:
        """Runs physical line diagnostics on a given line ID."""
        return f"Diagnostics for {line_id}: FAILED. Error code 'ERR-77' detected at local node."

    @tool
    def check_regional_outages(region: str) -> str:
        """Checks if there is an active network outage in the specified region."""
        return f"Region {region}: ACTIVE OUTAGE verified. Estimated downtime so far: 5 hours."

    @tool
    def monitor_stream_health(stream_id: object) -> str:
        """Checks the health and buffering rates of a specific live stream ID."""
        return f"Stream {stream_id}: WARNING - High buffering detected. Bitrate dropping below 10Mbps."

    @tool
    def raise_escalation(issue_desc: str) -> str:
        """Raises an on-call escalation to human engineers for severe issues. Returns a Ticket ID."""
        ticket_id = f"TKT-{random.randint(10000, 99999)}"
        return f"Escalation successful. Official Ticket ID: {ticket_id}"

    @tool
    def issue_refund(account_id: object, amount: float) -> str:
        """Issues a billing refund to the specified account ID."""
        return f"Successfully processed ${amount} refund for account {account_id}."

    tools = [
        search_billing_policy, search_network_policy, search_streaming_ops,
        lookup_account, run_line_diagnostics, check_regional_outages,
        monitor_stream_health, raise_escalation, issue_refund
    ]

    llm = ChatOllama(model=model_name, temperature=0, num_ctx=2048)

    system_prompt = """You are TelcoPilot, a Multi-Domain Agentic Assistant for Telecom Operations.
    You act rather than answer. You have access to tools for billing, network, and streaming ops.
    Plan your steps based purely on the tool descriptions provided.

    CRITICAL INSTRUCTION TO PREVENT HALLUCINATIONS:
    If you need to escalate an issue and provide a ticket ID, you MUST invoke the `raise_escalation` tool.
    DO NOT fabricate, guess, or make up Ticket IDs. You are strictly constrained to surface ONLY 
    the exact Ticket ID returned to you by the `raise_escalation` tool.

    Always think step-by-step. If a customer asks for a refund due to an outage, verify the outage 
    and the billing policy first before issuing the refund.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


try:
    agent_executor = setup_agent_environment(selected_model)
except Exception as e:
    st.error(f"Failed to initialize Ollama framework: {e}")
    st.stop()

st.title("📡 TelcoPilot Dashboard")
st.subheader("Autonomous Telecom Diagnostics & Operations Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am TelcoPilot. How can I assist with your network, billing, or live stream operations today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_prompt := st.chat_input("Describe the issue (e.g., 'Account A-123 is complaining about stream S-99...'):"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner():
            try:
                result = agent_executor.invoke({"input": user_prompt})
                response_text = result["output"]
                st.write(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                error_msg = f"An error occurred while executing the task: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})