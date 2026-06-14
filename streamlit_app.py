import os
import streamlit as st
import requests

st.set_page_config(page_title="SkyWings AI Support", page_icon="✈️")

st.title("✈️ SkyWings Customer Support")
st.markdown("""
Welcome to the **SkyWings AI Support Agent**.
Ask me about flight statuses, baggage policies, refunds, or general airline information.
""")

with st.sidebar:
    st.header("System Status")
    st.success("Backend: Connected")
    st.info("Capabilities: SQL Flight Data, RAG Policy Knowledge, Safety Guardrails")

user_query = st.text_input(
    "How can we help you today?",
    placeholder="e.g., What is the status of flight 6E477?"
)

if st.button("Send Query"):
    if user_query:
        with st.spinner("Processing your request..."):
            try:
                # FastAPI runs on port 8000 in the same Codespace
                api_url = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/chat")
                payload = {"query": user_query}
                response = requests.post(api_url, json=payload, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("SkyWings AI Assistant:")
                    st.write(data["response"])
                else:
                    st.error(f"Backend returned status code {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend. Make sure cell 85 ran successfully (server on port 8000).")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a query before clicking send.")

st.divider()
st.caption("© 2024 SkyWings Airlines - Powered by Agentic AI")
