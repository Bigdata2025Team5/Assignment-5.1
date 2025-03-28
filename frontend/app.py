import streamlit as st
import requests
import pandas as pd
import altair as alt
from datetime import datetime

# FastAPI endpoint for generating report
API_URL = "https://fastapi-app-1057230376331.us-central1.run.app"

# Streamlit UI setup with custom style
st.set_page_config(page_title="NVIDIA Research Assistant", page_icon=":bar_chart:", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .stButton>button {
            background-color: #0073e6;
            color: white;
            border-radius: 5px;
            padding: 10px 30px;
            font-size: 16px;
            font-weight: bold;
            width: 100%;
        }
        .stTextInput input {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
        }
        .stSelectbox select {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
        }
        h1 {
            font-size: 40px;
            color: #0073e6;
            text-align: center;
        }
        h2 {
            color: #2a2a2a;
            text-align: center;
        }
        .report-section {
            padding: 20px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            max-width: 800px;
            margin: 0 auto;
        }
        .stSelectbox, .stTextInput, .stButton {
            margin-bottom: 15px;
        }
        footer {
            text-align: center;
            font-size: 14px;
            color: #888;
        }
        .chat-history {
            padding: 10px;
            background-color: #f7f7f7;
            border-radius: 5px;
            margin-top: 20px;
            max-width: 800px;
            margin: 0 auto;
            max-height: 300px;
            overflow-y: scroll;
        }
        .chat-message {
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.title("NVIDIA Research Assistant :bar_chart:")
    st.write("This tool generates comprehensive NVIDIA research reports by leveraging financial data, historical performance, and real-time web insights.")
    st.write("---")

    # Step 1: Display the query text box first
    st.subheader("Research Query")
    query = st.text_area("Enter your research query", height=150)

    # Step 2: Select agent with conditional display for year and quarter
    agent = st.selectbox("Select Agent", ["Snowflake Agent", "RAG Agent", "Web Search Agent", "All Agents"], key="agent")

    # Conditionally display year and quarter options based on agent selection
    if agent != "Web Search Agent":
        year = st.selectbox("Select Year", [2021, 2022, 2023, 2024, 2025], key="year")
        quarter = st.selectbox("Select Quarter", [1, 2, 3, 4], key="quarter")
    else:
        year = None
        quarter = None

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Step 3: Generate the report
    if st.button("Generate Report", use_container_width=True):
        payload = {
            "query": query,
            "year": year,
            "quarter": quarter,
            "agent": agent
        }
        url = f"{API_URL}/generate_report"
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()

            # Add the query and response to the chat history
            report = data.get("report", "No report generated.")
            st.session_state.chat_history.append({"message": query, "response": report})

            # Display the generated report
            st.header("Generated Research Report")
            st.write(report)

            # Display the valuation charts if available
            if data.get("chart_data"):
                st.write("### Charts:")
                df_chart = pd.DataFrame(data["chart_data"])
                chart = alt.Chart(df_chart).mark_bar().encode(
                    x="metric:N",
                    y="value_num:Q",
                    tooltip=["metric", "value_num"]
                ).properties(
                    width=600,
                    height=400,
                    title="Valuation Metrics"
                )
                st.altair_chart(chart, use_container_width=True)

            # Display web insights and links
            if data.get("web_insights"):
                st.write("### Web Insights:")
                st.write(data.get("web_insights"))
            if data.get("web_links"):
                st.write("### Source Links:")
                for link in data.get("web_links"):
                    st.markdown(f"[{link}]({link})")
        else:
            st.error("Failed to generate report. Please try again.")

    # Step 4: Display the chat history at the bottom, after report generation
    if st.session_state.chat_history:
        st.write("### Chat History")
        with st.container():
            for chat in st.session_state.chat_history:
                st.markdown(f"**Query:** {chat['message']}")
                st.markdown(f"**Response:** {chat['response']}")
                st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)
