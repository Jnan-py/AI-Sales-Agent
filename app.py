import streamlit as st
import pandas as pd
import sqlite3
import google.generativeai as genai
from datetime import datetime

class MemoryManager:
    def __init__(self, db_file="agent_memory.db"):
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.c = self.conn.cursor()
        self.c.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            timestamp TEXT,
            agent_name TEXT,
            user_query TEXT,
            agent_response TEXT
        )
        ''')
        self.conn.commit()

    def save(self, agent_name, query, response):
        self.c.execute(
            "INSERT INTO memory VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), agent_name, query, response)
        )
        self.conn.commit()

    def fetch_recent(self, limit=5):
        self.c.execute("SELECT * FROM memory ORDER BY timestamp DESC LIMIT ?", (limit,))
        return self.c.fetchall()

class GeminiAgent:
    def __init__(self, name, role, instructions, model_name="gemini-1.5-flash"):
        self.name = name
        self.role = role
        self.instructions = instructions
        self.model = genai.GenerativeModel(model_name)

    def run(self, query, context=None):
        prompt = f"""
        You are {self.name}, role: {self.role}.
        Instructions: {self.instructions}

        User Query: {query}

        Context: {context if context else "No extra context"}

        Note : You can check the context for relevant information to help answer the query. If it is not relevant, you can use your own knowledge, skills, and reasoning to answer the query.
        For example, in the context, if there is a list of companies with their industries and locations, you can use that information to suggest potential leads based on the user's query.
        And also, in the context, the state is mentioned like "CA", "NY", etc. You should interpret these as the full state names like "California", "New York", etc. when providing your response.

        You can use your own knowledge, skills, and reasoning to answer the query, but always try to provide the answer, despite low information in the context.
        """
        response = self.model.generate_content(prompt)
        return response.text

st.set_page_config(page_title="AI Sales Agents", layout="wide")
st.title("AI-Powered Sales Assistant")

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

model_name = st.sidebar.selectbox(
    "Choose Gemini Model",
    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
)

uploaded_file = st.sidebar.file_uploader("Upload Dataset (Excel)", type=["xlsx"])

if api_key:
    genai.configure(api_key=api_key)
    memory = MemoryManager()

    df = None
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("### Dataset Preview", df.head())

    prospecting_agent = GeminiAgent(
        name="Prospecting Agent",
        role="Helps sales reps discover high-potential leads",
        instructions="Use dataset and all columns and reason based on the column values to suggest businesses and answer user queries. Explain reasoning.",
        model_name=model_name
    )
    insights_agent = GeminiAgent(
        name="Prospect Insights Agent",
        role="Analyzes selected prospects",
        instructions="Use dataset and all columns and reason based on the column values Perform SWOT analysis, review SEO, social presence, and identify opportunities. Suggest tailored engagement strategies.",
        model_name=model_name
    )
    communication_agent = GeminiAgent(
        name="Communication Agent",
        role="Crafts personalized communication",
        instructions="Use dataset and all columns and reason based on the column values, Write personalized emails, LinkedIn messages, or scripts. Adapt tone to industry and prospect’s needs.",
        model_name=model_name
    )
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Prospecting",
        "Insights",
        "Communication",
        "Memory"
    ])

    with tab1:
        st.subheader("Find Potential Prospects")
        query1 = st.text_area("Enter your prospecting query")
        if st.button("Run Prospecting", key="prospecting"):
            context = df.head().to_string() if df is not None else None
            response1 = prospecting_agent.run(query1, context=context)
            st.write("### Prospecting Agent Response")
            st.write(response1)
            memory.save(prospecting_agent.name, query1, response1)

    with tab2:
        st.subheader("Analyze Prospects")
        query2 = st.text_area("Enter your insights query")
        if st.button("Run Insights", key="insights"):
            context = df.head().to_string() if df is not None else None
            response2 = insights_agent.run(query2, context=context)
            st.write("### Insights Agent Response")
            st.write(response2)
            memory.save(insights_agent.name, query2, response2)

    with tab3:
        st.subheader("Craft Communication")
        query3 = st.text_area("Enter your communication request")
        if st.button("Run Communication", key="communication"):
            context = df.head().to_string() if df is not None else None
            response3 = communication_agent.run(query3, context=context)
            st.write("### Communication Agent Response")
            st.write(response3)
            memory.save(communication_agent.name, query3, response3)

    with tab4:
        st.subheader("Recent Memory")
        for row in memory.fetch_recent(limit=5):
            st.write(f"{row[0]} | **{row[1]}**")
            st.write(f"**Query:** {row[2]}")
            st.write(f"**Response:** {row[3]}")
            st.markdown("---")

else:
    st.warning("Please enter your Gemini API key in the sidebar to continue.")
