import streamlit as st
import os
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Setting up the page configuration with title and icon
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="")

st.markdown(
    """
    <style>
    /* Page title styling */
    .page-title {
        font-size: 2.5em;
        text-align: center;
        color: #333;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 4px;
    }
    
    .stButton>button:hover {
        background-color: white;
        color: black;
        border: 2px solid blue;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Setting up the title of the app
st.markdown('<div class="page-title">LangChain: Chat with SQL DB</div>', unsafe_allow_html=True)

# Database connection options
radio_opt = ["Use SQLite 3 Database - analytics_db"]

# Sidebar options for database selection
selected_opt = st.sidebar.radio(label="Choose the DB you want to chat with", options=radio_opt)

# API key input for Groq
api_key = os.getenv("GROQ_API_KEY")

# Info messages if API key is not provided
if not api_key:
    st.info("Please add the Groq API key")

# Initialize the Groq LLM
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

# Function to configure SQLite database
@st.cache_resource(ttl="2h")
def configure_db():
    dbfilepath = (Path(__file__).parent / "analytics_db").absolute()
    creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
    return SQLDatabase(create_engine("sqlite:///", creator=creator))

# Configure DB
db = configure_db()

# SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Creating an agent with SQL DB and Groq LLM
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Session state for messages and chat histories
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "chat_histories" not in st.session_state:
    st.session_state["chat_histories"] = []

# Dropdown for selecting chat history
chat_sessions = [f"Session {i+1}" for i in range(len(st.session_state["chat_histories"]))]
selected_chat = st.sidebar.selectbox("Select a chat history", ["Select Chat History"] + chat_sessions)

# Load selected chat history
if selected_chat != "Select Chat History":
    st.session_state["messages"] = st.session_state["chat_histories"][int(selected_chat.split(" ")[1]) - 1]

# Button to start a new chat session
if st.sidebar.button("New Chat"):
    # Save the current chat history
    st.session_state["chat_histories"].append(st.session_state["messages"])
    # Start a new chat
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    # st.experimental_rerun()

# Button to clear the current chat
if st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    st.stop() 
   

# Display chat history messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input for user query
user_query = st.chat_input(placeholder="Ask anything from the database")

# If user query is submitted
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Generate response from agent
    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            # Run the query through the agent and get the response
            response = agent.run(user_query)
            
            # Add the response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Display the response directly
            st.write(response)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Save chat history when "Share Chat" is clicked
def save_chat_history():
    chat_history = st.session_state.get("messages", [])
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    # Save the chat history to a file
    history_file = "chat_history.txt"
    with open(history_file, "w") as file:
        file.write(history_text)
    
    return history_file

if st.sidebar.button("Share Chat"):
    chat_file = save_chat_history()
    st.sidebar.download_button(
        label="Download Chat History",
        data=open(chat_file, "rb").read(),
        file_name=chat_file,
        mime="text/plain"
    )
