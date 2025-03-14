import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="Data Science Tutor", layout="wide")
st.title("Data Science Tutor")

# Initialize session state variables
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'submitted_input' not in st.session_state:
    st.session_state.submitted_input = ""
if 'needs_rerun' not in st.session_state:
    st.session_state.needs_rerun = False

# API and database configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_PATH = "chat_history.db"
SESSION_ID = "user_123"

# Initialize chat history
chat_history = SQLChatMessageHistory(
    session_id=SESSION_ID,
    connection_string=f"sqlite:///{DB_PATH}"
)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-2.0-flash-exp",
    streaming=True
)

# Create embeddings with the correct model name
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Fixed model name format
    api_key=GEMINI_API_KEY
)

# Function to process uploaded files
def process_uploaded_files(uploaded_files):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for file in uploaded_files:
        file_ext = os.path.splitext(file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(file.getvalue())
            temp_path = temp_file.name
        
        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(temp_path)
                documents.extend(loader.load())
            elif file_ext == '.csv':
                loader = CSVLoader(temp_path)
                documents.extend(loader.load())
            elif file_ext in ['.txt', '.py', '.ipynb', '.r', '.sql']:
                loader = TextLoader(temp_path)
                documents.extend(loader.load())
            else:
                st.warning(f"Unsupported file type: {file_ext}")
        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")
        finally:
            os.unlink(temp_path)
    
    if documents:
        # Add a try-except block to catch embedding errors
        try:
            split_docs = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            st.session_state.vectorstore = vectorstore
            st.session_state.documents = documents
            st.success(f"Successfully processed {len(uploaded_files)} files with {len(split_docs)} text chunks.")
        except Exception as e:
            st.error(f"Error during document processing: {str(e)}")
            # Continue without vector search if embedding fails
            st.session_state.documents = documents
            st.warning("File content is available but advanced search capabilities are limited.")
    return documents

# Create the chat template
standard_template = ChatPromptTemplate(
    messages=[
        ("system", "You are a highly knowledgeable Data Science tutor. You must strictly answer only Data Science-related queries. If a user asks about anything unrelated, firmly respond with: 'I can only assist with Data Science topics. Please ask a question related to Data Science. Other topics will not be addressed.provide answers in clear and detailed way'"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

# Create the output parser
output_parser = StrOutputParser()

# Define the conversation chain
standard_chain = standard_template | llm | output_parser

# Wrap the chain with message history
conversation_chain = RunnableWithMessageHistory(
    standard_chain,
    lambda session_id: chat_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

def generate_response(user_input):
    if st.session_state.vectorstore:
        try:
            # Create a retrieval chain that uses the vectorstore
            retrieval_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=st.session_state.vectorstore.as_retriever(),
                return_source_documents=True
            )
            result = retrieval_chain.invoke({
                "question": user_input,
                "chat_history": [(msg.content, chat_history.messages[i+1].content) 
                                for i, msg in enumerate(chat_history.messages[:-1:2]) if i+1 < len(chat_history.messages)]
            })
            response = result["answer"]
            
            # Add messages to chat history
            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(response)
            return response
        except Exception as e:
            # Fallback to standard chain if retrieval fails
            st.warning(f"Advanced search failed, using standard response: {str(e)}")
            return generate_standard_response(user_input)
    else:
        # Use the standard chain if no documents are uploaded
        return generate_standard_response(user_input)

def generate_standard_response(user_input):
    response = ""
    for chunk in conversation_chain.stream(
        {"question": user_input},
        config={"configurable": {"session_id": SESSION_ID}}
    ):
        response += chunk
    return response

# Callback function for the form submission
def handle_input_submit():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.submitted_input = user_input
        st.session_state.needs_rerun = True

# Sidebar for file upload
with st.sidebar:
    st.header("Tools")
    
    # File upload section
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader("Upload data science related files", 
                                    accept_multiple_files=True, 
                                    type=["pdf", "csv", "txt", "py", "ipynb", ".r", "sql"])
    
    if uploaded_files and uploaded_files != st.session_state.uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        process_uploaded_files(uploaded_files)
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.subheader("Uploaded Files")
        for file in st.session_state.uploaded_files:
            st.write(f"- {file.name}")

# Main chat interface
st.subheader("Chat with the Data Science Tutor")

# Use a form to handle the input submission properly
with st.form(key="input_form", clear_on_submit=True):
    user_input = st.text_input("Ask me anything related to Data Science...", key="user_input")
    submit_button = st.form_submit_button("Submit", on_click=handle_input_submit)

# Process the submitted input
if st.session_state.needs_rerun:
    response_container = st.empty()
    with st.spinner("Generating..."):
        response = generate_response(st.session_state.submitted_input)
        response_container.markdown(response)
    st.session_state.needs_rerun = False
    st.session_state.submitted_input = ""

# Display chat history in reverse order
st.divider()
st.subheader("Chat History")
for msg in reversed(chat_history.messages):  # Reverse the order to show the latest messages first
    if isinstance(msg, HumanMessage):
        st.write(f"**You:** {msg.content}")
    if isinstance(msg, AIMessage):
        st.write(f"**Tutor:** {msg.content}")

# Display file analysis if files are uploaded
if st.session_state.documents:
    st.divider()
    st.subheader("File Analysis")
    st.write(f"Number of documents processed: {len(st.session_state.documents)}")
    st.write("You can ask questions about the content of your uploaded files.")
    
    # Preview first few documents
    if st.checkbox("Show document preview"):
        for i, doc in enumerate(st.session_state.documents[:3]):
            st.write(f"**Document {i+1}** (Preview):")
            st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
            if i == 2 and len(st.session_state.documents) > 3:
                st.write(f"...and {len(st.session_state.documents) - 3} more documents")
                break   