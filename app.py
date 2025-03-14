import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import time
import sys

# Set up basic page
st.set_page_config(page_title="Data Science Tutor", layout="wide")
st.title("Data Science Tutor")

# Debug checkpoint
st.write("App initialization started")

# Load environment variables - works for local development
load_dotenv()

# For Streamlit Cloud deployment, get API key from secrets
# Use a try-except to handle both local and cloud environments
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    st.write("API key loaded from secrets")
except Exception as e:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        st.write("API key loaded from environment variables")
    else:
        st.error("GEMINI_API_KEY is not set. Please set it in your .env file or Streamlit secrets.")
        st.stop()

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
if 'chat_history_messages' not in st.session_state:
    st.session_state.chat_history_messages = []
if 'llm_initialized' not in st.session_state:
    st.session_state.llm_initialized = False
if 'embeddings_initialized' not in st.session_state:
    st.session_state.embeddings_initialized = False

# Generate a unique session ID for each user
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"user_{np.random.randint(10000)}"

# Use a database path that works in Streamlit Cloud
# For Streamlit Cloud, we'll use in-memory SQLite database
DB_PATH = ":memory:"  # Use in-memory SQLite database for Streamlit Cloud
SESSION_ID = st.session_state.session_id

# Lazy loading function for LLM
def initialize_llm():
    if not st.session_state.llm_initialized:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            # Set a timeout for the API call
            start_time = time.time()
            llm = ChatGoogleGenerativeAI(
                api_key=GEMINI_API_KEY,
                model="gemini-1.5-flash",
                streaming=True,
                timeout=10  # 10 second timeout
            )
            st.session_state.llm = llm
            st.session_state.llm_initialized = True
            
            # Create the chat template
            from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.output_parsers import StrOutputParser
            
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
            st.session_state.standard_chain = standard_template | llm | output_parser
            
            return True
        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
            return False
    return True

# Lazy loading function for embeddings
def initialize_embeddings():
    if not st.session_state.embeddings_initialized:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            # Set a timeout for the API call
            embeddings = GoogleGenerativeAIEmbeddings(
                model="embedding-001",
                api_key=GEMINI_API_KEY,
                timeout=10  # 10 second timeout
            )
            st.session_state.embeddings = embeddings
            st.session_state.embeddings_initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {str(e)}")
            return False
    return True

# Initialize chat history
def initialize_chat_history():
    if 'chat_history' not in st.session_state:
        try:
            from langchain_core.messages import HumanMessage, AIMessage
            from langchain_community.chat_message_histories import SQLChatMessageHistory
            
            chat_history = SQLChatMessageHistory(
                session_id=SESSION_ID,
                connection=f"sqlite:///{DB_PATH}"
            )
            st.session_state.chat_history = chat_history
            return True
        except Exception as e:
            st.error(f"Failed to initialize chat history: {str(e)}")
            return False
    return True

# Function to process uploaded files
def process_uploaded_files(uploaded_files):
    if not initialize_embeddings():
        st.warning("Embeddings could not be initialized. File search capabilities will be limited.")
        return []
        
    documents = []
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        for file in uploaded_files:
            file_ext = os.path.splitext(file.name)[1].lower()
            try:
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
                    # Clean up temp file with error handling
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        st.warning(f"Could not delete temporary file: {str(e)}")
            except Exception as e:
                st.error(f"Error creating temporary file for {file.name}: {str(e)}")
        
        if documents and st.session_state.embeddings_initialized:
            # Add a try-except block to catch embedding errors
            try:
                split_docs = text_splitter.split_documents(documents)
                
                from langchain_community.vectorstores import FAISS
                vectorstore = FAISS.from_documents(split_docs, st.session_state.embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.documents = documents
                st.success(f"Successfully processed {len(uploaded_files)} files with {len(split_docs)} text chunks.")
            except Exception as e:
                st.error(f"Error during document processing: {str(e)}")
                # Continue without vector search if embedding fails
                st.session_state.documents = documents
                st.warning("File content is available but advanced search capabilities are limited.")
        return documents
    except Exception as e:
        st.error(f"Error in file processing: {str(e)}")
        return []

def generate_response(user_input):
    # Add to session state history (backup)
    st.session_state.chat_history_messages.append({"role": "user", "content": user_input})
    
    # Initialize LLM if not already done
    if not initialize_llm():
        return "I apologize, but I couldn't initialize the AI model. Please check your API key and try again."
    
    # Initialize chat history if not already done
    initialize_chat_history()
    
    if st.session_state.vectorstore and st.session_state.embeddings_initialized:
        try:
            from langchain.chains import ConversationalRetrievalChain
            
            # Create a retrieval chain that uses the vectorstore
            retrieval_chain = ConversationalRetrievalChain.from_llm(
                llm=st.session_state.llm,
                retriever=st.session_state.vectorstore.as_retriever(),
                return_source_documents=True
            )
            
            # Prepare conversation history for the chain
            if 'chat_history' in st.session_state and hasattr(st.session_state.chat_history, 'messages'):
                history = [(msg.content, st.session_state.chat_history.messages[i+1].content) 
                          for i, msg in enumerate(st.session_state.chat_history.messages[:-1:2]) 
                          if i+1 < len(st.session_state.chat_history.messages)]
            else:
                # Use session state as backup
                history = []
                for i in range(0, len(st.session_state.chat_history_messages)-1, 2):
                    if i+1 < len(st.session_state.chat_history_messages):
                        history.append((
                            st.session_state.chat_history_messages[i]["content"],
                            st.session_state.chat_history_messages[i+1]["content"]
                        ))
            
            result = retrieval_chain.invoke({
                "question": user_input,
                "chat_history": history
            })
            response = result["answer"]
            
            # Add to both history systems
            if 'chat_history' in st.session_state:
                st.session_state.chat_history.add_user_message(user_input)
                st.session_state.chat_history.add_ai_message(response)
            st.session_state.chat_history_messages.append({"role": "assistant", "content": response})
            
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
    try:
        if 'standard_chain' in st.session_state:
            # Use the standard chain
            for chunk in st.session_state.standard_chain.stream({"question": user_input}):
                response += chunk
        else:
            response = "I apologize, but the AI model couldn't be initialized. Please check your API key and try again."
        
        # Add to session state history (backup)
        st.session_state.chat_history_messages.append({"role": "assistant", "content": response})
        
        return response
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        st.error(error_msg)
        return f"I apologize, but I encountered an error. Please try again or check your API key configuration. Error: {str(e)}"

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
                                    type=["pdf", "csv", "txt", "py", "ipynb", "r", "sql"])
    
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

# Use session state as the source of truth for displaying history
for msg in reversed(st.session_state.chat_history_messages):
    if msg["role"] == "user":
        st.write(f"**You:** {msg['content']}")
    else:
        st.write(f"**Tutor:** {msg['content']}")

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

# Debug checkpoint
st.write("App initialization completed")