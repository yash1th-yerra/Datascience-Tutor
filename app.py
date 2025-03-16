import streamlit as st
import sqlite3
import os
# import chromadb
import warnings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Fix for deprecation warnings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Prevent runtime errors with event loops
os.environ['PYTHONUNBUFFERED'] = '1'

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="Gemma2-9b-It"
)

# Hugging Face Embeddings with updated import
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Database Setup
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

def authenticate(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

def register(username, password):
    if not username or not password:
        return False
    
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# ChromaDB Initialization
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def load_vectorstore():
    try:
        if os.path.exists("./chroma_db"):
            # Updated Chroma import usage
            return Chroma(client=chroma_client, embedding_function=embedding_model)
        return None
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def save_vectorstore(docs):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)
        # Updated Chroma import usage
        return Chroma.from_documents(split_docs, embedding_model, client=chroma_client)
    except Exception as e:
        st.error(f"Error saving to vector store: {str(e)}")
        return None

def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        connection_string="sqlite:///chat_history.db",
        session_id=session_id
    )

def process_document(uploaded_file):
    try:
        # Create temp directory if it doesn't exist
        os.makedirs("temp_files", exist_ok=True)
        
        temp_file_path = os.path.join("temp_files", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        # Cleanup
        os.remove(temp_file_path)
        
        return save_vectorstore(docs)
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

# Function to handle form submission
def handle_form_submit():
    if st.session_state.user_question:  # Check if there's input
        st.session_state.submit_question = True
        # Store the current question and clear the input
        st.session_state.current_question = st.session_state.user_question
        st.session_state.user_question = ""  # Clear the input field

# Streamlit UI setup with error handling
try:
    st.title("Data Science Tutor 🤖")
    init_db()
    
    # Initialize session state variables
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vectorstore()
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # For handling form submission
    if "submit_question" not in st.session_state:
        st.session_state.submit_question = False
    
    # For storing current question
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    
except Exception as setup_error:
    st.error(f"Application setup error: {str(setup_error)}")

# Main application logic with try-except blocks
try:
    # Login/Register Screen
    if st.session_state.user_id is None:
        st.subheader("Login / Register")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                user = authenticate(username, password)
                if user:
                    st.session_state.user_id = user[0]
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        with tab2:
            new_username = st.text_input("New Username", key="register_username")
            new_password = st.text_input("New Password", type="password", key="register_password")
            if st.button("Register"):
                if not new_username or not new_password:
                    st.error("Username and password cannot be empty")
                elif register(new_username, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")
    # Main Application Screen
    else:
        with st.sidebar:
            st.header("Upload Documents")
            uploaded_file = st.file_uploader("Upload a PDF for RAG-based QA", type=["pdf"])
            if uploaded_file is not None:
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        vectorstore = process_document(uploaded_file)
                        if vectorstore is not None:
                            st.session_state.vectorstore = vectorstore
                            st.success("Document processed successfully!")
            
            # Add logout button to sidebar
            if st.button("Logout"):
                st.session_state.clear()
                st.rerun()
        
        st.subheader("Chat with the Tutor")
        
        # User input for new questions - BEFORE displaying chat history
        with st.form(key="question_form"):
            user_input = st.text_input("Ask a question:", key="user_question")
            submit_button = st.form_submit_button("Submit", on_click=handle_form_submit)
        
        # Process the question if submitted
        if st.session_state.submit_question and st.session_state.current_question:
            user_input = st.session_state.current_question
            try:
                if st.session_state.vectorstore:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    
                    # Create the system message prompt
                    system_prompt = """
                    You are a highly knowledgeable AI tutor specializing in Data Science.
                    You must only answer questions related to Data Science, Machine Learning,
                    Deep Learning, Statistics, Data Analytics, and related topics.
                    If a user asks about something unrelated, politely refuse to answer.
                    
                    Use the following pieces of retrieved context to answer the question.
                    If you don't know the answer, just say that you don't know.
                    
                    Context: {context}
                    
                    Question: {question}
                    """
                    
                    # Create the QA prompt
                    qa_prompt = ChatPromptTemplate.from_template(system_prompt)
                    
                    # Create the conversation chain
                    chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=retriever,
                        return_source_documents=False,
                        combine_docs_chain_kwargs={"prompt": qa_prompt},
                        chain_type="stuff",
                        get_chat_history=lambda h: h,
                        verbose=False,  # Reduce debug output
                    )
                    
                    with st.spinner("Thinking..."):
                        # Format chat history for the chain
                        formatted_history = []
                        for q, a in st.session_state.chat_history:
                            formatted_history.append(HumanMessage(content=q))
                            formatted_history.append(AIMessage(content=a))
                        
                        # Process the response
                        try:
                            response = chain({"question": user_input, "chat_history": formatted_history})
                            response_text = response.get("answer", "I'm sorry, I couldn't generate a response.")
                            
                            # Add to chat history
                            st.session_state.chat_history.append((user_input, response_text))
                        except Exception as chain_error:
                            st.error(f"Error in response generation: {str(chain_error)}")
                else:
                    st.warning("Please upload a document first for the tutor to use as reference.")
            except Exception as process_error:
                st.error(f"Error processing question: {str(process_error)}")
            
            # Reset the submit flag without refreshing
            st.session_state.submit_question = False
            # Remove the current question from session state
            st.session_state.current_question = None
        
        # Chat history display - AFTER the input form and processing
        if st.session_state.chat_history:
            st.subheader("Conversation History")
            for q, a in st.session_state.chat_history:
                with st.container():
                    st.markdown(f"**You**: {q}")
                    st.markdown(f"**Tutor**: {a}")
                    st.markdown("---")

except Exception as app_error:
    st.error(f"Application error: {str(app_error)}")
    st.info("Please refresh the page and try again.")