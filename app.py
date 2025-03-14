import os
import streamlit as st
import tempfile
import asyncio
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.memory import ConversationBufferMemory

# Load API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Cache LLM instance
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model="gemini-1.5-flash", streaming=True)

llm = get_llm()

# Cache embeddings to avoid repeated initialization
@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="embedding-001", api_key=GEMINI_API_KEY)

embeddings = get_embeddings()

# Load FAISS index
@st.cache_resource
def load_vectorstore():
    return FAISS.load_local("faiss_index", embeddings)

# Function to process uploaded file asynchronously
async def process_file(file):
    file_ext = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(file.getbuffer())
        temp_file_path = temp_file.name
    
    # Load document
    if file_ext == ".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_ext in [".docx", ".txt"]:
        loader = Docx2txtLoader(temp_file_path)
    else:
        st.error("Unsupported file format")
        return None
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Create FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

# Load or process document
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = load_vectorstore()

# Initialize chat history with limit
MAX_CHAT_HISTORY = 10
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def add_to_chat_history(user_input, response):
    st.session_state["chat_history"].append((user_input, response))
    if len(st.session_state["chat_history"]) > MAX_CHAT_HISTORY:
        st.session_state["chat_history"] = st.session_state["chat_history"][1:]

# Streamlit UI
st.title("Optimized Chat with RAG")
file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"], key="file_uploader")

if file:
    st.session_state["vectorstore"] = asyncio.run(process_file(file))
    st.success("Document processed successfully!")

user_input = st.text_input("Ask a question:")
if user_input:
    retriever = st.session_state["vectorstore"].as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    
    response = qa_chain.run(user_input)
    add_to_chat_history(user_input, response)
    
    response_container = st.empty()
    response_container.markdown(response)

# Display chat history
for query, resp in st.session_state["chat_history"]:
    st.markdown(f"**You:** {query}")
    st.markdown(f"**Bot:** {resp}")
