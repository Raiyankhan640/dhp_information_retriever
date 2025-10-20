import streamlit as st
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda 
from dotenv import load_dotenv
import os
import tempfile
import shutil

# Load environment variables
load_dotenv()

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

def load_and_chunk_documents_from_uploaded_files(uploaded_files, chunk_size=1200, chunk_overlap=300):
    """Load and chunk documents from uploaded files"""
    if not uploaded_files:
        return []
    
    # Create a temporary directory to store uploaded files
    temp_dir = tempfile.mkdtemp()
    
    try:
        pdf_files = []
        word_files = []
        
        # Save uploaded files to temporary directory
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if uploaded_file.name.lower().endswith('.pdf'):
                pdf_files.append(file_path)
            elif uploaded_file.name.lower().endswith('.docx'):
                word_files.append(file_path)
        
        all_documents = []
        
        # Load PDF documents
        if pdf_files:
            pdf_loader = DirectoryLoader(
                temp_dir,
                glob="**/*.pdf",
                loader_cls=UnstructuredPDFLoader,
                show_progress=False
            )
            pdf_documents = pdf_loader.load()
            all_documents.extend(pdf_documents)
            st.sidebar.success(f"Loaded {len(pdf_documents)} PDF documents")
        
        # Load Word documents
        if word_files:
            word_loader = DirectoryLoader(
                temp_dir,
                glob="**/*.docx",
                loader_cls=UnstructuredWordDocumentLoader,
                show_progress=False
            )
            word_documents = word_loader.load()
            all_documents.extend(word_documents)
            st.sidebar.success(f"Loaded {len(word_documents)} Word documents")
        
        if not all_documents:
            st.sidebar.warning("No valid documents found")
            return []
        
        # Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_documents(all_documents)
        st.sidebar.success(f"Created {len(chunks)} chunks")
        
        return chunks
        
    except Exception as e:
        st.sidebar.error(f"Error loading documents: {e}")
        return []
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def create_vector_store(chunks):
    """Create vector store from document chunks"""
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.sidebar.success(f"Vector store created successfully with {vector_store.index.ntotal} vectors")
        return vector_store
    except Exception as e:
        st.sidebar.error(f"Error creating vector store: {e}")
        return None

def initialize_rag_chain(vector_store):
    """Initialize the RAG chain"""
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        
        prompt_template = """
Here is a revised version of your prompt that incorporates your requests for a friendlier (yet formal) tone and the built-in understanding of time and day abbreviations.

This new prompt explicitly instructs the AI to standardize times (like "9-10") and to interpret common day abbreviations (like "MW" or "ST") using its general knowledge, even if those definitions are not in the provided text.

-----

### Revised RAG System Prompt

```
You are an intelligent, helpful, and formal assistant for a document retrieval system. Your tone should be friendly and approachable, while remaining professional and clear.

Use *only* the following pieces of retrieved context to answer the user's question.

Follow these specific instructions:

1.  **Interpret and Standardize Information:**
    * **Times:** When presenting times from the context (e.g., "9-10" or "1-3"), standardize them for clarity. Infer 'AM' or 'PM' based on typical academic or business hours (e.g., "9-10" should be presented as "9:00 AM - 10:00 AM", and "1-3" as "1:00 PM - 3:00 PM").
    * **Days:** You may use your general knowledge to interpret common academic abbreviations for days of the week, such as 'MW' (Monday/Wednesday), 'ST' (Sunday/Tuesday), or 'RA' (Thursday/Saturday), even if these abbreviations are not explicitly defined in the context.

2.  **Format for Schedule Details:**
    * If the answer requires specific personal or schedule details (like office hours, contact info, or availability), present them in a structured and easy-to-read format (e.g., bullet points or a table).
    * Ensure you include **all** associated details found in the context, such as names, email addresses, phone numbers, and the standardized times/days.

3.  **Handle Specific Availability Questions:**
    * If the user asks about availability at a *specific time*, your answer should focus *only* on Teaching Assistant (TA) availability from the context.
    * List the names of the available TAs in a clear, structured format.

4.  **Fallback:**
    * If you cannot find the answer in the provided context, clearly and politely state that you do not have that information.

Context:
{context}

Question: {question}

Formal Answer:
"""
        qa_prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )

        def format_docs(retrieved_docs):
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context_text

        parser = StrOutputParser()

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        main_chain = parallel_chain | qa_prompt | llm | parser
        return main_chain
        
    except Exception as e:
        st.error(f"Error initializing RAG chain: {e}")
        return None

# Streamlit UI
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling with improved colors and visibility
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    .user-message {
        background-color: #dbeafe;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid #3b82f6;
        color: #1e40af;
    }
    .assistant-message {
        background-color: #f0fdf4;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid #10b981;
        color: #166534;
    }
    .upload-section {
        background-color: #fffbeb;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px dashed #f59e0b;
    }
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .clear-button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white !important;
    }
    .clear-button:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
    }
    .input-container {
        display: flex;
        gap: 10px;
        align-items: flex-end;
    }
    .stTextInput>div>div>input {
        padding: 0.75rem;
        border-radius: 8px;
        border: 2px solid #d1d5db;
        font-size: 1rem;
    }
    .stTextInput>div>div>input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown(
    '''<div class="main-header">
        <h1>🧑‍� Office Hour Information Retriever</h1>
        <p>
            Effortlessly find any details of staff or workers from your given documents through a LLM.<br>
            <b>Upload your documents, ask questions, and get instant, structured answers powered by AI.</b>
        </p>
        <p style="font-size:1.1em; color:#e0e7ff; margin-top:1em;">
            <span style="background:#2563eb; color:white; padding:0.3em 0.8em; border-radius:8px;">Step 1:</span> Upload your files &nbsp; 
            <span style="background:#2563eb; color:white; padding:0.3em 0.8em; border-radius:8px;">Step 2:</span> Ask your question &nbsp; 
            <span style="background:#2563eb; color:white; padding:0.3em 0.8em; border-radius:8px;">Step 3:</span> Get answers instantly
        </p>
    </div>''',
    unsafe_allow_html=True
)

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Upload")
    st.markdown("Upload your PDF and Word documents to get started!")
    
    uploaded_files = st.file_uploader(
        "Choose PDF or Word files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload PDF (.pdf) or Word (.docx) files"
    )
    
    if st.button("🚀 Process Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing documents... This may take a few moments."):
                chunks = load_and_chunk_documents_from_uploaded_files(uploaded_files)
                if chunks:
                    st.session_state.chunks = chunks
                    vector_store = create_vector_store(chunks)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.documents_loaded = True
                        st.session_state.chat_history = []  # Clear chat history when new documents are loaded
                        st.success("✅ Documents processed successfully!")
                    else:
                        st.error("❌ Failed to create vector store")
                else:
                    st.error("❌ No valid documents to process")
        else:
            st.warning("⚠️ Please upload at least one document")
    
    st.divider()
    st.markdown("### ℹ️ Instructions")
    st.markdown("""
    1. Upload PDF or Word documents
    2. Click "Process Documents" 
    3. Ask questions about your documents
    4. Get AI-powered answers based on your content
    """)
    
    # Add clear chat button in sidebar
    if st.session_state.documents_loaded and st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", key="clear_chat_sidebar"):
            st.session_state.chat_history = []
            st.rerun()

# Main content area
if not st.session_state.documents_loaded:
    st.info("👆 Please upload and process your documents using the sidebar to get started!")
else:
    # Initialize RAG chain
    if 'rag_chain' not in st.session_state:
        rag_chain = initialize_rag_chain(st.session_state.vector_store)
        if rag_chain:
            st.session_state.rag_chain = rag_chain
        else:
            st.error("Failed to initialize the AI assistant. Please try reprocessing your documents.")
            st.session_state.documents_loaded = False
    
    # Chat interface
    st.markdown("### 💬 Ask Questions")
    
    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="user-message"><strong>👤 You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message"><strong>🤖 Assistant:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.info("No conversation yet. Ask your first question!")
    
    # User input with proper state management for re-asking
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Use a unique key that doesn't change to maintain input state
    user_input = st.text_input(
        "Ask a question about your documents:",
        value="",
        key="user_input_main",
        placeholder="e.g., What are the office hours? Who is available on Monday?",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 6])
    
    with col1:
        send_button = st.button("📤 Send", type="primary", use_container_width=True)
    
    with col2:
        if st.session_state.chat_history:
            clear_button = st.button("🗑️ Clear Chat", type="secondary", use_container_width=True)
            if clear_button:
                st.session_state.chat_history = []
                st.session_state.current_question = ""
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle sending question
    if send_button and user_input.strip():
        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        st.session_state.current_question = ""  # Clear input after sending
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke(user_input)
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error getting response: {e}")
    
    # Do not update current_question here; it is cleared after sending a question and rerun

# Footer
st.markdown("---")
st.markdown("Powered by LangChain, Google Gemini, and Hugging Face Embeddings | Built with Streamlit")
