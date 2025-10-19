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


load_dotenv()
#Step 1: Load Documents and Chunking
def load_and_chunk_documents(directory_path="./documents", chunk_size=1000, chunk_overlap=200):
    # Load all PDF files
    pdf_loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=UnstructuredPDFLoader,
        show_progress=True
    )
    
    # Load all Word documents (.docx)
    word_loader = DirectoryLoader(
        directory_path,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
        show_progress=True
    )
    
    # Load documents
    pdf_documents = pdf_loader.load()
    word_documents = word_loader.load()
    
    # Combine all documents
    all_documents = pdf_documents + word_documents
    
    print(f"Loaded {len(pdf_documents)} PDF documents")
    print(f"Loaded {len(word_documents)} Word documents")
    print(f"Total documents: {len(all_documents)}")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(all_documents)
    
    print(f"Created {len(chunks)} chunks")
    
    return chunks

# Step 2: Embedding and Vector Store Creation. This step generates vector embeddings for the document chunks and stores them in a FAISS vector store for efficient similarity search.

# 1. Load and chunk the documents using the function defined above.
chunks = load_and_chunk_documents()
length = len(chunks)

# 2. Initialize the HuggingFace sentence transformer model for embedding generation.
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 3. (To be added) Create the FAISS vector store from the generated embeddings and document chunks.
vector_store = FAISS.from_documents(chunks, embeddings)
print(f"Vector store created successfully and stored {vector_store.index.ntotal} vectors.")


# Step 3: Retrieval and Augmentation. This step sets up a retrieval-based question-answering system using the FAISS vector store and a language model.
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Initialize the Language Model (using gemini-2.5-flash for efficiency)
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
except Exception as e:
    print(f"Error initializing ChatGoogleGenerativeAI. Do you have your GEMINI_API_KEY set? Error: {e}")
    exit()
    
# Custom Prompt for RAG    
prompt= """
You are an intelligent, formal, and helpful assistant for a document retrieval system.
Use the following pieces of retrieved context to answer the user's question.
If the answer requires specific personal or schedule details (like office hours, contact info, or availability), provide them **formally and structuredly**, making sure to include **all** associated details like names, email addresses, phone numbers, and specific times.
If the question asks for availability at a specific time, respond by listing the available individuals/entities in an easy-to-read, structured format (e.g., bullet points or a table).
If you cannot find the answer in the context, clearly state that you do not have that information.

Context:
{context}

Question: {question}

Formal Answer:
"""
qa_prompt = PromptTemplate(
    template=prompt, 
    input_variables=["context", "question"]
)

# Document Formatter
def format_docs(retrieved_docs):
    """Merges the page content of retrieved documents into a single string."""
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

# Parallel Chain (Combining Retriever and Question). This chain runs the retriever (piped to format_docs) and the user question in parallel.
parallel_chain = RunnableParallel({
    # 'context': retriever retrieves documents, which are then formatted into a single string.
    'context': retriever | RunnableLambda(format_docs),
    # 'question': The input (user query) is passed directly through.
    'question': RunnablePassthrough()
})


# Step 4: Final RAG Chain. This chain combines the parallel retrieval/question chain with the LLM to generate answers based on retrieved context.
# Flow: Input -> Parallel Chain (Context + Question) -> Prompt (using both) -> LLM -> Output Parser
parser = StrOutputParser()
main_chain = parallel_chain | qa_prompt | llm | parser



## Step 5: Implement Chat Loop
# ---------------------------------
print("\n--- RAG Chat System Initialized (Powered by Gemini) ---")
print("Ask a question about the documents. Type 'exit' to quit.")
print("-" * 50)

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        print("Exiting chat. Goodbye! 👋")
        break
    
    if not user_input.strip():
        continue

    try:
        # Run the chain with the user input.
        # The entire chain expects the raw user input as the argument.
        response = main_chain.invoke(user_input)
        
        # Print the response
        print(f"\nAssistant: {response}")
        print("-" * 50)

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        print("-" * 50)


