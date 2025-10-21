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

def load_and_chunk_documents(directory_path="./documents", chunk_size=1200, chunk_overlap=300):
    try:
        pdf_loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=UnstructuredPDFLoader,
            show_progress=True
        )
        word_loader = DirectoryLoader(
            directory_path,
            glob="**/*.docx",
            loader_cls=UnstructuredWordDocumentLoader,
            show_progress=True
        )
        
        pdf_documents = pdf_loader.load()
        word_documents = word_loader.load()
        
    except Exception as e:
        print(f"Error loading documents from '{directory_path}'. Ensure the directory exists and files are accessible.")
        print(f"Details: {e}")
        return []

    all_documents = pdf_documents + word_documents
    
    print(f"Loaded {len(pdf_documents)} PDF documents")
    print(f"Loaded {len(word_documents)} Word documents")
    print(f"Total documents: {len(all_documents)}")
    
    if not all_documents:
        print("No documents found or loaded. Chunking skipped.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(all_documents)
    
    print(f"Created {len(chunks)} chunks")
    
    return chunks

# Step 2: Embedding and Vector Store Creation
chunks = load_and_chunk_documents()

if not chunks:
    exit()

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

vector_store = FAISS.from_documents(chunks, embeddings)
print(f"Vector store created successfully and stored {vector_store.index.ntotal} vectors.")

# Step 3: Retrieval and Augmentation
# Increased k to 8 for better retrieval coverage.
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
except Exception as e:
    print(f"Error initializing ChatGoogleGenerativeAI. Do you have your GEMINI_API_KEY set? Error: {e}")
    exit()
    
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

# Step 4: Final RAG Chain
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

main_chain = parallel_chain | qa_prompt | llm | parser

# Step 5: Implement Chat Loop
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
        response = main_chain.invoke(user_input)
        
        print(f"\nAssistant: {response}")
        print("-" * 50)

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        print("-" * 50)