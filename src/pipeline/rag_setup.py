import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

# --- Configuration ---
DATA_PATH = "data/unstructured_reports"
CHROMA_PATH = "chroma_db"
# ---

def load_documents():
    """Loads all text documents (.txt) from the specified data path."""
    print(f"Loading documents from: {DATA_PATH}")
    
    # Use TextLoader for each .txt file found in the directory
    documents = []
    for f in os.listdir(DATA_PATH):
        if f.endswith(".txt"):
            file_path = os.path.join(DATA_PATH, f)
            loader = TextLoader(file_path, encoding='utf-8')
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Could not load {file_path}: {e}")
                
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_text(documents):
    """Splits documents into smaller, overlapping chunks."""
    # RecursiveCharacterTextSplitter is recommended as it tries multiple separators 
    # (like \n\n, \n, space) to maintain meaningful chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # Max size of each chunk
        chunk_overlap=50,   # Overlap between chunks to maintain context
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} text chunks.")
    return chunks

def create_vector_store():
    """Initializes and populates the Chroma Vector Store."""
    print("\n--- Starting RAG Setup Process ---")
    
    # 1. Load and Split
    documents = load_documents()
    chunks = split_text(documents)

    # 2. Initialize Embeddings
    # NOTE: This requires the OPENAI_API_KEY to be set in your environment.
    embeddings = OpenAIEmbeddings()

    # 3. Create and Persist Vector Store
    print(f"Creating and persisting ChromaDB to '{CHROMA_PATH}'...")
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    # The persist call saves the index and files to the chroma_db directory
    vector_store.persist() 
    
    print(f"RAG Indexing Complete! Vector store saved to '{CHROMA_PATH}'")
    return vector_store

if __name__ == "__main__":
    # When you run this file, it will generate the 'chroma_db' folder
    create_vector_store()