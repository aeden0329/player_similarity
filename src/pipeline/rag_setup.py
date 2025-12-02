import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings # Using OpenAI for embeddings
from dotenv import load_dotenv

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

DATA_PATH = "data/unstructured_reports"
CHROMA_PATH = "chroma_db"

def load_documents():
    """Loads all text documents from the specified data path."""
    loaders = [
        TextLoader(os.path.join(DATA_PATH, f), encoding='utf-8')
        for f in os.listdir(DATA_PATH)
        if f.endswith(".txt")
    ]
    # Load all documents from the list of loaders
    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Could not load {loader.file_path}: {e}")
            
    return documents

def split_text(documents):
    """Splits documents into smaller chunks for RAG."""
    # Since our reports are short, we use a simple splitter to ensure context integrity.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def create_vector_store():
    """Initializes and populates the Chroma Vector Store."""
    print("Starting RAG setup: Loading and splitting documents...")
    documents = load_documents()
    chunks = split_text(documents)

    # Initialize the embedding function
    # NOTE: Ensure OPENAI_API_KEY is set in your environment or a .env file.
    embeddings = OpenAIEmbeddings()

    print(f"Creating {CHROMA_PATH} with {len(chunks)} chunks...")
    
    # Create the vector store from chunks and persist it to disk
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    vector_store.persist()
    print(f"✅ Vector store created and saved to '{CHROMA_PATH}'")
    return vector_store

if __name__ == "__main__":
    # When this script is run, it will generate the 'chroma_db' folder
    create_vector_store()