# Access environment variables (os.getenv()). Work with file paths. Interact with the system
import os

# load documents into LangChain                # single text file , # multiple files from a folder.
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# splits large text into smaller chunks
from langchain_text_splitters import CharacterTextSplitter

# Imports OpenAI’s embedding model through LangChain
from langchain_openai import OpenAIEmbeddings

# Imports the Chroma vector database integration
from langchain_chroma import Chroma


from dotenv import load_dotenv

load_dotenv()

### Document Loading Function
def load_documents(docs_path="docs"):
    """Load all text files from a docs directory."""
    print(f"Loading documents from {docs_path}...")


    # Check if docs directory exists or not
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} does not exist.")
    
    loader = DirectoryLoader(
        docs_path, 
        # glob="**/*.txt",  # It search subfolder also
        glob = "*.txt",  # It search only txt file
        loader_cls=lambda path: TextLoader(path, encoding="utf-8")
    )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No .txt files found in {docs_path}.")

    for i, doc in enumerate(documents[:2]):
        print(f"Document {i+1}: ")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")  # Print first 100 characters
        print(f"  metadata: {doc.metadata}")
    return documents


### Chunks Making Function
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks with overlap."""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
         chunk_size=chunk_size, 
         chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n----- Chunk {i+1} -----")
            print(f"  Source: {chunk.metadata['source']}")
            print(f"  Chunk length: {len(chunk.page_content)} characters")
            print(f" Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"... and {len(chunks) - 5} more chunks.")
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embedding and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small") # dimension can be up and down based on your needs and cost considerations

    # Create ChromaDB vector store
    print("---- Creating vector store ----")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity for HNSW index
    )
    print("---- Finished creating vector store ----")
    print(f"vector store created and persisted at {persist_directory}")
    return vectorstore

def main():
    print("Main Function Started")

    documents = load_documents(docs_path="docs")
    
    chunks = split_documents(documents)

    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()
# main()