from langchain.document_loaders import TextLoader
from langchain.text_splitter import SpacyTextSplitter, TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from pathlib import Path

# ----------------------------
# 1. Design logic:
# ----------------------------
# - Load documents from source (e.g. text files, PDFs).
# - First apply a semantic text splitter (Spacy) to break into coherent sentence-based chunks.
# - Then apply a token-based text splitter to ensure each chunk is within model limits.
# - Compute embeddings for final chunks.
# - Store embeddings in FAISS vector store for fast nearest-neighbor retrieval.
# - Persist the index to disk for reuse.

# ----------------------------
# 2. Implementation
# ----------------------------

def build_faiss_index(
    docs_path: str,
    spacy_model: str = "en_core_web_sm",
    semantic_chunk_size: int = 1000,
    semantic_overlap: int = 100,
    token_chunk_size: int = 500,
    token_overlap: int = 50,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_path: str = "faiss_index"
) -> FAISS:
    # 2.1 Load raw documents
    loader = TextLoader(docs_path)
    docs = loader.load()

    # 2.2 Semantic splitting (sentence-level coherence)
    semantic_splitter = SpacyTextSplitter(
        model=spacy_model,
        chunk_size=semantic_chunk_size,
        chunk_overlap=semantic_overlap
    )
    sem_chunks = semantic_splitter.split_documents(docs)

    # 2.3 Token-based splitting (ensure token limits)
    token_splitter = TokenTextSplitter(
        chunk_size=token_chunk_size,
        chunk_overlap=token_overlap
    )
    final_chunks = token_splitter.split_documents(sem_chunks)

    # 2.4 Embed and index
    embed = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(final_chunks, embed)

    # 2.5 Persist
    Path(index_path).mkdir(exist_ok=True)
    faiss_index_file = Path(index_path) / "index.faiss"
    vectorstore.save_local(str(index_path))
    print(f"FAISS index saved to {index_path}")

    return vectorstore

# Example usage
if __name__ == "__main__":
    INDEX = build_faiss_index(
        docs_path="./data/my_docs.txt",
        spacy_model="en_core_web_sm",
        semantic_chunk_size=1000,
        semantic_overlap=100,
        token_chunk_size=500,
        token_overlap=50,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path="./faiss_index"
    )
    # Now you can perform retrieval for RAG:
    query = "What are the main benefits of AI-powered smart glasses?"
    docs = INDEX.similarity_search(query, k=5)
    for d in docs:
        print(d.page_content)
