# RAG ETL Pipeline using unstructured, FAISS, and Hybrid Search

import os
from typing import List, Tuple
from unstructured.partition.auto import partition
from unstructured.documents.elements import Table
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import json
from rank_bm25 import BM25Okapi

# Load and partition document using sections and extract tables

def load_and_partition(file_path: str):
    elements = partition(filename=file_path, strategy="hi_res")
    content_elements = []
    for el in elements:
        if isinstance(el, Table):
            content_elements.append(el.text.strip())
        elif el.text and el.category not in ("Figure"):
            content_elements.append(el)
    return elements

# Chunking by title sections and then by semantic split with metadata tracking

def chunk_elements_by_title_semantic(elements, max_chars=512, overlap=50):
    chunks = []
    metadata = []
    current_section = []
    current_title = "Introduction"
    current_page = None

    for el in elements:
        if hasattr(el, "category") and "title" in el.category.lower():
            if current_section:
                section_chunks = semantic_chunk(current_section, max_chars, overlap)
                for chunk in section_chunks:
                    metadata.append({
                        "section_title": current_title,
                        "page_number": current_page,
                        "element_type": "text"
                    })
                    chunks.append(chunk)
                current_section = []
            current_title = el.text.strip()
            current_page = getattr(el.metadata, "page_number", None)

        if hasattr(el, "text"):
            current_section.append(el.text.strip())
            if current_page is None:
                current_page = getattr(el.metadata, "page_number", None)
        elif isinstance(el, str):
            current_section.append(el)

    if current_section:
        section_chunks = semantic_chunk(current_section, max_chars, overlap)
        for chunk in section_chunks:
            metadata.append({
                "section_title": current_title,
                "page_number": current_page,
                "element_type": "text"
            })
            chunks.append(chunk)

    return chunks, metadata

def semantic_chunk(texts: List[str], max_chars: int, overlap: int):
    full_text = "\n".join(texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=overlap)
    return splitter.split_text(full_text)

# Embed chunks

def embed_chunks(chunks: List[str], model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings, model

# Store embeddings in FAISS with metadata

def store_faiss_index(embeddings, chunks, metadata, index_path="faiss.index", meta_path="metadata.json"):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "metadata": metadata}, f)

    return index

# Build BM25 index for hybrid search

def build_bm25_index(chunks: List[str]):
    tokenized_corpus = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized_corpus)

# Hybrid Search (BM25 + FAISS)

def hybrid_search(query: str, model, faiss_index, bm25, chunks: List[str], top_k=5):
    # FAISS semantic search
    q_vec = model.encode([query])
    D, I = faiss_index.search(np.array(q_vec).astype('float32'), top_k)
    semantic_hits = [chunks[i] for i in I[0]]

    # BM25 keyword search
    bm25_hits = bm25.get_top_n(query.split(), chunks, n=top_k)

    # Combine and rerank (naive merge)
    combined = list(dict.fromkeys(semantic_hits + bm25_hits))[:top_k]
    return combined

# === Example Usage ===
if __name__ == "__main__":
    file_path = "sample.docx"  # or .pdf
    elements = load_and_partition(file_path)
    chunks, metadata = chunk_elements_by_title_semantic(elements)
    embeddings, model = embed_chunks(chunks)
    index = store_faiss_index(embeddings, chunks, metadata)
    bm25 = build_bm25_index(chunks)

    # Test hybrid query
    results = hybrid_search("What are the side effects?", model, index, bm25, chunks)
    for i, res in enumerate(results, 1):
        print(f"[{i}] {res[:200]}\n")
