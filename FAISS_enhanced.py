from pathlib import Path

from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.helpers.data import docs_to_data
from langflow.io import BoolInput, HandleInput, IntInput, StrInput
from langflow.schema import Data


class FaissVectorStoreComponent(LCVectorStoreComponent):
    """FAISS Vector Store with search capabilities."""

    display_name: str = "FAISS"
    description: str = "FAISS Vector Store with search capabilities"
    name = "FAISS"
    icon = "FAISS"

    inputs = [
        StrInput(
            name="index_name",
            display_name="Index Name",
            value="langflow_index",
        ),
        StrInput(
            name="persist_directory",
            display_name="Persist Directory",
            info="Path to save the FAISS index. It will be relative to where Langflow is running.",
        ),
        *LCVectorStoreComponent.inputs,
        BoolInput(
            name="allow_dangerous_deserialization",
            display_name="Allow Dangerous Deserialization",
            info="Set to True to allow loading pickle files from untrusted sources. "
            "Only enable this if you trust the source of the data.",
            advanced=True,
            value=True,
        ),
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"]),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            advanced=True,
            value=4,
        ),
        BoolInput(
            name="use_reranking",
            display_name="Use Cross-Encoder Reranking",
            info="Use Cross-Encoder to rerank results for better accuracy.",
            advanced=True,
            value=False,
        ),
        StrInput(
            name="cross_encoder_model",
            display_name="Cross-Encoder Model",
            info="Cross-Encoder model to use for reranking.",
            advanced=True,
            value="cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        IntInput(
            name="initial_candidates",
            display_name="Initial Candidates",
            info="Number of initial candidates to retrieve before reranking.",
            advanced=True,
            value=20,
        ),
    ]

    # ...existing code...

    @staticmethod
    def resolve_path(path: str) -> str:
        """Resolve the path relative to the Langflow root.

        Args:
            path: The path to resolve
        Returns:
            str: The resolved path as a string
        """
        return str(Path(path).resolve())
    
    def get_persist_directory(self) -> Path:
        """Returns the resolved persist directory path or the current directory if not set."""
        if self.persist_directory:
            return Path(self.resolve_path(self.persist_directory))
        return Path()
    
    @check_cached_vector_store
    def build_vector_store(self) -> FAISS:
        """Builds the FAISS vector store."""
        path = self.get_persist_directory()
        path.mkdir(parents=True, exist_ok=True)

        # Convert DataFrame to Data if needed using parent's method
        self.ingest_data = self._prepare_ingest_data()

        documents = []
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                documents.append(_input)

        faiss = FAISS.from_documents(documents=documents, embedding=self.embedding)
        faiss.save_local(str(path), self.index_name)
        return faiss

    def search_documents(self) -> list[Data]:
        """Search for documents in the FAISS vector store."""
        path = self.get_persist_directory()
        index_path = path / f"{self.index_name}.faiss"

        if not index_path.exists():
            vector_store = self.build_vector_store()
        else:
            vector_store = FAISS.load_local(
                folder_path=str(path),
                embeddings=self.embedding,
                index_name=self.index_name,
                allow_dangerous_deserialization=self.allow_dangerous_deserialization,
            )

        if not vector_store:
            msg = "Failed to load the FAISS index."
            raise ValueError(msg)

        if self.search_query and isinstance(self.search_query, str) and self.search_query.strip():
            if not getattr(self, 'use_reranking', False):
                # Standard similarity search without reranking
                docs = vector_store.similarity_search(
                    query=self.search_query,
                    k=self.number_of_results,
                )
                return docs_to_data(docs)
            else:
                # With Cross-Encoder reranking
                # First retrieve more candidates than needed
                initial_candidates = getattr(self, 'initial_candidates', 20)
                docs = vector_store.similarity_search(
                    query=self.search_query,
                    k=initial_candidates,
                )
                
                # Apply Cross-Encoder reranking
                try:
                    model_name = getattr(self, 'cross_encoder_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                    cross_encoder = CrossEncoder(model_name)
                    
                    # Create query-document pairs for reranking
                    doc_pairs = [[self.search_query, doc.page_content] for doc in docs]
                    
                    # Get relevance scores
                    scores = cross_encoder.predict(doc_pairs)
                    
                    # Sort documents by relevance score
                    ranked_results = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
                    
                    # Take the top k results
                    reranked_docs = [doc for _, doc in ranked_results[:self.number_of_results]]
                    
                    return docs_to_data(reranked_docs)
                    
                except Exception as e:
                    # Fallback to standard search if reranking fails
                    print(f"Cross-encoder reranking failed: {e}. Falling back to standard search.")
                    return docs_to_data(docs[:self.number_of_results])
        return []