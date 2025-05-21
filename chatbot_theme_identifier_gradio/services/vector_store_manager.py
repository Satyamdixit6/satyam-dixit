# services/vector_store_manager.py
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import logging
import shutil

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from langchain_core.documents import Document as LangchainCoreDocument
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore # Added VectorStore

from core.settings import settings # Adjusted import path

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, force_new_embeddings: bool = False): # Added flag for re-embedding
        self.persist_directory = Path(settings.VECTOR_STORE_PATH)
        self.chroma_add_batch_size = settings.CHROMA_ADD_BATCH_SIZE
        self.embedding_function = self._initialize_embedding_function()

        self.vector_store: Optional[VectorStore] = None
        if force_new_embeddings:
            logger.warning("force_new_embeddings is True. Deleting existing vector store if present.")
            self.delete_vector_store() # Delete existing store to ensure fresh embeddings

        self._ensure_vector_store_initialized() # Load or initialize on startup
        logger.info(f"VectorStoreManager initialized. DB path: {self.persist_directory}")

    def _initialize_embedding_function(self):
        if settings.EMBEDDING_PROVIDER == "local_hf":
            logger.info(f"VectorStore: Using local HuggingFace embeddings: {settings.LOCAL_HF_EMBEDDING_MODEL_NAME}")
            # Determine device: prioritize CUDA, fallback to CPU if issues or not available
            device = "cuda"
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU for HuggingFace embeddings.")
                    device = "cpu"
            except ImportError:
                logger.warning("PyTorch not found, HuggingFaceEmbeddings will likely default to CPU or fail if CUDA was intended.")
                device = "cpu" # Assuming CPU if torch isn't even there (though it's a dependency)

            # Check for an override in settings (e.g., add EMBEDDING_DEVICE="cpu" to .env)
            embedding_device_override = getattr(settings, 'EMBEDDING_DEVICE', None)
            if embedding_device_override:
                device = embedding_device_override
                logger.info(f"Overriding embedding device to: {device} based on settings.")

            if device == "cpu": # Explicitly inform if using CPU
                 logger.info("HuggingFaceEmbeddings will run on CPU.")

            return HuggingFaceEmbeddings(
                model_name=settings.LOCAL_HF_EMBEDDING_MODEL_NAME,
                encode_kwargs={'normalize_embeddings': True},
                model_kwargs={'device': device} # Explicitly set the device
            )
        elif settings.EMBEDDING_PROVIDER == "openai_api":
            if not settings.OPENAI_EMBEDDING_API_KEY:
                logger.error("OPENAI_EMBEDDING_API_KEY must be set for OpenAI embeddings.")
                raise ValueError("OPENAI_EMBEDDING_API_KEY is not set.")
            logger.info(f"VectorStore: Using OpenAI embeddings: {settings.OPENAI_EMBEDDING_MODEL_NAME}")
            return LangchainOpenAIEmbeddings(
                openai_api_key=settings.OPENAI_EMBEDDING_API_KEY,
                model=settings.OPENAI_EMBEDDING_MODEL_NAME
            )
        else:
            raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")

    def _get_vector_store_path_for_index(self, index_name: str = "default_index") -> str:
        return str(self.persist_directory / index_name)

    def _load_or_initialize_store(self, index_name: str = "default_index") -> VectorStore:
        db_path = self._get_vector_store_path_for_index(index_name)

        if Path(db_path).exists() and any(Path(db_path).iterdir()):
            try:
                logger.info(f"Loading existing Chroma vector store from: {db_path} for index '{index_name}'")
                return Chroma(
                    persist_directory=db_path,
                    embedding_function=self.embedding_function,
                    collection_name=index_name # Important for loading specific collection
                )
            except Exception as e:
                logger.error(f"Error loading vector store from {db_path} for index '{index_name}': {e}. Re-initializing.", exc_info=True)
                # If loading fails, delete potentially corrupt dir and re-initialize
                try:
                    shutil.rmtree(db_path)
                    logger.info(f"Removed potentially corrupt store at {db_path} for re-initialization.")
                except OSError as ose:
                    logger.error(f"Could not remove corrupt store directory {db_path}: {ose}")
                    raise # Re-raise if cleanup fails, as it's a critical state

        logger.info(f"No existing or loadable vector store at {db_path} for index '{index_name}'. Initializing new empty store.")
        return Chroma(
            persist_directory=db_path,
            embedding_function=self.embedding_function,
            collection_name=index_name
        )

    def _ensure_vector_store_initialized(self, index_name: str = "default_index") -> VectorStore:
        if self.vector_store is None:
            self.vector_store = self._load_or_initialize_store(index_name)
        return self.vector_store

    def add_documents_to_store(self,
                               langchain_documents: List[LangchainCoreDocument],
                               original_doc_id: str, # Added for logging
                               original_file_name: str, # Added for logging
                               index_name: str = "default_index"):
        if not langchain_documents:
            logger.warning(f"No Langchain documents provided to add for '{original_file_name}' (ID: {original_doc_id}).")
            return

        current_vector_store = self._ensure_vector_store_initialized(index_name)

        total_added_count = 0
        num_docs_to_add = len(langchain_documents)
        logger.info(f"Preparing to add {num_docs_to_add} chunks from '{original_file_name}' (ID: {original_doc_id}) in batches of {self.chroma_add_batch_size}.")

        for i in range(0, num_docs_to_add, self.chroma_add_batch_size):
            batch = langchain_documents[i:i + self.chroma_add_batch_size]
            num_in_batch = len(batch)
            current_batch_num = i // self.chroma_add_batch_size + 1
            total_batches = (num_docs_to_add + self.chroma_add_batch_size - 1) // self.chroma_add_batch_size

            logger.info(f"Adding batch {current_batch_num}/{total_batches} ({num_in_batch} chunks) for '{original_file_name}'.")
            try:
                added_ids = current_vector_store.add_documents(documents=batch)
                if added_ids and isinstance(added_ids, list):
                    total_added_count += len(added_ids)
                # Chroma with persist_directory should auto-persist.
            except Exception as e:
                 logger.error(f"Error adding batch {current_batch_num} for '{original_file_name}': {e}", exc_info=True)
                 raise ValueError(f"Failed to add document batch to vector store for '{original_file_name}': {e}")

        logger.info(f"Successfully added {total_added_count} chunks from '{original_file_name}' (ID: {original_doc_id}) to vector store '{index_name}'.")

    def similarity_search(self, query: str, k: int = 5, index_name: str = "default_index", filter_criteria: Optional[Dict[str, Any]] = None) -> List[LangchainCoreDocument]: # Returns docs only for RAG chain
        current_vector_store = self._ensure_vector_store_initialized(index_name)
        try:
            search_kwargs = {}
            if filter_criteria:
                search_kwargs['filter'] = filter_criteria
                logger.debug(f"Performing similarity search with filter: {filter_criteria}")

            results_with_scores = current_vector_store.similarity_search_with_score(query, k=k, **search_kwargs)
            logger.info(f"Found {len(results_with_scores)} relevant chunks for query '{query[:50]}...'.")
            return [doc for doc, score in results_with_scores] # Extract just the documents
        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=True)
            return []

    def get_retriever(self, k: int = 5, score_threshold: Optional[float] = None, filter_criteria: Optional[Dict[str, Any]] = None, index_name: str = "default_index") -> VectorStoreRetriever:
        current_vector_store = self._ensure_vector_store_initialized(index_name)

        search_kwargs = {'k': k}
        if filter_criteria:
            search_kwargs['filter'] = filter_criteria

        search_type = "similarity"
        if score_threshold is not None:
            search_type = "similarity_score_threshold" # Chroma specific search_type for this
            search_kwargs['score_threshold'] = score_threshold

        logger.debug(f"Configuring retriever for index '{index_name}' with search_type='{search_type}', search_kwargs={search_kwargs}")
        return current_vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def delete_vector_store(self, index_name: str = "default_index"):
        db_path = Path(self._get_vector_store_path_for_index(index_name))
        if db_path.exists() and db_path.is_dir():
            try:
                shutil.rmtree(db_path)
                logger.info(f"Successfully deleted vector store index '{index_name}' from {db_path}")
                if self.vector_store: # Check if an instance exists
                    if index_name == "default_index": # Simplistic check
                        self.vector_store = None
            except OSError as e:
                logger.error(f"Error deleting vector store index '{index_name}' from {db_path}: {e}", exc_info=True)
                raise ValueError(f"Could not delete vector store: {e}")
        else:
            logger.info(f"Vector store index '{index_name}' not found at {db_path}, nothing to delete.")

    def get_document_count(self, index_name: str = "default_index") -> int:
        """Returns the approximate number of documents/chunks in the store."""
        try:
            store = self._ensure_vector_store_initialized(index_name)
            if hasattr(store, '_collection'): # Chroma specific
                return store._collection.count()
            logger.warning("get_document_count not fully implemented for this vector store type beyond Chroma.")
            return 0 # Fallback
        except Exception as e:
            logger.error(f"Error getting document count for index '{index_name}': {e}")
            return 0