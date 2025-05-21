# services/document_processor.py
import os
import uuid
from pathlib import Path
from typing import List, Tuple, Optional # Added Optional

from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainCoreDocument

from core.settings import settings # Adjusted import path

import logging
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS_UNSTRUCTURED = [".pdf", ".txt", ".md", ".docx", ".doc", ".pptx", ".html", ".eml", ".msg"]
SUPPORTED_EXTENSIONS_OCR_IMAGES = [".png", ".jpeg", ".jpg", ".tiff", ".bmp"]

class DocumentProcessorService:
    def __init__(self):
        self.document_dir = Path(settings.DOCUMENT_DIR)
        self.document_dir.mkdir(parents=True, exist_ok=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        self.current_doc_total_pages: Optional[int] = None # To store page count temporarily
        logger.info(f"DocumentProcessorService initialized. Doc dir: {self.document_dir}, Chunk size: {settings.CHUNK_SIZE}, Overlap: {settings.CHUNK_OVERLAP}")

    def save_temp_uploaded_file(self, temp_file_obj) -> Tuple[Path, str]:
        """
        Saves a temporary file object (from Gradio upload) to the configured document directory.
        Gradio's File component provides a temp file path. We'll copy it to our persistent storage.
        Returns the path to the saved file and the original filename.
        """
        original_filename = getattr(temp_file_obj, 'name', None) # Gradio file object has a .name attribute
        if not original_filename:
            original_filename = f"unknown_file_{uuid.uuid4().hex[:6]}"
            logger.warning(f"Uploaded file object missing 'name' attribute. Using generated name: {original_filename}")
        
        temp_file_path = Path(temp_file_obj.name) # Path to Gradio's temporary file

        safe_basename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in Path(original_filename).stem)
        safe_extension = Path(original_filename).suffix.lower()
        unique_persistent_filename = f"{uuid.uuid4()}_{safe_basename}{safe_extension}"
        persistent_file_path = self.document_dir / unique_persistent_filename
        
        try:
            import shutil
            shutil.copyfile(temp_file_path, persistent_file_path)
            logger.info(f"File '{original_filename}' (from temp: {temp_file_path}) saved as '{unique_persistent_filename}' to {persistent_file_path}")
            return persistent_file_path, original_filename
        except Exception as e:
            logger.error(f"Error copying temp file {temp_file_path} to {persistent_file_path} for '{original_filename}': {e}", exc_info=True)
            raise IOError(f"Could not save uploaded file '{original_filename}': {e}") # Raise standard IOError

    def _determine_unstructured_mode(self, file_extension: str) -> str:
        if file_extension == ".pdf":
            return settings.UNSTRUCTURED_PDF_MODE
        elif file_extension in [".txt", ".md", ".html"]:
            return settings.UNSTRUCTURED_TEXT_MODE
        return "elements"

    def extract_and_chunk_document(self, file_path: Path, original_filename: str) -> Tuple[List[LangchainCoreDocument], Optional[int]]:
        """
        Loads, OCRs (if applicable via Unstructured), and chunks a document.
        Returns LangchainCoreDocument chunks and the number of pages processed.
        """
        file_extension = file_path.suffix.lower()
        
        if file_extension not in SUPPORTED_EXTENSIONS_UNSTRUCTURED and file_extension not in SUPPORTED_EXTENSIONS_OCR_IMAGES:
            logger.warning(f"File type '{file_extension}' for '{original_filename}' is not explicitly handled. Unstructured will attempt generic load.")

        langchain_core_documents_raw: List[LangchainCoreDocument] = []
        try:
            mode = self._determine_unstructured_mode(file_extension)
            loader_kwargs = {"mode": mode}
            if settings.OCR_ENABLED and (file_extension == ".pdf" or file_extension in SUPPORTED_EXTENSIONS_OCR_IMAGES):
                # Unstructured with `local-inference` extras should attempt OCR if tesseract/poppler are installed.
                # You can force a strategy like "hi_res" or "ocr_only" if needed.
                # loader_kwargs['strategy'] = "hi_res" 
                logger.info(f"OCR may be applied for '{original_filename}' using Unstructured (mode: {mode}).")

            loader = UnstructuredLoader(str(file_path), **loader_kwargs)
            langchain_core_documents_raw = loader.load()

            if not langchain_core_documents_raw:
                logger.warning(f"No content elements loaded from '{original_filename}' by UnstructuredLoader.")
                return [], 0
            
            logger.info(f"UnstructuredLoader (mode: {mode}) returned {len(langchain_core_documents_raw)} raw element(s) for '{original_filename}'. Splitting...")
            
            split_langchain_core_documents = self.text_splitter.split_documents(langchain_core_documents_raw)
            
            # Determine number of pages processed (best effort from metadata)
            total_pages_processed_set = set()
            for lc_doc_raw in langchain_core_documents_raw: # Check raw docs from loader for page count
                page_num_val = lc_doc_raw.metadata.get("page_number", lc_doc_raw.metadata.get("page"))
                if page_num_val is not None:
                    try:
                        total_pages_processed_set.add(int(page_num_val))
                    except ValueError:
                        pass
            
            num_pages = len(total_pages_processed_set) if total_pages_processed_set else None
            if num_pages is None and len(langchain_core_documents_raw) > 0 and mode == "paged":
                num_pages = len(langchain_core_documents_raw)


            # Standardize metadata for final chunks
            final_chunks_with_meta: List[LangchainCoreDocument] = []
            for i, lc_chunk in enumerate(split_langchain_core_documents):
                chunk_metadata_dict = lc_chunk.metadata or {}
                page_num = chunk_metadata_dict.get("page_number", chunk_metadata_dict.get("page"))
                try:
                    page_num = int(page_num) if page_num is not None else None
                except ValueError:
                    page_num = None
                
                final_metadata = {
                    "source_file": original_filename, # Original filename
                    "processed_file_path": str(file_path), # Path to the uniquely named saved file
                    "page_number": page_num,
                    "chunk_index_raw_element": chunk_metadata_dict.get("element_id", i), # From Unstructured if available
                    "doc_id_chunk": str(uuid.uuid4()) # Unique ID for this specific chunk
                }
                final_chunks_with_meta.append(
                    LangchainCoreDocument(page_content=lc_chunk.page_content, metadata=final_metadata)
                )

            logger.info(f"Processed '{original_filename}' into {len(final_chunks_with_meta)} final chunks. Approx. {num_pages or 'N/A'} pages.")
            return final_chunks_with_meta, num_pages

        except ImportError as ie:
            logger.error(f"ImportError during document processing for '{original_filename}': {ie}. Dependencies missing?", exc_info=True)
            raise ValueError(f"Missing dependency for processing '{original_filename}': {ie}.")
        except Exception as e:
            logger.error(f"Error loading/splitting document '{original_filename}' (path: {file_path}): {e}", exc_info=True)
            if "tesseract" in str(e).lower() or "poppler" in str(e).lower() or "gs" in str(e).lower():
                 raise ValueError(f"External tool error (Tesseract/Poppler/Ghostscript) for '{original_filename}': {e}.")
            raise ValueError(f"Error processing document '{original_filename}': {e}")

    def process_uploaded_file_object(self, temp_file_obj) -> Tuple[str, str, List[LangchainCoreDocument], int, Optional[int]]:
        """
        High-level method for Gradio: saves temp file, extracts, chunks.
        Returns: (persistent_doc_id, original_filename, chunks, num_chunks, num_pages)
        """
        persistent_file_path, original_filename = self.save_temp_uploaded_file(temp_file_obj)
        
        # doc_id for the processed document can be derived from the unique saved filename's stem
        persistent_doc_id = persistent_file_path.stem 

        # CPU-bound, Gradio runs this in a thread by default for regular functions.
        # If this becomes very slow, advanced Gradio might need async/threading considerations.
        langchain_core_document_chunks, num_pages = self.extract_and_chunk_document(
            file_path=persistent_file_path, original_filename=original_filename
        )

        if not langchain_core_document_chunks:
            try:
                os.remove(persistent_file_path)
                logger.info(f"Cleaned up file '{original_filename}' with no processable content: {persistent_file_path}")
            except OSError as e:
                logger.error(f"Error removing file {persistent_file_path} after failed processing: {e}")
            raise ValueError(f"No processable text content found in file '{original_filename}'.")

        # Add original_doc_id (persistent_doc_id) to each chunk's metadata *before* sending to vector store
        for chunk in langchain_core_document_chunks:
            chunk.metadata["original_doc_id"] = persistent_doc_id
            # original_file_name is already set as source_file in chunk metadata

        return persistent_doc_id, original_filename, langchain_core_document_chunks, len(langchain_core_document_chunks), num_pages
