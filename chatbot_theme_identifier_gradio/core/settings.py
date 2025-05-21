# core/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ValidationError  # Import for catching Pydantic errors
from pathlib import Path
from typing import Literal, Optional
import sys  # For exiting if settings fail critically
import os  # For checking .env file existence

# Determine project root dynamically
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Pre-check for .env file ---
env_file_path = PROJECT_ROOT_DIR / ".env"
if not os.path.exists(env_file_path):
    print(f"WARNING (core/settings.py): .env file not found at expected location: {env_file_path}", file=sys.stderr)
    print(f"INFO (core/settings.py): Proceeding with default values or other environment variables.", file=sys.stderr)
# --- End Pre-check ---

class AppSettings(BaseSettings):
    APP_NAME: str = "Document Research & Theme Chatbot (Gradio Edition)"
    APP_VERSION: str = "0.1.0-gradio"
    LOG_LEVEL: str = "INFO"  # Default log level

    LLM_PROVIDER: Literal["local_llama_server", "openai_api"] = "local_llama_server"
    LOCAL_LLM_API_BASE_URL: str = "http://localhost:8080/v1"
    LOCAL_LLM_MODEL_NAME: str = "local-model/gguf-model-name"  # User needs to set this in .env or here
    LOCAL_LLM_API_KEY: str = "not-needed"
    
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_LLM_MODEL_NAME: str = "gpt-3.5-turbo"

    EMBEDDING_PROVIDER: Literal["local_hf", "openai_api"] = "local_hf"
    LOCAL_HF_EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    OPENAI_EMBEDDING_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    UNSTRUCTURED_PDF_MODE: Literal["paged", "elements", "single"] = "paged"
    UNSTRUCTURED_TEXT_MODE: Literal["paged", "elements", "single"] = "single"
    OCR_ENABLED: bool = True
    OCR_LANGUAGE: str = 'eng'

    _vector_store_path_segment: str = "data/vector_store"
    _document_dir_segment: str = "data/uploaded_documents"

    @property
    def VECTOR_STORE_PATH(self) -> str:
        return str(PROJECT_ROOT_DIR / self._vector_store_path_segment)

    @property
    def DOCUMENT_DIR(self) -> str:
        return str(PROJECT_ROOT_DIR / self._document_dir_segment)

    CHROMA_ADD_BATCH_SIZE: int = 1000

    GRADIO_SERVER_NAME: str = "0.0.0.0"
    GRADIO_SERVER_PORT: int = 7860
    GRADIO_SHARE: bool = False
    GRADIO_MAX_FILE_UPLOADS: int = 20  # Gradio UI might limit this per interaction

    model_config = SettingsConfigDict(
        env_file=str(env_file_path),  # Use the checked path
        extra='ignore',
        case_sensitive=False
    )

# Global variable to hold the settings instance
settings: Optional[AppSettings] = None  # Initialize as None

try:
    print(f"INFO (core/settings.py): Attempting to load AppSettings... Env file target: {env_file_path}", file=sys.stderr)
    settings_instance = AppSettings()  # Attempt instantiation
    settings = settings_instance  # Assign to global 'settings' only if successful
    print(f"INFO (core/settings.py): AppSettings loaded successfully. LOG_LEVEL set to: {settings.LOG_LEVEL}", file=sys.stderr)

    # This part will only run if 'settings' was successfully created.
    try:
        Path(settings.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
        Path(settings.DOCUMENT_DIR).mkdir(parents=True, exist_ok=True)
        print(f"INFO (core/settings.py): Data directories ensured/created.", file=sys.stderr)
    except Exception as e:
        print(f"WARNING (core/settings.py): Could not create data directories specified in settings: {e}", file=sys.stderr)

except ValidationError as e:
    print(f"FATAL ERROR (core/settings.py): Settings validation failed. Check your .env file (path: {env_file_path}) or default values in AppSettings for type mismatches or missing required fields: \n{str(e)}", file=sys.stderr)
    sys.exit(1)  # Exit immediately
except Exception as e:  # Catch any other unexpected error during settings instantiation
    print(f"FATAL ERROR (core/settings.py): An unexpected error occurred while initializing AppSettings: {e}", file=sys.stderr)
    sys.exit(1)  # Exit immediately

# Safeguard check
if settings is None:
    print("FATAL ERROR (core/settings.py): 'settings' object is None after attempting instantiation. This should not happen if error handling above worked. Exiting.", file=sys.stderr)
    sys.exit(1)