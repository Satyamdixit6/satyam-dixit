# Chatbot Theme Identifier

This project is a Gradio-based application that allows users to upload documents, interact with them using a Retrieval Augmented Generation (RAG) chatbot, and identify key themes within the document corpus.

## Features

This application provides the following key features:

*   **Document Upload**: Users can upload documents in various formats (e.g., PDF, TXT, MD, DOCX). The system can perform OCR on scanned images within PDF documents if Tesseract and Poppler are installed.
*   **Conversational RAG (Retrieval Augmented Generation)**: Engage in a conversation with an AI-powered chatbot that uses the uploaded documents as a knowledge base. The chatbot provides answers to your questions and cites the sources from the documents.
*   **Theme Analysis**: Identify and extract key themes from the document corpus. Theme analysis can be performed on:
    *   A specific document.
    *   The context of a user query.
    *   All documents processed in the knowledge base.

## Setup and Installation

Follow these steps to set up and run the application:

1.  **Prerequisites**:
    *   Python 3.8 or higher.
    *   (Optional but Recommended for full PDF support) Tesseract OCR and Poppler:
        *   **Tesseract OCR**: Installation varies by OS. Refer to the [official Tesseract documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html).
        *   **Poppler**: Needed for PDF processing.
            *   On macOS (via Homebrew): `brew install poppler`
            *   On Debian/Ubuntu: `sudo apt-get install poppler-utils`
            *   On Windows: Download from [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/) and add the `bin/` directory to your PATH.

2.  **Clone the Repository**:
    (If you haven't already)
    ```bash
    git clone <repository_url>
    cd <repository_directory> 
    ```
    (Replace `<repository_url>` and `<repository_directory>` with the actual URL and directory name of the main project)

3.  **Navigate to Application Directory**:
    All subsequent commands assume you are in the `chatbot_theme_identifier_gradio` directory.
    ```bash
    cd chatbot_theme_identifier_gradio 
    ```
    (If you cloned the project and are in its root, you'll need to `cd` into this directory)

4.  **Create a Virtual Environment** (Recommended, inside `chatbot_theme_identifier_gradio`):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

5.  **Install Dependencies**:
    With the virtual environment activated and while inside the `chatbot_theme_identifier_gradio` directory:
    ```bash
    pip install -r requirements.txt
    ```

6.  **Environment Variables**:
    The application uses environment variables for configuration, managed via a `.env` file within the `chatbot_theme_identifier_gradio` directory.
    *   Rename `.env.example` (if provided) to `.env` or create a new `.env` file in this directory.
    *   Update the `.env` file with your specific configurations (API keys, model paths, etc.). Refer to `core/settings.py` for all possible variables.
    *   A typical minimal `.env` might look like this:
        ```env
        # LLM Configuration
        LLM_PROVIDER="local_llama_server" # or "openai"
        # OPENAI_API_KEY="your_openai_api_key_here" # If using openai
        # LOCAL_LLM_SERVER_URL="http://localhost:11434" # If using local Ollama server

        # Embedding Model Configuration
        EMBEDDING_PROVIDER="local_hf" # or "openai_api"
        # OPENAI_EMBEDDING_MODEL_NAME="text-embedding-ada-002" # If using openai for embeddings
        # LOCAL_HF_EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2" # If using local HuggingFace model

        # Data Paths (defaults are usually relative to chatbot_theme_identifier_gradio)
        # UPLOAD_DIR_PATH="data/uploaded_documents"
        # DB_DIR_PATH="data/vector_store"
        # LOG_FILE_PATH="logs/app.log"

        # Gradio Settings
        # GRADIO_SERVER_NAME="0.0.0.0"
        # GRADIO_SERVER_PORT=7860
        ```
    *   **Important**: The application, by default, might try to create `data/uploaded_documents/`, `data/vector_store/`, and `logs/` directories within the `chatbot_theme_identifier_gradio` directory if they don't exist (based on settings in `core/settings.py`). Ensure the application has write permissions for its working directory.

## Running the Application

Once the setup is complete:

1.  Ensure your virtual environment (created inside `chatbot_theme_identifier_gradio`) is activated.
2.  Make sure you are in the `chatbot_theme_identifier_gradio` directory.
3.  Run the Gradio application:
    ```bash
    python app.py
    ```
4.  The application will typically be available at `http://127.0.0.1:7860` (or the host/port specified in your settings). Open this URL in your web browser.

## Project Structure

Here's a brief overview of the key directories and files within `chatbot_theme_identifier_gradio/`:

```
chatbot_theme_identifier_gradio/
├── app.py                    # Main Gradio application script
├── core/                     # Core components
│   ├── __init__.py           # Makes 'core' a Python package
│   ├── settings.py           # Configuration (LLM endpoints, model names, etc.)
│   └── utils.py              # Utility functions
├── services/                 # Business logic modules
│   ├── __init__.py           # Makes 'services' a Python package
│   ├── document_processor.py # Handles file loading, OCR, chunking
│   ├── vector_store_manager.py# Manages ChromaDB and embeddings
│   ├── rag_chain_builder.py  # Builds and manages Langchain RAG chains
│   └── theme_service.py      # For theme identification logic
├── data/                     # Default location for application data (usually auto-created)
│   ├── uploaded_documents/   # Default storage for uploaded files
│   └── vector_store/         # Default storage for the vector database
├── static/                   # If Gradio needs any static assets like custom CSS (optional)
│   └── custom.css
├── README.md                 # This file: Detailed README for the Gradio application
└── requirements.txt          # Python package dependencies
```

*   **`app.py`**: The entry point for the Gradio web application. It defines the UI and handles user interactions.
*   **`core/`**: Contains the central configuration (`settings.py`), utilities (`utils.py`), and `__init__.py` to make it a Python package. `settings.py` is crucial for defining LLM endpoints, model names, and other operational parameters from environment variables.
*   **`services/`**: This directory houses the modules responsible for the application's main functionalities, along with an `__init__.py`.
    *   `document_processor.py`: Handles file loading, OCR, and text chunking.
    *   `vector_store_manager.py`: Manages the ChromaDB instance and text embeddings.
    *   `rag_chain_builder.py`: Constructs and manages Langchain RAG (Retrieval Augmented Generation) chains.
    *   `theme_service.py`: Contains the logic for theme identification.
*   **`data/`**: This directory is the default location for persistent application data.
    *   `uploaded_documents/`: Stores files uploaded by the user.
    *   `vector_store/`: Contains the ChromaDB vector store.
    *(Note: These subdirectories are typically created automatically by the application if they don't exist, based on paths configured in `core/settings.py`.)*
*   **`static/`**: (Optional) For any static files (e.g., `custom.css`) used by the Gradio interface.
*   **`README.md`**: (This file) The comprehensive guide for understanding, setting up, and using the Chatbot Theme Identifier application.
*   **`requirements.txt`**: Lists all Python dependencies required to run the application.
*   **`.env` file (not listed in structure but important)**: Located in `chatbot_theme_identifier_gradio/`, this user-created file (from `.env.example` or manually) stores local environment variable settings like API keys, model names, and paths. It is crucial for configuring the application and should not be committed to version control.

## How to Use the Application

The application interface is organized into tabs for different functionalities:

1.  **Tab 1: Upload & Manage Documents**:
    *   **Upload Files**: Click the "Select Documents to Upload" area or drag and drop files.
    *   **Process Files**: Click "Process Uploaded Files". Status will appear below.
    *   **Knowledge Base Status**: Click "Refresh Knowledge Base Status & Document List" to update status and document lists.

2.  **Tab 2: Chat & Research**:
    *   **Ask Questions**: Type questions in "Your Question:".
    *   **Send Message**: Click "Send Message" or press Enter.
    *   **View Response & Sources**: Chatbot's answer appears in "Conversation Log"; sources in "Cited Sources & Context".

3.  **Tab 3: Identify Themes**:
    *   **Select Scope**: Choose "Specific Document", "Query Context", or "All Processed Documents".
    *   **Input**: Select a document or enter a query if needed.
    *   **Identify Themes**: Click "Identify Themes".
    *   **View Results**: Themes appear in "Identified Themes".

**Typical Workflow**:

1.  Upload documents via the "Upload & Manage Documents" tab.
2.  Chat about documents in the "Chat & Research" tab.
3.  Analyze themes using the "Identify Themes" tab.

## Language Models and Embeddings

This application uses LLMs and text embedding models, configurable via environment variables (see `core/settings.py` and your `.env` file).

*   **LLM Providers**: Local servers (Ollama) or OpenAI API.
*   **Embedding Providers**: Local Hugging Face models or OpenAI API.

Configure via `.env` variables like `LLM_PROVIDER`, `OPENAI_API_KEY`, `EMBEDDING_PROVIDER`, etc.

## Contributing & Feedback

This README aims to be clear and helpful. For suggestions or issues, please open an issue or submit a pull request.

---

We hope this guide helps you get started!
