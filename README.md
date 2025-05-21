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
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
    (Replace `<repository_url>` and `<repository_directory>` with the actual URL and directory name)

3.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    Navigate to the `chatbot_theme_identifier_gradio` directory and install the required packages:
    ```bash
    cd chatbot_theme_identifier_gradio
    pip install -r requirements.txt
    ```

5.  **Environment Variables**:
    The application uses environment variables for configuration, managed via a `.env` file.
    *   In the `chatbot_theme_identifier_gradio` directory, rename the `.env.example` file (if provided) to `.env` or create a new `.env` file.
    *   Update the `.env` file with your specific configurations, such as API keys for LLM providers (e.g., OpenAI) or paths for local models. Refer to the `core/settings.py` file to see all possible environment variables.
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

        # Paths (adjust if needed, defaults are usually fine)
        # UPLOAD_DIR_PATH="uploads"
        # DB_DIR_PATH="vector_db"
        # LOG_FILE_PATH="logs/app.log"

        # Gradio Settings
        # GRADIO_SERVER_NAME="0.0.0.0"
        # GRADIO_SERVER_PORT=7860
        ```
    *   **Important**: Ensure the paths specified (like `UPLOAD_DIR_PATH`, `DB_DIR_PATH`, `LOG_FILE_PATH`) are writable by the application. The application will attempt to create these directories if they don't exist, based on the paths in `core/settings.py` or your `.env` overrides.

## Running the Application

Once the setup is complete:

1.  Ensure your virtual environment is activated.
2.  Navigate to the `chatbot_theme_identifier_gradio` directory if you are not already there.
3.  Run the Gradio application:
    ```bash
    python app.py
    ```
4.  The application will typically be available at `http://127.0.0.1:7860` (or the host/port specified in your settings). Open this URL in your web browser.

## Project Structure

Here's a brief overview of the key directories and files:

```
.
├── chatbot_theme_identifier_gradio/
│   ├── app.py                    # Main Gradio application script
│   ├── core/                     # Core components like settings and utilities
│   │   ├── settings.py           # Application settings and environment variable handling
│   │   └── utils.py              # Utility functions (if any)
│   ├── services/                 # Business logic for document processing, RAG, themes
│   │   ├── document_processor.py # Handles file parsing, chunking, OCR
│   │   ├── vector_store_manager.py # Manages vector database interactions
│   │   ├── rag_chain_builder.py  # Builds and manages the RAG chain
│   │   └── theme_service.py      # Logic for theme identification
│   ├── static/                   # Static assets (e.g., custom.css)
│   │   └── custom.css            # Custom CSS for Gradio interface
│   ├── .env                      # Local environment variable configurations (user-created)
│   ├── .env.example              # Example environment file (if provided)
│   ├── requirements.txt          # Python package dependencies
│   └── README.md                 # Detailed README for the Gradio app itself (currently empty)
├── LICENSE                       # Project license file
└── README.md                     # This file: Main project README
```

*   **`chatbot_theme_identifier_gradio/`**: Contains all the source code and specific files for the Gradio application.
    *   **`app.py`**: The entry point for the Gradio web application. It defines the UI and handles user interactions.
    *   **`core/`**: Contains the central configuration (`settings.py`) and potentially other core utilities. `settings.py` is crucial as it defines how the application loads its configuration from environment variables (and the `.env` file).
    *   **`services/`**: This directory houses the modules responsible for the application's main functionalities:
        *   `document_processor.py`: Handles everything related to loading, parsing, and preparing documents (including OCR).
        *   `vector_store_manager.py`: Manages the creation, updating, and querying of the vector database where document embeddings are stored.
        *   `rag_chain_builder.py`: Constructs and manages the RAG conversational chain, integrating the LLM with the vector store.
        *   `theme_service.py`: Contains the logic for analyzing content and identifying themes.
    *   **`static/`**: For any static files (like CSS or JavaScript) used by the Gradio interface.
    *   **`.env`**: (User-created from `.env.example` or manually) Stores local environment variable settings, like API keys, model names, and paths. This file is crucial for configuring the application without hardcoding sensitive information. It's listed in `.gitignore` and should not be committed.
    *   **`requirements.txt`**: Lists all Python dependencies required to run the application.
*   **`LICENSE`**: Contains the license information for the project.
*   **`README.md`**: (This file) The main README providing an overview of the entire project.

## How to Use the Application

The application interface is organized into tabs for different functionalities:

1.  **Tab 1: Upload & Manage Documents**:
    *   **Upload Files**: Click the "Select Documents to Upload" area or drag and drop files (PDF, TXT, MD, DOCX, etc.).
    *   **Process Files**: After selecting files, click the "Process Uploaded Files" button. The status of the upload and processing will appear below the button.
    *   **Knowledge Base Status**: You can click "Refresh Knowledge Base Status & Document List" to update the application's status display and the document list used in the "Identify Themes" tab. This is useful if you've added many documents or want to ensure the display is current.

2.  **Tab 2: Chat & Research**:
    *   **Ask Questions**: Once documents are processed and added to the knowledge base, you can ask questions about their content in the "Your Question:" textbox.
    *   **Send Message**: Click "Send Message" or press Enter in the textbox.
    *   **View Response**: The chatbot's answer will appear in the "Conversation Log".
    *   **View Sources**: Relevant snippets and source documents for the chatbot's answer will be displayed in the "Cited Sources & Context" area on the right.

3.  **Tab 3: Identify Themes**:
    *   **Select Scope**: Choose how you want to identify themes:
        *   **Specific Document**: Analyzes a single document you select from the dropdown.
            *   A dropdown list of "Select Document" will appear. Choose the document you want to analyze.
        *   **Query Context**: Identifies themes based on a query you provide.
            *   A textbox "Enter Query to Define Context" will appear. Type your query.
        *   **All Processed Documents**: Analyzes all documents currently in the knowledge base to find common themes.
    *   **Identify Themes**: Click the "Identify Themes" button.
    *   **View Results**: The identified themes, along with supporting snippets and source document IDs, will be displayed in the "Identified Themes" area.

**Typical Workflow**:

1.  Start by going to the "Upload & Manage Documents" tab.
2.  Upload one or more documents and wait for them to be processed.
3.  Navigate to the "Chat & Research" tab to ask questions about the content of your documents.
4.  Use the "Identify Themes" tab to gain insights into the main topics or themes present, either within a specific document, across all documents, or related to a particular query.

## Language Models and Embeddings

This application leverages Large Language Models (LLMs) and text embedding models for its core functionalities. The specific models used are configurable via environment variables, as defined in `chatbot_theme_identifier_gradio/core/settings.py`.

You can configure the application to use:
*   **LLM Providers**:
    *   Local LLM servers (e.g., Ollama with models like Llama2, Mistral)
    *   OpenAI API (e.g., GPT-3.5-turbo, GPT-4)
*   **Embedding Providers**:
    *   Local Hugging Face sentence-transformer models (e.g., `all-MiniLM-L6-v2`)
    *   OpenAI API (e.g., `text-embedding-ada-002`)

Refer to the `.env` file setup instructions and the `core/settings.py` file for details on which environment variables to set for your desired model providers and names (e.g., `LLM_PROVIDER`, `OPENAI_API_KEY`, `LOCAL_LLM_SERVER_URL`, `EMBEDDING_PROVIDER`, `LOCAL_HF_EMBEDDING_MODEL_NAME`, `OPENAI_EMBEDDING_MODEL_NAME`).

## Contributing & Feedback

This README aims to be clear, concise, and easy to follow. If you have suggestions for improving this documentation or encounter any issues, please feel free to open an issue or submit a pull request.

---

We hope this guide helps you get started with the Chatbot Theme Identifier!
