# app.py
import sys 
import logging 

# --- Initial Bootstrap Logging ---
bootstrap_logger = logging.getLogger("bootstrap")
logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
bootstrap_logger.info("app.py: Script execution started.")
# --- End Bootstrap Logging ---

try:
    bootstrap_logger.info("app.py: Attempting to import 'settings' from 'core.settings'...")
    from core.settings import settings 
    bootstrap_logger.info(f"app.py: Successfully imported 'settings'. LOG_LEVEL from settings: {settings.LOG_LEVEL}")
except ImportError as e:
    bootstrap_logger.critical(f"app.py: CRITICAL - Failed to import 'settings' from 'core.settings'. Error: {e}. This usually means 'core/settings.py' has an issue or a PYTHONPATH issue. Check 'core/settings.py' output.", exc_info=True)
    sys.exit(1)
except Exception as e: 
    bootstrap_logger.critical(f"app.py: CRITICAL - An unexpected error occurred during 'settings' import: {e}. Check 'core/settings.py'.", exc_info=True)
    sys.exit(1)

# --- Configure Global Logging ---
try:
    log_level_from_settings = settings.LOG_LEVEL.upper()
    numeric_level = getattr(logging, log_level_from_settings, logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(numeric_level)
    root_logger.addHandler(stream_handler)
    logger = logging.getLogger(__name__)
    logger.info(f"app.py: Global logging reconfigured to level: {log_level_from_settings} based on settings.")
except Exception as e:
    bootstrap_logger.error(f"app.py: Error configuring global logging with settings: {e}. Using bootstrap logging.", exc_info=True)
    logger = bootstrap_logger
# --- End Global Logging Configuration ---

logger.info("app.py: Importing Gradio and other services...")
try:
    import gradio as gr
    import time 
    import uuid
    from typing import List, Tuple, Optional, Dict, Any
    from pathlib import Path
    from services.document_processor import DocumentProcessorService
    from services.vector_store_manager import VectorStoreManager
    from services.rag_chain_builder import RAGChainBuilder, get_session_history
    from services.theme_service import ThemeService
    logger.info("app.py: Gradio and service modules imported successfully.")
except ImportError as e:
    logger.critical(f"app.py: Failed to import a required module (Gradio or a service): {e}. Please check installations (requirements.txt) and Python paths.", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.critical(f"app.py: Unexpected error during service module imports: {e}", exc_info=True)
    sys.exit(1)

# --- Initialize Core Services ---
SERVICES_INITIALIZED = False
doc_processor: Optional[DocumentProcessorService] = None
vector_store_manager: Optional[VectorStoreManager] = None
rag_builder: Optional[RAGChainBuilder] = None
theme_service: Optional[ThemeService] = None

try:
    logger.info("app.py: Initializing core services...")
    doc_processor = DocumentProcessorService()
    vector_store_manager = VectorStoreManager(force_new_embeddings=False) 
    rag_builder = RAGChainBuilder(vector_store_manager)
    theme_service = ThemeService(vector_store_manager)
    SERVICES_INITIALIZED = True
    logger.info("app.py: All services initialized successfully.")
except Exception as e:
    logger.error(f"app.py: FATAL - Failed to initialize one or more core services during app setup: {e}", exc_info=True)
# --- End Core Services ---

PROCESSED_DOC_IDS: List[Dict[str, Any]] = []

def format_chat_history_for_gradio(session_id: str) -> List[Tuple[Optional[str], Optional[str]]]:
    history = get_session_history(session_id)
    gradio_chat = []
    for msg in history.messages:
        if msg.type == "human": gradio_chat.append((msg.content, None))
        elif msg.type == "ai":
            ai_content = msg.content
            if isinstance(msg.content, dict) and "answer" in msg.content: ai_content = msg.content["answer"]
            gradio_chat.append((None, ai_content))
    return gradio_chat

def format_sources_for_display(sources: List[Dict[str, Any]]) -> str:
    if not sources: return "_No sources cited for this response._"
    source_strs = ["**Cited Sources:**"]
    for i, src in enumerate(sources):
        f_name = src.get('file_name', src.get('doc_id', 'Unknown Document'))
        page = src.get('page_number', 'N/A')
        preview = src.get('content_preview', '')
        source_item_md = f"{i+1}. **{f_name} (Page: {page})**"
        if preview: source_item_md += f"\n   *Preview: {preview}*"
        source_strs.append(source_item_md)
    return "\n\n".join(source_strs)

def format_themes_for_display(themes: List[Dict[str, Any]]) -> str:
    if not themes: return "_No themes were identified for this context._"
    theme_strs = ["**Identified Themes:**"]
    for i, theme_data in enumerate(themes):
        name = theme_data.get("theme_name", "Unnamed Theme")
        snippets = theme_data.get("supporting_snippets", [])
        docs = theme_data.get("source_document_ids", [])
        theme_str = f"**Theme {i+1}: {name}**\n"
        if snippets:
            theme_str += "  *Supporting Snippets:*\n"
            for snip_idx, snip in enumerate(snippets): theme_str += f"    {snip_idx+1}. \"{snip}\"\n"
        if docs: theme_str += f"  *Relevant Document(s): {', '.join(docs)}*\n"
        theme_strs.append(theme_str)
    return "\n\n".join(theme_strs)

# --- Gradio Interface Functions ---
def handle_file_uploads(files_list: Optional[List[Any]]) -> Tuple[str, Any]:
    global PROCESSED_DOC_IDS
    if not SERVICES_INITIALIZED or not doc_processor or not vector_store_manager:
        return "ERROR: Core services failed to initialize. File processing disabled.", update_doc_dropdown_choices()
    if not files_list:
        return "No files provided. Please select files to upload.", update_doc_dropdown_choices()

    logger.info(f"Gradio: Received {len(files_list)} file(s) for processing.")
    upload_summaries = []
    processed_files_count_this_upload = 0
    
    total_files = len(files_list)
    
    for i, temp_file_obj in enumerate(files_list):
        original_filename = getattr(temp_file_obj, 'name', f"file_{i+1}")
        logger.info(f"Gradio: Processing file {i+1}/{total_files}: {original_filename}")
        upload_summaries.append(f"Processing '{original_filename}' ({i+1}/{total_files})...") # Add desc to summaries
        try:
            persistent_doc_id, orig_name, lc_core_chunks, num_chunks, num_pages = \
                doc_processor.process_uploaded_file_object(temp_file_obj)
            if not lc_core_chunks:
                summary = f"File '{orig_name}': No content extracted or processed."
                logger.warning(summary); upload_summaries.append(summary); continue
            vector_store_manager.add_documents_to_store(
                langchain_documents=lc_core_chunks,
                original_doc_id=persistent_doc_id,
                original_file_name=orig_name
            )
            doc_info = {"doc_id": persistent_doc_id, "file_name": orig_name, "num_chunks": num_chunks, "num_pages": num_pages}
            existing_doc_index = next((idx for idx, d in enumerate(PROCESSED_DOC_IDS) if d["doc_id"] == persistent_doc_id), -1)
            if existing_doc_index != -1: PROCESSED_DOC_IDS[existing_doc_index] = doc_info
            else: PROCESSED_DOC_IDS.append(doc_info)
            processed_files_count_this_upload += 1
            summary = f"OK: '{orig_name}' ({num_chunks} chunks, {num_pages or 'N/A'} pages) processed."
            logger.info(summary); upload_summaries.append(summary)
        except ValueError as ve: 
            summary = f"ERROR processing '{original_filename}': {ve}"
            logger.error(f"Gradio: {summary}", exc_info=False); upload_summaries.append(summary)
        except Exception as e: 
            summary = f"UNEXPECTED ERROR with '{original_filename}': {e}"
            logger.error(f"Gradio: {summary}", exc_info=True); upload_summaries.append(summary)
    
    logger.debug(f"Updated PROCESSED_DOC_IDS: {PROCESSED_DOC_IDS}")  # Debugging log
    final_status = f"File processing finished for {total_files} file(s).\n" + "\n".join(upload_summaries)
    if processed_files_count_this_upload > 0:
        final_status += f"\n\nSuccessfully processed {processed_files_count_this_upload} file(s). Knowledge base updated."
    logger.info(f"Gradio: {final_status}")
    return final_status, update_doc_dropdown_choices()

async def chat_interface_fn(message: str, history: List[Tuple[str, str]], session_state: gr.State) -> Tuple[List[Tuple[str, str]], str, gr.State]:
    if not SERVICES_INITIALIZED or not rag_builder:
        error_msg = "ERROR: Chat services not available. Initialization may have failed."
        history.append((message, error_msg)); return history, "Chat service unavailable.", session_state
    if not message.strip():
        return history, "Please type a question.", session_state
    session_id = session_state.get("session_id")
    if not session_id: 
        session_id = f"gr_sess_{uuid.uuid4().hex[:12]}"; session_state["session_id"] = session_id
        logger.info(f"Gradio: New chat session initiated: {session_id}")
    logger.info(f"Gradio Chat (Session: {session_id}): User asked: '{message[:100]}...'")
    history.append((message, None))
    
    chain_output = await rag_builder.invoke_chain(user_question=message, session_id=session_id)
    ai_answer = chain_output.get("answer", "Sorry, I could not generate a response.")
    if not isinstance(ai_answer, str): ai_answer = str(ai_answer)
    retrieved_docs = chain_output.get("context", [])
    formatted_sources = rag_builder.get_formatted_sources(retrieved_docs)
    sources_markdown = format_sources_for_display(formatted_sources)
    history[-1] = (message, ai_answer) 
    logger.info(f"Gradio Chat (Session: {session_id}): AI responded: '{ai_answer[:100]}...' Sources: {len(formatted_sources)}")
    return history, sources_markdown, session_state

async def identify_themes_fn(scope: str, selected_doc_id: Optional[str], query_context: Optional[str]) -> str:
    if not SERVICES_INITIALIZED or not theme_service or not vector_store_manager:
        return "ERROR: Theme services are not available. Initialization may have failed."
    logger.info(f"Gradio Theme request. Scope: {scope}, Doc ID: {selected_doc_id}, Query: '{query_context[:50] if query_context else 'N/A'}'")

    num_themes = getattr(settings, 'THEME_SERVICE_NUM_THEMES_UI', 3) 
    identified_themes_data: List[Dict[str, Any]] = []
    try:
        if scope == "Specific Document":
            if not selected_doc_id: return "Please select a document for 'Specific Document' theme analysis."
            doc_info = next((doc for doc in PROCESSED_DOC_IDS if doc["doc_id"] == selected_doc_id), None)
            if not doc_info: return f"Error: Document with ID '{selected_doc_id}' not found."
            # Corrected method name
            identified_themes_data = await theme_service.get_themes_for_specific_document(
                doc_id=selected_doc_id, file_name_hint=doc_info['file_name'], num_themes=num_themes
            )
        elif scope == "Query Context":
            if not query_context or not query_context.strip(): return "Please enter a query for 'Query Context' theme analysis."
            identified_themes_data = await theme_service.get_themes_from_query_context(query=query_context, num_themes=num_themes)
        elif scope == "All Processed Documents":
            if not PROCESSED_DOC_IDS: return "No documents processed yet."
            doc_ids_to_analyze = [doc['doc_id'] for doc in PROCESSED_DOC_IDS]
            identified_themes_data = await theme_service.identify_common_themes_across_docs(doc_ids=doc_ids_to_analyze, num_common_themes=num_themes)
        else:
            return "Invalid scope selected for theme identification."
    except Exception as e:
        logger.error(f"Error during theme identification for scope '{scope}': {e}", exc_info=True)
        return f"An error occurred during theme identification: {str(e)}"

    formatted_themes_md = format_themes_for_display(identified_themes_data)
    logger.info(f"Gradio Theme ID complete. Found {len(identified_themes_data)} themes for scope '{scope}'.")
    return formatted_themes_md

def update_doc_dropdown_choices():
    choices = [(f"{doc['file_name']} (ID: {doc['doc_id'][:8]}...)", doc['doc_id']) for doc in PROCESSED_DOC_IDS]
    if not choices:
        # Always return an interactive dropdown, but disabled if no docs
        return gr.update(choices=[("No documents processed yet", None)], value=None, interactive=False)
    # Set value to first doc if available, and interactive
    return gr.update(choices=choices, value=choices[0][1], interactive=True)

def get_initial_status_and_update_dropdown_on_load() -> Tuple[str, Any]:
    status_str = ""
    dropdown_update_val = update_doc_dropdown_choices()

    if not SERVICES_INITIALIZED:
        status_str = "App Status: CRITICAL ERROR - Core services failed. Check server logs."
    elif not vector_store_manager :
         status_str = "App Status: CRITICAL ERROR - Vector store manager not initialized."
    elif vector_store_manager.get_document_count() == 0:
        status_str = "App Status: Ready. Knowledge base is empty. Upload documents."
    else:
        status_str = f"App Status: Ready. KB has ~{vector_store_manager.get_document_count()} chunks."
    return status_str, dropdown_update_val

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky), title=settings.APP_NAME) as demo:
    current_session_state = gr.State(value={"session_id": f"gr_sess_{uuid.uuid4().hex[:12]}"})
    gr.Markdown(f"<h1 style='text-align: center; color: #0056b3;'>{settings.APP_NAME}</h1>")
    gr.Markdown(f"<p style='text-align: center; font-size: 0.9em;'>Version: {settings.APP_VERSION} | LLM: {settings.LLM_PROVIDER} ({settings.LOCAL_LLM_MODEL_NAME if settings.LLM_PROVIDER=='local_llama_server' else settings.OPENAI_LLM_MODEL_NAME}) | Embeddings: {settings.EMBEDDING_PROVIDER} ({settings.LOCAL_HF_EMBEDDING_MODEL_NAME if settings.EMBEDDING_PROVIDER=='local_hf' else settings.OPENAI_EMBEDDING_MODEL_NAME})</p>")
    app_status_display = gr.Markdown("App Status: Initializing...", elem_id="app_status_display")

    with gr.Tabs():
        with gr.TabItem("1. Upload & Manage Documents", id="upload_tab"):
            gr.Markdown("### Document Upload\nUpload documents (PDF, TXT, MD, DOCX, etc.). Scanned images in PDFs will be processed with OCR if Tesseract & Poppler are installed.")
            file_uploader = gr.File(label="Select Documents to Upload", file_count="multiple", file_types=[".pdf", ".txt", ".md", ".docx", ".doc", ".pptx", ".html", ".eml", ".msg", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"])
            upload_button = gr.Button("Process Uploaded Files", variant="primary", elem_id="upload_button_elem")
            upload_output_status_markdown = gr.Markdown(label="Upload & Processing Status", elem_id="upload_status_md", value="_Upload status will appear here._")
            gr.Markdown("### Knowledge Base Management")
            refresh_kb_status_button = gr.Button("Refresh Knowledge Base Status & Document List")
            
        with gr.TabItem("2. Chat & Research", id="chat_tab"):
            gr.Markdown("### Conversational RAG\nAsk questions about your documents. Answers will cite sources.")
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot_display = gr.Chatbot(label="Conversation Log", height=550, show_copy_button=True, bubble_full_width=False, avatar_images=(None, "https://raw.githubusercontent.com/gradio-app/gradio/main/gradio/icons/robot.svg"))
                    chat_input_textbox = gr.Textbox(label="Your Question:", placeholder="e.g., What are the key challenges mentioned regarding AI ethics?", lines=3, show_copy_button=True) 
                with gr.Column(scale=2):
                    gr.Markdown("### Cited Sources & Context"); sources_display_markdown = gr.Markdown(label="Sources for Last Response", elem_id="sources_display_md", value="_Sources will appear here._")
            chat_submit_button = gr.Button("Send Message", variant="primary", elem_id="chat_send_button")

        with gr.TabItem("3. Identify Themes", id="theme_tab"):
            gr.Markdown("### Theme Analysis\nIdentify themes from documents or query context.")
            theme_scope_selector = gr.Radio(["Specific Document", "Query Context", "All Processed Documents"], label="Analyze Themes From:", value="Specific Document", interactive=True)
            with gr.Group(visible=True) as specific_doc_group:
                 processed_docs_dropdown = gr.Dropdown(label="Select Document", choices=[("No documents processed yet", None)], value=None, interactive=False)
            with gr.Group(visible=False) as query_context_group:
                theme_query_textbox = gr.Textbox(label="Enter Query to Define Context", placeholder="e.g., common concerns about data privacy")
            
            def toggle_theme_inputs(scope_choice):
                if scope_choice == "Specific Document": return gr.update(visible=True), gr.update(visible=False) # CORRECTED
                elif scope_choice == "Query Context": return gr.update(visible=False), gr.update(visible=True) # CORRECTED
                else: return gr.update(visible=False), gr.update(visible=False) # CORRECTED

            theme_scope_selector.change(fn=toggle_theme_inputs, inputs=theme_scope_selector, outputs=[specific_doc_group, query_context_group])
            identify_themes_button = gr.Button("Identify Themes", variant="primary", elem_id="identify_themes_btn")
            themes_output_markdown = gr.Markdown(label="Identified Themes", value="_Theme analysis results appear here._")

    # --- Event Handlers Wiring ---
    upload_button.click(
        handle_file_uploads,
        inputs=[file_uploader],
        outputs=[upload_output_status_markdown, processed_docs_dropdown]
    ).then(
        fn=lambda: get_initial_status_and_update_dropdown_on_load()[0],
        inputs=None, outputs=[app_status_display]
    )
    
    refresh_kb_status_button.click(
        fn=get_initial_status_and_update_dropdown_on_load,
        inputs=None, outputs=[app_status_display, processed_docs_dropdown]
    )

    chat_submit_button.click(
        chat_interface_fn,
        inputs=[chat_input_textbox, chatbot_display, current_session_state],
        outputs=[chatbot_display, sources_display_markdown, current_session_state],
        api_name="send_chat_message"
    ).then(fn=lambda: gr.update(value=""), inputs=None, outputs=[chat_input_textbox]) # CORRECTED

    chat_input_textbox.submit(
        chat_interface_fn,
        inputs=[chat_input_textbox, chatbot_display, current_session_state],
        outputs=[chatbot_display, sources_display_markdown, current_session_state]
    ).then(fn=lambda: gr.update(value=""), inputs=None, outputs=[chat_input_textbox]) # CORRECTED

    identify_themes_button.click(
        identify_themes_fn,
        inputs=[theme_scope_selector, processed_docs_dropdown, theme_query_textbox],
        outputs=[themes_output_markdown], api_name="identify_themes"
    )
    
    demo.load(
        fn=get_initial_status_and_update_dropdown_on_load,
        inputs=None, outputs=[app_status_display, processed_docs_dropdown]
    )

if __name__ == "__main__":
    if not SERVICES_INITIALIZED:
        logger.critical("Core services did not initialize. Gradio app functionality will be severely limited. Check logs for errors during service instantiation.")
    
    logger.info(f"Launching Gradio app: Name='{settings.APP_NAME}', Version='{settings.APP_VERSION}'")
    logger.info(f"Server settings: Host='{settings.GRADIO_SERVER_NAME}', Port={settings.GRADIO_SERVER_PORT}, Share={settings.GRADIO_SHARE}")
    logger.info(f"To access the app, open your browser to: http://{settings.GRADIO_SERVER_NAME if settings.GRADIO_SERVER_NAME != '0.0.0.0' else '127.0.0.1'}:{settings.GRADIO_SERVER_PORT}")
    
    demo.queue().launch(
        server_name=settings.GRADIO_SERVER_NAME,
        server_port=settings.GRADIO_SERVER_PORT,
        share=settings.GRADIO_SHARE,
    )
    logger.info("Gradio application has been launched.")

