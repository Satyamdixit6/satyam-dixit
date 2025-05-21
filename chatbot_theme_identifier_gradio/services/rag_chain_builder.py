# services/rag_chain_builder.py
import logging
from typing import Dict, Any, List # Added List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory # For session history
from langchain_core.documents import Document as LangchainCoreDocument # For type hinting

from core.settings import settings # Adjusted import path
from services.vector_store_manager import VectorStoreManager # Adjusted import path

logger = logging.getLogger(__name__)

# In-memory session store for chat history (for demonstration)
# For production, consider a more persistent store (Redis, DB, etc.)
SESSION_CHATS: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in SESSION_CHATS:
        logger.debug(f"RAGChainBuilder: Creating new chat history for session_id: {session_id}")
        SESSION_CHATS[session_id] = ChatMessageHistory()
    else:
        logger.debug(f"RAGChainBuilder: Reusing existing chat history for session_id: {session_id}")
    return SESSION_CHATS[session_id]

class RAGChainBuilder:
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.llm = self._initialize_llm()
        self.conversational_rag_chain = self._build_chain()
        logger.info("RAGChainBuilder initialized with local LLM and conversational RAG chain.")

    def _initialize_llm(self) -> ChatOpenAI:
        if settings.LLM_PROVIDER == "local_llama_server":
            logger.info(f"RAGChainBuilder: Initializing ChatOpenAI for local server. Base URL: {settings.LOCAL_LLM_API_BASE_URL}, Model: {settings.LOCAL_LLM_MODEL_NAME}")
            # Ensure your llama.cpp server is running and accessible at this URL
            # The model parameter might be used by the server to select a model if it serves multiple,
            # or it might be ignored if the server is started with a single specific model.
            return ChatOpenAI(
                model=settings.LOCAL_LLM_MODEL_NAME, 
                base_url=settings.LOCAL_LLM_API_BASE_URL,
                api_key=settings.LOCAL_LLM_API_KEY, # Usually "not-needed" or any string for local servers
                temperature=0.6, # Adjust for desired creativity/factuality balance
                # request_timeout=120 # Optional: For slower local models
                # streaming=True # Enable if Gradio interface will handle streaming output
            )
        elif settings.LLM_PROVIDER == "openai_api":
             if not settings.OPENAI_API_KEY:
                logger.error("OPENAI_API_KEY must be set in .env for OpenAI LLM provider.")
                raise ValueError("OPENAI_API_KEY is not set for OpenAI LLM provider.")
             logger.info(f"RAGChainBuilder: Initializing ChatOpenAI for OpenAI API. Model: {settings.OPENAI_LLM_MODEL_NAME}")
             return ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model_name=settings.OPENAI_LLM_MODEL_NAME,
                temperature=0.7
            )
        else:
            raise ValueError(f"RAGChainBuilder: Unsupported LLM_PROVIDER '{settings.LLM_PROVIDER}' in settings.")

    def _build_chain(self) -> Runnable:
        try:
            # Configure the retriever
            # Task PDF mentions: "precise citations clearly indicating locations (page, paragraph, sentence)" -> start with page
            # "minimum document-level granularity to support each synthesized theme"
            # For smarter RAG, we might want more diverse context:
            retriever = self.vector_store_manager.get_retriever(
                k=7, # Number of chunks to retrieve, might need tuning
                score_threshold=0.45 # Adjust threshold to filter less relevant chunks
            )
        except Exception as e: # Catch errors if vector store isn't ready
            logger.error(f"RAGChainBuilder: Failed to get retriever from VectorStoreManager: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize retriever for RAG chain: {e}")

        # Contextualize question based on chat history
        contextualize_q_system_prompt = (
            "Given the chat history and a follow-up question, rephrase the "
            "follow-up question to be a standalone question that can be understood "
            "without the chat history. Do NOT answer the question; only rephrase it."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # Answering chain (takes question and retrieved documents)
        qa_system_prompt_template = (
            "You are an expert AI assistant specializing in document research and theme identification. "
            "Your task is to answer the user's question based *solely* on the provided context (excerpts from documents).\n"
            "Follow these instructions carefully:\n"
            "1. Detailed Answer: Provide a comprehensive and detailed answer that directly addresses the user's question.\n"
            "2. Context-Bound: Base your entire answer strictly on the information found in the 'Context provided' section. Do not use any external knowledge or make assumptions beyond the text.\n"
            "3. Citation: For every piece of information or claim in your answer that comes from the context, you MUST provide a precise citation. Format citations as (Source: [original_file_name], Page: [page_number]). If page number is not available, cite as (Source: [original_file_name]). If multiple sources support a point, cite them all.\n"
            "4. Theme Integration (If Applicable): If the question implies theme identification or if distinct themes emerge from the context relevant to the question, clearly state these themes. For each theme, provide a brief explanation supported by information from the context, along with citations for that supporting information.\n"
            "5. Unknown Information: If the provided context does not contain information to answer the question or identify relevant themes, clearly state 'Based on the provided documents, I cannot answer this question.' or 'No relevant themes were identified in the provided context for this query.'\n"
            "6. Synthesis: If necessary, synthesize information from multiple context excerpts to form a coherent answer. Ensure all synthesized points are properly cited.\n"
            "7. Clarity and Conciseness: While being detailed, strive for clarity and avoid unnecessary jargon unless it's part of the document's terminology (in which case, explain it if possible).\n"
            "\n"
            "Context provided:\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(qa_system_prompt_template),
                MessagesPlaceholder(variable_name="chat_history"), # For conversational flow and tone
                HumanMessagePromptTemplate.from_template("{input}"), # User's (potentially rephrased) question
            ]
        )
        Youtube_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever, # Input: user question, chat_history. Output: rephrased_question, context_docs
            Youtube_chain    # Input: rephrased_question, context_docs, chat_history. Output: answer
        )
        
        # Wrap with message history management
        conversational_rag_chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer", # The key in rag_chain's output that contains the final LLM answer
        )
        return conversational_rag_chain_with_history

    async def invoke_chain(self, user_question: str, session_id: str) -> Dict[str, Any]:
        """
        Invokes the conversational RAG chain with the user's question and session ID.
        Returns the full chain output dictionary.
        """
        if not self.conversational_rag_chain:
            logger.error("RAGChainBuilder: Conversational RAG chain is not initialized.")
            return {"answer": "Error: RAG chain not available.", "context": [], "chat_history": get_session_history(session_id).messages}

        logger.debug(f"RAGChainBuilder: Invoking chain for session_id '{session_id}' with question: '{user_question[:100]}...'")
        
        payload_for_chain = {"input": user_question}
        config_for_chain = {"configurable": {"session_id": session_id}}
        
        try:
            # ainvoke returns a dictionary, e.g., {"input": ..., "chat_history": ..., "context": ..., "answer": ...}
            chain_output = await self.conversational_rag_chain.ainvoke(payload_for_chain, config=config_for_chain)
            return chain_output
        except Exception as e:
            logger.error(f"RAGChainBuilder: Error invoking conversational RAG chain for session '{session_id}': {e}", exc_info=True)
            # Return a structured error response
            return {
                "answer": f"An error occurred while processing your request: {str(e)[:150]}",
                "context": [], # No context could be retrieved or processed
                "chat_history": get_session_history(session_id).messages, # Return current history
                "error": str(e)
            }

    def get_formatted_sources(self, retrieved_docs: List[LangchainCoreDocument]) -> List[Dict[str, Any]]:
        """Helper to format retrieved Langchain documents into a list of dictionaries for UI display."""
        sources: List[Dict[str, Any]] = []
        if not retrieved_docs:
            return sources
        
        unique_source_keys = set() # To avoid nearly identical source entries from very similar chunks
        for doc in retrieved_docs:
            metadata = doc.metadata or {}
            file_name = metadata.get("original_file_name", metadata.get("source_file", "Unknown Source"))
            page_number = metadata.get("page_number")
            try:
                page_number_str = str(int(page_number)) if page_number is not None else "N/A"
            except ValueError:
                page_number_str = str(page_number) if page_number is not None else "N/A"

            # Create a unique key for source to avoid duplicates in response if desired
            source_key = f"{file_name}_p{page_number_str}"
            
            if source_key not in unique_source_keys:
                unique_source_keys.add(source_key)
                sources.append({
                    "doc_id": metadata.get("original_doc_id"),
                    "file_name": file_name,
                    "page_number": page_number_str, # Keep as string for display consistency
                    "content_preview": doc.page_content[:200] + "..." if doc.page_content else "",
                    # "relevance_score": metadata.get('relevance_score') # If your retriever adds this
                })
        return sources

