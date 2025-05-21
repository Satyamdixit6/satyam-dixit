# services/theme_service.py
import logging
from typing import List, Dict, Any
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document as LangchainCoreDocument


from core.settings import settings
from services.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

class ThemeService:
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.llm = self._initialize_llm()
        logger.info("ThemeService initialized with local LLM.")

    def _initialize_llm(self) -> ChatOpenAI:
        if settings.LLM_PROVIDER == "local_llama_server":
            logger.info(f"ThemeService: Initializing ChatOpenAI for local server. Base URL: {settings.LOCAL_LLM_API_BASE_URL}, Model: {settings.LOCAL_LLM_MODEL_NAME}")
            return ChatOpenAI(
                model=settings.LOCAL_LLM_MODEL_NAME,
                base_url=settings.LOCAL_LLM_API_BASE_URL,
                api_key=settings.LOCAL_LLM_API_KEY,
                temperature=0.1,
            )
        elif settings.LLM_PROVIDER == "openai_api":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY must be set for OpenAI LLM provider in ThemeService.")
            logger.info(f"ThemeService: Initializing ChatOpenAI for OpenAI API. Model: {settings.OPENAI_LLM_MODEL_NAME}")
            return ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model_name=settings.OPENAI_LLM_MODEL_NAME,
                temperature=0.1
            )
        else:
            raise ValueError(f"ThemeService: Unsupported LLM_PROVIDER '{settings.LLM_PROVIDER}' in settings.")

    def _parse_llm_theme_json_output(self, llm_json_str: str, context_doc_ids: List[str]) -> List[Dict[str, Any]]:
        themes = []
        try:
            json_start = llm_json_str.find('[')
            json_end = llm_json_str.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                clean_json_str = llm_json_str[json_start:json_end]
                parsed_output = json.loads(clean_json_str)
                if isinstance(parsed_output, list):
                    for item in parsed_output:
                        if isinstance(item, dict) and ("theme_name" in item or "theme" in item):
                            themes.append({
                                "theme_name": item.get("theme_name", item.get("theme", "Unnamed Theme")),
                                "supporting_snippets": item.get("supporting_snippets", item.get("snippets", [])),
                                "source_document_ids": context_doc_ids
                            })
                    logger.info(f"Successfully parsed {len(themes)} themes from LLM JSON output.")
                    return themes
                else:
                    logger.warning(f"LLM output was valid JSON but not a list as expected for themes. Output: {parsed_output}")
            else:
                logger.warning(f"Could not find JSON array structure in LLM output for themes. Raw: {llm_json_str[:300]}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM theme output as JSON. Error: {e}. Output: {llm_json_str[:500]}")

        if not themes and llm_json_str.strip():
            logger.warning("Could not parse themes as JSON, using raw output as a single theme description.")
            return [{"theme_name": f"Unparsed Theme Analysis: {llm_json_str[:250]}...", "supporting_snippets": [], "source_document_ids": context_doc_ids}]
        return []


    async def identify_themes_from_text_context(
        self,
        full_text_context: str,
        context_source_doc_ids: List[str],
        num_themes_to_identify: int = 3
    ) -> List[Dict[str, Any]]:

        if not full_text_context.strip():
            logger.info("No text context provided for theme identification.")
            return []

        MAX_CONTEXT_FOR_THEMES_CHAR = 15000
        processed_text_context = full_text_context
        if len(full_text_context) > MAX_CONTEXT_FOR_THEMES_CHAR:
            processed_text_context = full_text_context[:MAX_CONTEXT_FOR_THEMES_CHAR] + "\n...(Context Truncated for Theme Analysis)..."
            logger.warning(f"Combined text context for theme analysis was truncated to {MAX_CONTEXT_FOR_THEMES_CHAR} characters.")

        prompt_template_str = (
            f"You are an AI assistant specialized in identifying common themes from provided text excerpts. "
            f"Based ONLY on the text below, identify up to {num_themes_to_identify} distinct and coherent common themes. "
            f"For each theme, provide:\n"
            f"1. 'theme_name': A clear and concise name for the theme (e.g., 'Impact of X on Y', 'Key Challenges in Z').\n"
            f"2. 'supporting_snippets': A list of 1-2 direct short quotes or phrases from the provided text that exemplify this theme.\n"
            f"Your entire response MUST be a single, valid JSON array where each element is an object representing a theme. Each object must have 'theme_name' (string) and 'supporting_snippets' (list of strings) keys.\n"
            f"Example JSON array format:\n"
            f"[\n"
            f"  {{{{\n"  # Quadruple braces to produce literal {{ for Langchain, which becomes { for LLM
            f"    \"theme_name\": \"Innovation in Renewable Energy\",\n"
            f"    \"supporting_snippets\": [\"Solar panel efficiency has increased by 20%...\", \"New battery technologies are emerging...\"]\n"
            f"  }}}},\n" # Quadruple braces to produce literal }} for Langchain, which becomes } for LLM
            f"  {{{{\n"
            f"    \"theme_name\": \"Regulatory Hurdles for Startups\",\n"
            f"    \"supporting_snippets\": [\"The lengthy approval process often deters new companies...\", \"Compliance costs are a significant barrier...\"]\n"
            f"  }}}}\n"
            f"]\n"
            f"If no significant common themes can be identified, or if the text is insufficient, return an empty JSON array [].\n"
            f"Analyze the following text excerpts ONLY:\n"
            f"---------------------\n"
            f"{{text_to_analyze}}\n"  # This is correctly a Langchain template variable
            f"---------------------\n"
            f"JSON Array of Themes:"
        )
        theme_identification_prompt = ChatPromptTemplate.from_template(template=prompt_template_str)
        chain = theme_identification_prompt | self.llm | StrOutputParser()

        logger.info(f"Requesting theme identification from LLM, targeting {num_themes_to_identify} themes.")
        try:
            llm_response_str = await chain.ainvoke({"text_to_analyze": processed_text_context})
            return self._parse_llm_theme_json_output(llm_response_str, context_source_doc_ids)
        except Exception as e:
            logger.error(f"LLM call for theme identification failed: {e}", exc_info=True)
            error_theme_name = f"Error during theme identification: {str(e)}"
            error_source_ids = context_source_doc_ids if isinstance(context_source_doc_ids, list) else [str(context_source_doc_ids)]
            return [{"theme_name": error_theme_name, "supporting_snippets": [], "source_document_ids": error_source_ids}]

    async def identify_themes_for_documents(self, doc_ids: List[str], num_themes: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        results = {}
        if not self.vector_store_manager:
            logger.error("VectorStoreManager not available in ThemeService.")
            for doc_id in doc_ids:
                 results[doc_id] = [{"theme_name": "Error: Vector store not available.", "supporting_snippets": [], "source_document_ids": [doc_id]}]
            return results

        for doc_id in doc_ids:
            logger.info(f"Identifying themes for document ID: {doc_id}")
            retriever = self.vector_store_manager.get_retriever(k=200, filter_criteria={"original_doc_id": doc_id})
            doc_chunks = await retriever.aget_relevant_documents(query=doc_id)

            if not doc_chunks:
                logger.warning(f"No chunks found for document ID '{doc_id}'. Skipping theme identification for it.")
                results[doc_id] = [{"theme_name": "No content found for this document.", "supporting_snippets": [], "source_document_ids": [doc_id]}]
                continue

            context_text = "\n\n".join([chunk.page_content for chunk in doc_chunks])
            original_file_name = doc_chunks[0].metadata.get("original_file_name", doc_id)

            themes_for_doc = await self.identify_themes_from_text_context(
                full_text_context=context_text,
                context_source_doc_ids=[original_file_name],
                num_themes_to_identify=num_themes
            )
            results[doc_id] = themes_for_doc
        return results

    async def identify_common_themes_across_docs(self, doc_ids: List[str], num_common_themes: int = 3) -> List[Dict[str, Any]]:
        logger.info(f"Identifying common themes across document IDs: {doc_ids}")
        if not doc_ids: return []

        all_relevant_chunks: List[LangchainCoreDocument] = []
        context_source_doc_ids_map = {}

        for doc_id in doc_ids:
            retriever = self.vector_store_manager.get_retriever(k=20, filter_criteria={"original_doc_id": doc_id})
            chunks = await retriever.aget_relevant_documents(query=doc_id)
            for chunk in chunks:
                if "original_file_name" not in chunk.metadata:
                    chunk.metadata["original_file_name"] = doc_id
                context_source_doc_ids_map[chunk.metadata["original_doc_id"]] = chunk.metadata["original_file_name"]
            all_relevant_chunks.extend(chunks)

        if not all_relevant_chunks:
            logger.warning("No content found for any of the specified document IDs to find common themes.")
            return []

        combined_context_parts = []
        for chunk in all_relevant_chunks:
            file_name = chunk.metadata.get("original_file_name", "Unknown Document")
            page_num = chunk.metadata.get("page_number", "N/A")
            combined_context_parts.append(f"Excerpt from '{file_name}' (Page: {page_num}):\n{chunk.page_content}\n---")

        full_text_context = "\n".join(combined_context_parts)
        unique_source_filenames_for_context = sorted(list(set(context_source_doc_ids_map.values())))

        common_themes = await self.identify_themes_from_text_context(
            full_text_context=full_text_context,
            context_source_doc_ids=unique_source_filenames_for_context,
            num_themes_to_identify=num_common_themes
        )
        return common_themes

    async def get_themes_for_specific_document(self, doc_id: str, file_name_hint: str, num_themes: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Getting themes for specific document ID: '{doc_id}', hint: '{file_name_hint}'")
        retriever = self.vector_store_manager.get_retriever(
            k=500,
            filter_criteria={"original_doc_id": doc_id}
        )
        doc_specific_chunks = await retriever.aget_relevant_documents(query=file_name_hint or doc_id)

        if not doc_specific_chunks:
            logger.warning(f"No chunks found for document ID '{doc_id}' (hint: {file_name_hint}). Cannot identify themes.")
            return []

        logger.info(f"Found {len(doc_specific_chunks)} chunks for document '{file_name_hint}' (ID: '{doc_id}'). Identifying themes.")

        full_text_context = "\n\n".join([chunk.page_content for chunk in doc_specific_chunks if chunk.page_content])

        if not full_text_context.strip():
            logger.warning(f"Combined text from chunks for document ID '{doc_id}' is empty. Cannot identify themes.")
            return []

        context_source_doc_ids = [file_name_hint]

        return await self.identify_themes_from_text_context(
            full_text_context=full_text_context,
            context_source_doc_ids=context_source_doc_ids,
            num_themes_to_identify=num_themes
        )