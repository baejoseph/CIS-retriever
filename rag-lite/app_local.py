import streamlit as st
from ollama_services import OllamaEmbeddingService, OllamaGenerationService, LocalCacheService
from openai_services import OpenAIEmbeddingService, OpenAIGenerationService
from rag_pipeline import (
    Corpus, RetrievalService, PromptAugmenter, QueryProcessor,
    ProcessorConfig, RetrievalConfig, CosineSimilarity
)
from parser_local import DocumentParser
from dotenv import load_dotenv
import os
import re
from log_time import ProcessTimer
from helpers import load_config


pt = ProcessTimer()

load_dotenv()


api_key = os.environ["OPENAI_API_KEY"]
# === Streamlit Setup ===
st.set_page_config(page_title="RAG Chat MVP (Local)", layout="wide")
st.title("🔍📚 Retrieval-Augmented Chatbot (Local Demo)")

# === Local Cache & Model Setup ===
cache_dir    = os.getenv("LOCAL_CACHE_DIR", "local_cache")
bucket_name  = 'test-bucket'
  
# === Session State Initialization ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "corpus" not in st.session_state:
    st.session_state.corpus = Corpus()

if "base_services" not in st.session_state:
    with st.spinner("Initializing local models…"):
        embedding_service = OpenAIEmbeddingService(api_key) #OllamaEmbeddingService(load_config('embedding_model'))
        generation_service = OpenAIGenerationService(api_key) #OllamaGenerationService(load_config('inference_model'))
        similarity_metric  = CosineSimilarity()
        retrieval_service  = RetrievalService(
            st.session_state.corpus,
            similarity_metric,
            load_config('reranker_model'),
        )
        augmenter     = PromptAugmenter('rag_prompt.md')
        cache_service = LocalCacheService()

        st.session_state.base_services = {
            "embedding_service": embedding_service,
            "generation_service": generation_service,
            "retrieval_service": retrieval_service,
            "augmenter": augmenter,
            "cache_service": cache_service,
        }
    st.success(f"Local LLM initialized: {generation_service.model}", icon="✅")

# === File Upload ===
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Upload pdf or docx files")
uploaded_files = st.sidebar.file_uploader("Upload pdf or docx (<200MB)", type=["pdf", "docx"], accept_multiple_files=True)

for uploaded_file in uploaded_files:
    if uploaded_file and uploaded_file.size < 200 * 1024 * 1024:
        st.sidebar.success(f"Uploaded: {uploaded_file.name}")
        with st.spinner("Parsing…"):
            parser = DocumentParser(
                st.session_state.base_services["embedding_service"],
                st.session_state.base_services["cache_service"],
                bucket_name
            )
            new_chunks = parser.parse(uploaded_file)
        st.sidebar.success("Document parsed", icon="✅")

        # Add to corpus
        added = st.session_state.corpus.add_chunks(new_chunks)
        if added:
            st.sidebar.success(f"Added {added} chunks")
        else:
            st.sidebar.info("No new chunks (already cached)")
        st.sidebar.info(f"Total chunks: {len(st.session_state.corpus.get_all_chunks())}")

    elif uploaded_file:
        st.sidebar.error("File too large. Must be <2MB.")

# === Retrieval Settings ===
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Retrieval Settings")
top_k               = st.sidebar.slider("Top K Chunks", 1, 10, 3)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.42)

# === Chat Input ===
user_input = st.chat_input("Ask a question…")
if user_input:
    config = ProcessorConfig(
            retrieval=RetrievalConfig(
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
        )
    processor = QueryProcessor(
            corpus=st.session_state.corpus,
            embedding_service=st.session_state.base_services["embedding_service"],
            retrieval_service=st.session_state.base_services["retrieval_service"],
            prompt_augmenter=st.session_state.base_services["augmenter"],
            generation_service=st.session_state.base_services["generation_service"],
            config=config
        )
    pt.mark("Answer Generation")
    with st.spinner("Thinking..."):
        # Augment user query using retrieve relevant chunks
        augmented_query = processor.pre_gen_process(user_input)

        gen = st.session_state.base_services["generation_service"]
        # adapted for streaming
        response_chunks = gen.generate_response(augmented_query)
        collected = ""
        assistant_area = st.empty()
        inside_think = False
        think_text = ""
        final_text = ""
        
        pt.mark("Thinking process")
        for partial in response_chunks:
            collected += partial

            # check for think tags
            if "<think>" in collected:
                inside_think = True
            if inside_think:
                # accumulate thinking text
                think_text += partial
                # stream it but hide the <think> tags
                cleaned = re.sub(r"<\/?think>", "", think_text)
                assistant_area.markdown(cleaned)
                if "</think>" in collected:
                    inside_think = False
                    # store final text after </think>
                    after_think = collected.split("</think>")[-1]
                    final_text = after_think.strip()
                    assistant_area.markdown(final_text)  # replace entire area
            else:
                # if no <think> tags at all, just stream normally
                pt.done("Thinking process")
                assistant_area.markdown(collected)

        # after stream, ensure final_text shows alone
        if final_text:
            assistant_area = st.empty()
            st.session_state.chat_history.append({"user": user_input, "bot": final_text})
            pt.done("Answer Generation")
        else:
            st.session_state.chat_history.append({"user": user_input, "bot": collected})

# === Display Chat ===
for exchange in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(exchange["user"])
    with st.chat_message("assistant"):
        st.markdown(exchange["bot"])
