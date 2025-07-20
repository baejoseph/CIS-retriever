import streamlit as st
from openai_services import OpenAIEmbeddingService, OpenAIGenerationService
from rag_pipeline import (
    Corpus, RetrievalService, PromptAugmenter, QueryProcessor,
    ProcessorConfig, RetrievalConfig, CosineSimilarity
)
from parser import DocumentParser
from dotenv import load_dotenv
import os
import json
import boto3
from log_time import ProcessTimer
from helpers import load_config

pt = ProcessTimer()

load_dotenv()

# === Streamlit Setup ===
st.set_page_config(page_title="CIS Benchmarks Retrieval", layout="wide")
st.title("🔍📚 Retrieval-Augmented Chatbot (MVP)")

api_key = os.environ["OPENAI_API_KEY"]
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
bucket_name = os.getenv("AWS_S3_BUCKET")

# === Session State Initialization ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "corpus" not in st.session_state:
    st.session_state.corpus = Corpus()

if "base_services" not in st.session_state:
    with st.spinner("Initializing LLM..."):
        embedding_service = OpenAIEmbeddingService(api_key, load_config('embedding_model'))
        generation_service = OpenAIGenerationService(api_key, load_config('inference_model'))
        similarity_metric = CosineSimilarity()
        retrieval_service = RetrievalService(
            st.session_state.corpus, 
            similarity_metric,
            load_config('reranker_model')
            )
        augmenter = PromptAugmenter('rag_prompt.md')
        s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region
            )
        
        st.session_state.base_services = {
            "embedding_service": embedding_service,
            "generation_service": generation_service,
            "retrieval_service": retrieval_service,
            "augmenter": augmenter,
            "s3_client": s3_client,
        }
    st.success(f"LLM initialized: {generation_service.model}", icon="✅")

# === Pre-processed PDFs Selector ===
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Select Documents")

s3 = st.session_state.base_services["s3_client"]
bucket = bucket_name

# 1. List all metadata JSONs
res = s3.list_objects_v2(Bucket=bucket)
keys = [o["Key"] for o in res.get("Contents", []) if o["Key"].endswith("_md_meta.json")]

# 2. Read titles
titles = {}
hashes = []
for key in keys:
    hash_ = key.removesuffix("_md_meta.json")
    hashes.append(hash_)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    meta = json.loads(body)
    titles[hash_] = meta.get("title", hash_)

# 3. Render with big 📄 + checkbox
selected = []
for h in hashes:
    col1, col2 = st.sidebar.columns([4, 1])
    col1.markdown(
        f"<span style='font-size:1.5rem'>📄</span>  **{titles[h]}**",
        unsafe_allow_html=True,
    )
    if col2.checkbox("", key=f"doc_{h}"):
        selected.append(h)

# 4. Mutate the corpus in place
corpus = st.session_state.corpus
corpus.clear()

added = 0
for h in selected:
    chunks_key = f"{h}_chunks_embedded.json"
    dicts = json.loads(
        s3.get_object(Bucket=bucket, Key=chunks_key)["Body"].read().decode('utf-8')
    )
    chunks = [DocumentParser._reconstruct_chunk_from_dict(dic) for dic in dicts]
    added += corpus.add_chunks(chunks)

# 5. Feedback
if selected:
    st.sidebar.success(f"Loaded {added} chunks from {len(selected)} doc(s)", icon="✅")
else:
    st.sidebar.info("No documents selected; corpus is empty.")
st.sidebar.info(f"Total chunks in corpus: {len(corpus.get_all_chunks())}")

# === Retrieval Configuration ===
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Retrieval Settings")
top_k = st.sidebar.slider("Top K Chunks", 1, 10, 3)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.32)

# === Chat Input ===
user_input = st.chat_input("Ask a question...")

if user_input and api_key:
    with st.spinner("Generating response..."):
        # Create new processor with current config values
        current_config = ProcessorConfig(
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
            config=current_config
        )
        
        pt.mark("RAG Processing Query")
        response = processor.process_query(user_input)
        pt.mark("RAG Processing Query")
        st.session_state.chat_history.append({"user": user_input, "bot": response})

# === Display Chat ===
for exchange in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(exchange["user"])
    with st.chat_message("assistant"):
        st.markdown(exchange["bot"])
