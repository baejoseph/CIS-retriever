#!/usr/bin/env python3
import os
import io
import json
import logging

# Adjust this import to wherever your parser lives:
from parser_local import DocumentParser
from ollama_services import OllamaEmbeddingService, LocalCacheService 
from openai_services import OpenAIEmbeddingService
from helpers import load_config
from log_time import ProcessTimer
from dotenv import load_dotenv

pt = ProcessTimer()
load_dotenv()


api_key = os.environ["OPENAI_API_KEY"]


def main():
    logging.basicConfig(level=logging.INFO)
    embed_service = OpenAIEmbeddingService(api_key) # OllamaEmbeddingService(load_config('embedding_model'))
    cache_service = LocalCacheService()
    parser = DocumentParser(
        embedding_service=embed_service,
        cache_service=cache_service,
        bucket_name='test-bucket'
    )

    pt.mark("Document parsing")
    pdf_dir = 'temp'
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(pdf_dir, fname)
        print(f"Parsing PDF at {pdf_path!r}…")
        chunks = parser.parse_pdf(pdf_path)
        print(f"→ Parsed {len(chunks)} chunks.\n")
    pt.done("Document parsing")

    if not chunks:
        print("No chunks produced.")
        return

    # The document_id is the hash used for the JSON filename:
    doc_id = chunks[0].metadata.document_id
    json_path = os.path.join(parser.cache_root, f"{doc_id}_chunks_embedded.json")

    if not os.path.exists(json_path):
        print(f"❌ Expected JSON output not found at {json_path!r}")
        return

    print(f"✅ JSON output written to {json_path!r}\n")

    # Load and pretty-print the first 3 entries so you can inspect them:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    preview = data[:3]  # first three chunks
    # truncate embeddings to only the first 3 floats
    for chunk in preview:
        chunk['embedding'] = chunk['embedding'][:3]
    print("Preview of first 3 chunks:")
    print(json.dumps(preview, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()