#!/usr/bin/env python3
import os
import boto3
import json
import logging

# Adjust this import to wherever your parser lives:
from parser import DocumentParser
from openai_services import OpenAIEmbeddingService
from helpers import load_config
from log_time import ProcessTimer
from dotenv import load_dotenv

pt = ProcessTimer()
load_dotenv()


api_key = os.environ["OPENAI_API_KEY"]
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
bucket_name = os.getenv("AWS_S3_BUCKET")

def main():
    logging.basicConfig(level=logging.INFO)
    embed_service = OpenAIEmbeddingService(api_key) # OllamaEmbeddingService(load_config('embedding_model'))
    cache_service = boto3.client("s3",
                                 aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key,
                                 region_name=aws_region,
    )
    parser = DocumentParser(
        embedding_service=embed_service,
        cache_service=cache_service,
        bucket_name=bucket_name
    )

    pt.mark("Document parsing")
    pdf_dir = 'tmp'
    list_of_chunks = []
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(pdf_dir, fname)
        print(f"Parsing PDF at {pdf_path!r}…")
        chunks = parser.parse_pdf(pdf_path)
        list_of_chunks.append(chunks)
        print(f"→ Parsed {len(chunks)} chunks.\n")
    pt.done("Document parsing")

if __name__ == '__main__':
    main()