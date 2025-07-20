# rag-lite/test_local_retrieval.py
"""
Standalone end-to-end retrieval & reranking tester.
Run manually (not via pytest):
  $ python rag-lite/run_local_retrieval.py
Generates `tests/test_local_retrieval_report.md` and exits with code 1
if any expected section is missing.
"""
import sys
from pathlib import Path
import yaml

from rag_pipeline import (
    Corpus,
    RetrievalService,
    RetrievalConfig,
    Query,
    CosineSimilarity
)
from parser_local import DocumentParser
from ollama_services import OllamaEmbeddingService, LocalCacheService
from helpers import load_config

TOP_K = 3
SIMILARITY_THRESHOLD = 0.42

def main():
    tests_dir = Path("tests/")

    # 1. Load test definitions
    yaml_path = tests_dir / "retrieval_tests.yaml"
    with open(yaml_path, 'r') as f:
        test_cases = yaml.safe_load(f)

    # 2. Initialize services
    embedding_service = OllamaEmbeddingService(load_config('embedding_model'))
    similarity_metric = CosineSimilarity()
    reranker_model = load_config('reranker_model')

    cache = LocalCacheService(str(tests_dir / '.cache'))
    parser = DocumentParser(embedding_service, cache, bucket_name='tests')

    # 3. Build corpus from all docs under tests/documents
    corpus = Corpus()
    docs_dir = tests_dir / "documents"
    for doc_path in docs_dir.glob("*.docx"):
        chunks = parser.parse_docx(doc_path)
        corpus.add_chunks(chunks)

    # 4. Initialize retrieval service (with BGE reranker)
    retriever = RetrievalService(corpus, similarity_metric, reranker_model)
    # to test recall, retrieve all candidates
    cfg = RetrievalConfig(top_k=TOP_K, similarity_threshold=SIMILARITY_THRESHOLD)

    # 5. Execute each case, compute recall
    results = []
    missing = []
    for case in test_cases:
        q_text = case['question']
        expected = [
            (e['file_name'], e['section_number'])
            for e in case['expected_sections']
        ]

        q_embed = embedding_service.embed_text(q_text)
        query = Query(text=q_text, embedding=q_embed)
        retrieved = retriever.retrieve_similar_chunks(query, cfg)

        retrieved_pairs = [
            (rc.chunk.metadata.file_name, rc.chunk.metadata.section_number)
            for rc in retrieved
        ]

        tp = set(retrieved_pairs) & set(expected)
        recall = len(tp) / len(expected) if expected else 0.0
        results.append({
            'question': q_text,
            'expected': expected,
            'retrieved': retrieved_pairs,
            'recall': recall
        })

        for exp in expected:
            if exp not in retrieved_pairs:
                missing.append((q_text, exp))

    # 6. Emit markdown report
    report_path = tests_dir / "test_local_retrieval_report.md"
    with open(report_path, 'w') as rpt:
        rpt.write("# Retrieval Test Report\n\n")
        rpt.write("| Question | Expected | Retrieved | Recall |\n")
        rpt.write("|---|---|---|---|\n")
        for r in results:
            exp_str = ", ".join(f"{f}:{s}" for f, s in r['expected'])
            ret_str = ", ".join(f"{f}:{s}" for f, s in r['retrieved'])
            rpt.write(f"| {r['question']} | {exp_str} | {ret_str} | {r['recall']:.2f} |\n")
        avg = sum(r['recall'] for r in results) / len(results) if results else 0.0
        rpt.write(f"\n**Average Recall:** {avg:.2f}\n")

    print(f"Report written to {report_path}")
    if missing:
        print("\nMissing expected sections:")
        for q, exp in missing:
            print(f"  - Query '{q}' missing: {exp}")
        sys.exit(1)

    print("All expected sections retrieved ✓")
    sys.exit(0)

if __name__ == '__main__':
    main()