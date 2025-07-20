from dataclasses import dataclass
from typing import List, Protocol, Optional, Dict
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from logger import logger
from log_time import log_time

# === Data Classes ===

@dataclass(frozen=True)  # Make immutable for better testing
class DocumentMetadata:
    file_name: str
    file_version: str
    file_date: datetime
    section_number: str
    section_heading: str
    section_page: Optional[int] = None
    document_id: Optional[str] = None
    document_tags: Optional[List[str]] = None

    def __post_init__(self):
        if not self.file_name:
            raise ValueError("file_name cannot be empty")
        if not self.file_version:
            raise ValueError("file_version cannot be empty")
        if not isinstance(self.file_date, datetime):
            raise ValueError("file_date must be a datetime object")
        if not self.section_number:
            raise ValueError("section_number cannot be empty")

@dataclass(frozen=True)  # Make immutable for better testing
class DocumentChunk:
    content: str
    metadata: DocumentMetadata
    embedding: List[float]

    def __post_init__(self):
        if not self.content.strip():
            raise ValueError("content cannot be empty")
        if not self.embedding:
            raise ValueError("embedding cannot be empty")
        if not isinstance(self.embedding, list):
            raise ValueError("embedding must be a list")
        if not all(isinstance(x, float) for x in self.embedding):
            raise ValueError("embedding must contain only floats")

@dataclass(frozen=True)
class Query:
    text: str
    embedding: List[float]

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("query text cannot be empty")
        if not self.embedding:
            raise ValueError("embedding cannot be empty")
        if not isinstance(self.embedding, list):
            raise ValueError("embedding must be a list")
        if not all(isinstance(x, float) for x in self.embedding):
            raise ValueError("embedding must contain only floats")

@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    similarity_threshold: float

    def __post_init__(self):
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    similarity_score: float

    def __post_init__(self):
        if not 0 <= self.similarity_score <= 1:
            raise ValueError("similarity_score must be between 0 and 1")

@dataclass(frozen=True)
class ProcessorConfig:
    retrieval: RetrievalConfig

# === Service Interfaces ===

class EmbeddingService(Protocol):
    def embed_text(self, text: str) -> List[float]:
        ...

class SimilarityMetric(Protocol):
    def compute(self, a: List[float], b: List[float]) -> float:
        ...

class GenerationService(Protocol):
    def generate_response(self, augmented_prompt: str) -> str:
        ...

# === Core Corpus ===

class Corpus:
    """Repository for document chunks with duplicate prevention."""
    
    def __init__(self):
        self._chunks: List[DocumentChunk] = []
        self._chunk_ids: Dict[str, bool] = {}  # Using dict for O(1) lookup

    def _make_chunk_id(self, chunk: DocumentChunk) -> str:
        """Create a unique identifier for a chunk."""
        return f"{chunk.metadata.document_id}:{chunk.metadata.section_number}"

    def add_chunk(self, chunk: DocumentChunk) -> bool:
        """
        Add a chunk to the corpus if it doesn't exist.
        Returns True if added, False if duplicate.
        """
        chunk_id = self._make_chunk_id(chunk)
        if chunk_id in self._chunk_ids:
            logger.debug("Skipping duplicate chunk: %s", chunk_id)
            return False
        
        self._chunk_ids[chunk_id] = True
        self._chunks.append(chunk)
        return True

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add multiple chunks to the corpus.
        Returns number of chunks actually added (excluding duplicates).
        """
        return sum(1 for chunk in chunks if self.add_chunk(chunk))

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Get all chunks in the corpus."""
        return self._chunks.copy()  # Return a copy to prevent modification

    def clear(self) -> None:
        """Clear all chunks from the corpus."""
        self._chunks.clear()
        self._chunk_ids.clear()

    def __len__(self) -> int:
        return len(self._chunks)

# === Similarity Implementation ===

class CosineSimilarity:
    """Compute cosine similarity between two vectors."""
    
    def compute(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            raise ValueError("Vectors must have same dimension")
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            raise ValueError("Zero vectors are not allowed")
            
        return dot / (norm_a * norm_b)

# === Retrieval ===

class RetrievalService:
    """Service for retrieving relevant chunks based on query similarity."""
    
    def __init__(self, corpus: Corpus, similarity_metric: SimilarityMetric, reranker_model_name: str = None):
        if not isinstance(corpus, Corpus):
            raise ValueError("corpus must be an instance of Corpus")
            
        self.corpus = corpus
        self.similarity_metric = similarity_metric
        logger.info("Initialized RetrievalService")
        
        # initialize BGE reranker if requested
        if reranker_model_name:
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
            logger.info(f"Loaded BGE re-ranker model: {reranker_model_name}")
        else:
            self.reranker_tokenizer = None
            self.reranker_model = None

    @log_time("rerank_with_bge")
    def rerank_with_bge(
        self,
        query_text: str,
        retrieved_chunks: List[RetrievedChunk],
        top_n: int = 3
    ) -> List[RetrievedChunk]:
        """
        Re-rank retrieved chunks with BGE cross-encoder
        """
        if not self.reranker_model:
            logger.warning("BGE re-ranker is not configured, skipping reranking.")
            return retrieved_chunks
        
        pairs = [(query_text, rc.chunk.content) for rc in retrieved_chunks]
        
        inputs = self.reranker_tokenizer(
            [q for q, c in pairs],
            [c for q, c in pairs],
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.reranker_model(**inputs)
            scores = outputs.logits.squeeze(-1)
        
        # rank the retrieved chunks by BGE re-ranker scores
        sorted_indices = torch.topk(scores, k=min(top_n, len(retrieved_chunks))).indices.tolist()
        
        sorted_indices = [x for x in sorted_indices if scores[x].item() > scores[sorted_indices[0]].item()*0.5]
        
        reranked_chunks = [retrieved_chunks[i] for i in sorted_indices]
        
        for idx,rc in enumerate(reranked_chunks):
            logger.info("Re-ranked chunk from section %s with BGE score %.3f", 
                        rc.chunk.metadata.section_number, scores[sorted_indices[idx]].item())
        
        logger.info("Re-ranked chunks with BGE cross-encoder.")
        return reranked_chunks

    @log_time("retrieve_similar_chunks")
    def retrieve_similar_chunks(
        self, query: Query, config: RetrievalConfig
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks similar to the query based on the config.
        Returns list of chunks sorted by similarity score.
        """
        results = []
        all_chunks = self.corpus.get_all_chunks()
        logger.info("Searching through %d chunks in corpus", len(all_chunks))
        
        for chunk in all_chunks:
            try:
                score = self.similarity_metric.compute(query.embedding, chunk.embedding)
                if score >= config.similarity_threshold:
                    results.append(RetrievedChunk(chunk=chunk, similarity_score=score))
            except ValueError as e:
                logger.error("Error computing similarity: %s", str(e))
                continue
        
        results.sort(key=lambda rc: rc.similarity_score, reverse=True)
        if self.reranker_model:
            results = results[:config.top_k]
        else:
            results = results[:config.top_k]
        
        logger.info("Retrieved %d chunks above similarity threshold %.2f", 
                   len(results), config.similarity_threshold)
        for rc in results:
            logger.info("Retrieved chunk from section %s with score %.3f", 
                        rc.chunk.metadata.section_number, rc.similarity_score)
        
        # pass to reranker if configured
        if self.reranker_model:
            results = self.rerank_with_bge(query.text, results, top_n=config.top_k)
        
        return results

# === Prompt Augmentation ===

class PromptAugmenter:
    """Service for augmenting queries with retrieved context."""
    
    def __init__(self, prompt_template_path: Optional[str] = None):
        """
        Initialize with optional prompt template path.
        If not provided, uses default template.
        """
        self.prompt_template = self._load_template(prompt_template_path)

    def _load_template(self, template_path: Optional[str]) -> str:
        """Load prompt template from file or use default."""
        if template_path:
            with open(template_path, 'r') as f:
                return f.read()
        return "{user_query}\n\nContext:\n{retrieved_chunks_text}"

    def augment_query(self, query: Query, retrieved_chunks: List[RetrievedChunk]) -> str:
        """
        Augment query with retrieved chunks using template.
        """
        logger.info("Augmenting query with %d retrieved chunks", len(retrieved_chunks))
        
        retrieved_chunk_text = ""
        for rc in retrieved_chunks:
            md = rc.chunk.metadata
            context_line = f"[Section {md.section_number} of {md.file_name}]: {rc.chunk.content}"
            retrieved_chunk_text += context_line + "\n"
            
        prompt = self.prompt_template.format(
            user_query=query.text,
            retrieved_chunks_text=retrieved_chunk_text
        )
        
        estimated_tokens = len(prompt) // 4
        logger.debug("Generated augmented prompt with ~%d tokens", estimated_tokens)
        return prompt

# === Query Processor ===

class QueryProcessor:
    """Main service for processing queries using RAG pipeline."""
    
    def __init__(
        self,
        corpus: Corpus,
        embedding_service: EmbeddingService,
        retrieval_service: RetrievalService,
        prompt_augmenter: PromptAugmenter,
        generation_service: GenerationService,
        config: ProcessorConfig
    ):
        self.corpus = corpus
        self.embedding_service = embedding_service
        self.retrieval_service = retrieval_service
        self.prompt_augmenter = prompt_augmenter
        self.generation_service = generation_service
        self.config = config
        logger.info("Initialized QueryProcessor with config: %s", config)

    def pre_gen_process(self, query_text: str) -> str:
        """
        Process a query through the RAG pipeline until the generation.
        Returns augmented response.
        """
        if not query_text.strip():
            raise ValueError("Query text cannot be empty")

        logger.info("Processing query: %s", query_text)
        query_embedding = self.embedding_service.embed_text(query_text)
        query = Query(text=query_text, embedding=query_embedding)

        retrieved_chunks = self.retrieval_service.retrieve_similar_chunks(
            query, self.config.retrieval
        )
        augmented_prompt = self.prompt_augmenter.augment_query(query, retrieved_chunks)
        logger.info("Augmented prompt: %s", augmented_prompt)
        return augmented_prompt

    def process_query(self, query_text: str) -> str:
        """
        Process a query through the RAG pipeline after retrieval.
        Returns generated response.
        """
        augmented_prompt = self.pre_gen_process(query_text)
        response = self.generation_service.generate_response(augmented_prompt)
        logger.info("Query processing completed")
        logger.info("Response: %s", response)
        return response
