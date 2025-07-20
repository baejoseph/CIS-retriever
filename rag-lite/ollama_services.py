import os
import json
import shutil
from typing import List, Protocol, Dict
from io import BytesIO
import ollama
from logger import logger


# === Embedding Service Protocol ===
class EmbeddingService(Protocol):
    def embed_text(self, text: str) -> List[float]:
        ...

# === Generation Service Protocol ===
class GenerationService(Protocol):
    def generate_response(self, augmented_prompt: str) -> str:
        ...


class OllamaEmbeddingService:
    def __init__(self, model: str = "mxbai-embed-large"):
        self.model = model
        logger.info("Initialized OpenAIEmbeddingService with model: %s", model)

    def embed_text(self, texts)-> List[float]:
        """
        texts: str → List[float]
        """
        return ollama.embeddings(model=self.model, prompt=texts).embedding


class OllamaGenerationService:
    def __init__(self, model: str = "deepseek-r1:latest"):
        self.model = model

    def generate_response(self, prompt: str) -> str:
        response_chunks = ollama.generate(model=self.model, prompt=prompt, stream=True)
        collected = ""
        for chunk in response_chunks:
            token = chunk['response']
            collected += token
            # you could yield or push token to Streamlit here
            yield token
        


class LocalCacheService:
    """
    A drop-in replacement for boto3 S3 client methods: get_object, put_object,
    upload_file, download_file, upload_fileobj. Stores files under cache_dir/<Bucket>/<Key>.
    """
    def __init__(self, cache_dir="local_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, Bucket, Key):
        return os.path.join(self.cache_dir, Bucket, Key)

    def get_object(self, Bucket, Key):
        path = self._path(Bucket, Key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{Bucket}/{Key} not found in local cache")
        return {"Body": open(path, "rb")}

    def put_object(self, Bucket, Key, Body):
        path = self._path(Bucket, Key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Body may be bytes, str, or file-like
        if hasattr(Body, "read"):
            data = Body.read()
        elif isinstance(Body, str):
            data = Body.encode()
        elif isinstance(Body, (bytes, bytearray)):
            data = Body
        else:
            data = json.dumps(Body).encode()
        with open(path, "wb") as f:
            f.write(data)

    def upload_file(self, Filename, Bucket, Key):
        dest = self._path(Bucket, Key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(Filename, dest)

    def download_file(self, Bucket, Key, Filename):
        src = self._path(Bucket, Key)
        if not os.path.exists(src):
            raise FileNotFoundError(f"{Bucket}/{Key} not in cache")
        os.makedirs(os.path.dirname(Filename), exist_ok=True)
        shutil.copyfile(src, Filename)

    def upload_fileobj(self, Fileobj, Bucket, Key):
        data = Fileobj.read()
        self.put_object(Bucket, Key, data)
