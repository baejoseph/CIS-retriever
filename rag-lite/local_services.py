import os
import json
import shutil
from io import BytesIO

class LocalEmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, texts):
        """
        texts: List[str] → List[List[float]]
        """
        embs = self.model.encode(texts, convert_to_numpy=True)
        return embs.tolist()


class LocalGenerationService:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            Text2TextGenerationPipeline
        )
        self.model = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_obj  = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.generator  = Text2TextGenerationPipeline(
            model=self.model_obj,
            tokenizer=self.tokenizer,
            device=-1,          # CPU
            max_length=256,
            do_sample=False
        )

    def generate_response(self, prompt: str) -> str:
        out = self.generator(prompt)
        return out[0]["generated_text"].strip()


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
