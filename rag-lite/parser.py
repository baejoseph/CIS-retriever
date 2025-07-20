import os
import re
import subprocess
import json
import hashlib
from typing import List, Dict, Any, Tuple
from datetime import datetime
import mammoth
import html2text  
from docx import Document as DocxDocument
from rag_pipeline import DocumentChunk, DocumentMetadata
from logger import logger


class DocumentParser:
    def __init__(self, embedding_service, s3_client, bucket_name):
        self.embedding_service = embedding_service
        self.cache_root = "tmp/cache/"
        os.makedirs(self.cache_root, exist_ok=True)
        self.bucket = bucket_name
        self.s3 = s3_client
        logger.info("DocumentParser initialized with s3 bucket: %s", self.bucket)

    def _hash_docx_metadata(self, docx_file: DocxDocument) -> str:
        core_props = docx_file.core_properties
        key_data = f"{core_props.title}-{core_props.modified}-{core_props.author}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _serialize_chunk(self, chunk: DocumentChunk) -> Dict[str, Any]:
        return {
            'content': chunk.content,
            'metadata': {
                'file_name': chunk.metadata.file_name,
                'file_version': chunk.metadata.file_version,
                'file_date': chunk.metadata.file_date.isoformat(),
                'section_number': chunk.metadata.section_number,
                'section_heading': chunk.metadata.section_heading,
                'document_id': chunk.metadata.document_id
            },
            'embedding': chunk.embedding
        }

    def _reconstruct_chunk_from_dict(self, chunk_dict: Dict[str, Any]) -> DocumentChunk:
        metadata_dict = chunk_dict['metadata']
        if isinstance(metadata_dict, str):
            # Handle old format if needed
            logger.warning("Found old format metadata, clearing cache recommended")
            return DocumentChunk(
                content=chunk_dict['content'],
                metadata=DocumentMetadata(
                    file_name="",
                    file_version="v1",
                    file_date=datetime.now(),
                    section_number="1",
                    section_heading="",
                    document_id=""
                ),
                embedding=chunk_dict['embedding']
            )
        
        metadata_dict['file_date'] = datetime.fromisoformat(metadata_dict['file_date'])
        metadata = DocumentMetadata(**metadata_dict)
        
        return DocumentChunk(
            content=chunk_dict['content'],
            metadata=metadata,
            embedding=chunk_dict['embedding']
        )

    def _process_heading(self, chunk: str) -> Tuple[str, str, str]:
        """Extract heading information from a chunk.
        Returns: (cleaned_content, section_number, section_heading)"""
        heading_match = re.match(r"^(#{1,6})\s+(.+?)(?:\n|$)(.*)", chunk.strip(), re.DOTALL)
        if heading_match:
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            content = (heading + "\n" + heading_match.group(3)).strip()
            return content, str(level), heading
        return chunk.strip(), "", ""

    def parse_docx(self, docx_file: DocxDocument) -> List[DocumentChunk]:
        file_name = docx_file.name
        docx_obj = DocxDocument(docx_file)
        file_hash = self._hash_docx_metadata(docx_obj)
        cache_prefix = f"cache/{file_hash}/"

        chunks_key = cache_prefix + "chunks.json"
        markdown_key = cache_prefix + "converted.md"
        uploaded_key = cache_prefix + "uploaded.docx"

        # Return from cache if exists
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=chunks_key)
            chunk_dicts = json.load(response['Body'])
            logger.info("Loaded chunks from S3: %s", chunks_key)
            return [self._reconstruct_chunk_from_dict(chunk) for chunk in chunk_dicts]
        except self.s3.exceptions.NoSuchKey:
            logger.info("No cached chunks found in S3 for hash: %s", file_hash)

        # Save file to file_path for pandoc subprocess
        local_docx_path = os.path.join(self.cache_root, f"{file_hash}_uploaded.docx")
        docx_obj.save(local_docx_path)
        
        logger.info("Processing new document: %s", file_name)
        self.s3.upload_file(local_docx_path, self.bucket, uploaded_key)
        
        # Load file again for fresh metadata
        core_props = docx_obj.core_properties
        file_date = core_props.modified or datetime.now()

        # Convert DOCX to Markdown using pandoc
        

        with open(local_docx_path, "rb") as f:
            html = mammoth.convert_to_html(f).value

        # use html2text for much cleaner markdown
        converter = html2text.HTML2Text()
        converter.body_width = 0       # no forced wraps
        markdown = converter.handle(html)

        # write out for inspection & cache
        local_md_path = os.path.join(self.cache_root, f"{file_hash}_converted.md")
        with open(local_md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
            
        self.s3.upload_file(local_md_path, self.bucket, markdown_key)

        pattern = r"(?=^#{1,3} .*)"  # Split on headings (up to ###)
        raw_chunks = re.split(pattern, markdown, flags=re.MULTILINE)
        logger.info("Split markdown into %d initial chunks", len(raw_chunks))

        chunks: List[DocumentChunk] = []
        section_counter = {}
        current_heading = ""

        for chunk in raw_chunks:
            if not chunk.strip():
                continue

            chunk_text, level_str, heading = self._process_heading(chunk)
            
            if level_str:  # This is a section with a heading
                level = int(level_str)
                if level not in section_counter:
                    section_counter[level] = 1
                else:
                    section_counter[level] += 1
                for k in list(section_counter.keys()):
                    if k > level:
                        del section_counter[k]
                section_numbers = [str(section_counter[l]) for l in sorted(section_counter.keys()) if l <= level]
                section_number = ".".join(section_numbers)
                current_heading = heading
            else:
                section_number = ".".join(str(section_counter[l]) for l in sorted(section_counter.keys()))

            estimated_tokens = len(chunk_text) // 4
            logger.debug("Processing chunk with ~%d tokens from section %s: %s", 
                        estimated_tokens, section_number, current_heading)
            
            embedding = self.embedding_service.embed_text(chunk_text)
            metadata = DocumentMetadata(
                file_name=file_name,
                file_version="v1",
                file_date=file_date,
                section_number=section_number,
                section_heading=current_heading,
                document_id=file_hash,
            )
            chunks.append(DocumentChunk(content=chunk_text, metadata=metadata, embedding=embedding))

        logger.info("Successfully processed %d chunks with embeddings", len(chunks))

        chunk_dicts = [self._serialize_chunk(chunk) for chunk in chunks]
        self.s3.put_object(
            Bucket=self.bucket,
            Key=chunks_key,
            Body=json.dumps(chunk_dicts, indent=2, default=str),
            ContentType="application/json"
        )
        logger.info("Saved chunks to S3: %s", chunks_key)

        return chunks
