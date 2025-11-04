from typing import List, Any
import os
from pypdf import PdfReader
import docx2txt


class SimpleDoc:
    def __init__(self, page_content: str):
        self.page_content = page_content


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _read_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext in [".docx", ".doc"]:
            return docx2txt.process(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = max(end - self.chunk_overlap, end)
        return chunks

    def process_file(self, file_path: str) -> List[SimpleDoc]:
        text = self._read_text(file_path)
        parts = self._chunk_text(text)
        return [SimpleDoc(p) for p in parts]

    def process_directory(self, directory_path: str) -> List[SimpleDoc]:
        docs: List[SimpleDoc] = []
        for root, _, files in os.walk(directory_path):
            for fname in files:
                path = os.path.join(root, fname)
                try:
                    docs.extend(self.process_file(path))
                except Exception:
                    # skip unsupported/errored files
                    continue
        return docs

    def save_upload(self, file: Any, upload_dir: str) -> str:
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        return file_path