import os
from typing import List, Optional, Any
import faiss
import numpy as np
import pickle
import hashlib
from typing import Iterable

from .utils.document_processor import DocumentProcessor


class SimpleHashEmbeddings:
    """Deterministic, dependency-light text -> vector transformation.

    This uses repeated SHA256 hashing to produce a fixed-length float vector.
    It's not a semantic embedding like OpenAI/Gemini, but it avoids external
    embedding APIs while allowing FAISS storage and retrieval to work.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _text_to_vector(self, text: str) -> np.ndarray:
        # produce 'dim' floats deterministically from the text
        out = np.zeros(self.dim, dtype=np.float32)
        i = 0
        counter = 0
        while i < self.dim:
            m = hashlib.sha256()
            m.update(text.encode("utf-8"))
            m.update(counter.to_bytes(4, "little", signed=False))
            digest = m.digest()
            # expand digest bytes into floats
            for b in digest:
                if i >= self.dim:
                    break
                out[i] = (b - 128) / 128.0
                i += 1
            counter += 1
        # normalize
        norm = np.linalg.norm(out)
        if norm > 0:
            out = out / norm
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vector(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._text_to_vector(text).tolist()


class SimpleFAISSStore:
    """A tiny FAISS-backed store that holds texts and vectors and can persist to disk.

    This is intentionally minimal: it stores a list of texts and a corresponding FAISS index.
    """
    def __init__(self, persist_directory: str, embeddings: SimpleHashEmbeddings):
        self.persist_directory = persist_directory
        self.embeddings = embeddings
        os.makedirs(persist_directory, exist_ok=True)
        self.index_path = os.path.join(persist_directory, "index.faiss")
        self.meta_path = os.path.join(persist_directory, "meta.pkl")
        self.texts: list[str] = []
        self.index = None
        # try to load
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self.texts = pickle.load(f)
            except Exception:
                # fallback to empty
                self.index = None

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

    def add_texts(self, texts: Iterable[str]):
        texts = list(texts)
        if not texts:
            return
        vectors = self.embeddings.embed_documents(texts)
        vecs = np.array(vectors).astype("float32")
        dim = vecs.shape[1]
        self._ensure_index(dim)
        self.index.add(vecs)
        self.texts.extend(texts)

    def save_local(self, path: str | None = None):
        p = path or self.persist_directory
        os.makedirs(p, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.texts, f)

    def search(self, query: str, k: int = 4):
        if self.index is None or len(self.texts) == 0:
            return []
        qv = np.array([self.embeddings.embed_query(query)]).astype("float32")
        D, I = self.index.search(qv, k)
        results = []
        for idx in I[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])
        return results


class RAGSystem:
    def __init__(self, persist_directory: str = "vectorstore", chunk_size: int = 1000, chunk_overlap: int = 200, model_name: str = "chat-bison-001", temperature: float = 0.0):
        self.persist_directory = persist_directory
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        # use local, deterministic embeddings (no OpenAI dependency)
        self.embeddings = SimpleHashEmbeddings(dim=256)
        self.store = SimpleFAISSStore(persist_directory, self.embeddings)
        self.model_name = model_name
        self.temperature = temperature

    def add_documents(self, documents: list[str]):
        self.store.add_texts(documents)
        self.store.save_local(self.persist_directory)

    def process_and_add_file(self, file_path: str):
        docs = self.document_processor.process_file(file_path)
        texts = [d.page_content for d in docs]
        self.add_documents(texts)

    def process_and_add_directory(self, directory_path: str):
        docs = self.document_processor.process_directory(directory_path)
        texts = [d.page_content for d in docs]
        self.add_documents(texts)

    def query(self, question: str) -> dict:
        # retrieve
        hits = self.store.search(question, k=4)
        context = "\n\n".join(hits)
        # Require Gemini (google-generativeai) to be configured. No OpenAI fallback.
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY not set. Please set GEMINI_API_KEY in your environment to use the Gemini API.")
        try:
            import google.generativeai as genai
        except Exception:
            raise RuntimeError("google-generativeai is not installed. Run: pip install google-generativeai")
        genai.configure(api_key=gemini_key)
        gemini_model = os.getenv("GEMINI_MODEL", self.model_name)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        # Use the high-level text generation API
        resp = genai.generate_text(model=gemini_model, input=prompt)
        # Try to extract textual response
        answer = None
        if hasattr(resp, "text"):
            answer = resp.text
        elif isinstance(resp, dict):
            if "candidates" in resp and len(resp["candidates"])>0:
                c = resp["candidates"][0]
                answer = c.get("content") or c.get("output") or c.get("text")
        if answer is None:
            answer = str(resp)
        return {"answer": answer, "source_documents": hits}

    def save_uploaded_file(self, file: Any, upload_dir: str = "uploads") -> str:
        file_path = self.document_processor.save_upload(file, upload_dir)
        self.process_and_add_file(file_path)
        return file_path