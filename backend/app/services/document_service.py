import os
import re
import shutil
from pathlib import Path
import json
from typing import List, Optional

from filelock import FileLock

from fastapi import UploadFile

import faiss

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    ParentDocumentRetriever,
)
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore

from app.models.documents import ProcessResult
from app.services.llm_factory import build_embeddings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/data/sessions")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))


# ---------------------------------------------------------------------------
# Document Service
# ---------------------------------------------------------------------------

class DocumentService:

    # Class-level singleton for the reranker (expensive to load on every request)
    _reranker_instance: Optional[FlashrankRerank] = None

    def __init__(self, session_id: str):
        self._validate_session_id(session_id)

        self.session_id = session_id
        self.session_dir = os.path.join(UPLOAD_DIR, session_id)
        self.upload_dir = os.path.join(self.session_dir, "uploads")
        self.index_dir = os.path.join(self.session_dir, "index")
        self.parent_store_path = os.path.join(self.session_dir, "parents.json")
        self.lock_path = os.path.join(self.session_dir, ".process.lock")

        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    @staticmethod
    def _validate_session_id(session_id: str):
        if not re.match(r"^[a-f0-9\-]{8,64}$", session_id):
            raise ValueError("Invalid session_id")

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        return Path(filename).name

    # -----------------------------------------------------------------------
    # Upload
    # -----------------------------------------------------------------------

    def save_upload(self, file: UploadFile) -> str:
        filename = self._sanitize_filename(file.filename)

        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are allowed")

        file_path = os.path.join(self.upload_dir, filename)
        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        total_size = 0
        chunk_size = 1024 * 1024  # 1 MB

        try:
            with open(file_path, "wb") as buffer:
                while True:
                    chunk = file.file.read(chunk_size)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > max_bytes:
                        buffer.close()
                        os.remove(file_path)
                        raise ValueError(f"File exceeds {MAX_FILE_SIZE_MB} MB limit")
                    buffer.write(chunk)
        finally:
            file.file.close()

        return filename

    # # ---------------------------
    # # Embedding Factory
    # # ---------------------------

    # def get_embeddings(self, model : str, provider: str = "ollama", api_key: Optional[str] = None):
    #     # TODO : Validate that the provider supports the model
    #     if provider == "google":
    #         if not api_key:
    #             raise ValueError("Google API key required")
    #         return GoogleGenerativeAIEmbeddings(
    #             model=model,
    #             google_api_key=api_key
    #         )

    #     if provider == "ollama":
    #         return OllamaEmbeddings(
    #             base_url=OLLAMA_BASE_URL,
    #             model=model
    #         )

    #     raise ValueError("Unsupported embedding provider")

    # -----------------------------------------------------------------------
    # Processing / Indexing
    # -----------------------------------------------------------------------

    def process_documents(
        self,
        model: str,
        provider: str = "ollama",
        api_key: Optional[str] = None,
    ) -> ProcessResult:
        """
        Load all uploaded PDFs, chunk them with a parent-child strategy,
        embed with the chosen provider, and save a FAISS index to disk.
        """
        with FileLock(self.lock_path):
            files = [
                f for f in os.listdir(self.upload_dir)
                if f.lower().endswith(".pdf")
            ]

            if not files:
                return ProcessResult(status="empty", chunks=0, documents=0,
                                     parent_chunks=0, corrupted_docs=0)

            documents: List[Document] = []
            corrupted_docs = 0

            for f in files:
                file_path = os.path.join(self.upload_dir, f)
                try:
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    # Sanitise unicode (avoids encoding errors downstream)
                    print("Sanitising documents")
                    for page in pages:
                        page.page_content = (
                            page.page_content
                            .encode("utf-8", "ignore")
                            .decode("utf-8", "ignore")
                        )
                    documents.extend(pages)
                except Exception as e:
                    print("Failed to load PDF '%s': %s", f, e)
                    corrupted_docs += 1
                    continue

            if not documents:
                return ProcessResult(status="empty", chunks=0, documents=0,
                                     parent_chunks=0, corrupted_docs=corrupted_docs)

            # Parent-child chunking strategy
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
            )

            # Build embeddings via factory
            api_keys = {provider: api_key} if api_key else {}
            embeddings = build_embeddings(
                provider=provider,
                model=model,
                api_keys=api_keys,
            )

            # Wipe and recreate the index directory
            shutil.rmtree(self.index_dir, ignore_errors=True)
            os.makedirs(self.index_dir, exist_ok=True)

            # Initialise empty FAISS store
            dim = len(embeddings.embed_query("dimension check"))
            index = faiss.IndexFlatL2(dim)
            vectorstore = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )

            docstore = InMemoryStore()

            parent_retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=docstore,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs={"k": 15},
            )
            parent_retriever.add_documents(documents)

            # Persist FAISS index
            vectorstore.save_local(self.index_dir)

            # Persist parent documents (for BM25 + reconstruction)
            serialized_chunks = {
                k: {"page_content": v.page_content, "metadata": v.metadata}
                for k, v in docstore.store.items()
            }
            with open(self.parent_store_path, "w", encoding="utf-8") as f:
                json.dump(serialized_chunks, f)

            print(
                "Indexed %d files → %d FAISS vectors, %d parent chunks",
                len(files),
                vectorstore.index.ntotal,
                len(serialized_chunks),
            )

            return ProcessResult(
                status="success",
                chunks = vectorstore.index.ntotal,
                documents = len(files),
                parent_chunks = len(serialized_chunks),
                corrupted_docs = corrupted_docs
            )

    # -----------------------------------------------------------------------
    # Retriever construction
    # -----------------------------------------------------------------------

    def get_retriever(
        self,
        model: str,
        provider: str = "ollama",
        api_key: Optional[str] = None,
    ):
        """
        Load the saved FAISS index and build a hybrid retriever:
            FAISS (ParentDocumentRetriever) + BM25 → EnsembleRetriever
            → FlashRank reranker → ContextualCompressionRetriever

        Returns None if no index exists yet.
        """
        index_file = os.path.join(self.index_dir, "index.faiss")
        if not os.path.exists(index_file):
            return None

        api_keys = {provider: api_key} if api_key else {}
        embeddings = build_embeddings(
            provider=provider,
            model=model,
            api_keys=api_keys,
        )

        vectorstore = FAISS.load_local(
            self.index_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Rebuild in-memory docstore from JSON
        docstore = InMemoryStore()
        if os.path.exists(self.parent_store_path):
            with open(self.parent_store_path, "r", encoding="utf-8") as f:
                docstore_data = json.load(f)
            docstore.mset([
                (k, Document(page_content=v["page_content"], metadata=v["metadata"]))
                for k, v in docstore_data.items()
            ])

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
        )
        
        base_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": 15},
        )

        # parent_splitter_keys = list(docstore.yield_keys())
        parent_splitter_docs = [docstore.store[k] for k in docstore.store]

        if parent_splitter_docs :
            bm25_retriever = BM25Retriever.from_documents(parent_splitter_docs)
            bm25_retriever.k = 5

            base_retriever = EnsembleRetriever(
                retrievers = [bm25_retriever, base_retriever],
                weights = [0.3, 0.7]
            )

        # Reranker (class-level singleton to avoid reloading the model)
        if not DocumentService._reranker_instance:
            DocumentService._reranker_instance = FlashrankRerank(
                model="ms-marco-MiniLM-L-12-v2",
                top_n=5,
            )

        return ContextualCompressionRetriever(
            base_compressor=DocumentService._reranker_instance,
            base_retriever = base_retriever
        )

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def list_documents(self) -> List[dict]:
        if not os.path.exists(self.upload_dir):
            return []
        result = []
        for f in os.listdir(self.upload_dir):
            path = os.path.join(self.upload_dir, f)
            if os.path.isfile(path):
                result.append({
                    "filename": f,
                    "size_bytes": os.path.getsize(path),
                })
        return result

    def delete_document(self, filename: str):
        filename = self._sanitize_filename(filename)
        file_path = os.path.join(self.upload_dir, filename)

        if os.path.exists(file_path):
            os.remove(file_path)

        # Invalidate the index so stale embeddings don't get served
        shutil.rmtree(self.index_dir, ignore_errors=True)
        os.makedirs(self.index_dir, exist_ok=True)

        if os.path.exists(self.parent_store_path):
            os.remove(self.parent_store_path)