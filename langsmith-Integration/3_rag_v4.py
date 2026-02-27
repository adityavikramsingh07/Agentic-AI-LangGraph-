# pip install -U langchain langchain-google-genai langchain-community faiss-cpu pypdf python-dotenv langsmith

# ========================================================================
# FREE TIER EMBEDDING OPTIMIZATIONS:
# -----------------------------------------------------------------------
# Google Gemini free tier limits:
#   - 100 embedding requests per MINUTE
#   - 1,000 embedding requests per DAY
#
# Optimizations applied:
#   1. Larger chunk_size (2000 vs 1000) → fewer chunks → fewer API calls
#   2. Batch embedding (50 chunks/batch) with 60s delay between batches
#      to stay under the per-minute rate limit
#   3. FAISS index is saved to disk after first run (".indices/" folder)
#      so subsequent runs load instantly with ZERO embedding API calls
#   4. Uses "models/gemini-embedding-001" — the only free embedding model
# ========================================================================

import os
import json
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # expects GOOGLE_API_KEY in .env

PDF_PATH = "islr.pdf"  # change to your file
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

EMBED_MODEL = "models/gemini-embedding-001"
emb = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

# ----------------- helpers (traced) -----------------
@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()  # list[Document]

@traceable(name="split_documents")
def split_documents(docs, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    # Batch embedding to respect free tier rate limits
    batch_size = 50
    vs = FAISS.from_documents(splits[:batch_size], emb)
    print(f"Embedded {min(batch_size, len(splits))}/{len(splits)} chunks...")
    for i in range(batch_size, len(splits), batch_size):
        time.sleep(60)  # wait 60s between batches for free tier
        batch = splits[i:i + batch_size]
        vs.add_documents(batch)
        print(f"Embedded {min(i + batch_size, len(splits))}/{len(splits)} chunks...")
    return vs

# ----------------- cache key / fingerprint -----------------
def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str) -> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()

# ----------------- explicitly traced load/build runs -----------------
@traceable(name="load_index", tags=["index"])
def load_index_run(index_dir: Path):
    return FAISS.load_local(
        str(index_dir),
        emb,
        allow_dangerous_deserialization=True
    )

@traceable(name="build_index", tags=["index"])
def build_index_run(pdf_path: str, index_dir: Path, chunk_size: int, chunk_overlap: int):
    docs = load_pdf(pdf_path)  # child
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # child
    vs = build_vectorstore(splits)  # child
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    (index_dir / "meta.json").write_text(json.dumps({
        "pdf_path": os.path.abspath(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": EMBED_MODEL,
    }, indent=2))
    print("FAISS index saved to disk.")
    return vs

# ----------------- dispatcher (not traced) -----------------
def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    force_rebuild: bool = False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, EMBED_MODEL)
    index_dir = INDEX_ROOT / key
    cache_hit = index_dir.exists() and not force_rebuild
    if cache_hit:
        print("Loading cached FAISS index...")
        return load_index_run(index_dir)
    else:
        print("Building index (this may take a while on free tier)...")
        return build_index_run(pdf_path, index_dir, chunk_size, chunk_overlap)

# ----------------- model, prompt, and pipeline -----------------
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str, chunk_size=2000, chunk_overlap=200, force_rebuild=False):
    return load_or_build_index(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force_rebuild=force_rebuild,
    )

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    force_rebuild: bool = False,
):
    vectorstore = setup_pipeline(pdf_path, chunk_size, chunk_overlap, force_rebuild)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    chain = parallel | prompt | llm | StrOutputParser()

    return chain.invoke(
        question,
        config={"run_name": "pdf_rag_query", "tags": ["qa"], "metadata": {"k": 4}}
    )

# ----------------- CLI -----------------
if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(PDF_PATH, q)
    print("\nA:", ans)
