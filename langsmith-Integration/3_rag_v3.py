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
#   3. FAISS index is saved to disk after first run ("faiss_index_v3/" folder)
#      so subsequent runs load instantly with ZERO embedding API calls
#   4. Uses "models/gemini-embedding-001" — the only free embedding model
# ========================================================================

import os
import time
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

PDF_PATH = "islr.pdf"  # <- change to your file
FAISS_INDEX_PATH = "faiss_index_v3"  # local folder to cache the index

emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# ----------------- helpers (not traced individually) -----------------
@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()  # list[Document]

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
    # Save index so we don't have to re-embed
    vs.save_local(FAISS_INDEX_PATH)
    print("FAISS index saved to disk.")
    return vs

# ----------------- parent setup function (traced) -----------------
@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str, chunk_size=2000, chunk_overlap=200):
    # ✅ These three steps are "clubbed" under this parent function
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits)
    return vs

# ----------------- model, prompt, and run -----------------
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ----------------- one top-level (root) run -----------------
@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(pdf_path: str, question: str):
    # Load cached index if available, otherwise build from PDF
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading cached FAISS index...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, emb, allow_dangerous_deserialization=True)
    else:
        print("Building index (this may take a while on free tier)...")
        vectorstore = setup_pipeline(pdf_path, chunk_size=2000, chunk_overlap=200)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    chain = parallel | prompt | llm | StrOutputParser()

    # This LangChain run stays under the same root (since we're inside this traced function)
    lc_config = {"run_name": "pdf_rag_query"}
    return chain.invoke(question, config=lc_config)

# ----------------- CLI -----------------
if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(PDF_PATH, q)
    print("\nA:", ans)
