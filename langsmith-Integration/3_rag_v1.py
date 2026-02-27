# pip install -U langchain langchain-google-genai langchain-community faiss-cpu pypdf python-dotenv

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
#   3. FAISS index is saved to disk after first run ("faiss_index/" folder)
#      so subsequent runs load instantly with ZERO embedding API calls
#   4. Uses "models/gemini-embedding-001" — the only free embedding model
# ========================================================================

import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_PROJECT'] = 'RAG App'

load_dotenv()  # expects GOOGLE_API_KEY in .env

PDF_PATH = "islr.pdf"  # <-- change to your PDF filename
FAISS_INDEX_PATH = "faiss_index"  # local folder to cache the index

emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Load cached index if available, otherwise build from PDF
if os.path.exists(FAISS_INDEX_PATH):
    print("Loading cached FAISS index...")
    vs = FAISS.load_local(FAISS_INDEX_PATH, emb, allow_dangerous_deserialization=True)
else:
    # 1) Load PDF
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    # 2) Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks. Embedding (this may take a while on free tier)...")

    # 3) Embed + index in batches to respect rate limits
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

retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)
