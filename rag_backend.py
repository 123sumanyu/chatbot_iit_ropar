
import os
import logging
import hashlib
import time
from functools import lru_cache
from threading import Lock

import markdown
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# =================================================
# LOGGING
# =================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =================================================
# CONFIG — tweak here, not scattered across code
# =================================================
PDF_PATH       = "merged_docs.pdf"
PERSIST_DIR    = "./chroma_db"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO       = "mistralai/Mistral-7B-Instruct-v0.2"
CHUNK_SIZE     = 800
CHUNK_OVERLAP  = 150
TOP_K          = 3          # docs retrieved per query
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.1
CACHE_SIZE     = 256        # number of query→response pairs to cache in RAM
MAX_QUERY_LEN  = 1000       # reject suspiciously long inputs early


# =================================================
# EMBEDDINGS  (loaded once, reused)
# =================================================
logger.info("Loading embedding model …")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    # Encode in batches for speed; cache model in memory
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
)
logger.info("Embedding model ready.")


# =================================================
# VECTOR STORE
# =================================================
def _build_vectorstore() -> Chroma:
    """Load PDF → split → embed → persist, or reload from disk."""
    if os.path.exists(PERSIST_DIR):
        logger.info("Loading existing ChromaDB from %s", PERSIST_DIR)
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )

    logger.info("Building ChromaDB from %s …", PDF_PATH)
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found: '{PDF_PATH}'. "
            "Place merged_docs.pdf next to rag_backend.py."
        )

    loader   = PyPDFLoader(PDF_PATH)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vs.persist()
    logger.info("Stored %d vectors in ChromaDB.", len(chunks))
    return vs


vectorstore = _build_vectorstore()
retriever   = vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# =================================================
# LLM  (single instance, thread-safe via LangChain)
# =================================================
logger.info("Connecting to HuggingFace endpoint …")
_llm_endpoint = HuggingFaceEndpoint(
    repo_id=LLM_REPO,
    temperature=TEMPERATURE,
    max_new_tokens=MAX_NEW_TOKENS,
    # Slightly faster: skip special tokens in the streamed output
    streaming=False,
)
llm = ChatHuggingFace(llm=_llm_endpoint)
logger.info("LLM ready.")


# =================================================
# PROMPT
# =================================================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are IPR-Assist, an expert AI assistant on Intellectual Property Rights (IPR) in India.

Rules:
- Answer directly and concisely; no filler phrases like "based on the context"
- Use plain language; define legal terms when first used
- If the answer is not available, say exactly: "I don't have enough information to answer that."
- For greetings, respond warmly and briefly, then offer to help with IPR questions
- Never fabricate case numbers, dates, or statute sections

Context:
{context}

Question:
{question}

Answer:""",
)


# =================================================
# RAG CHAIN
# =================================================
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)


# =================================================
# RESPONSE CACHE  (keyed on normalised query hash)
# =================================================
_cache: dict[str, str] = {}   # { sha256_hex: html_string }
_cache_lock = Lock()


def _cache_key(query: str) -> str:
    """Normalise + hash so minor whitespace differences still hit cache."""
    normalised = " ".join(query.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()


def _get_cached(key: str) -> str | None:
    with _cache_lock:
        return _cache.get(key)


def _set_cached(key: str, value: str) -> None:
    with _cache_lock:
        if len(_cache) >= CACHE_SIZE:
            # Evict the oldest inserted key (Python 3.7+ dict preserves order)
            oldest = next(iter(_cache))
            del _cache[oldest]
        _cache[key] = value


# =================================================
# INPUT VALIDATION
# =================================================
def _validate_query(query: str) -> str:
    """Clean and validate; raise ValueError for bad input."""
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    if len(query) > MAX_QUERY_LEN:
        raise ValueError(
            f"Query too long ({len(query)} chars). "
            f"Please keep it under {MAX_QUERY_LEN} characters."
        )
    return query


# =================================================
# PUBLIC FUNCTION  (called by Flask)
# =================================================
def get_response(query: str) -> str:
    """
    Returns an HTML string with the assistant's answer.

    Raises:
        ValueError  – bad / empty input  → Flask should return 400
        RuntimeError – LLM / chain error → Flask should return 500
    """
    # 1. Validate
    query = _validate_query(query)

    # 2. Cache hit — instant return
    key = _cache_key(query)
    cached = _get_cached(key)
    if cached:
        logger.info("Cache HIT for query (len=%d)", len(query))
        return cached

    # 3. Call the RAG chain with timing + error handling
    logger.info("Cache MISS — invoking RAG chain …")
    t0 = time.perf_counter()
    try:
        response = rag_chain.invoke(query)
    except Exception as exc:
        logger.exception("RAG chain failed: %s", exc)
        raise RuntimeError(
            "The assistant encountered an error while processing your request. "
            "Please try again in a moment."
        ) from exc

    elapsed = time.perf_counter() - t0
    logger.info("RAG chain responded in %.2fs", elapsed)

    # 4. Post-process
    raw_text = (response.content or "").strip()
    if not raw_text:
        raw_text = "I don't have enough information to answer that."

    html_response = markdown.markdown(
        raw_text,
        extensions=["nl2br", "sane_lists"],
    )

    # 5. Cache and return
    _set_cached(key, html_response)
    return html_response