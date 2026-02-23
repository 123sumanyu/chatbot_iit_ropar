from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import os

# =================================================
# EMBEDDINGS (LOCAL – NO API)
# =================================================
def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =================================================
# LOAD & SPLIT PDF
# =================================================
loader = PyPDFLoader("merged_docs.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
text_chunks = text_splitter.split_documents(documents)


# =================================================
# CHROMA VECTOR STORE (PERSISTENT)
# =================================================
embeddings = download_hugging_face_embeddings()
PERSIST_DIR = "./chroma_db"

if not os.path.exists(PERSIST_DIR):
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectorstore.persist()
    print(f"Stored {len(text_chunks)} vectors in ChromaDB")
else:
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# =================================================
# LLM INITIALIZATION (MISTRAL)
# =================================================
# =========================
# LLM INITIALIZATION
# =========================



llm_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.1,
    max_new_tokens=512
)

llm = ChatHuggingFace(llm=llm_endpoint)



# =================================================
# PROMPT TEMPLATE (YOUR FORMAT – MODIFIED FOR IPR)
# =================================================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant.
You are an expert in Intellectual Property Rights (IPR).

Rules:
- Never say "based on the context"
- Never explain the source
- Answer directly and clearly
- If the answer is not in the context, say "I don't know"
- If the user greets you, greet them politely

Context:
{context}

Question:
{question}

Answer:
"""
)


# =================================================
# RAG CHAIN (MODERN LANGCHAIN)
# =================================================
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# =================================================
# FUNCTION USED BY FLASK
# =================================================
import markdown

def get_response(query: str) -> str:
    response = rag_chain.invoke(query)

    html_response = markdown.markdown(response.content)

    return html_response
