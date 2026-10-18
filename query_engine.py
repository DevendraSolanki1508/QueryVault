import chromadb
import ollama
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever

# ── Configuration ──────────────────────────────────────────────
CHROMA_DIR  = "./chroma_db"
COLLECTION  = "notesmind"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gemma3:270m"

def load_query_engine():
    print("Loading embedding model...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.embed_model = embed_model
    Settings.llm = None

    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection    = chroma_client.get_collection(COLLECTION)
    vector_store  = ChromaVectorStore(chroma_collection=collection)
    index         = VectorStoreIndex.from_vector_store(vector_store)

    # ── BM25 nodes from ChromaDB ───────────────────────────────
    result = collection.get(include=["documents", "metadatas"])
    nodes = [
        TextNode(text=doc, metadata=meta)
        for doc, meta in zip(result["documents"], result["metadatas"])
    ]

    # ── Retrievers ─────────────────────────────────────────────
    vector_retriever = index.as_retriever(similarity_top_k=5)
    bm25_retriever   = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=5
    )
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False
    )

    # ── Reranker ───────────────────────────────────────────────
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        top_n=4
    )

    print("✓ Query engine ready!\n")
    return hybrid_retriever, reranker, nodes

def chat(query: str, retriever, reranker, history: list):
    from llama_index.core.schema import QueryBundle

    nodes = retriever.retrieve(query)
    nodes = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query))

    # remove duplicate chunks
    seen, unique = set(), []
    for n in nodes:
        if n.text not in seen:
            seen.add(n.text)
            unique.append(n)
    nodes = unique

    context = "\n\n---\n\n".join([n.text for n in nodes])

    prompt = (
        "Use the context below to answer the question. "
        "Write 2-4 sentences in your own words. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    print("Thinking...")
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 2048, "temperature": 0.3, "repeat_penalty": 1.2}
    )
    answer = response["message"]["content"]

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return answer, nodes