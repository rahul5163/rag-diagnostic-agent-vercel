import os
import time
from dotenv import load_dotenv
import pinecone
from pinecone import ServerlessSpec

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
#from langchain_community.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from backend.app.data_loader import load_item_documents, load_kb_documents

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "item-diagnostics-v2"

def build_retrievers():
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create index if needed
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"🌲 Creating Pinecone Index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Ingest if empty
    stats = pc.Index(INDEX_NAME).describe_index_stats()
    if stats['total_vector_count'] == 0:
        print("🟡 Ingesting documents into Pinecone...")
        docs = load_item_documents() + load_kb_documents()
        vectorstore.add_documents(docs)
        print(f"✅ Ingested {len(docs)} documents.")

    # -------------------------------
    # 1️⃣ Item Retriever (unchanged)
    # -------------------------------
    item_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4, "filter": {"type": "item"}}
    )

    # -------------------------------
    # 2️⃣ Dense Knowledge Retriever
    # -------------------------------
    dense_knowledge_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6, "filter": {"type": "knowledge"}}
    )

    # -------------------------------
    # 3️⃣ Cohere Cross-Encoder Reranker
    # -------------------------------
    compressor = CohereRerank(
        model="rerank-v3.5",
        top_n=3
    )

    knowledge_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=dense_knowledge_retriever
    )

    return item_retriever, knowledge_retriever