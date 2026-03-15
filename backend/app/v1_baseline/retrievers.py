import os
import time
from dotenv import load_dotenv
#from pinecone import Pinecone, ServerlessSpec

import pinecone 
#from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from backend.app.data_loader import load_item_documents, load_kb_documents

load_dotenv()


# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "item-diagnostics" # Pinecone likes hyphens, not underscores

def build_retrievers():
    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1. Create Index if it doesn't exist
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"🌲 Creating Pinecone Index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1") # Great for Vercel/Serverless
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)

    # 2. Connect to the Index
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # 3. Ingest Data if Empty
    stats = pc.Index(INDEX_NAME).describe_index_stats()
    if stats['total_vector_count'] == 0:
        print("🟡 Ingesting documents into Pinecone...")
        docs = load_item_documents() + load_kb_documents()
        
        # Pinecone handles LangChain Document metadata perfectly
        vectorstore.add_documents(docs)
        print(f"✅ Ingested {len(docs)} documents.")

    # 4. Build Retrievers
    # No "metadata.metadata" - just standard metadata filtering
    item_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4, "filter": {"type": "item"}}
    )
    
    knowledge_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4, "filter": {"type": "knowledge"}}
    )

    return item_retriever, knowledge_retriever