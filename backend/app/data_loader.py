# backend/app/data_loader.py

import json
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Resolve path relative to backend directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_item_documents():
    """
    Load item narrative documents from JSON.
    Returns list[Document]
    """
    with open(DATA_DIR / "item_cases_narrative.json", "r") as f:
        narrative_data = json.load(f)

    documents = [
        Document(
            page_content=entry["text"],
            metadata={
                "item_id": entry["metadata"]["item_id"],
                "type": "item"
            }
        )
        for entry in narrative_data
    ]

    return documents


def load_kb_documents():
    """
    Load KB articles and chunk them.
    Returns list[Document]
    """
    #with open(DATA_DIR / "intervention_kb.json", "r") as f:
    with open(DATA_DIR / "noisy_intervention_kb.json", "r") as f:
        kb_data = json.load(f)

    base_docs = [
        Document(
            page_content=entry["text"],
            metadata={
                "source": entry["id"],
                "type": "knowledge"
            }
        )
        for entry in kb_data
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunked_docs = []

    for doc in base_docs:
        chunks = splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": doc.metadata["source"],
                        "type": "knowledge",
                        "chunk_index": i
                    }
                )
            )

    return chunked_docs