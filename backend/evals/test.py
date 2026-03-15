import json
import os
from pathlib import Path
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from app.agent import build_agent

# Load env
load_dotenv()

# Build agent
agent = build_agent()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "synthetic_evaluation_set.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision

dummy = Dataset.from_list([
    {
        "question": "Test?",
        "answer": "Test answer",
        "contexts": ["Some context"],
        "ground_truth": "Test answer"
    }
])

print("Running tiny test...")
result = evaluate(dummy, metrics=[context_precision])
print("Tiny test finished.")