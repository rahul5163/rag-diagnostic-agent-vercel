import json
import os
from pathlib import Path
from dotenv import load_dotenv

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)


#from app.agent import build_agent
from backend.app.v2_rerank.agent import build_agent

# Load env
load_dotenv()

print("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))

# Build agent
agent = build_agent()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "synthetic_evaluation_set.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("🔵 Loading synthetic evaluation set...")

with open(DATA_PATH, "r") as f:
    synthetic_eval = json.load(f)

#synthetic_eval = synthetic_eval[:30]

print(f"Loaded {len(synthetic_eval)} synthetic examples.")

agent_answers = []
agent_contexts = []

print("🔵 Running agent on synthetic dataset...")

for case in synthetic_eval:

    result = agent.invoke(
        {"question": case["question"]},
        config={
            "tags": ["AGENT_EVAL", case.get("query_type"), case.get("difficulty")],
            "metadata": {
                "query_type": case.get("query_type"),
                "difficulty": case.get("difficulty"),
            }
        }
    )

    agent_answers.append(result.get("final_answer", ""))

    contexts = []

    # Item context
    for doc in result.get("item_context", []) or []:
        contexts.append(doc.page_content)

    # Knowledge context
    for doc in result.get("knowledge_context", []) or []:
        contexts.append(doc.page_content)

    # External context (if exists)
    for ext in result.get("external_context", []) or []:
        contexts.append(ext)

    agent_contexts.append(contexts)


print("🔵 Building RAGAS dataset...")

records = []

for i, case in enumerate(synthetic_eval):
    records.append({
    "user_input": case["question"],
    "response": agent_answers[i],
    "retrieved_contexts": agent_contexts[i],
    "reference": case["ground_truth"],
})

eval_dataset = EvaluationDataset.from_list(records)

print("🔵 Running RAGAS evaluation...")

result = evaluate(
    eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)

print("\n✅ Evaluation Complete\n")
print(result)

results_df = result.to_pandas()

overall_metrics = results_df[[
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall"
]].mean()

print("\n📊 Baseline Metrics:")
print(overall_metrics)

# Save outputs
results_df.to_csv(RESULTS_DIR / "reranker_plus_prompt_results.csv", index=False)

with open(RESULTS_DIR / "reranker_plus_prompt_metrics.json", "w") as f:
    json.dump(overall_metrics.to_dict(), f, indent=2)

print("\n✅ Reranker with Stronger Prompt metrics saved.")