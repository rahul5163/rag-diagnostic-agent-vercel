import re
from typing import TypedDict, List, Dict, Any

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
import os

from .retrievers import build_retrievers
from langchain_core.tools import tool


# --------------------------------------------------
# Agent State Definition
# --------------------------------------------------

class AgentState(TypedDict):
    question: str
    item_id: str
    item_context: List[Document]
    item_metrics: Dict[str, Any]
    knowledge_context: List[Document]
    external_context: List[str]
    final_answer: str


# --------------------------------------------------
# Initialize Dependencies (Single Initialization)
# --------------------------------------------------

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Tavily
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def tavily_search(query: str) -> str:
    """
    Search external web for strategic context related to retail or category risk.
    """
    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=3
    )

    results = [r["content"] for r in response.get("results", [])]

    return "\n\n".join(results)

# Pinecone Retrievers
item_retriever, knowledge_retriever = build_retrievers()


# --------------------------------------------------
# Graph Builder Function
# --------------------------------------------------

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("item_lookup", item_lookup_node)
    graph.add_node("signal_extraction", signal_extraction_node)
    graph.add_node("planner", planner_node)
    graph.add_node("knowledge_retrieval", knowledge_retrieval_node)
    graph.add_node("external_retrieval", external_retrieval_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("item_lookup")

    graph.add_edge("item_lookup", "signal_extraction")
    graph.add_edge("signal_extraction", "planner")

    # Conditional routing
    def route_after_planner(state: AgentState):
        if state.get("external_context") is None:
            return "external_retrieval"
        if state.get("knowledge_context") is None:
            return "knowledge_retrieval"
        return "synthesis"

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "external_retrieval": "external_retrieval",
            "knowledge_retrieval": "knowledge_retrieval",
            "synthesis": "synthesis"
        }
    )

    graph.add_edge("knowledge_retrieval", "synthesis")
    graph.add_edge("external_retrieval", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()

# --------------------------------------------------
# Deterministic Item Lookup Node
# --------------------------------------------------

def item_lookup_node(state: AgentState):

    question = state["question"]

    # Extract ITEM_XXX
    match = re.search(r"ITEM_\d+", question)

    if not match:
        return {
            "item_id": None,
            "item_context": []
        }

    item_id = match.group(0)

    # Deterministic Pinecone lookup
    results = item_retriever.vectorstore.similarity_search(
        query="ignore",
        k=1,
        filter={
            "type": "item",
            "item_id": item_id
        }
    )

    return {
        "item_id": item_id,
        "item_context": results
    }

# --------------------------------------------------
# Signal Extraction Node
# --------------------------------------------------

def signal_extraction_node(state: AgentState):

    docs = state.get("item_context", [])

    if not docs:
        return {"item_metrics": {}}

    text = docs[0].page_content

    def extract(pattern):
        match = re.search(pattern, text)
        if not match:
            return None
        value = match.group(1).strip().rstrip(".")
        try:
            return float(value)
        except:
            return None

    metrics = {
        "impressions": extract(r"(\d+)\s+impressions"),
        "rank": extract(r"rank of\s+(\d+)"),
        "ctr": extract(r"click-through rate is\s+(\d+\.\d+)%"),
        "conversion": extract(r"conversion rate is\s+(\d+\.\d+)%"),
        "sales": extract(r"sales are\s+(\d+)"),
        "overlap": extract(r"overlap score is\s+(\d+\.\d+)")
    }

    return {"item_metrics": metrics}

# --------------------------------------------------
# Planner Node (Decides Knowledge Retrieval)
# --------------------------------------------------

from langchain.prompts import PromptTemplate
import json


planner_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a planning agent.

Determine whether external knowledge retrieval is required
to answer the question.

Return STRICT JSON:

{{
  "retrieve_knowledge": true or false
}}

Question:
{question}
"""
)


# Toggle for evaluation experiments
MANDATORY_KNOWLEDGE_RETRIEVAL = True


def planner_node(state: AgentState):

    question = state["question"].lower()

    abstract_keywords = [
        "strategy",
        "business",
        "market",
        "industry",
        "trend",
        "long term",
        "positioning"
    ]

    is_abstract = any(k in question for k in abstract_keywords)

    if is_abstract:
        return {"external_context": None}

    if MANDATORY_KNOWLEDGE_RETRIEVAL:
        return {"knowledge_context": []}

    formatted = planner_prompt.format(question=state["question"])
    response = llm.invoke(formatted)

    try:
        decision = json.loads(response.content)
        retrieve = decision.get("retrieve_knowledge", False)
    except:
        retrieve = False


    if retrieve:
        return {"knowledge_context": None}
    else:
        return {}


# --------------------------------------------------
# Knowledge Retrieval Node
# --------------------------------------------------

def knowledge_retrieval_node(state: AgentState):

    docs = knowledge_retriever.invoke(state["question"])

    return {"knowledge_context": docs}


# --------------------------------------------------
# Synthesis Node
# --------------------------------------------------

from langchain.prompts import PromptTemplate

synthesis_prompt = PromptTemplate(
    input_variables=["item_metrics", "knowledge_context", "question"],
    template="""
You are a Retail Decision Intelligence Engine.

Your objective is to diagnose whether item underperformance is caused by:
- Discoverability constraints
- Weak customer demand
- Cannibalization from similar items
- Mixed or ambiguous signals
- Structural weakness (both low demand and low visibility)

You must reason strictly from quantitative signals provided.
Do not hallucinate missing metrics.

----------------------------------------------------
INPUT SIGNALS
----------------------------------------------------

Item Metrics:
{item_metrics}

Knowledge Context (if provided):
{knowledge_context}

User Question:
{question}

----------------------------------------------------
DIAGNOSTIC FRAMEWORK
----------------------------------------------------

Interpret signals using the following logic:

VISIBILITY SIGNALS
- Impressions
- Search Rank
- CTR

DEMAND SIGNALS
- Conversion Rate
- Sales Volume

CANNIBALIZATION SIGNAL
- Overlap Score

----------------------------------------------------
THRESHOLD GUIDELINES
----------------------------------------------------

Use the following heuristics unless context suggests otherwise:

Discoverability Issue:
- Conversion >= 6%
- Impressions < 5000 OR Rank > 30
- Overlap < 0.5

Demand Weakness:
- Impressions >= 5000
- Conversion < 4%
- Overlap < 0.5

Cannibalization:
- Overlap >= 0.5
- Sales diluted relative to impressions
- Conversion not critically low

Structural Weakness:
- Conversion < 4%
- Impressions < 5000

Mixed:
- Conflicting signals across visibility and demand

If metrics are missing, explain limitations clearly.

----------------------------------------------------
RECOMMENDATION RULES
----------------------------------------------------

Discoverability → Recommend re-ranking or featured placement.
Demand Weakness → Recommend content improvement, pricing review, or delisting test.
Cannibalization → Recommend assortment rationalization or clustering test.
Structural Weakness → Recommend deeper review before intervention.
Mixed → Recommend controlled experiment.

----------------------------------------------------
OUTPUT FORMAT (STRICT)
----------------------------------------------------

Diagnosis Category:
(One of: Discoverability | Demand | Cannibalization | Structural Weakness | Mixed)

Signal Breakdown:
- Visibility Analysis:
- Demand Analysis:
- Cannibalization Analysis:

Recommended Primary Action:
(Concrete operational step)

Alternative Actions:
(Optional)

Business Risk Assessment:
(Operational and revenue risks)

Confidence Level:
(Low | Medium | High)

If confidence is below High, explain uncertainty drivers.

----------------------------------------------------
CRITICAL INSTRUCTIONS
----------------------------------------------------

- Do NOT repeat raw metrics without interpretation.
- Do NOT hallucinate missing numbers.
- Do NOT invent external data.
- Be precise and concise.
- Think step-by-step internally but do not reveal chain-of-thought.

Generate final answer only in the specified format.
"""
)



def synthesis_node(state: AgentState):

    knowledge_docs = state.get("knowledge_context") or []
    knowledge_text = "\n\n".join([doc.page_content for doc in knowledge_docs])

    external_text = "\n\n".join(state.get("external_context") or [])

    formatted = synthesis_prompt.format(
        item_metrics=state.get("item_metrics", {}),
        knowledge_context=knowledge_text + "\n\n" + external_text,
        question=state["question"]
    )

    response = llm.invoke(formatted)

    return {"final_answer": response.content}


# --------------------------------------------------
# External Retrieval Node (Tavily)
# --------------------------------------------------

def external_retrieval_node(state: AgentState):

    question = state["question"]

    try:
        print("🌐 Tavily tool invoked")

        external_text = tavily_search.invoke(question)

        return {
            "external_context": [external_text] if external_text else []
        }

    except Exception as e:
        print("❌ Tavily error:", e)
        return {"external_context": []}
    
agent = build_agent()