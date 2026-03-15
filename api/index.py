from fastapi import FastAPI
from backend.main import build_agent
from pydantic import BaseModel

app = FastAPI()

agent = None

def get_agent():
    global agent
    if agent is None:
        agent = build_agent()
    return agent


class QueryRequest(BaseModel):
    question: str


@app.get("/api/health")
def health():
    return {"status": "agent running"}


@app.post("/api/query")
async def query_rag(request: QueryRequest):
    agent = get_agent()

    result = agent.invoke({
        "question": request.question
    })

    return {
        "answer": result.get("final_answer", "No response generated")
    }