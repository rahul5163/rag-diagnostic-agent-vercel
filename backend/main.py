from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
#from backend.app.v1_baseline.agent import build_agent
from backend.app.v2_rerank.agent import build_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = build_agent()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    result = agent.invoke({
        "question": request.question
    })

    return {
        "answer": result.get("final_answer", "No response generated")
    }