from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag import rag_system  # lazy-loaded
import os

app = FastAPI(title="Portfolio RAG Backend")

# CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://portfolio-qllp3nx2x-muhammad-umair-farooqs-projects-8b12a4bf.vercel.app/"],  # or set to your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        answer = rag_system.query(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: favicon to remove browser 500s
@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static/favicon.ico"))
