from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import rag_system
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(os.path.dirname(BASE_DIR), "dist")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    answer = rag_system.query(request.query)
    return {"answer": answer}

# Serve static files from the dist directory
if os.path.exists(DIST_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")
    
    # Also serve files in the root of dist (like favicon.ico, resume.pdf)
    @app.get("/{file_path:path}")
    async def serve_static(file_path: str):
        # API requests should bypass this
        if file_path.startswith("api/"):
            raise HTTPException(status_code=404)
            
        full_path = os.path.join(DIST_DIR, file_path)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            return FileResponse(full_path)
        
        # Fallback to index.html for SPA routing
        return FileResponse(os.path.join(DIST_DIR, "index.html"))
else:
    print(f"Warning: {DIST_DIR} not found. Static files will not be served.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
