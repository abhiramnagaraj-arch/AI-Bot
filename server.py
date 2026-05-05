from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import time
import os
from main import run_pipeline

app = FastAPI(title="RAG Bot API")

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat API Models
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    time_ms: int

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    try:
        # Note: The current run_pipeline doesn't return confidence, 
        # so we'll mock it or provide a heuristic.
        answer = run_pipeline(request.query)
        
        end_time = time.time()
        duration_ms = int((end_time - start_time) * 1000)
        
        # Simple confidence mock for now as requested by the schema
        confidence = 0.95 if "I don't know" not in answer else 0.1
        
        return ChatResponse(
            answer=answer,
            confidence=confidence,
            time_ms=duration_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files from the 'frontend' directory
# This will serve index.html at '/' and other files like styles.css at their paths
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
