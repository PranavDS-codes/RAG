import os
import json
import queue
import asyncio
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any

# Import workflow and configs
from graph import build_graph
from services import recorder, curator, traced_retriever
from config import REPORTS_DIR, PENDING_KNOWLEDGE_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Diagnostics diagnostics on application startup.
    """
    print("\n🔍 Performing Graph-RAG System Diagnostics...")
    print("====================================================")
    
    # 1. Verify Neo4j Connection
    try:
        with traced_retriever.graph_engine.driver.session() as session:
            session.run("RETURN 1")
        print("🟢 NEO4J GRAPH DB: Connection successful")
    except Exception as e:
        print(f"🔴 NEO4J GRAPH DB: Connection failed! ({str(e)})")
        
    # 2. Check FAISS Vector Indexes
    has_nvidia = hasattr(traced_retriever, 'nvidia_index') and traced_retriever.nvidia_index is not None
    print(f"{'🟢' if has_nvidia else '🔴'} FAISS NVIDIA INDEX: {'Loaded' if has_nvidia else 'Failed/Missing'}")
    
    has_local = hasattr(traced_retriever, 'local_index') and traced_retriever.local_index is not None
    print(f"{'🟢' if has_local else '🔴'} FAISS LOCAL INDEX: {'Loaded' if has_local else 'Failed/Missing'}")
    
    # 3. Check Configured API Keys
    from config import NVIDIA_API_KEY, TAVILY_API_KEY
    print(f"{'🟢' if NVIDIA_API_KEY else '🔴'} NVIDIA API KEY: {'Configured' if NVIDIA_API_KEY else 'Missing'}")
    print(f"{'🟢' if TAVILY_API_KEY else '🔴'} TAVILY API KEY: {'Configured' if TAVILY_API_KEY else 'Missing'}")
    
    print("====================================================\n")
    yield

app = FastAPI(
    title="Graph-RAG Agent API",
    description="Backend API endpoints for the Hybrid Vector & Graph RAG Agentic Pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile LangGraph Workflow
compiled_graph = build_graph()

class QueryRequest(BaseModel):
    query: str

class CurateRequest(BaseModel):
    query: str
    source_type: str = "web_scout"

@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    Serves the modern single-page dashboard.
    """
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Graph-RAG Console online. UI file static/index.html not found.</h1>"

@app.get("/api/health")
async def health_check():
    """
    Diagnostics healthcheck for UI status alerts.
    """
    checks = {}
    
    # Neo4j Check
    try:
        with traced_retriever.graph_engine.driver.session() as session:
            session.run("RETURN 1")
        checks["neo4j"] = {"status": "active", "message": "Connected successfully"}
    except Exception as e:
        checks["neo4j"] = {"status": "inactive", "message": str(e)}

    # FAISS Checks
    has_nvidia = hasattr(traced_retriever, 'nvidia_index') and traced_retriever.nvidia_index is not None
    checks["faiss_nvidia"] = {
        "status": "active" if has_nvidia else "inactive",
        "message": "Loaded" if has_nvidia else "Not initialized"
    }

    has_local = hasattr(traced_retriever, 'local_index') and traced_retriever.local_index is not None
    checks["faiss_local"] = {
        "status": "active" if has_local else "inactive",
        "message": "Loaded" if has_local else "Not initialized"
    }

    # API Keys Configuration Check
    from config import NVIDIA_API_KEY, TAVILY_API_KEY
    checks["api_keys"] = {
        "nvidia": "configured" if NVIDIA_API_KEY else "missing",
        "tavily": "configured" if TAVILY_API_KEY else "missing"
    }
    
    # Core system active check
    all_ok = checks["neo4j"]["status"] == "active" and has_local and has_nvidia
    
    return {
        "status": "healthy" if all_ok else "unhealthy",
        "details": checks
    }

@app.post("/api/query")
async def run_query(request: QueryRequest):
    """
    Submits a query to the agentic RAG pipeline.
    Executes the LangGraph workflow with deep tracing active.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Start tracing
    recorder.start_run(request.query)
    run_id = recorder.current_run_id
    
    try:
        # Run workflow
        result = compiled_graph.invoke({"query": request.query})
        
        # Save final answer to trace metadata
        recorder.run_data["meta"]["final_answer"] = result.get("final_answer", "")
        
        # Save trace report to file
        recorder.save_report()
        
        return {
            "query": request.query,
            "run_id": run_id,
            "final_answer": result.get("final_answer", ""),
            "trace_log": recorder.run_data.get("trace_log", [])
        }
    except Exception as e:
        # Save report even if it fails
        recorder.log_event("SYSTEM_ERROR", "API_ROUTER", {"error": str(e)})
        recorder.save_report()
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

@app.get("/api/query/stream")
async def run_query_stream(query: str):
    """
    Submits a query and streams trace events as Server-Sent Events (SSE).
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    q = queue.Queue()

    def callback(entry):
        q.put({"event": "step_update", "data": entry})

    recorder.register_callback(callback)

    def run_graph():
        try:
            recorder.start_run(query)
            q.put({"event": "run_start", "run_id": recorder.current_run_id, "query": query})
            
            result = compiled_graph.invoke({"query": query})
            
            # Save final answer to trace metadata
            recorder.run_data["meta"]["final_answer"] = result.get("final_answer", "")
            
            recorder.save_report()
            
            q.put({
                "event": "run_complete",
                "run_id": recorder.current_run_id,
                "final_answer": result.get("final_answer", "")
            })
        except Exception as e:
            recorder.log_event("SYSTEM_ERROR", "API_ROUTER", {"error": str(e)})
            recorder.save_report()
            q.put({"event": "run_error", "error": str(e)})
        finally:
            recorder.clear_callbacks()

    # Run in a background thread
    thread = threading.Thread(target=run_graph)
    thread.start()

    async def sse_generator():
        # Yield 2KB of comment padding to force-flush browser network buffers immediately
        yield f": {' ' * 2048}\n\n".encode("utf-8")
        
        while True:
            try:
                while not q.empty():
                    item = q.get_nowait()
                    yield f"data: {json.dumps(item)}\n\n".encode("utf-8")
                    if item.get("event") in ["run_complete", "run_error"]:
                        return
            except queue.Empty:
                pass
            await asyncio.sleep(0.05)

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/api/history")
async def get_history():
    """
    Retrieves the list of previous runs sorted by timestamp (descending).
    """
    if not os.path.exists(REPORTS_DIR):
        return []
    
    history = []
    try:
        for file_name in os.listdir(REPORTS_DIR):
            if file_name.endswith("_deep_trace.json"):
                filepath = os.path.join(REPORTS_DIR, file_name)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        meta = data.get("meta", {})
                        if meta:
                            history.append(meta)
                except Exception:
                    # Skip corrupt files
                    continue
        
        # Sort by timestamp descending
        # Format of timestamp is YYYYMMDD_HHMMSS
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.get("/api/traces/{run_id}")
async def get_trace(run_id: str):
    """
    Retrieves the full trace log for a specific query run.
    """
    filename = f"{run_id}_deep_trace.json"
    filepath = os.path.join(REPORTS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Trace with ID {run_id} not found")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read trace file: {str(e)}")

@app.get("/api/curation/pending")
async def get_pending_curation():
    """
    Retrieves the queue of pending knowledge artifacts for reviewer approval.
    """
    if not os.path.exists(PENDING_KNOWLEDGE_PATH):
        return []
    
    artifacts = []
    try:
        with open(PENDING_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            # Handle potential JSONL vs standard JSON formats
            content = f.read().strip()
            if not content:
                return []
            
            # Try parsing as JSON array first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                return [data]
            except json.JSONDecodeError:
                # Fallback to reading JSON Lines
                f.seek(0)
                for line in f:
                    clean_line = line.strip().rstrip(",")
                    if clean_line:
                        try:
                            artifacts.append(json.loads(clean_line))
                        except Exception:
                            continue
        return artifacts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read curation queue: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
