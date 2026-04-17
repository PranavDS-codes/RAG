import os
from dotenv import load_dotenv
load_dotenv(override=True)

# --- API KEYS ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

# New Neo4j Keys
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- MODEL CONFIGURATION ---

# 1. Premise Model
# PREMISE_MODEL = "llama-3.3-70b-versatile" 
PREMISE_PROVIDER = "groq"
PREMISE_MODEL = "openai/gpt-oss-120b"

# 1. RETRIEVAL (Query Decomposer)
QUERY_PROVIDER = "groq"
QUERY_MODEL = "openai/gpt-oss-120b"

# 2. AUDIT (Fact Checker)
AUDIT_PROVIDER = "groq"
AUDIT_MODEL = "openai/gpt-oss-120b"
# AUDIT_MODEL = "llama-3.3-70b-versatile" 

# 3. VERIFY (Search Quality)
VERIFY_PROVIDER = "groq"
VERIFY_MODEL = "openai/gpt-oss-120b"
# VERIFY_MODEL = "llama-3.3-70b-versatile" 

# 4. REFINE (Failed Search -> Search Strategy Refiner)
REFINE_PROVIDER = "groq"
REFINE_MODEL = "openai/gpt-oss-120b"
# REFINE_MODEL = "llama-3.3-70b-versatile" 

# 5. SCOUT (Web Researcher)
SCOUT_PROVIDER = "groq"
SCOUT_MODEL = "openai/gpt-oss-120b" 
# SCOUT_MODEL = "llama-3.3-70b-versatile"

# 6. SYNTHESIS (Final Answer Writer)
SYNTHESIZE_PROVIDER = "groq"
SYNTHESIZE_MODEL = "openai/gpt-oss-120b"
# SYNTHESIZE_MODEL = "llama-3.3-70b-versatile"

# 7. CURATOR (Pending Knowledge Base Builder)
VECTOR_CURATOR_PROVIDER = "groq"
VECTOR_CURATOR_MODEL = "openai/gpt-oss-120b"
GRAPH_CURATOR_PROVIDER = "nvidia"
GRAPH_CURATOR_MODEL = "qwen/qwen3-coder-480b-a35b-instruct"
# CURATOR_MODEL = "llama-3.3-70b-versatile"

# --- PATHS ---
# # CRITICAL: Point to the CLEAN graph from your surgery
# GRAPH_PATH = "./models/knowledge_graph.pkl"
# # GRAPH_PATH = "./models/knowledge_graph_clean.pkl" 
# CHUNKS_PATH = "./models/chunk_metadata.pkl"
# VECTOR_INDEX_PATH = "./models/faiss_index.bin"
BM25_INDEX_PATH = "./models/bm25_index.pkl"
PENDING_KNOWLEDGE_PATH = "./models/pending_knowledge.json"
NEW_FAISS_INDEX_PATH = "/Users/pranavpant/Desktop/code/RAG/new_data/nvidia_faiss_index.bin"
NEW_CHUNK_METADATA_PATH = "/Users/pranavpant/Desktop/code/RAG/new_data/nvidia_chunk_metadata.pkl"
OLD_FAISS_INDEX_PATH =  "/Users/pranavpant/Desktop/code/RAG/models/faiss_index.bin"
OLD_CHUNK_METADATA_PATH = "/Users/pranavpant/Desktop/code/RAG/models/chunk_metadata.pkl"

# --- SETTINGS ---
SUPER_NODE_THRESHOLD = 50
LOG_FILE_PATH = "./models/brain_activity.log"
REPORTS_DIR = "./models/run_reports/new"  # <--- NEW: Folder for detailed JSON reports

# --- FILTERS ---
# STOP_RELATIONS = {
#     "is", "are", "has", "have", "related_to", "part_of", "includes", 
#     "involved_in", "associated_with", "type_of"
# }
STOP_RELATIONS = {
    "is", "are", "am", "was", "were", 
    "has", "have", "had", 
    "be", "been", "being"
}

STOP_NODES = {
    "it", "they", "he", "she", "who", "that", "this", "which", 
    "him", "her", "them", "there", "where"
}

# --- WEB SEARCH CONFIG ---
SEARCH_CONFIG = {
    "search_depth": "advanced",      # Deep search for quality
    "topic": "general",              # General knowledge
    "max_results": 5,                # Top 5 sources
    "include_answer": False,          # Get the AI-generated summary
    "include_raw_content": True,     # Get full page text
    "include_images": False,
    "chunks_per_source": 3
}

MAX_RAW_CHARS = 1000 # Truncate raw content per source to avoid context overflow




