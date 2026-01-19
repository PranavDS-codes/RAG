import os
from dotenv import load_dotenv
load_dotenv(override=True)

# --- API KEYS ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# --- MODEL CONFIGURATION ---

# 1. Premise Model
PREMISE_MODEL = "llama-3.3-70b-versatile" 
# PREMISE_MODEL = "openai/gpt-oss-120b"

# 1. RETRIEVAL (Query Decomposer)
# Needs to be instruction-following and structured JSON capable.
QUERY_MODEL = "llama-3.3-70b-versatile" 
# QUERY_MODEL = "openai/gpt-oss-120b"

# 2. AUDIT (Fact Checker)
# Needs to be logical and strict.
# AUDIT_MODEL = "llama-3.3-70b-versatile" 
AUDIT_MODEL = "openai/gpt-oss-120b"

# 3. VERIFY (Search Quality)
# Needs to be logical and strict.
# VERIFY_MODEL = "llama-3.3-70b-versatile" 
VERIFY_MODEL = "openai/gpt-oss-120b"

# 4. REFINE (Failed Search -> Search Strategy Refiner)
# Needs to be logical and strict.
# REFINE_MODEL = "llama-3.3-70b-versatile" 
REFINE_MODEL = "openai/gpt-oss-120b"

# 5. SCOUT (Web Researcher)
# Used by Tavily/Curator for summarization.
SCOUT_MODEL = "llama-3.3-70b-versatile"
# SCOUT_MODEL = "openai/gpt-oss-120b" 

# 6. SYNTHESIS (Final Answer Writer)
# Needs to be creative, professional, and good at citations.
# SYNTHESIZE_MODEL = "llama-3.3-70b-versatile"
SYNTHESIZE_MODEL = "openai/gpt-oss-120b"

# --- PATHS ---
# CRITICAL: Point to the CLEAN graph from your surgery
GRAPH_PATH = "./models/knowledge_graph.pkl"
# GRAPH_PATH = "./models/knowledge_graph_clean.pkl" 
CHUNKS_PATH = "./models/chunk_metadata.pkl"
VECTOR_INDEX_PATH = "./models/faiss_index.bin"
BM25_INDEX_PATH = "./models/bm25_index.pkl"
PENDING_KNOWLEDGE_PATH = "./models/pending_knowledge.json"

# --- SETTINGS ---
SUPER_NODE_THRESHOLD = 50
LOG_FILE_PATH = "./models/brain_activity.log"
REPORTS_DIR = "./models/run_reports"  # <--- NEW: Folder for detailed JSON reports

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
    "include_answer": True,          # Get the AI-generated summary
    "include_raw_content": True,     # Get full page text
    "include_images": False,
    "chunks_per_source": 3
}

MAX_RAW_CHARS = 4000 # Truncate raw content per source to avoid context overflow




