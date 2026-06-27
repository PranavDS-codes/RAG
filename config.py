import os
from dotenv import load_dotenv

# Load main API environment variables
load_dotenv(override=True)
# Load model configuration environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.models"), override=True)

# --- API KEYS ---
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

# New Neo4j Keys
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- MODEL CONFIGURATION ---

PREMISE_PROVIDER = os.getenv("PREMISE_PROVIDER", "nvidia")
PREMISE_MODEL = os.getenv("PREMISE_MODEL", "meta/llama-3.3-70b-instruct")

QUERY_PROVIDER = os.getenv("QUERY_PROVIDER", "nvidia")
QUERY_MODEL = os.getenv("QUERY_MODEL", "meta/llama-3.3-70b-instruct")

AUDIT_PROVIDER = os.getenv("AUDIT_PROVIDER", "nvidia")
AUDIT_MODEL = os.getenv("AUDIT_MODEL", "meta/llama-3.3-70b-instruct")

VERIFY_PROVIDER = os.getenv("VERIFY_PROVIDER", "nvidia")
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "meta/llama-3.3-70b-instruct")

REFINE_PROVIDER = os.getenv("REFINE_PROVIDER", "nvidia")
REFINE_MODEL = os.getenv("REFINE_MODEL", "meta/llama-3.3-70b-instruct")

SCOUT_PROVIDER = os.getenv("SCOUT_PROVIDER", "nvidia")
SCOUT_MODEL = os.getenv("SCOUT_MODEL", "meta/llama-3.3-70b-instruct")

SYNTHESIZE_PROVIDER = os.getenv("SYNTHESIZE_PROVIDER", "nvidia")
SYNTHESIZE_MODEL = os.getenv("SYNTHESIZE_MODEL", "meta/llama-3.3-70b-instruct")

VECTOR_CURATOR_PROVIDER = os.getenv("VECTOR_CURATOR_PROVIDER", "nvidia")
VECTOR_CURATOR_MODEL = os.getenv("VECTOR_CURATOR_MODEL", "meta/llama-3.3-70b-instruct")

GRAPH_CURATOR_PROVIDER = os.getenv("GRAPH_CURATOR_PROVIDER", "nvidia")
GRAPH_CURATOR_MODEL = os.getenv("GRAPH_CURATOR_MODEL", "meta/llama-3.3-70b-instruct")

# --- PATHS ---
BM25_INDEX_PATH = "./models/bm25_index.pkl"
PENDING_KNOWLEDGE_PATH = "./models/pending_knowledge.json"
NEW_FAISS_INDEX_PATH = "/Users/pranavpant/Desktop/code/RAG/new_data/nvidia_faiss_index.bin"
NEW_CHUNK_METADATA_PATH = "/Users/pranavpant/Desktop/code/RAG/new_data/nvidia_chunk_metadata.pkl"
OLD_FAISS_INDEX_PATH =  "/Users/pranavpant/Desktop/code/RAG/models/faiss_index.bin"
OLD_CHUNK_METADATA_PATH = "/Users/pranavpant/Desktop/code/RAG/models/chunk_metadata.pkl"

# --- SETTINGS ---
SUPER_NODE_THRESHOLD = 50
LOG_FILE_PATH = "./models/brain_activity.log"
REPORTS_DIR = "./models/run_reports/new"  
ENABLE_KNOWLEDGE_CURATION = os.getenv("ENABLE_KNOWLEDGE_CURATION", "True").lower() in ("true", "1", "yes")

# --- FILTERS ---
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
TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")
TAVILY_TOPIC = os.getenv("TAVILY_TOPIC", "general")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
TAVILY_INCLUDE_ANSWER = os.getenv("TAVILY_INCLUDE_ANSWER", "False").lower() in ("true", "1", "yes")
TAVILY_INCLUDE_RAW_CONTENT = os.getenv("TAVILY_INCLUDE_RAW_CONTENT", "True").lower() in ("true", "1", "yes")

SEARCH_CONFIG = {
    "search_depth": TAVILY_SEARCH_DEPTH,
    "topic": TAVILY_TOPIC,
    "max_results": TAVILY_MAX_RESULTS,
    "include_answer": TAVILY_INCLUDE_ANSWER,
    "include_raw_content": TAVILY_INCLUDE_RAW_CONTENT,
    "include_images": False,
    "chunks_per_source": 3
}

MAX_RAW_CHARS = 1000 # Truncate raw content per source to avoid context overflow




