import json
import pickle
import faiss
import unicodedata
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Dict, Any

# IMPORTS FROM CONFIG
from config import (
    GROQ_API_KEY, 
    STOP_NODES, 
    STOP_RELATIONS, 
    SUPER_NODE_THRESHOLD
)

# ==========================================
# DEEP WEB SCOUT ENGINE (TAVILY)
# ==========================================

class DeepWebScout:
    def __init__(self):
        if not TAVILY_API_KEY:
            raise ValueError("❌ TAVILY_API_KEY not found in environment variables.")
        self.client = TavilyClient(api_key=TAVILY_API_KEY)

    def _clean_raw_content(self, text: str) -> str:
        """
        Helper to clean and truncate the massive raw_content strings.
        Removes excessive newlines and limits length.
        """
        if not text: return ""
        # Collapse whitespace
        clean = re.sub(r'\s+', ' ', text).strip()
        # Truncate to keep context window healthy
        if len(clean) > MAX_RAW_CHARS:
            return clean[:MAX_RAW_CHARS] + "... [TRUNCATED]"
        return clean

    def search_and_extract(self, sub_query: str) -> Dict[str, Any]:
        """
        Executes Advanced Search and formats the JSON for the Agent/Curator.
        """
        print(f"   🔎 Scouting External Cortex for: '{sub_query}'...")
        
        try:
            # 1. CALL TAVILY API
            response = self.client.search(query=sub_query, **SEARCH_CONFIG)
            
            # 2. EXTRACT THE "ADVANCED ANSWER" (The Executive Summary)
            # Tavily's LLM generates this based on the search results.
            # This is extremely high-value for our Storyteller/Synthesizer.
            ai_summary = response.get("answer", "")
            
            # 3. PROCESS THE EVIDENCE (The "Results" List)
            results = response.get("results", [])
            
            formatted_context = []
            curation_data = []
            
            # If we have an AI summary, put it at the very top of the context
            if ai_summary:
                formatted_context.append(f"★ EXECUTIVE SUMMARY (AI GENERATED):\n{ai_summary}\n{'-'*40}")

            print(f"   👀 Retrieved {len(results)} high-fidelity sources...")
            
            for i, res in enumerate(results):
                # Extract Metadata
                title = res.get("title", "Unknown Title")
                url = res.get("url", "No URL")
                score = res.get("score", 0.0)
                
                # We prefer the high-quality snippet 'content', but we back it up
                # with 'raw_content' if the snippet is too short.
                snippet = res.get("content", "")
                raw_text = self._clean_raw_content(res.get("raw_content", ""))
                
                # 4. CONSTRUCT CONTEXT STRING (For the Agent)
                # We prioritize the Title/URL/Snippet.
                # We append a chunk of raw text only if it adds value.
                entry = (
                    f"SOURCE [{i+1}]: {title}\n"
                    f"LINK: {url} (Relevance: {score:.2f})\n"
                    f"SUMMARY: {snippet}\n"
                    f"EXTRACT: {raw_text[:500]}...\n" # Give agent a peek at raw text
                    f"{'-'*40}"
                )
                formatted_context.append(entry)
                
                # 5. PREPARE CURATION OBJECT (For the JSON File)
                # The Curator gets the FULL raw text to extract graph triples.
                curation_data.append({
                    "url": url,
                    "title": title,
                    "relevance_score": score,
                    "snippet": snippet,
                    "full_text": raw_text # Curator gets the big chunk
                })

            return {
                "status": "success",
                "tavily_answer": ai_summary,      # The direct answer
                "agent_context": "\n".join(formatted_context), # The string for the Prompt
                "curation_data": curation_data,   # The list for the JSON file
                "original_response": response     # Keep full metadata just in case
            }

        except Exception as e:
            print(f"   ❌ Search Engine Error: {e}")
            return {"status": "failed", "content": str(e)}

# ==========================================
# KNOWLEDGE CURATOR (UPDATED)
# ==========================================
class KnowledgeCurator:
    def __init__(self, pending_file="./models/pending_knowledge.json", model_name="llama-3.3-70b-versatile"):
        self.pending_file = pending_file
        self.llm = ChatGroq(
            temperature=0, 
            model_name=model_name, 
            api_key=os.environ.get("GROQ_API_KEY"),
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    def curate(self, query: str, scout_result: Dict):
        """
        Takes the scout result, analyzes the 'curation_data', and saves a knowledge artifact.
        """
        if scout_result["status"] != "success":
            return

        print("   🧠 Curating knowledge from raw content...")
        
        # Prepare a rich context from the top 3 results for the Curator
        # We combine the raw text from the best sources
        best_sources = scout_result["curation_data"][:3] 
        combined_text = "\n\n".join([f"Source ({s['url']}): {s['full_text']}" for s in best_sources])
        
        sys_msg = """
            You are the **Graph RAG Knowledge Architect**.
            Your goal is to transform raw, noisy web content into a pristine, structured Knowledge Artifact optimized for both vector search and graph traversal.

            ### INSTRUCTIONS

            1. **VECTOR CONTENT (The Summary)**:
            - Synthesize a **dense, information-rich paragraph** that directly answers the User Query based *only* on the Scraped Content.
            - Remove conversational fluff ("The article states...", "It is important to note...").
            - Focus on factual density: include dates, numbers, names, and specific technical details.
            - This text will be embedded; ensure it is semantically complete and self-contained.

            2. **GRAPH TRIPLES (The Knowledge Graph)**:
            - Extract 5-15 semantic triples: `{"head": "Subject", "relation": "Predicate", "tail": "Object"}`.
            - **Entity Rules (Head/Tail)**: Use precise Proper Nouns or technical concepts. Keep them atomic (e.g., "Elon Musk" instead of "The CEO of Tesla Elon Musk").
            - **Relation Rules**: Use active, directed verbs (e.g., "founded", "acquired", "located_in", "author_of"). Avoid generic relations like "is" or "has" if a more specific one exists.
            - **Canonicalization**: Resolve pronouns and aliases to their full names (e.g., replace "he" with the person's name).

            3. **METADATA**:
            - `confidence_score`: 0.0 (Irrelevant/Garbage) to 1.0 (Perfect, Factual Match).
            - `category`: Classify the content into one specific domain tag (e.g., "Market Data", "Technical Documentation", "Biography", "News").

            ### OUTPUT SCHEMA (Strict JSON)
            {
                "vector_content": "Dense text summary...",
                "graph_triples": [
                    {"head": "Entity A", "relation": "relationship_verb", "tail": "Entity B"},
                    {"head": "Entity B", "relation": "relationship_verb", "tail": "Entity C"}
                ],
                "metadata": {
                    "confidence_score": 0.85,
                    "category": "Domain Tag"
                }
            }
            """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=sys_msg),
                HumanMessage(content=f"QUERY: {query}\n\nCONTENT:\n{combined_text[:6000]}") # Context limit
            ])
            artifact_data = json.loads(response.content)
            
            final_artifact = {
                "status": "pending_review",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "original_query": query,
                "data": artifact_data
            }
            
            self._save(final_artifact)
            print(f"   💾 Knowledge Artifact Saved to {self.pending_file}")
            
        except Exception as e:
            print(f"   ❌ Curation Failed: {e}")

    def _save(self, artifact):
        data = []
        if os.path.exists(self.pending_file):
            try:
                with open(self.pending_file, "r") as f: data = json.load(f)
            except: data = []
        data.append(artifact)
        with open(self.pending_file, "w") as f: json.dump(data, f, indent=2)
# example usage
# curator = KnowledgeCurator(model_name=SCOUT_MODEL, pending_file=PENDING_KNOWLEDGE_PATH)