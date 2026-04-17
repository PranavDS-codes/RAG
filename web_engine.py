import os
import json
import re
import time
from typing import List, Dict, Any
from tavily import TavilyClient
import wikipedia
from prompts import VECTOR_CHUNK_CURATION_PROMPT, GRAPH_CURATION_PROMPT
from llm_client import llm_client

# IMPORTS FROM CONFIG
from config import (
    TAVILY_API_KEY, 
    SEARCH_CONFIG, 
    MAX_RAW_CHARS,
    PENDING_KNOWLEDGE_PATH
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
class WikiScout:
    def __init__(self, lang="en"):
        self.lang = lang
        wikipedia.set_lang(lang)

    def search_and_extract(self, topic: str) -> Dict:
        print(f"   📖 [WikiScout] Looking up '{topic}'...")
        try:
            search_results = wikipedia.search(topic, results=1)
            if not search_results:
                return {"status": "error", "reason": "No results found"}
            
            best_match = search_results[0]
            page = wikipedia.page(best_match, auto_suggest=False)
            
            return {
                "status": "success",
                "topic": topic,
                "title": page.title,
                "summary": page.summary,
                "url": page.url,
                "answer": page.summary + "...", 
                "curation_data": [{
                    "url": page.url,
                    "full_text": page.content[:5000] 
                }]
            }
        except Exception as e:
            return {"status": "error", "reason": str(e)}

# curator = KnowledgeCurator(graph_provider=, graph_model=, vector_provider=, vector_model=)
# ==========================================
# KNOWLEDGE CURATOR (DUAL-LLM POWERED)
# ==========================================
class KnowledgeCurator:
    def __init__(self, pending_file=PENDING_KNOWLEDGE_PATH, 
                graph_provider='nvidia', 
                graph_model='qwen/qwen3-coder-480b-a35b-instruct', 
                vector_provider='groq', 
                vector_model='llama-3.3-70b-versatile'
                ):
        self.pending_file = pending_file
        os.makedirs(os.path.dirname(self.pending_file), exist_ok=True)
        
        # # 1. Groq Client (For Vector Chunks)
        # self.groq_llm = ChatGroq(
        #     temperature=0, 
        #     model_name=groq_model, 
        #     api_key=GROQ_API_KEY,
        #     model_kwargs={"response_format": {"type": "json_object"}}
        # )

        # # 2. NVIDIA NIM Client (For Graph Extractions)
        # self.nvidia_client = OpenAI(
        #     api_key=NVIDIA_API_KEY,
        #     base_url="https://integrate.api.nvidia.com/v1"
        # )
        self.graph_provider = graph_provider
        self.graph_model = graph_model
        self.vector_provider = vector_provider
        self.vector_model = vector_model

    def curate(self, query: str, scout_result: Dict, source_type: str = "web_scout"):
        """
        Executes a dual-LLM curation: Vector chunks via Groq, Graph triples via NVIDIA.
        """
        if scout_result.get("status") != "success":
            return

        print(f"   🧠 Curating Knowledge for [{query}]...")
        
        # 1. Prepare Content
        sources = scout_result.get("curation_data", [])
        combined_text = ""
        for s in sources[:5]:
            text = s.get("full_text") or s.get("raw_content") or s.get("snippet") or ""
            combined_text += f"\n\nSource ({s.get('url')}):\n{text[:2000]}"
            
        content_payload = f"TOPIC: {query}\n\nCONTENT:\n{combined_text[:6000]}"

        vector_data = {}
        graph_data = {}

        # 2. EXECUTE Vector Chunking
        print("      ↳ Generating Vector Chunks (Groq)...")
        try:
            vector_data = llm_client.generate_json(
                system_prompt=VECTOR_CHUNK_CURATION_PROMPT,
                user_prompt=content_payload,
                provider=self.vector_provider,
                model=self.vector_model,
            )
        except Exception as e:
            print(f"      ⚠️ Vector Curation Failed: {e}")

        # 3. EXECUTE NVIDIA (Graph Extraction)
        print("      ↳ Extracting Graph Topology (NVIDIA NIM)...")
        try:
            graph_data = llm_client.generate_json(
                system_prompt=GRAPH_CURATION_PROMPT,
                user_prompt=content_payload,
                provider=self.graph_provider,
                model=self.graph_model,
            )
        except Exception as e:
            print(f"      ⚠️ Graph Curation Failed: {e}")

        # 4. Construct & Save JSONL Artifact
        final_artifact = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "original_query": query,
            "source_type": source_type,
            "vector_chunks": vector_data.get("chunks", []),
            "graph_extractions": {
                "entities": graph_data.get("entities", []),
                "relationships": graph_data.get("relationships", [])
            }
        }
        
        self._save_jsonl(final_artifact)
        print(f"   💾 Dual-Artifact appended to {self.pending_file}")

    def _save_jsonl(self, artifact):
        """Appends a single JSON object to the JSONL file."""
        with open(self.pending_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(artifact, ensure_ascii=False) + "\n")

# example usage
# curator = KnowledgeCurator(model_name=SCOUT_MODEL, pending_file=PENDING_KNOWLEDGE_PATH)