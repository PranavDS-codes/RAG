import json
import pickle
import faiss
import unicodedata
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Dict, Any
from prompts import QUERY_DECOMPOSITION_PROMPT

# IMPORTS FROM CONFIG
from config import (
    GROQ_API_KEY, 
    STOP_NODES, 
    STOP_RELATIONS, 
    SUPER_NODE_THRESHOLD
)

# ==========================================
# ROBUST QUERY OPTIMIZER (AUTO-FIXING)
# ==========================================
class QueryOptimizer:
    def __init__(self, model_name="llama-3.3-70b-versatile", api_key=None):
        self.llm = ChatGroq(
            temperature=0, 
            model_name=model_name, 
            api_key=api_key,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    def _clean_text(self, text):
        """Standardizes text to ASCII-compatible format."""
        if not isinstance(text, str): return text
        return unicodedata.normalize('NFKC', text).strip()

    def _normalize_response(self, raw_json: Dict) -> Dict:
        """
        ADAPTER: Now expects the 'tasks' list directly from the LLM.
        """
        # Case 1: Ideal Scenario - LLM gave us the list of task objects
        if "tasks" in raw_json and isinstance(raw_json["tasks"], list):
            return raw_json

        # Case 2: Legacy/Fallback - If LLM still outputs flat format, we adapt it
        if "sub_queries" in raw_json:
            print("⚠️ LLM reverted to flat format. Distributing global metadata...")
            generated_tasks = []
            global_hyde = raw_json.get("hyde_passage", "")
            global_entities = raw_json.get("graph_entities", [])
            global_keywords = raw_json.get("keywords", [])
            
            for sub_q in raw_json["sub_queries"]:
                generated_tasks.append({
                    "sub_query": sub_q,
                    "hyde_passage": global_hyde,
                    "graph_entities": global_entities,
                    "keywords": global_keywords
                })
            return {"tasks": generated_tasks}

        return {"tasks": []}

    def optimize(self, query: str) -> Dict[str, Any]:
        """
        Generates a Multi-Task Omni-Query Object with ISOLATED metadata per task.
        """
        system_prompt = QUERY_DECOMPOSITION_PROMPT
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ])
            raw_result = json.loads(response.content)
            
            # 1. RUN THE ADAPTER
            result = self._normalize_response(raw_result)
            
            # 2. RUN THE CLEANER
            clean_tasks = []
            for task in result.get("tasks", []):
                clean_tasks.append({
                    "sub_query": self._clean_text(task.get("sub_query", "")),
                    "hyde_passage": self._clean_text(task.get("hyde_passage", "")),
                    "graph_entities": [self._clean_text(e) for e in task.get("graph_entities", [])],
                    "keywords": [self._clean_text(k) for k in task.get("keywords", [])]
                })
            
            if not clean_tasks:
                raise ValueError("Structure empty after normalization")
                
            return {"tasks": clean_tasks}
            
        except Exception as e:
            print(f"⚠️ Optimization Failed: {e}")
            return {
                "tasks": [{
                    "sub_query": query,
                    "hyde_passage": query,
                    "graph_entities": [],
                    "keywords": query.split()
                }]
            }

# ==========================================
# GRAPH SEARCH MODULE
# ==========================================
class GraphSearcher:
    def __init__(self, graph_path, threshold=50):
        with open(graph_path, "rb") as f:
            self.G = pickle.load(f)
        self.node_degrees = dict(self.G.degree())
        self.threshold = threshold

    def get_neighbors(self, start_node, depth=1):
        if start_node not in self.G: return []
        facts = set()
        queue = [(start_node, 0)]
        visited = set()
        
        while queue:
            current, dist = queue.pop(0)
            if dist >= depth: continue
            visited.add(current)
            
            # Super-Node Logic
            if dist > 0 and self.node_degrees.get(current, 0) > self.threshold:
                continue
            
            for neighbor in self.G.neighbors(current):
                if neighbor in visited or neighbor.lower() in STOP_NODES: continue
                edge_data = self.G.get_edge_data(current, neighbor)
                relation = edge_data.get('relation', 'related_to')
                if relation.lower() in STOP_RELATIONS: continue
                
                facts.add(f"{current} --{relation}--> {neighbor}")
                queue.append((neighbor, dist + 1))
        
        return list(facts)

# ==========================================
# CELL 4: THE OMNI-RETRIEVER (ATOMIC CONTEXTS)
# ==========================================
class OmniRetriever:
    def __init__(self, graph_path, chunks_path, vector_path, bm25_path, model_name):
        # Tools
        self.optimizer = QueryOptimizer(api_key=GROQ_API_KEY, model_name=model_name)
        self.graph_engine = GraphSearcher(graph_path, threshold=SUPER_NODE_THRESHOLD)
        
        # Load Resources
        print("📂 Loading Resources...")
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        self.chunk_texts = [c['text'] for c in self.chunks]
        
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
            
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index(vector_path)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("🚀 Omni-Retriever Ready.")

    def _retrieve_atomic(self, task, top_k=5, verbose=False):
        """
        Executes search for ONE sub-query and returns its own Top K results.
        Now leverages the strict, per-task metadata generated by the Optimizer.
        """
        candidates = {} # text -> score
        
        # 1. VECTOR (HyDE) - Now uses the task-specific HyDE passage
        if task.get('hyde_passage'):
            hyde_emb = self.embedder.encode([task['hyde_passage']], convert_to_numpy=True)
            # Increased candidate pool to 3x to allow ReRanker more choices
            D, I = self.index.search(hyde_emb, k=top_k*3) 
            for idx in I[0]:
                if idx < len(self.chunks):
                    candidates[self.chunks[idx]['text']] = 0.0

        # 2. BM25 (Keywords) - Now uses task-specific keywords
        # We combine sub_query + keywords for a rich lexical query
        bm25_query = f"{task['sub_query']} {' '.join(task.get('keywords', []))}"
        tokenized_query = bm25_query.split()
        bm25_docs = self.bm25.get_top_n(tokenized_query, self.chunk_texts, n=top_k*3)
        for txt in bm25_docs:
            candidates[txt] = 0.0

        # 3. GRAPH (Entities) - Now uses task-specific entities
        graph_facts = []
        for entity in task.get('graph_entities', []):
            # Exact Match
            if entity in self.graph_engine.G:
                facts = self.graph_engine.get_neighbors(entity)
                graph_facts.extend(facts)
            else:
                # Fuzzy Fallback (Simple case-insensitive check)
                for node in self.graph_engine.G.nodes():
                    if str(node).lower() == entity.lower():
                        facts = self.graph_engine.get_neighbors(node)
                        graph_facts.extend(facts)
                        break
        
        # Add graph facts to candidates
        for fact in graph_facts[:20]: # Cap graph facts to prevent flooding
            candidates[fact] = 0.0
            
        # 4. RE-RANKING (Per Task)
        unique_docs = list(candidates.keys())
        if not unique_docs: return []
        
        # Rerank candidates against the specific SUB-QUERY (not the global query)
        pairs = [[task['sub_query'], doc] for doc in unique_docs]
        scores = self.reranker.predict(pairs)
        
        # Sort and take top K
        final_ranked = sorted(list(zip(unique_docs, scores)), key=lambda x: x[1], reverse=True)
        
        return final_ranked[:top_k]

    def _deduplicate_and_flatten(self, task_results):
        """
        Merges results from all tasks into a single unique list, 
        keeping the highest score for any duplicate document.
        """
        unique_map = {} # text -> max_score
        
        for task in task_results:
            for doc, score in task['results']:
                if doc not in unique_map:
                    unique_map[doc] = score
                else:
                    # Keep the higher score if retrieved by multiple tasks
                    unique_map[doc] = max(unique_map[doc], score)
        
        # Sort globally by score
        final_list = sorted(unique_map.items(), key=lambda x: x[1], reverse=True)
        return final_list

    def retrieve(self, query, top_k_per_task=5, verbose=True):
        # 1. OPTIMIZE
        # This triggers the NEW system prompt with strict isolation
        omni = self.optimizer.optimize(query)
        
        if "tasks" not in omni:
            print("❌ Error: Optimizer returned invalid structure:", omni.keys())
            return {"original_query": query, "tasks": [], "combined_context": []}

        if verbose:
            print(f"\n🧠 OMNI-QUERY: Generated {len(omni['tasks'])} Atomic Tasks")
            for t in omni['tasks']:
                print(f"  - Sub-Query: {t['sub_query']}")
                print(f"    Keywords: {t['keywords']}")
                print(f"    Entities: {t['graph_entities']}")
        
        final_structure = {
            "original_query": query,
            "tasks": []
        }
        
        # 2. EXECUTE ATOMIC TASKS
        for i, task in enumerate(omni['tasks']):
            if verbose: 
                print(f"\n⚡ Executing Task {i+1}: '{task['sub_query']}'")
            
            results = self._retrieve_atomic(task, top_k=top_k_per_task, verbose=verbose)
            
            task_result = {
                "sub_query": task['sub_query'],
                "results": results 
            }
            final_structure["tasks"].append(task_result)
            
            if verbose:
                # Show top match for this specific atomic query
                if results:
                    print(f"   ↳ Retrieved {len(results)} contexts. Top Match: {results[0][1]:.4f}")
                else:
                    print(f"   ↳ Retrieved 0 contexts.")

        # 3. GLOBAL MERGE (New Step)
        # Create a single 'context' list for the final answer generator
        combined_context = self._deduplicate_and_flatten(final_structure["tasks"])
        final_structure["combined_context"] = combined_context
        
        if verbose:
            print(f"\n✅ Final Consolidated Contexts: {len(combined_context)} unique items.")

        return final_structure

# Initialize
# omni_tool = OmniRetriever(GRAPH_PATH, CHUNKS_PATH, VECTOR_INDEX_PATH, BM25_INDEX_PATH, MODEL_NAME)