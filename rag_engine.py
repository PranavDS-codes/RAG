import json
from openai import OpenAI
import numpy as np
import pickle
import faiss
import unicodedata
import requests
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
    SUPER_NODE_THRESHOLD,
    NVIDIA_API_KEY
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
# CELL 4: THE OMNI-RETRIEVER (DUAL-ENGINE VECTOR & RERANK)
# ==========================================
class OmniRetriever:
    def __init__(self, graph_path, bm25_path, model_name):
        # Tools
        self.optimizer = QueryOptimizer(api_key=GROQ_API_KEY, model_name=model_name)
        self.graph_engine = GraphSearcher(graph_path, threshold=SUPER_NODE_THRESHOLD)
        
        print("📂 Loading Resources...")
        
        # 1. Setup NVIDIA NIM (Primary Vector & Rerank Engine)
        self.nvidia_client = OpenAI(
            api_key=NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        self.nvidia_model_name = "nvidia/nv-embed-v1"
        self.nvidia_rerank_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
        self.nvidia_rerank_model = "nv-rerank-qa-mistral-4b:1"
        
        try:
            print("🧠 Loading NVIDIA Vector Store (Primary)...")
            self.nvidia_index = faiss.read_index("../models/nvidia_faiss_index.bin")
            with open("../models/nvidia_chunk_metadata.pkl", "rb") as f:
                self.nvidia_chunks = pickle.load(f)
            self.nvidia_active = True
        except Exception as e:
            print(f"⚠️ Could not load NVIDIA store: {e}")
            self.nvidia_active = False

        # 2. Setup BGE-Small (Local Fallback Vector Engine)
        try:
            print("💾 Loading BGE Local Vector Store (Fallback)...")
            self.local_embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
            self.local_index = faiss.read_index("../models/faiss_index.bin")
            with open("../models/chunk_metadata.pkl", "rb") as f:
                self.local_chunks = pickle.load(f)
            self.local_active = True
        except Exception as e:
            print(f"⚠️ Could not load Local Fallback store: {e}")
            self.local_active = False

        # 3. Setup BM25 (Keywords)
        # BM25 needs a list of texts. We prefer the NVIDIA chunks if available.
        active_chunks = self.nvidia_chunks if self.nvidia_active else self.local_chunks
        self.chunk_texts = [c['text'] for c in active_chunks]
        
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
            
        # 4. Setup Local Reranker (Fallback)
        try:
            print("⚖️ Loading Local MS-MARCO Reranker (Fallback)...")
            self.local_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.local_reranker_active = True
        except Exception as e:
            print(f"⚠️ Could not load Local Reranker: {e}")
            self.local_reranker_active = False
            
        print("🚀 Dual-Engine Omni-Retriever Ready.")

    def _retrieve_atomic(self, task, top_k=5, verbose=False):
        """
        Executes search for ONE sub-query and returns its own Top K results.
        Features automatic NVIDIA API fallback to local processing for both Vectors and Reranking.
        """
        candidates = {} # text -> score
        
        # ---------------------------------------------------------
        # 1. VECTOR (HyDE) - DUAL ENGINE
        # ---------------------------------------------------------
        if task.get('hyde_passage'):
            hyde_text = task['hyde_passage']
            vector_success = False
            
            # --- ATTEMPT 1: NVIDIA PRIMARY ---
            if self.nvidia_active:
                try:
                    response = self.nvidia_client.embeddings.create(
                        input=[hyde_text],
                        model=self.nvidia_model_name,
                        encoding_format="float",
                        extra_body={"input_type": "query", "truncate": "NONE"}
                    )
                    hyde_emb = np.array([response.data[0].embedding], dtype=np.float32)
                    
                    D, I = self.nvidia_index.search(hyde_emb, k=top_k*3)
                    for idx in I[0]:
                        if idx < len(self.nvidia_chunks):
                            candidates[self.nvidia_chunks[idx]['text']] = 0.0
                    vector_success = True
                    if verbose: print("   [Vector] 🟢 Used NVIDIA NIM embeddings.")
                except Exception as e:
                    if verbose: print(f"   [Vector] 🔴 NVIDIA API Failed ({e}). Initiating Fallback...")
            
            # --- ATTEMPT 2: LOCAL FALLBACK ---
            if not vector_success and self.local_active:
                try:
                    query_prompt = f"Represent this sentence for searching relevant passages: {hyde_text}"
                    hyde_emb = self.local_embedder.encode([query_prompt], convert_to_numpy=True)
                    
                    D, I = self.local_index.search(hyde_emb, k=top_k*3)
                    for idx in I[0]:
                        if idx < len(self.local_chunks):
                            candidates[self.local_chunks[idx]['text']] = 0.0
                    if verbose: print("   [Vector] 🟡 Used Local BGE embeddings (Fallback).")
                except Exception as e:
                    if verbose: print(f"   [Vector] ❌ Vector Search Failed Completely: {e}")

        # ---------------------------------------------------------
        # 2. BM25 (Keywords)
        # ---------------------------------------------------------
        bm25_query = f"{task['sub_query']} {' '.join(task.get('keywords', []))}"
        tokenized_query = bm25_query.split()
        bm25_docs = self.bm25.get_top_n(tokenized_query, self.chunk_texts, n=top_k*3)
        for txt in bm25_docs:
            candidates[txt] = 0.0

        # ---------------------------------------------------------
        # 3. GRAPH (Entities)
        # ---------------------------------------------------------
        graph_facts = []
        for entity in task.get('graph_entities', []):
            if entity in self.graph_engine.G:
                facts = self.graph_engine.get_neighbors(entity)
                graph_facts.extend(facts)
            else:
                for node in self.graph_engine.G.nodes():
                    if str(node).lower() == entity.lower():
                        facts = self.graph_engine.get_neighbors(node)
                        graph_facts.extend(facts)
                        break
        
        for fact in graph_facts[:20]: 
            candidates[fact] = 0.0
            
        # ---------------------------------------------------------
        # 4. RE-RANKING - DUAL ENGINE
        # ---------------------------------------------------------
        unique_docs = list(candidates.keys())
        if not unique_docs: return []
        
        scores = [0.0] * len(unique_docs)
        rerank_success = False
        
        # --- ATTEMPT 1: NVIDIA RERANKER ---
        if self.nvidia_active:
            try:
                passages_for_reranker = [{"text": doc} for doc in unique_docs]
                headers = {
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Accept": "application/json",
                }
                payload = {
                    "model": self.nvidia_rerank_model,
                    "query": {"text": task['sub_query']},
                    "passages": passages_for_reranker
                }
                
                response = requests.post(self.nvidia_rerank_url, headers=headers, json=payload, timeout=15)
                response.raise_for_status()
                
                # Map returned logits back to the original unique_docs indices
                rankings = response.json()['rankings']
                for rank_data in rankings:
                    idx = rank_data['index']
                    scores[idx] = rank_data['logit']
                    
                rerank_success = True
                if verbose: print("   [Rerank] 🟢 Used NVIDIA Mistral-4B Reranker.")
            except Exception as e:
                if verbose: print(f"   [Rerank] 🔴 NVIDIA Reranker Failed ({e}). Initiating Fallback...")

        # --- ATTEMPT 2: LOCAL RERANKER FALLBACK ---
        if not rerank_success and self.local_reranker_active:
            try:
                pairs = [[task['sub_query'], doc] for doc in unique_docs]
                # CrossEncoder predict returns a numpy array, convert to list
                scores = self.local_reranker.predict(pairs).tolist()
                if verbose: print("   [Rerank] 🟡 Used Local MS-MARCO Reranker (Fallback).")
            except Exception as e:
                if verbose: print(f"   [Rerank] ❌ Reranking Failed Completely: {e}")

        # Zip documents with scores and sort
        final_ranked = sorted(list(zip(unique_docs, scores)), key=lambda x: x[1], reverse=True)
        return final_ranked[:top_k]

    def _deduplicate_and_flatten(self, task_results):
        unique_map = {} 
        for task in task_results:
            for doc, score in task['results']:
                if doc not in unique_map:
                    unique_map[doc] = score
                else:
                    unique_map[doc] = max(unique_map[doc], score)
        
        final_list = sorted(unique_map.items(), key=lambda x: x[1], reverse=True)
        return final_list

    def retrieve(self, query, top_k_per_task=5, verbose=True):
        omni = self.optimizer.optimize(query)
        
        if "tasks" not in omni:
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
                if results:
                    print(f"   ↳ Retrieved {len(results)} contexts. Top Match: {results[0][1]:.4f}")
                else:
                    print(f"   ↳ Retrieved 0 contexts.")

        combined_context = self._deduplicate_and_flatten(final_structure["tasks"])
        final_structure["combined_context"] = combined_context
        
        if verbose:
            print(f"\n✅ Final Consolidated Contexts: {len(combined_context)} unique items.")

        return final_structure

# Initialize
# omni_tool = OmniRetriever(GRAPH_PATH, BM25_INDEX_PATH, MODEL_NAME)