import numpy as np
import pickle
import faiss
import unicodedata
from sentence_transformers import SentenceTransformer, CrossEncoder

from typing import List, Dict, Any
from prompts import QUERY_DECOMPOSITION_PROMPT
from neo4j import GraphDatabase
from collections import defaultdict
from llm_client import llm_client

# IMPORTS FROM CONFIG
from config import ( 
    STOP_NODES, 
    STOP_RELATIONS, 
    SUPER_NODE_THRESHOLD,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    BM25_INDEX_PATH,
    NEW_FAISS_INDEX_PATH, NEW_CHUNK_METADATA_PATH,
    OLD_FAISS_INDEX_PATH, OLD_CHUNK_METADATA_PATH
)


# ==========================================
# ROBUST QUERY OPTIMIZER (AUTO-FIXING)
# ==========================================
class QueryOptimizer:
    def __init__(self, provider="groq", model_name="llama-3.3-70b-versatile"):
        self.provider = provider
        self.model_name = model_name

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
            # response = self.llm.invoke([
            #     SystemMessage(content=system_prompt),
            #     HumanMessage(content=query)
            # ])
            # raw_result = json.loads(response.content)

            # 🚀 NEW: Call the centralized LLM client!
            raw_result = llm_client.generate_json(
                system_prompt=QUERY_DECOMPOSITION_PROMPT,
                user_prompt=query,
                provider=self.provider,
                model=self.model_name
            )

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
# GRAPH SEARCH MODULE (NEO4J UPGRADE)
# ==========================================
class Neo4jGraphSearcher:
    def __init__(self, uri, username, password, threshold=50):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.threshold = threshold

    def close(self):
        self.driver.close()

    def get_neighbors(self, entity_name):
        """
        Executes a 1-hop fuzzy search in Neo4j and returns rich semantic facts.
        """
        if not entity_name or len(entity_name) < 2:
            return []

        query = """
        // 1. Fuzzy match the starting entity
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($entity_name)
        
        // 2. Expand 1-hop in any direction
        MATCH (n)-[r]-(m:Entity)
        
        // 3. Apply Stop-Words (converted to lower case in Python before passing)
        WHERE NOT toLower(m.name) IN $stop_nodes
          AND NOT toLower(type(r)) IN $stop_relations
          
        // 4. Return top results to prevent Super-Node context blowouts
        WITH n, r, m
        LIMIT $threshold
        
        RETURN n.name AS source, 
               type(r) AS rel_type, 
               m.name AS target, 
               r.description AS description
        """
        
        facts = []
        with self.driver.session() as session:
            # We pass the STOP lists from your config directly into the query
            result = session.run(
                query, 
                entity_name=entity_name,
                stop_nodes=[sn.lower() for sn in STOP_NODES],
                stop_relations=[sr.lower() for sr in STOP_RELATIONS],
                threshold=self.threshold
            )
            
            for record in result:
                src = record["source"]
                rel = record["rel_type"]
                tgt = record["target"]
                desc = record["description"]
                
                # Format the rich fact for the LLM
                fact_str = f"{src} --[{rel}]--> {tgt}"
                if desc:
                    fact_str += f" | Context: {desc}"
                    
                facts.append(fact_str)
                
        return facts

# ==========================================
# CELL 4: THE OMNI-RETRIEVER (DUAL-ENGINE VECTOR & RERANK)
# ==========================================
class OmniRetriever:
    def __init__(self, provider, model_name): # Removed graph_path
        # Tools
        self.optimizer = QueryOptimizer(provider=provider, model_name=model_name)
        # self.graph_engine = GraphSearcher(graph_path, threshold=SUPER_NODE_THRESHOLD) old graph engine
        print("🔗 Connecting to Neo4j Knowledge Graph...")
        self.graph_engine = Neo4jGraphSearcher(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            threshold=SUPER_NODE_THRESHOLD
        )
        
        print("📂 Loading Resources...")
        
        # 1. Setup NVIDIA NIM (Primary Vector & Rerank Engine)
        self.nvidia_model_name = "nvidia/nv-embed-v1"
        self.nvidia_rerank_model = "nv-rerank-qa-mistral-4b:1"
        
        try:
            print("🧠 Loading NVIDIA Vector Store (Primary)...")
            self.nvidia_index = faiss.read_index(NEW_FAISS_INDEX_PATH)
            with open(NEW_CHUNK_METADATA_PATH, "rb") as f:
                self.nvidia_chunks = pickle.load(f)
            self.nvidia_active = True
        except Exception as e:
            print(f"⚠️ Could not load NVIDIA store: {e}")
            self.nvidia_active = False

        # 2. Setup BGE-Small (Local Fallback Vector Engine)
        try:
            print("💾 Loading BGE Local Vector Store (Fallback)...")
            self.local_embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
            self.local_index = faiss.read_index(OLD_FAISS_INDEX_PATH)
            with open(OLD_CHUNK_METADATA_PATH, "rb") as f:
                self.local_chunks = pickle.load(f)
            self.local_active = True
        except Exception as e:
            print(f"⚠️ Could not load Local Fallback store: {e}")
            self.local_active = False

        # 3. Setup BM25 (Keywords)
        # BM25 needs a list of texts. We prefer the NVIDIA chunks if available.
        active_chunks = self.nvidia_chunks if self.nvidia_active else self.local_chunks
        self.chunk_texts = [c['text'] for c in active_chunks]
        
        with open(BM25_INDEX_PATH, "rb") as f:
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
        source_tracker = defaultdict(set) # NEW: text -> set of sources

        # ---------------------------------------------------------
        # 1. VECTOR (HyDE) - DUAL ENGINE
        # ---------------------------------------------------------
        if task.get('hyde_passage'):
            hyde_text = task['hyde_passage']
            vector_success = False
            
            # --- ATTEMPT 1: NVIDIA PRIMARY ---
            if self.nvidia_active:
                try:
                    embedding_data = llm_client.generate_embeddings(text=hyde_text, model=self.nvidia_model_name)
                    hyde_emb = np.array([embedding_data], dtype=np.float32)
                    
                    D, I = self.nvidia_index.search(hyde_emb, k=top_k*3)
                    for idx in I[0]:
                        if idx < len(self.nvidia_chunks):
                            txt = self.nvidia_chunks[idx]['text']
                            candidates[txt] = 0.0
                            source_tracker[txt].add('Vector (NVIDIA)')
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
                            txt = self.local_chunks[idx]['text']
                            candidates[txt] = 0.0
                            source_tracker[txt].add('Vector (Local BGE)')
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
            source_tracker[txt].add('BM25 Keyword')

        # ---------------------------------------------------------
        # 3. GRAPH (Entities)
        # ---------------------------------------------------------
        # graph_facts = []
        # for entity in task.get('graph_entities', []):
        #     if entity in self.graph_engine.G:
        #         facts = self.graph_engine.get_neighbors(entity)
        #         graph_facts.extend(facts)
        #     else:
        #         for node in self.graph_engine.G.nodes():
        #             if str(node).lower() == entity.lower():
        #                 facts = self.graph_engine.get_neighbors(node)
        #                 graph_facts.extend(facts)
        #                 break
        
        # for fact in graph_facts[:20]: 
        #     candidates[fact] = 0.0
        
        # ---------------------------------------------------------
        # 3. GRAPH (Entities) via Neo4j
        # ---------------------------------------------------------
        graph_facts = []
        for entity in task.get('graph_entities', []):
            # Neo4j handles the fuzzy matching and stop-words directly!
            facts = self.graph_engine.get_neighbors(entity)
            graph_facts.extend(facts)
        for fact in graph_facts[:20]:
            candidates[fact] = 0.0
            source_tracker[fact].add('Graph (Neo4j)')
            
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
                rankings = llm_client.rerank_passages(
                    query=task['sub_query'],
                    passages=unique_docs,
                    model=self.nvidia_rerank_model
                )
                
                # Map returned logits back to the original unique_docs indices
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

        # Zip documents with scores AND sources
        final_ranked = []
        for i, doc in enumerate(unique_docs):
            final_ranked.append((doc, scores[i], list(source_tracker[doc])))
            
        final_ranked.sort(key=lambda x: x[1], reverse=True)
        return final_ranked[:top_k]

    def _deduplicate_and_flatten(self, task_results):
        unique_map = {} 
        for task in task_results:
            for doc, score, sources in task['results']: # Unpack 3 items now
                if doc not in unique_map:
                    unique_map[doc] = {"score": score, "sources": set(sources)}
                else:
                    unique_map[doc]["score"] = max(unique_map[doc]["score"], score)
                    unique_map[doc]["sources"].update(sources)
        
        # Convert back to a sorted list of tuples
        final_list = sorted(
            [(k, v["score"], list(v["sources"])) for k, v in unique_map.items()], 
            key=lambda x: x[1], 
            reverse=True
        )
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