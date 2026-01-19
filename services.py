import time
from config import GRAPH_PATH, CHUNKS_PATH, VECTOR_INDEX_PATH, BM25_INDEX_PATH, QUERY_MODEL
from tracing import DeepFlightRecorder

# Import your original engines (assuming these files are in your root or python path)
from rag_engine import OmniRetriever
from web_engine import DeepWebScout, WikiScout, KnowledgeCurator

# 1. Initialize Recorder Singleton
recorder = DeepFlightRecorder()

class TracedOmniRetriever(OmniRetriever):
    def retrieve(self, query, top_k_per_task=5, verbose=False):
        # 1. OPTIMIZATION TRACE
        t_start = time.perf_counter()
        omni = self.optimizer.optimize(query)
        duration = (time.perf_counter() - t_start) * 1000
        
        # Log the full breakdown so we can see the isolated metadata in the UI
        recorder.log_event("QUERY_DECOMPOSITION", "QueryOptimizer", omni, duration)
        
        final_structure = {"original_query": query, "tasks": []}
        
        # 2. ATOMIC EXECUTION TRACE
        for i, task in enumerate(omni['tasks']):
            sub_q = task['sub_query']
            candidates = {} 
            
            # --- A. VECTOR SEARCH (HyDE) ---
            t0 = time.perf_counter()
            vector_hits = []
            if task.get('hyde_passage'):
                emb = self.embedder.encode([task['hyde_passage']], convert_to_numpy=True)
                # Matches the new OmniRetriever logic (k*3)
                D, I = self.index.search(emb, k=top_k_per_task*3)
                for idx in I[0]: 
                    if idx < len(self.chunks): 
                        txt = self.chunks[idx]['text']
                        candidates[txt] = 0.0
                        vector_hits.append(txt) 
            
            vec_dur = (time.perf_counter() - t0) * 1000
            recorder.log_event("SEARCH_VECTOR", f"Task_{i}", {
                "hyde_used": task.get('hyde_passage'), 
                "hit_count": len(vector_hits), 
                "full_results": vector_hits 
            }, vec_dur)

            # --- B. BM25 SEARCH (Keywords) --- 
            # [NEW] Added this block to capture the keyword search
            t0 = time.perf_counter()
            keyword_hits = []
            if task.get('keywords'):
                bm25_query = f"{sub_q} {' '.join(task['keywords'])}"
                tokenized_query = bm25_query.split()
                bm25_docs = self.bm25.get_top_n(tokenized_query, self.chunk_texts, n=top_k_per_task*3)
                for txt in bm25_docs:
                    candidates[txt] = 0.0
                    keyword_hits.append(txt)

            bm25_dur = (time.perf_counter() - t0) * 1000
            recorder.log_event("SEARCH_BM25", f"Task_{i}", {
                "keywords_used": task.get('keywords'),
                "hit_count": len(keyword_hits),
                "full_results": keyword_hits
            }, bm25_dur)

            # --- C. GRAPH SEARCH (Entities) ---
            t0 = time.perf_counter()
            graph_hits = []
            for entity in task.get('graph_entities', []):
                # Exact Match
                if entity in self.graph_engine.G: 
                    facts = self.graph_engine.get_neighbors(entity)
                    for f in facts: 
                        candidates[f] = 0.0
                        graph_hits.append(f)
                # Fuzzy Match (Quick check)
                else:
                    for node in self.graph_engine.G.nodes():
                        if str(node).lower() == entity.lower():
                            facts = self.graph_engine.get_neighbors(node)
                            for f in facts:
                                candidates[f] = 0.0
                                graph_hits.append(f)
                            break
            
            graph_dur = (time.perf_counter() - t0) * 1000
            recorder.log_event("SEARCH_GRAPH", f"Task_{i}", {
                "entities_used": task.get('graph_entities'), 
                "hit_count": len(graph_hits), 
                "full_facts": graph_hits 
            }, graph_dur)
            
            # --- D. RERANKING (Full Scores) ---
            t0 = time.perf_counter()
            unique_docs = list(candidates.keys())
            score_log = []
            results = []

            if unique_docs:
                pairs = [[sub_q, doc] for doc in unique_docs]
                scores = self.reranker.predict(pairs)
                ranked = sorted(list(zip(unique_docs, scores)), key=lambda x: x[1], reverse=True)
                results = ranked[:top_k_per_task]
                
                # LOG FULL SCORES for visualization
                score_log = [{"text": r[0], "score": float(r[1])} for r in results]
                
            rerank_dur = (time.perf_counter() - t0) * 1000
            recorder.log_event("RERANKER_SCORES", f"Task_{i}", score_log, rerank_dur)

            final_structure["tasks"].append({"sub_query": sub_q, "results": results})

        # 3. GLOBAL MERGE (New Step)
        # We use the parent class method for consistency
        combined_context = self._deduplicate_and_flatten(final_structure["tasks"])
        final_structure["combined_context"] = combined_context
        
        return final_structure

class TracedWebScout(DeepWebScout):
    def search_and_extract(self, sub_query: str):
        t0 = time.perf_counter()
        result = super().search_and_extract(sub_query)
        duration = (time.perf_counter() - t0) * 1000
        
        # Capture raw Tavily output structure
        recorder.log_event("WEB_SEARCH_RAW", "DeepWebScout", result, duration)
        return result

class TracedWikiScout(WikiScout):
    def search_and_extract(self, entity_title: str):
        t0 = time.perf_counter()
        result = super().search_and_extract(entity_title)
        duration = (time.perf_counter() - t0) * 1000
        recorder.log_event("WIKI_SEARCH_RAW", "WikiScout", result, duration)
        return result

# 3. Instantiate Resources
print("⚙️  Injecting High-Fidelity Probes...")
traced_retriever = TracedOmniRetriever(
    GRAPH_PATH, CHUNKS_PATH, VECTOR_INDEX_PATH, BM25_INDEX_PATH, QUERY_MODEL
)
traced_scout = TracedWebScout()
traced_wiki = TracedWikiScout()
curator = KnowledgeCurator()
print("🚀 Probes Active.")