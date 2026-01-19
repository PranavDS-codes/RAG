import time
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import ( 
    GROQ_API_KEY, AUDIT_MODEL, SYNTHESIZE_MODEL, 
    VERIFY_MODEL, REFINE_MODEL )
from state import BrainState
from prompts import ( AUDIT_NODE_PROMPT, SYNTHESIZE_NODE_PROMPT, 
    VERIFY_NODE_PROMPT, REFINE_NODE_PROMPT )

# Import the initialized tools and recorder
from services import traced_retriever, traced_scout, traced_wiki, curator, recorder

def retrieve_node(state: BrainState):
    query = state["query"]
    print(f"\n📚 [LIBRARIAN] Executing Traced Retrieval...")
    
    t0 = time.perf_counter()
    results = traced_retriever.retrieve(query, top_k_per_task=5)
    duration = (time.perf_counter() - t0) * 1000
    
    evidence = []
    # We also keep a raw list for the Verifier/Synthesizer if needed
    raw_context = [] 
    
    for task in results.get("tasks", []):
        sub_q = task["sub_query"]
        for txt, score in task["results"]:
            formatted = f"[Score: {score:.2f}] {txt}"
            evidence.append(formatted)
            raw_context.append(formatted)
    
    recorder.log_event("NODE_OUTPUT", "retrieve_node", {"evidence_count": len(evidence)}, duration)
    
    # Return BOTH the specific key for Audit AND the combined context
    return {
        "internal_knowledge": evidence,
        # We initialize combined_context here so it's not None
        "combined_context": raw_context 
    }

def audit_node(state: BrainState):
    print("🕵️‍♂️ [AUDITOR] Auditing Evidence & Freshness...")
    t0 = time.perf_counter()
    
    query = state["query"]
    evidence_text = "\n".join(state["internal_knowledge"][:20]) # Increased context limit
    
    sys_msg = AUDIT_NODE_PROMPT
    user_msg = f"QUERY: {query}\n\nINTERNAL EVIDENCE:\n{evidence_text}"
    
    # Init LLM
    audit_llm = ChatGroq(temperature=0, model_name=AUDIT_MODEL, api_key=GROQ_API_KEY)
    
    response = audit_llm.invoke([
        SystemMessage(content=sys_msg),
        HumanMessage(content=user_msg)
    ])
    
    duration = (time.perf_counter() - t0) * 1000
    
    # LOG FULL PROMPT AND RESPONSE
    recorder.log_event("LLM_AUDIT", "AuditNode", {
        "full_system_prompt": sys_msg,
        "full_user_prompt": user_msg,
        "full_response": response.content
    }, duration)
    
    try:
        analysis = json.loads(response.content)
    except:
        analysis = {"sufficient": False, "missing_topics": [query]}
        
    if analysis.get("sufficient"):
        print("   ✅ Evidence is sufficient & fresh.")
    else:
        print(f"   ❌ Gaps/Stale Data detected: {analysis.get('missing_topics')}")

    return {"gap_analysis": analysis}


def synthesize_node(state: BrainState):
    print("✍️ [SYNTHESIZER] Writing Final Answer...")
    t0 = time.perf_counter()
    
    synth_llm = ChatGroq(temperature=0, model_name=SYNTHESIZE_MODEL, api_key=GROQ_API_KEY)
    
    # [FIX 1] Get the unified context bucket
    context_list = state.get("combined_context", [])
    
    # [FIX 2] Format it clearly so the LLM isn't reading a messy raw list
    # This handles cases where context might be mixed types
    formatted_context = ""
    for i, item in enumerate(context_list):
        formatted_context += f"--- Source {i+1} ---\n{str(item)}\n\n"
        
    if not formatted_context:
        formatted_context = "No relevant context found in Internal or External sources."

    sys_msg = SYNTHESIZE_NODE_PROMPT
    
    # [FIX 3] Update the user message to use the new formatted context
    user_msg = f"""
    QUESTION: {state['query']}
    
    ALL GATHERED INTELLIGENCE:
    {formatted_context}
    """
    
    response = synth_llm.invoke([
        SystemMessage(content=sys_msg),
        HumanMessage(content=user_msg)
    ])
    
    duration = (time.perf_counter() - t0) * 1000
    
    recorder.log_event("LLM_SYNTHESIS", "SynthesizeNode", {
        "full_system_prompt": sys_msg,
        "full_user_prompt": user_msg,
        "full_response": response.content
    }, duration)
    
    return {"final_answer": response.content}

def verify_node(state: BrainState):
    print("🧐 [VERIFIER] Inspecting evidence quality...")
    t0 = time.perf_counter()
    
    query = state["query"]
    # Get the combined context (Search + Wiki + Internal)
    context_list = state.get("combined_context", [])
    
    # Safety check: If no context, fail immediately
    if not context_list:
        print("   ❌ Verification Failed: No context available.")
        return {
            "gap_analysis": {
                "sufficient": False,
                "reasoning": "No context was retrieved from any source."
            },
            "loop_count": state.get("loop_count", 0)
        }

    # Format context clearly with delimiters so the LLM can distinguish sources
    formatted_context = ""
    for i, item in enumerate(context_list):
        # Ensure we are handling strings
        text = str(item)
        formatted_context += f"--- EVIDENCE FRAGMENT {i+1} ---\n{text}\n\n"
    
    # Import the NEW prompt from your prompts file
    from prompts import VERIFY_NODE_PROMPT 
    
    user_msg = f"""
    USER QUERY: {query}
    
    RETRIEVED CONTEXT:
    {formatted_context} 
    """ 
    # Note: Keep a high character limit (15k-20k) for the Verifier so it sees details.
    
    verifier_llm = ChatGroq(
        temperature=0, 
        model_name=VERIFY_MODEL, 
        api_key=GROQ_API_KEY,
        model_kwargs={"response_format": {"type": "json_object"}} # Force JSON mode
    )
    
    try:
        response = verifier_llm.invoke([
            SystemMessage(content=VERIFY_NODE_PROMPT),
            HumanMessage(content=user_msg)
        ])
        verification = json.loads(response.content)
        
    except Exception as e:
        print(f"   ⚠️ Verifier Error: {e}")
        # Default to False on error to be safe
        verification = {"sufficient": False, "reasoning": f"Verifier crashed: {str(e)}"}

    duration = (time.perf_counter() - t0) * 1000
    
    recorder.log_event("LLM_VERIFY", "VerifyNode", verification, duration)
    
    if verification.get("sufficient"):
        print("   ✅ Quality Check Passed.")
    else:
        print(f"   ❌ Quality Check Failed: {verification.get('reasoning')}")

    return {
        "gap_analysis": {
            "sufficient": verification.get("sufficient"),
            # Preserve the routing decision if it exists, or default to WEB
            "routing_decision": state.get("gap_analysis", {}).get("routing_decision", "WEB"), 
            "reasoning": verification.get("reasoning")
        },
        # We generally do not increment loop_count here; that happens in the Refine Node
        # But we pass it through just in case.
        "loop_count": state.get("loop_count", 0)
    }

def refine_node(state: BrainState):
    print("🔄 [REFINER] Search failed. Re-strategizing...")
    t0 = time.perf_counter()
    
    query = state["query"]
    last_decision = state["gap_analysis"].get("routing_decision", "WEB")
    failure_reason = state["gap_analysis"].get("reasoning", "Unknown gap")
    current_loop = state.get("loop_count", 0) + 1
    
    sys_msg = REFINE_NODE_PROMPT
    
    user_msg = f"""
    QUERY: {query}
    PREVIOUS SOURCE: {last_decision}
    FAILURE REASON: {failure_reason}
    """
    
    refiner_llm = ChatGroq(temperature=0.4, model_name=REFINE_MODEL, api_key=GROQ_API_KEY)
    
    try:
        response = refiner_llm.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=user_msg)
        ])
        new_strategy = json.loads(response.content)
    except Exception as e:
        print(f"   ⚠️ Refiner Error: {e}")
        # Fallback: Just switch sources and reuse the original query
        new_source = "WEB" if last_decision == "WIKI" else "WIKI"
        new_strategy = {
            "routing_decision": new_source,
            "wiki_topics": [query],
            "web_topics": [query]
        }

    duration = (time.perf_counter() - t0) * 1000
    
    recorder.log_event("LLM_REFINE", "RefineNode", {
        "loop": current_loop,
        "new_plan": new_strategy
    }, duration)
    
    print(f"   ↳ Loop {current_loop}: Switching to [{new_strategy['routing_decision']}] with new topics.")

    # We return the NEW plan into 'gap_analysis' so the Router sees it as fresh instructions
    return {
        "loop_count": current_loop,
        "gap_analysis": {
            "sufficient": False, # Still false until proven otherwise
            "routing_decision": new_strategy["routing_decision"],
            "wiki_topics": new_strategy.get("wiki_topics", []),
            "web_topics": new_strategy.get("web_topics", [])
        }
    }

def wiki_search_node(state: BrainState):
    gap_data = state.get("gap_analysis", {})
    topics = gap_data.get("wiki_topics") or gap_data.get("missing_topics", [])
    
    print(f"📚 [WIKI SCOUT] Consulting Encyclopedia for {len(topics)} topics...")
    t0 = time.perf_counter()
    wiki_facts = []
    
    for topic in topics:
        try:
            res = traced_wiki.search_and_extract(topic)
            if res["status"] == "success":
                # Wikipedia results usually have 'summary' as the main content
                summary = res.get('summary', '')[:500] 
                url = res.get('url', 'Wiki')
                
                fact_block = (
                    f"### WIKI TOPIC: {topic}\n"
                    f"**Source URL:** {url}\n"
                    f"**Excerpt:** {summary}...\n"
                )
                
                wiki_facts.append(fact_block)
                curator.curate(topic, res, source_type="wiki_librarian")
        except Exception as e:
            print(f"   ⚠️ Wiki Error on '{topic}': {e}")
            
    duration = (time.perf_counter() - t0) * 1000
    recorder.log_event("NODE_OUTPUT", "wiki_search_node", {"facts_found": len(wiki_facts)}, duration)
    
    return {
        "wiki_knowledge": wiki_facts,
        "combined_context": wiki_facts 
    }

def web_search_node(state: BrainState):
    gap_data = state.get("gap_analysis", {})
    # Handle both potential key names for robustness
    topics = gap_data.get("web_topics") or gap_data.get("missing_topics", [])
    
    print(f"🌐 [WEB SCOUT] Tracing Web Search for {len(topics)} topics...")
    t0 = time.perf_counter()
    web_facts = []
    
    for topic in topics:
        try:
            res = traced_scout.search_and_extract(topic)
            if res["status"] == "success":
                # 1. Capture the AI Summary (The "Claim")
                summary = res.get('tavily_answer', 'No summary provided.')
                
                # 2. Capture the Raw Evidence (The "Proof")
                # Extract top 2 snippets to ground the Verifier
                evidence_snippets = ""
                sources = res.get('curation_data', []) or res.get('results', [])
                for i, source in enumerate(sources[:2]): # Top 2 sources
                    content = source.get('snippet') or source.get('content') or source.get('raw_content') or ""
                    # Truncate content to avoid blowing up context window
                    evidence_snippets += f"   - [Source {i+1}]: {content[:400]}...\n"

                # 3. Format the Fact Block
                # We clearly separate Claim vs Evidence
                fact_block = (
                    f"### SEARCH TOPIC: {topic}\n"
                    f"**AI Summary:** {summary}\n"
                    f"**Raw Evidence Snippets:**\n{evidence_snippets}"
                )
                
                web_facts.append(fact_block)
                
                # Curate for the Knowledge Graph
                curator.curate(topic, res, source_type="web_scout")
        except Exception as e:
            print(f"   ⚠️ Search Error on '{topic}': {e}")
            
    duration = (time.perf_counter() - t0) * 1000
    recorder.log_event("NODE_OUTPUT", "web_search_node", {"facts_found": len(web_facts)}, duration)
    
    return {
        "web_knowledge": web_facts,
        "combined_context": web_facts 
    }