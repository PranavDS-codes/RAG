from typing import TypedDict, List, Dict, Annotated
import operator

# class BrainState(TypedDict):
#     query: str
#     internal_knowledge: List[str]
#     external_knowledge: List[str]
#     gap_analysis: Dict
#     final_answer: str

class BrainState(TypedDict):
    query: str
    tasks: List[Dict]
    
    # Storage for specific streams (The Synthesizer needs these separate)
    internal_knowledge: List[str]
    wiki_knowledge: List[str]
    web_knowledge: List[str]
    
    # Storage for the Verifier (The Merged View)
    # Annotated[List, operator.add] means "Append new items to this list"
    combined_context: Annotated[List, operator.add] 
    
    # Logic State
    gap_analysis: Dict
    loop_count: int
    feedback: str
    scout_decision: str
    final_answer: str

def route_scouts(state):
    """
    Determines next steps. Returns a LIST of nodes to run.
    """
    analysis = state["gap_analysis"]
    decision = analysis.get("routing_decision", "WEB")
    is_sufficient = analysis.get("sufficient", False)
    
    # 1. If sufficient, skip to synthesis
    if is_sufficient:
        return "synthesize"

    # 2. Handle "BOTH" (Parallel Execution)
    if decision == "BOTH":
        return ["wiki_search", "web_search"]
        
    # 3. Handle Single Paths
    if decision == "WIKI":
        return "wiki_search"
    
    # Default fallback
    return "web_search"

# 1. Routing Logic (Updated to handle Loop Limits)
def route_verification(state):
    """
    Decides: Synthesize (Success) OR Refine (Failure) OR Give Up (Max Loops).
    """
    is_sufficient = state["gap_analysis"].get("sufficient", False)
    loop_count = state.get("loop_count", 0)
    
    if is_sufficient:
        return "synthesize"
    
    # SAFETY LIMIT: Stop after 2 correction attempts
    if loop_count >= 2:
        print("⚠️ [SYSTEM] Max loops reached. Synthesizing best effort.")
        return "synthesize"
        
    return "refine"