from langgraph.graph import StateGraph, END
from IPython.display import Markdown, display

from state import BrainState, route_scouts, route_verification
from nodes import (
    retrieve_node, audit_node, web_search_node, 
    synthesize_node, verify_node, refine_node, 
    wiki_search_node, web_search_node
)
from services import recorder

def build_graph():
    workflow = StateGraph(BrainState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("audit", audit_node)
    workflow.add_node("wiki_search", wiki_search_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("verify", verify_node)      # NEW
    workflow.add_node("refine", refine_node)      # NEW
    workflow.add_node("synthesize", synthesize_node)

    # Flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "audit")

    # The First Fork (From Initial Audit)
    workflow.add_conditional_edges(
        "audit",
        route_scouts,
        {
            "wiki_search": "wiki_search",
            "web_search": "web_search",
            "synthesize": "synthesize"
        }
    )

    # The Convergence (Scouts -> Verifier)
    workflow.add_edge("wiki_search", "verify")
    workflow.add_edge("web_search", "verify")

    # The Second Fork (The Loop Back)
    workflow.add_conditional_edges(
        "verify",
        route_verification,
        {
            "synthesize": "synthesize", # Success or Give Up
            "refine": "refine"          # Failure -> Try Again
        }
    )

    # The Loop Edge (Refiner -> Router)
    # We send 'refine' back to the router logic. 
    # Since 'route_scouts' isn't a node, we connect Refiner to the scouts conditionally.
    workflow.add_conditional_edges(
        "refine",
        route_scouts,
        {
            "wiki_search": "wiki_search",
            "web_search": "web_search"
        }
    )
    workflow.add_edge("synthesize", END)
    return workflow.compile()


def ask_brain(question: str, app):
    recorder.start_run(question)
    
    print(f"\n❓ QUERY: {question}\n" + "="*40)
    
    try:
        result = app.invoke({"query": question})
        display(Markdown(result["final_answer"]))
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        path = recorder.save_report()
        print(f"\n📄 FULL FIDELITY LOG: {path}")