import sys
from llm_client import llm_client
from config import (
    PREMISE_MODEL, QUERY_MODEL, AUDIT_MODEL, VERIFY_MODEL,
    REFINE_MODEL, SCOUT_MODEL, SYNTHESIZE_MODEL,
    VECTOR_CURATOR_MODEL, GRAPH_CURATOR_MODEL
)

models_to_test = {
    "PREMISE_MODEL": PREMISE_MODEL,
    "QUERY_MODEL": QUERY_MODEL,
    "AUDIT_MODEL": AUDIT_MODEL,
    "VERIFY_MODEL": VERIFY_MODEL,
    "REFINE_MODEL": REFINE_MODEL,
    "SCOUT_MODEL": SCOUT_MODEL,
    "SYNTHESIZE_MODEL": SYNTHESIZE_MODEL,
    "VECTOR_CURATOR_MODEL": VECTOR_CURATOR_MODEL,
    "GRAPH_CURATOR_MODEL": GRAPH_CURATOR_MODEL
}

print("🧪 Starting Model Connectivity and Capability Diagnostic...")
print("================================================================")

all_pass = True
for name, model in models_to_test.items():
    print(f"Testing {name:23} | Model: {model:32} -> ", end="", flush=True)
    try:
        if "SYNTHESIZE" in name:
            res = llm_client.generate_text(
                system_prompt="You are a helpful assistant.",
                user_prompt="Say 'OK'",
                model=model,
                temperature=0.0,
                retries=1
            )
            if "OK" in res.upper():
                print("🟢 PASS (Text)")
            else:
                print(f"🟡 WEAK PASS (Text output: '{res}')")
        else:
            res = llm_client.generate_json(
                system_prompt="You are a helpful assistant.",
                user_prompt="Return JSON with key 'status' and value 'OK'",
                model=model,
                temperature=0.0,
                retries=1
            )
            if res.get("status") == "OK":
                print("🟢 PASS (JSON)")
            else:
                print(f"🟡 WEAK PASS (JSON output: {res})")
    except Exception as e:
        print(f"🔴 FAIL ({str(e).strip()})")
        all_pass = False

print("================================================================")
if all_pass:
    print("🎉 All configured models are ONLINE and fully functional!")
else:
    print("⚠️ Diagnostics finished with issues. See failures above.")
