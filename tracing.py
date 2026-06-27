import os
import json
import copy
import logging
from datetime import datetime
from typing import Any
from config import LOG_FILE_PATH, REPORTS_DIR

# Setup Logging (Moved from global scope to inside this module or a setup function)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("BRAIN")

class DeepFlightRecorder:
    def __init__(self):
        self.current_run_id = None
        self.run_data = {}
        self.callbacks = []
        if not os.path.exists(REPORTS_DIR): os.makedirs(REPORTS_DIR)

    def register_callback(self, callback):
        self.callbacks.append(callback)

    def clear_callbacks(self):
        self.callbacks = []

    def start_run(self, query):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_id = f"trace_{timestamp}"
        self.run_data = {
            "meta": { "run_id": self.current_run_id, "query": query, "timestamp": timestamp },
            "trace_log": []
        }
        print(f"📼 Deep Trace started: {self.current_run_id}")

    def log_event(self, event_type: str, component: str, data: Any, duration_ms: float = 0.0):
        entry = {
            "type": event_type,
            "component": component,
            "duration_ms": round(duration_ms, 2),
            "data": copy.deepcopy(data) if isinstance(data, (dict, list)) else str(data)
        }
        self.run_data["trace_log"].append(entry)
        # Notify SSE/WebSocket subscribers
        for cb in self.callbacks:
            try:
                cb(entry)
            except Exception:
                pass

    def save_report(self):
        filename = f"{self.current_run_id}_deep_trace.json"
        filepath = os.path.join(REPORTS_DIR, filename)
        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(self.run_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Deep Trace Report saved to: {filepath}")
        return filepath