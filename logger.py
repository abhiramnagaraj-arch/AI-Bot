import json
import os
from datetime import datetime

LOG_FILE = "pipeline_logs.jsonl"

def log_event(event_data):
    """
    Logs an event to a JSONL file.
    event_data should containing query, history, context, answer, scores, etc.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        **event_data
    }
    
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def get_logs():
    if not os.path.exists(LOG_FILE):
        return []
    
    logs = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            logs.append(json.loads(line))
    return logs
