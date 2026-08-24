import json
from datetime import datetime

from .config import LOG_FILE

def log_event(event_data):
    """
    Logs an event to a JSONL file.
    event_data should containing query, history, context, answer, scores, etc.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        **event_data
    }
    
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def get_logs():
    if not LOG_FILE.exists():
        return []
    
    logs = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            logs.append(json.loads(line))
    return logs
