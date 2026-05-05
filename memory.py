class ChatMemory:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history * 2:  # 5 pairs of user/assistant
            self.history = self.history[-self.max_history * 2:]

    def get_formatted_history(self):
        formatted = ""
        for msg in self.history:
            role = msg["role"].upper()
            formatted += f"{role}: {msg['content']}\n"
        return formatted

    def clear(self):
        self.history = []
