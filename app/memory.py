from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import messages_from_dict, messages_to_dict


class SQLiteHistory(BaseChatMessageHistory):
    """LangChain message history backed by the application's owner-checked store."""

    def __init__(self, store, owner, session_id):
        self.store, self.owner, self.session_id = store, owner, session_id

    @property
    def messages(self):
        return messages_from_dict(self.store.messages(self.owner, self.session_id))

    def add_messages(self, messages):
        self.store.append_messages(self.owner, self.session_id, messages_to_dict(messages))

    def clear(self):
        self.store.clear_messages(self.owner, self.session_id)


def bounded_history(messages, char_budget):
    selected, used = [], 0
    # Keep complete user/assistant pairs and strip stored citation metadata from prompts.
    from langchain_core.messages import AIMessage, HumanMessage

    for i in range(len(messages) - 2, -1, -2):
        pair = messages[i : i + 2]
        cost = sum(len(str(m.content).encode("utf-8")) for m in pair)
        if used + cost > char_budget:
            break
        selected[0:0] = [HumanMessage(str(pair[0].content)), AIMessage(str(pair[1].content))]
        used += cost
    return selected
