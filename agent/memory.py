from langchain_core.messages import HumanMessage, AIMessage

class ConversationMemory:
    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns
        # Store as plain dicts for session state serialization safety
        self._raw = []

    def add_turn(self, human: str, ai: str):
        self._raw.append({"human": human, "ai": ai})
        if len(self._raw) > self.max_turns:
            self._raw = self._raw[-self.max_turns:]

    def get_history(self):
        history = []
        for turn in self._raw:
            history.append(HumanMessage(content=turn["human"]))
            history.append(AIMessage(content=turn["ai"]))
        return history

    def clear(self):
        self._raw = []

    def to_dict(self) -> list:
        """Serialize for session state storage."""
        return self._raw.copy()

    @classmethod
    def from_dict(cls, data: list, max_turns: int = 4) -> "ConversationMemory":
        """Restore from session state storage."""
        mem = cls(max_turns=max_turns)
        mem._raw = data
        return mem