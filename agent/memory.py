from langchain_core.messages import HumanMessage, AIMessage

class ConversationMemory:
    def __init__(self, max_turns: int = 4):
        # max_turns = how many back-and-forth exchanges to remember
        self.max_turns = max_turns
        self.history = []

    def add_turn(self, human: str, ai: str):
        self.history.append(HumanMessage(content=human))
        self.history.append(AIMessage(content=ai))
        # Keep only last N turns to avoid context bloat
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []