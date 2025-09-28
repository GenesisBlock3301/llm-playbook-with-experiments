from typing import Callable, Dict
import asyncio
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ChatState:
    user_input: str
    conversation_history: list
    step: str


def update_state(state: ChatState, user_input: str, new_step: str = None):
    return replace(
        state,
        user_input=user_input,
        conversation_history=state.conversation_history + [user_input],
        step=new_step or state.step
    )

class Node:
    def __init__(self, name: str, handler: Callable[[ChatState], ChatState], next_nodes: Dict[str, str]):
        self.name = name
        self.handler = handler
        self.next_nodes = next_nodes

def greet_node(state: ChatState):
    print("Bot: Hi! How can I help you?")
    return update_state(state, state.user_input, "awaiting_input")

def farewell_node(state: ChatState):
    print("Bot: Goodbye!")
    return update_state(state, state.user_input, "end")

graph = {
    "start": Node("start", greet_node, {"bye": "farewell", "help": "assist"}),
    "farewell": Node("farewell", farewell_node, {}),
    "assist": Node("assist", lambda s: print("Bot: Here some help!"),{"bye": "farewell"}),
}

# Initial state
state = ChatState(user_input="", conversation_history=[], step="start")
# Step 1: User says hi
state = update_state(state, "hi", "greeted")
# Step 2: Bot asks for name (simulate user input)
state = update_state(state, "What is your name?", "asked_name")

# Step 3: User provides name
state = update_state(state, "Alice", "received_name")

current_node = graph["start"]

state = current_node.handler(state)
user_input = "help"
current_node = graph[current_node.next_nodes.get(user_input, "farewell")]
state = current_node.handler(state)