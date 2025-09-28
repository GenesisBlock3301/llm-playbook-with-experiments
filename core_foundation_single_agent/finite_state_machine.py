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

# Initial state
state = ChatState(user_input="", conversation_history=[], step="start")

# Step 1: User says hi
state = update_state(state, "hi", "greeted")

# Step 2: Bot asks for name (simulate user input)
state = update_state(state, "What is your name?", "asked_name")

# Step 3: User provides name
state = update_state(state, "Alice", "received_name")

print(state)

# Streaming bot response
async def stream_response(_state: ChatState):
    if _state.step == "greeted":
        reply = "Hello! Nice to meet you."
    elif _state.step == "asked_name":
        reply = "May I know your name?"
    elif _state.step == "received_name":
        reply = f"Nice to meet you, {_state.user_input}!"
    else:
        reply = f"Bot replying to: {_state.user_input}"

    for char in reply:
        await asyncio.sleep(0.05)
        yield char

async def main():
    async for char in stream_response(state):
        print(char, end="", flush=True)

asyncio.run(main())
