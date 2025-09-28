# Memory

We’ll build a simple agent with:

1. Short-term memory → Stores recent chat messages in memory.

2. Long-term memory → Searches knowledge from a Vector DB (like Chroma/FAISS).

3. RAG flow → Combines both memories to answer.

4. Self-correction → If an answer is incomplete, the agent re-checks the vector DB and refines the response.