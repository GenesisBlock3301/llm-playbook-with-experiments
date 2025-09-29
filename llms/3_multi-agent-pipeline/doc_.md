# POC Project: Multi-agent Pipeline with Supervisor.

## Goal:
### Create a pipeline:

- Supervisor Agent breaks a user request into subtasks.
- 2 - 3 worker Agent(e.g; researcher, summarizer, knowledge retriever) solve subtasks.
- Supervisor orchestrates execution, retries on failure and applies conditional branching.(if subtask is unclear, 
send to retriever first.)


### System design:

1. Supervisor agent:
    
   - Reads user query, decide which agents to invoke.
   - Handles failure (retries & clarification).
   - Perform branching(decides: "Do we need external knowledge?")
2. Research Agent
   
   - Uses Ollama2 + ChromaDB to retrieve relevant knowledge.

3. Summarize Agent:

   - Takes research output -- creates a clean, human friendly summary.
4. Answer agent:

   - Finalize the response into professional output.

### Flow:
1. User asks: "Explain how to multi-agent orchestration works in real world companies?"
2. Supervisor checks:

   - If query is knowledge heavy --> send to research agent.
   - if knowledge is found --> pass to summarizer agent.
   - If summarization fails --> Supervisor retries.
   - Finally, Answer aget polish output.