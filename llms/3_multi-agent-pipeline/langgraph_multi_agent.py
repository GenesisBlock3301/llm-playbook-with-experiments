from IPython.display import display
import ollama
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import diskcache as dc

cache = dc.Cache("./cache_dir")


# ---- Ollama wrapper ----
def query_ollama(prompt: str, model: str = "llama2"):
    try:
        response = ollama.chat(model, messages=[{
            "role": "user",
            "content": prompt,
        }])
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[QUERY ERROR] {e}"


# ---- Agents ----
class ResearchAgent:
    def __init__(self, chroma_collection):
        self.collection = chroma_collection

    def run(self, query):
        try:
            print("Querying OLLAMA....")
            if query in cache:
                print("Using cached query")
                return cache[query]

            results = self.collection.query(
                query_texts=[query],
                n_results=5,
                include=['documents']
            )
            docs = results["documents"][0]
            if docs:
                output = " ".join(docs)
            else:
                output = query_ollama(f"No docs found. Answer this directly: {query}")
            cache[query] = output
            return output
        except Exception as e:
            return f"[RESEARCH ERROR] {e}"


class SummarizerAgent:
    def run(self, text: str):
        print("Summarizing text....")
        if not text:
            return "No content to summarize."
        prompt = f"Summarize the following content clearly and concisely:\n\n{text}"
        return query_ollama(prompt)


class AnswerAgent:
    def run(self, summary: str):
        print("Final answer....")
        if not summary:
            return "No summary available to answer."
        prompt = f"Generate a clear and complete answer based on this summary:\n{summary}"
        return query_ollama(prompt)


# ---- State Definition ----
class MyState(TypedDict, total=False):
    query: str
    research: str
    summary: str
    final_answer: str


# ---- Build StateGraph ----
def build_state_graph(
        research_agent: ResearchAgent,
        summarizer_agent: SummarizerAgent,
        answer_agent: AnswerAgent) -> StateGraph[MyState]:
    graph = StateGraph(MyState)

    def research_node(state: MyState) -> MyState:
        q = state["query"]
        state['research'] = research_agent.run(q)
        return state

    def summary_node(state: MyState) -> MyState:
        state['summary'] = summarizer_agent.run(state.get('research', ''))
        return state

    def answer_node(state: MyState) -> MyState:
        state['final_answer'] = answer_agent.run(state.get('summary', ''))
        return state

    graph.add_node('research', research_node)
    graph.add_node('summary', summary_node)
    graph.add_node('answer', answer_node)

    graph.add_edge(START, 'research')
    graph.add_edge('research', 'summary')
    graph.add_edge('summary', 'answer')
    graph.add_edge('answer', END)

    return graph


# ---- MAIN ----
if __name__ == "__main__":
    print("🚀 Initializing LangGraph (StateGraph) pipeline...\n")

    # 1️⃣ Setup Chroma
    chroma_client = Client(Settings(persist_directory="./vector_db"))
    embedding_func = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text")
    collection = chroma_client.get_or_create_collection(
        name="docs",
        embedding_function=embedding_func
    )

    # 2️⃣ Seed data
    docs = [
        {"id": "1", "text": "Python is a programming language widely used in AI and web development."},
        {"id": "2", "text": "RAG means Retrieval-Augmented Generation which mixes search with LLMs."},
        {"id": "3", "text": "Ollama lets you run local LLMs like Llama3 and Mistral."},
        {"id": "4", "text": "LangGraph orchestrates multiple agents in a structured workflow."}
    ]
    for d in docs:
        try:
            collection.add(ids=[d["id"]], documents=[d["text"]])
        except:
            pass

    # 3️⃣ Initialize agents
    research_agent = ResearchAgent(collection)
    summarizer_agent = SummarizerAgent()
    answer_agent = AnswerAgent()

    # 4️⃣ Build & Compile graph
    graph = build_state_graph(research_agent, summarizer_agent, answer_agent).compile()

    display(graph)

    init_state: MyState = {"query": "What is LangGraph?"}
    final_state = graph.invoke({"query": "What is LangGraph?"})

    print("\n=== 🧩 FINAL ANSWER ===\n")
    print(final_state.get("final_answer"))
