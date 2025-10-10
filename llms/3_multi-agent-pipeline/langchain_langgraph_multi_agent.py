from IPython.display import display
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import HumanMessage

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import diskcache as dc

cache = dc.Cache("./cache_dir")


# ---- Initialize LangChain components ----
llm = ChatOllama(model="llama2", temperature=0.3)
embedding = OllamaEmbeddings(model="nomic-embed-text")

# ---- ChromaDB as a LangChain VectorStore ----
vectorstore = Chroma(
    collection_name="docs",
    embedding_function=embedding,
    persist_directory="./vector_db"
)

# ---- Seed the VectorStore ----
docs = [
    "Python is a programming language widely used in AI and web development.",
    "RAG means Retrieval-Augmented Generation which mixes search with LLMs.",
    "Ollama lets you run local LLMs like Llama3 and Mistral.",
    "LangGraph orchestrates multiple agents in a structured workflow.",
    "Shoe price is 320tk",
    "The book price is 100tk"
]

for text in docs:
    vectorstore.add_texts([text])


# ---- Agents ----
class ResearchAgent:
    def __init__(self, vs):
        self.vs = vs

    def run(self, query):
        print("Querying via LangChain retriever...")
        if query in cache:
            print("Using cached query")
            return cache[query]

        retriever = self.vs.as_retriever(search_kwargs={"k": 3})
        results = retriever.invoke(query)

        if results:
            output = " ".join([doc.page_content for doc in results])
        else:
            # fallback to LLM directly
            response = llm.invoke([HumanMessage(content=f"Answer this directly: {query}")])
            output = response.content

        cache[query] = output
        return output


class SummarizerAgent:
    def run(self, text: str):
        print("Summarizing text....")
        if not text:
            return "No content to summarize."
        prompt = f"Summarize this clearly and concisely:\n\n{text}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content


class AnswerAgent:
    def run(self, summary: str, query: str):
        print("Final answer....")
        if not summary:
            return "No summary available."
        prompt = f"Based on this summary, answer the question '{query}' directly and concisely:\n{summary}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content


# ---- State Definition ----
class MyState(TypedDict, total=False):
    query: str
    research: str
    summary: str
    final_answer: str


# ---- Build LangGraph ----
def build_state_graph(
        research_agent: ResearchAgent,
        summarizer_agent: SummarizerAgent,
        answer_agent: AnswerAgent) -> StateGraph[MyState]:
    graph = StateGraph(MyState)

    def research_node(state: MyState) -> MyState:
        state['research'] = research_agent.run(state['query'])
        return state

    def summary_node(state: MyState) -> MyState:
        state['summary'] = summarizer_agent.run(state.get('research', ''))
        return state

    def answer_node(state: MyState) -> MyState:
        state['final_answer'] = answer_agent.run(state.get('summary', ''), state['query'])
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
    print("🚀 Initializing LangGraph + LangChain hybrid pipeline...\n")

    # Initialize agents
    research_agent = ResearchAgent(vectorstore)
    summarizer_agent = SummarizerAgent()
    answer_agent = AnswerAgent()

    # Build and compile graph
    graph = build_state_graph(research_agent, summarizer_agent, answer_agent).compile()
    display(graph)

    init_state = {"query": "What is the price of book? "}
    final_state = graph.invoke(init_state)

    print("\n=== 🧩 FINAL ANSWER ===\n")
    print(final_state.get("final_answer"))
