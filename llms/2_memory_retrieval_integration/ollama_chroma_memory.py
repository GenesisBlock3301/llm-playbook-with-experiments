import ollama
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ---- Step 1: Setup Chroma DB (for Long-Term Memory) ----
client = Client(Settings(persist_directory="./vector_db"))
embedding_func = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text")

# Create a collection for documents
collection = client.get_or_create_collection("docs", embedding_function=embedding_func)

# Add some documents to long-term memory
docs = [
    {"id": "1", "text": "Python is a programming language popular for AI and web development."},
    {"id": "2", "text": "RAG stands for Retrieval-Augmented Generation, combining search with LLMs."},
    {"id": "3", "text": "Ollama allows running LLMs locally like Llama3, Mistral, etc."}
]
for d in docs:
    collection.add(ids=[d["id"]], documents=[d["text"]])

# ---- Step 2: Short-term memory (last N messages) ----
short_term_memory = []


def add_to_memory(role, content, max_len=5):
    short_term_memory.append({"role": role, "content": content})
    if len(short_term_memory) > max_len:
        short_term_memory.pop(0)


# ---- Step 3: RAG Agent ----
def rag_agent(query):
    # Add a user query to memory
    add_to_memory("user", query)

    # Retrieve from long-term memory
    results = collection.query(query_texts=[query], n_results=2)
    retrieved_docs = " ".join(results["documents"][0])

    # Build prompt
    stm_text = "\n".join([f"{m['role']}: {m['content']}" for m in short_term_memory])
    prompt = f"""
    You are a helpful AI agent with memory.

    Short-term memory:
    {stm_text}

    Retrieved knowledge:
    {retrieved_docs}

    Task:
    Answer the user query based on memory + documents. If not confident, refine by checking again.

    User query: {query}
    """

    # Query Ollama locally
    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]

    # Store assistant response
    add_to_memory("assistant", answer)

    return answer


# ---- Step 4: Test Agent ----
print(rag_agent("What is RAG?"))
print("==============")
print(rag_agent("And how does Ollama fit in?"))
