# 🧠 LLM Playbook with Experiments

A practical playground for exploring **Large Language Models (LLMs)**, **LangGraph**, and **Deep Learning** fundamentals — from linear algebra all the way to multi-agent orchestration and long-running durable systems.  

This repository represents my continuous journey as a **Machine Learning Engineer** and **Software Engineer** experimenting with the internals of AI systems and production-ready agent architectures.

---

## 🚀 Project Overview

This project bridges **core ML theory** with **applied LLM engineering**.  
Each folder focuses on a distinct research or engineering concept:

| Folder | Focus Area | Key Topics |
|:-------|:------------|:------------|
| **`deep_learning/core_foundation/`** | Mathematical fundamentals | Linear Algebra, Matrix operations, Backprop basics |
| **`llms/1_core_foundation_single_agent/`** | Building foundational LLM workflows | Attention mechanism, finite-state thinking, GPU checks |
| **`llms/2_memory_retrieval_integration/`** | Memory-augmented systems | RAG (Retrieval Augmented Generation), Chroma vector store, Ollama local LLM integration |
| **`llms/3_multi-agent-pipeline/`** | Multi-agent orchestration | LangGraph + LangChain, cooperative agents, research + summarization + answer pipeline |
| **`llms/4_long_running_process/`** | Durable, reliable agents | Queue management, retries, persistence, HITL (Human-in-the-Loop) execution |
| **`llms/5_all_about_langgraph/`** | LangGraph deep dive | Graph nodes, conditional edges, checkpointing, message passing patterns |

---

## 🧩 Key Experiments & Learnings

### 1️⃣ Core Foundations
- Implemented **attention mechanism** from scratch to understand transformer internals.
- Built simple **finite-state machines** to simulate agent reasoning.
- Practiced **LangGraph basics** (nodes, edges, conditions) to design structured reasoning flows.

### 2️⃣ Retrieval-Augmented Memory
- Integrated **Ollama** (local LLMs like Llama-2 / Mistral) with **ChromaDB** for semantic retrieval.
- Implemented a **RAG pipeline** to enrich context dynamically during LLM responses.

### 3️⃣ Multi-Agent Collaboration
- Created **Researcher → Summarizer → Answerer** workflow.
- Introduced caching, embedding persistence, and modular agents.
- Showcased **graph orchestration** using LangGraph to coordinate autonomous agents.

### 4️⃣ Long-Running & Durable Systems
- Implemented **durable execution** with `diskcache` and `tenacity` for retries.
- Added **Human-in-the-Loop (HITL)** corrections to simulate real feedback and preference learning.
- Designed persistent task states for resumable, production-grade pipelines.

### 5️⃣ LangGraph Internals
- Studied graph anatomy: **nodes vs edges**, **conditional edges**, **state channels**, **checkpointing**, and **message passing**.
- Visualized graphs using Mermaid and Graphviz to understand data flow.

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|:----------|:------------------|
| **LLMs & Agents** | [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), [Ollama](https://ollama.com/) |
| **Retrieval** | [ChromaDB](https://www.trychroma.com/) |
| **Persistence / Retry** | `diskcache`, `tenacity` |
| **Visualization** | `IPython.display`, `Graphviz`, `Mermaid` |
| **Core ML** | Python, NumPy |
| **Experiment Management** | Jupyter Notebooks |

---

## 🌱 Why I’m Building This

Modern AI systems (like ChatGPT, Claude, or Gemini) are **not single models** —  
they are **networks of reasoning agents**, **memory layers**, and **durable feedback loops**.

This repo is my effort to:
1. Understand **how LLM agents can be orchestrated** like microservices.  
2. Explore **long-running, stateful AI systems** beyond single-prompt inference.  
3. Combine **theory (math & DL)** with **practical agent engineering**.  
4. Prepare for **production-level AI development** (queue management, retries, persistence, feedback learning).  

---

## 🔭 Future Scope

- Integrate **Celery + Redis** for distributed task orchestration.  
- Add **async workflows** and **pa**
