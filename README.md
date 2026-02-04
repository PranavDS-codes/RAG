# Agentic RAG: "The Brain"

This project implements an advanced **Agentic Retrieval-Augmented Generation (RAG)** system designed to act as a self-correcting, multi-step reasoning engine. Unlike naive RAG systems, "The Brain" employs a graph-based workflow (built with **LangGraph**) to audit its own retrieval, actively scout for missing information (Web/Wikipedia), verify evidence quality, and iteratively refine its search strategy before synthesizing an answer.

## 🧠 Core Architecture

The system is built on a **StateGraph** that orchestrates specialized functional nodes (agents). The workflow allows for dynamic routing and self-correction:

### The Workflow (Graph)
1.  **Retrieve Node**: Queries internal knowledge using a hybrid **Omni-Retriever** (Vector + BM25 + Knowledge Graph).
2.  **Audit Node**: An LLM "Auditor" evaluates if the internal evidence is sufficient to answer the query.
    *   ✅ **Sufficient**: Proceeds to synthesis.
    *   ❌ **Insufficient**: Triggers a "Gap Analysis" and routes to external scouts.
3.  **Scout Nodes**:
    *   **Wiki Scout**: Searches Wikipedia for encyclopedic knowledge.
    *   **Web Scout (Deep Web)**: Uses **Tavily API** for live web search and extraction.
4.  **Verify Node**: An LLM "Verifier" inspects the retrieved external evidence for relevance and quality.
5.  **Refine Node**: If verification fails, this node re-strategizes, generating new search queries/keywords to try again (Looping mechanism).
6.  **Synthesize Node**: Consolidates all verified evidence to generate the final answer.

### The Omni-Retriever
The retrieval engine (`rag_engine.py`) is a sophisticated hybrid system:
*   **Vector Search**: Uses **FAISS** with **HyDE** (Hypothetical Document Embeddings) to find conceptually related chunks.
*   **Keyword Search**: Uses **BM25** for precise lexical matching.
*   **Relationship Search**: Queries a **NetworkX** knowledge graph to find entity relationships.
*   **Re-Ranking**: Uses a Cross-Encoder to score and filter the best results from all sources.

## ✨ Key Features

*   **Self-Correcting**: The system doesn't just hallucinate if it doesn't know. It detects gaps (`Audit`) and actively finds answers.
*   **Query Decomposition**: The `QueryOptimizer` breaks complex user queries into atomic sub-tasks with specific keywords and graph entities.
*   **Traffic Control**: A "Router" decides whether to use internal memory, Wikipedia, or the Web based on confidence scores.
*   **Knowledge Curation**: The system actively "learns" by curating new findings from the web into structured formats (saved to `pending_knowledge.json`) for future ingestion.
*   **Evaluation Framework**: Includes a robust `rag_evaluation.py` module with an **LLM Judge** to measure Faithfulness, Relevance, and Context Utility.

## 📂 Project Structure

```text
├── 01-10_*.ipynb        # Jupyter Notebooks for data processing, graph building, and demos
├── config.py            # Global configuration (API Keys, Model Selection)
├── rag_engine.py        # Core retrieval logic (OmniRetriever, GraphSearcher)
├── web_engine.py        # External search tools (Tavily, Wikipedia, Curator)
├── nodes.py             # Agent node definitions (Retrieve, Audit, Verify, etc.)
├── graph.py             # LangGraph workflow definition
├── state.py             # Shared state definition (BrainState)
├── prompts.py           # System prompts for all agents
├── rag_evaluation.py    # Evaluation metrics and LLM Judge
└── data/                # Dataset storage (Raw and Processed)
```

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   A **Groq** API Key (for LLM inference)
*   A **Tavily** API Key (for web search)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd RAG
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup**:
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_groq_key_here
    TAVILY_API_KEY=your_tavily_key_here
    ```

### Usage

The project is primarily notebook-driven for development and demonstration.

1.  **Data Pipeline (Optional)**:
    *   Run `01_data_extract.ipynb` to download SQuAD/Wiki data.
    *   Run `02_embed_vector.ipynb` to build the Vector Index.
    *   Run `05_graph_builder.ipynb` to build the Knowledge Graph.

2.  **Running the Brain**:
    *   Open `09_brain.ipynb`.
    *   This notebook initializes the `OmniRetriever` and compiles the `LangGraph`.
    *   Use the `ask_brain("Your Question")` function to interact with the system.

    ```python
    from graph import build_graph, ask_brain
    
    app = build_graph()
    ask_brain("What acts did the Beatles perform?", app)
    ```

## 📊 Evaluation

To run benchmarks:
1.  Configure your test set in `rag_evaluation.py`.
2.  Run the evaluation script or use the experimentation notebooks (`03_naive_rag.ipynb`, `04_adv_rag.ipynb`) to see comparative results between Naive RAG and this Agentic system.

## 🛠 Tech Stack

*   **Orchestration**: LangChain, LangGraph
*   **LLMs**: Llama-3 (via Groq)
*   **Vector DB**: FAISS
*   **Search**: Tavily, Wikipedia-API
*   **Data Science**: Pandas, NumPy, NetworkX
