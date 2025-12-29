# Autonomous RAG Agent (Web-Based) 🤖

<p align="center">
  <img src="https://img.shields.io/badge/Tech-LangChain-green" alt="LangChain">
  <img src="https://img.shields.io/badge/DB-FAISS-blue" alt="FAISS">
  <img src="https://img.shields.io/badge/Model-GPT--4o--mini-black" alt="OpenAI">
</p>

This module implements a **Retrieval-Augmented Generation (RAG)** pipeline. It creates an autonomous agent capable of scraping websites, ingesting their content, and answering user questions based *only* on that specific data. The goal is to understand (and show) how a simple RAG system is wired end-to-end.

---

## 🔍 Overview

The `web_rag_agent.py` script orchestrates a complete RAG lifecycle. Instead of relying on pre-trained knowledge, the agent builds a temporary, local vector database from a target URL (defaulting to the University of York Wikipedia page) and uses semantic search to find answers.

### Key Features
* **Live Ingestion:** Uses `BSHTMLLoader` to scrape and parse HTML content in real-time.
* **Smart Chunking:** Splits raw text into manageable pieces (300 chars) using `RecursiveCharacterTextSplitter` to preserve context.
* **Vector Memory:** Indexes embeddings in a local **FAISS** (Facebook AI Similarity Search) database for millisecond-speed retrieval.
* **Hallucination Control:** The prompt is engineered to strictly say *"I don't have enough information"* if the answer isn't found in the retrieved chunks.

---

## 🛠️ How It Works

The `WebRAGAgent` class follows this 5-step pipeline:

1.  **Fetch:** Downloads HTML from the target URL (`requests`).
2.  **Process:** Cleans HTML tags and splits text into chunks with a 50-character overlap.
3.  **Embed:** Converts text chunks into dense vectors using `OpenAIEmbeddings`.
4.  **Retrieve:** When you ask a question, the system finds the top 3 most similar chunks (Top-K Retrieval).
5.  **Generate:** `GPT-4o-mini` synthesizes the final answer using *only* the retrieved context.

---

## 📊 Interactive Session

The agent maintains a conversation loop with memory. It automatically logs your session to `outputs/.../chat_history.txt`.

**Example Interaction:**
```text
WELCOME TO WEB-RAG AGENT (University of York Edition)
Auto-loading University of York URL...
Created 142 text chunks.
RAG Pipeline is ready.

You: When was the university founded?
Agent: The University of York was founded in 1963.

You: What is the main campus called?
Agent: The main campus is known as Campus West.

You: Who won the World Cup in 2022?
Agent: I don't have enough information.
(Correct! This info is not in the University of York Wikipedia page)
```

---

## 🚀 Usage

This module expects `OPENAI_API_KEY` in your environment.

### Option A) `.env` (recommended)
Create `RAG/.env` (local only):

```env
OPENAI_API_KEY=your_key_here
```

### Option B) export the key
```bash
export OPENAI_API_KEY="your_key_here"
```

### Default behavior
The script auto-loads a target URL inside the interactive session:

```python
target_url = "https://en.wikipedia.org/wiki/University_of_York"
```

You can change `target_url` to any page you want to test.

---

## 📦 Outputs

Each run creates:

- `outputs/YYYY-MM-DD_HH-MM-SS_web_rag/chat_history.txt`

The log includes:
- session start time
- source URL
- all user questions and agent answers

---

## 🔧 Technical Details

- **LLM:** `ChatOpenAI` (default: `gpt-4o-mini`)
- **Embeddings:** `OpenAIEmbeddings`
- **Vector store:** FAISS (in-memory)
- **Retriever:** `vectorstore.as_retriever()`
- **Chain:** `RetrievalQA` (`chain_type="stuff"`) with a custom prompt
- **Memory:** `ConversationBufferMemory(return_messages=True)`
- **Chunking:** `RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)`

---
