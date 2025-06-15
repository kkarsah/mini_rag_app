# 🧠 Mini RAG App

A simple command-line **Retrieval-Augmented Generation (RAG)** chatbot using:

- 🧩 **Claude 3 Haiku (Anthropic)**
- 🔍 **HuggingFace Embeddings**
- 🗂️ **Chroma Vector Store**
- 📄 Support for `.txt`, `.pdf`, and **web pages**

---

## 📦 Features

- Loads content from:
  - Plain text files (`.txt`)
  - PDF documents (`.pdf`)
  - Web pages (via URL)
- Splits and embeds documents using HuggingFace models
- Stores embeddings in a local vector store using Chroma
- Queries are answered using Claude via LangChain's `RetrievalQA`

---

## 🛠 Requirements

- Python 3.10 or higher
- `conda` or `venv` for virtual environments
- macOS/Linux (PDF support may need additional dependencies)

---

## 📂 Project Structure
mini_rag_app/
├── rag_app.py           # Main script
├── data/
│   ├── knowledge.txt    # Plain text input
│   └── *.pdf            # Any PDF files
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
└── README.md            # This file


+------------------+
|   User Query     |
| (Natural Language|
|   Question)      |
+--------+---------+
         |
         v
+--------------------------+
|   Embedding Model        |
|  (e.g. MiniLM, BERT)     |
|  Converts query to vector|
+------------+-------------+
             |
             v
+------------------------------+
|     Vector Store / DB        |
| (e.g. FAISS, Chroma, Pinecone)|
| Stores document embeddings   |
+--------------+---------------+
               |
               v
+-----------------------------+
|  Top-k Similarity Search    |
|  (Find top-N relevant docs) |
+--------------+--------------+
               |
               v
+------------------------------+
| Retrieved Document Chunks   |
| (Context to inform LLM)     |
+--------------+--------------+
               |
               v
+------------------------------+
|   Large Language Model       |
| (e.g. Claude, GPT-4, etc.)   |
| Combines Query + Context     |
| => Generates Answer          |
+--------------+--------------+
               |
               v
+------------------------------+
|         Final Answer         |
+------------------------------+