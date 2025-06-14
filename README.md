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
