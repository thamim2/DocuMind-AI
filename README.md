# 🤖 Offline RAG Chatbot (PDF Based)

This project is an offline document-based chatbot built using Streamlit, LangChain, ChromaDB, and Ollama.

## 🚀 Features
- Upload PDF documents
- Store document embeddings locally
- Ask questions based on uploaded content
- Fully offline (no OpenAI / internet required)
- Uses Ollama with Mistral model

## 🧠 Architecture
1. PDF is loaded and split into chunks
2. Chunks are converted to embeddings
3. Embeddings are stored in ChromaDB
4. User queries retrieve relevant chunks
5. Ollama generates answers using context

## 🛠 Tech Stack
- Streamlit (UI)
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Ollama (Mistral)

## ▶ Run Locally
```bash
ollama run mistral
streamlit run chat_app.py

