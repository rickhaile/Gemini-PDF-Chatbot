# 🤖 AI-Powered Document Assistant (RAG Pipeline)

### 🚀 Project Overview
This project is a **Retrieval-Augmented Generation (RAG)** application that enables users to chat with multiple PDF documents in real-time. It uses a **Hybrid AI Architecture** to balance privacy, cost, and speed.

Unlike standard wrappers, this application decouples the **Embedding Layer** (running locally on CPU) from the **Inference Layer** (running on Google Gemini Cloud), ensuring zero cost for processing large documents while maintaining high-intelligence responses.

### ⚙️ Technical Architecture
*   **Frontend:** Streamlit (Interactive Web UI).
*   **Vector Database:** FAISS (Facebook AI Similarity Search) for local, high-speed semantic retrieval.
*   **Embedding Model:** `all-MiniLM-L6-v2` (HuggingFace) - Runs locally to avoid API rate limits.
*   **LLM:** Google Gemini 2.5 Flash - For high-speed, context-aware reasoning.

### 🛠️ Tech Stack
*   **Python 3.10+**
*   **LangChain** (Orchestration)
*   **Google Generative AI** (LLM)
*   **PyPDF2** (Document Parsing)
*   **FAISS-CPU** (Vector Store)

### ⚡ Key Features
*   ✅ **Hybrid RAG:** Local embeddings + Cloud generation prevents `429 Quota Exceeded` errors.
*   ✅ **Multi-File Support:** Upload and process multiple PDFs simultaneously.
*   ✅ **Context Preservation:** The AI remembers the context of the document to answer specific questions accurately.
*   ✅ **Cost Efficient:** 100% Free tier compatible architecture.

### 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rickhaile/Gemini-PDF-Chatbot.git
   cd Gemini-PDF-Chatbot
