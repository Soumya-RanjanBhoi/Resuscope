# Resuscope – AI Resume Analyzer

Resuscope is an AI-powered resume analysis backend built with **FastAPI**, designed to evaluate resumes against job descriptions using **semantic similarity**, **ATS-style scoring**, and **LLM-driven feedback**.  
It provides actionable insights on skills, structure, content quality, and tone to help candidates improve their resumes for specific roles.

---

## Features

- 📄 **PDF & DOCX Resume Parsing**
- 🧠 **Semantic Matching** using Sentence Transformers
- 🎯 **ATS Compatibility Scoring**
- 🛠️ **Skill Gap Analysis (Technical & Soft Skills)**
- ✍️ **Resume Content Quality Evaluation**
- 🧱 **Resume Structure Analysis**
- 🎭 **Tone & Style Feedback**
- 🚀 **Actionable Optimization Suggestions**
- ⚡ **FastAPI**

---

## Tech Stack

- **Backend Framework**: FastAPI  
- **Embedding Model**: Sentence Transformers  
- **LLM**: Google Gemini (`gemini-2.5-flash`) via LangChain  
- **Document Parsing**: pdfplumber, python-docx  
- **Vector Similarity**: Cosine Similarity  
- **Deployment**: Uvicorn  

---
---

## Repository layout (important files)
- app.py — FastAPI application and endpoints
- models/
  - extractor.py — PDF/DOCX parsing + LLM-based skill extraction
  - matcher.py — SemanticModel singleton using SentenceTransformer
  - rewriter.py — LLM chains that produce scores, feedback and suggestions
  - try.py — local experimentation / demo harness
- requirements.txt — pinned Python dependencies
- Dockerfile, .dockerignore

---


## Repository layout (important files)
- app.py — FastAPI application and endpoints
- models/
  - extractor.py — PDF/DOCX parsing + LLM-based skill extraction
  - matcher.py — SemanticModel singleton using SentenceTransformer
  - rewriter.py — LLM chains that produce scores, feedback and suggestions
  - try.py — local experimentation / demo harness
- requirements.txt — pinned Python dependencies
- Dockerfile, .dockerignore

---

## Quickstart (local)

Prerequisites:
- Python 3.11
- Git
- (Optional) CUDA-enabled GPU if you want to run sentence-transformers on GPU

1. Clone
   git clone https://github.com/Soumya-RanjanBhoi/Resuscope.git
   cd Resuscope

2. Create and activate a virtualenv (recommended)
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows (PowerShell)

3. Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt

4. Environment variables
   - MODEL_NAME (required): the sentence-transformers model name to use for semantic scoring.
     Example: export MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
   - Google GenAI credentials: Resuscope uses langchain_google_genai.ChatGoogleGenerativeAI.
     Provide Google credentials via one of:
     - GOOGLE_API_KEY environment variable (if supported in your Google GenAI setup), or
     - GOOGLE_APPLICATION_CREDENTIALS pointing to a service account JSON key, or
     - configure your environment as required by google-genai SDK.
   - Create a .env file in project root or export vars in your shell.

   Example (.env)
   MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   # GOOGLE_API_KEY=...
   # GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

5. Run
   python app.py
   - The server listens on port 8000 (app.py default). Uvicorn starts from the script.

6. API docs
   - Open http://localhost:8000/docs for Swagger UI
   - Open http://localhost:8000/redoc for ReDoc

---

## Docker

Build:
  docker build -t resuscope:latest .

Run (example):
  docker run -e MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2" \
             -e GOOGLE_API_KEY="YOUR_KEY_IF_USED" \
             -p 8080:8080 resuscope:latest

Notes:
- The Dockerfile sets PORT env to 8080 and exposes 8080. The app.py default runs on 8000, but the Dockerfile runs the Python script as-is — map ports appropriately when running.
- Ensure MODEL_NAME and Google credentials are provided at runtime. Large sentence-transformer models will be downloaded during the first run (requires internet and sufficient disk space).

---



