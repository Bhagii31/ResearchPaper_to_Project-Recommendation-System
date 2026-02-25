# Paper-to-Project Recommendation System

A full-stack LLM application that converts research papers (PDF or public URL) into **actionable, beginner-friendly project plans**—including a concise paper summary, extracted technologies/methods, recommended projects, relevant public datasets, and curated YouTube learning resources.

**Live Demo:** https://research-paper-to-project-recommend-three.vercel.app  
**API Health:** https://YOUR-RENDER-BACKEND.onrender.com/health  

---

## Overview
Research papers are often dense and implementation details are scattered across assumptions, references, and missing datasets. This project bridges that gap by transforming paper content into a structured “build plan” that helps students and early engineers quickly move from reading to building.

---

## Key Features
- **Paper input:** Upload a PDF or paste a public paper link
- **LLM extraction:** Generates a 3–4 sentence summary and identifies key technologies/methods
- **Project recommendations:** Produces 2–3 beginner-friendly project ideas with milestones, metrics, and suggested stack
- **Dataset discovery:** Recommends relevant public datasets (Hugging Face + PapersWithCode)
- **YouTube resources:** Displays related tutorials with thumbnails and in-app video playback

---

## Tech Stack
- **Frontend:** Next.js (App Router), TypeScript
- **Backend:** FastAPI (Python)
- **LLM:** OpenAI (`gpt-4.1-mini`)
- **APIs / Sources:** YouTube Data API v3, Hugging Face Datasets, PapersWithCode
- **Deployment:** Vercel (frontend), Render (backend)
- **CI:** GitHub Actions

---

## High-Level Architecture
1. Ingest paper via **PDF upload** or **URL**
2. Extract text from the document/source page
3. Use LLM to generate:
   - short summary
   - technology/method extraction
   - project recommendations (milestones, metrics, stack)
4. Fetch supporting resources:
   - datasets (HF + PapersWithCode)
   - YouTube tutorials
5. Render results in a structured UI

---

## Local Setup (Docker)
### 1) Configure environment variables
Create a `.env` file (do not commit):
```bash
OPENAI_API_KEY=your_key
YOUTUBE_API_KEY=your_key
OPENAI_MODEL=gpt-4.1-mini
