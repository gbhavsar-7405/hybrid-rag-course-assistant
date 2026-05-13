# Hybrid RAG Course Recommendation Assistant

A conversational AI-powered course recommendation system built using Hybrid Retrieval-Augmented Generation (RAG), combining semantic vector search with keyword-based retrieval for more accurate and context-aware recommendations.

The system supports conversational memory, query reformulation, metadata-aware reranking, hallucination reduction, and grounded URL-based recommendations across multiple course platforms.

---

## Features

- Hybrid Retrieval (BM25 + Vector Search)
- Conversational RAG Pipeline
- Query Reformulation & Typo Correction
- Metadata-Aware Reranking
- MMR-Based Retrieval Optimization
- URL Grounding
- Hallucination Reduction
- Dynamic Retrieval Depth
- Platform-Specific Filtering
- Conversational Memory Support
- Structured AI Responses

---

## Tech Stack

- Python
- LangChain
- ChromaDB
- NVIDIA AI Endpoints
- BM25 Retrieval
- EnsembleRetriever
- Pandas

---

## Architecture Overview

```text
User Query
   ↓
Query Reformulation
   ↓
Hybrid Retrieval
(BM25 + Vector Search)
   ↓
MMR Retrieval Optimization
   ↓
Metadata-Aware Reranking
   ↓
Context Construction
   ↓
LLM Response Generation

```

## Dataset Sources

The system currently uses multi-platform course datasets from:

- Coursera
- Udacity
- Udemy

Each course entry includes metadata such as:
- Course Name
- Platform
- Difficulty
- Rating
- URL
- Description

---

## Installation

```bash
git clone https://github.com/your-username/hybrid-rag-course-assistant.git

cd hybrid-rag-course-assistant

pip install -r requirements.txt
```

Create a `.env` file:

```env
NVIDIA_API_KEY=your_api_key
```

---

## Running the Project

### 1. Build the Vector Database

```bash
python 1_ingestion_pipeline.py
```

### 2. Run the Conversational RAG Assistant

```bash
python 4_history_aware_generation.py
```

---

## Example Queries

```text
Beginner AI courses on Udemy

Best machine learning courses

Only beginner courses from Coursera

React API development courses with URLs
```

---

## Key Improvements Implemented

- Hybrid search using BM25 + vector retrieval
- MMR-based retrieval diversification
- Metadata-aware heuristic reranking
- Conversational query reformulation
- Hallucination prevention guardrails
- URL grounding improvements
- Duplicate result filtering
- Conversational continuation handling

---

## Future Improvements

- Graph RAG integration
- Neo4j knowledge graph support
- Pinecone vector database migration
- Web-based UI
- Streaming responses
- Agentic retrieval pipelines

---

## Author

Geetanshu Bhavsar

---
