# RAG Pipeline from Scratch

A simple Retrieval-Augmented Generation (RAG) pipeline built with Python. This project demonstrates how to load a PDF document, split it into chunks, convert the chunks into embeddings, store them in a vector database, retrieve relevant information based on a user query, and use an LLM to generate a grounded answer.



# Project Overview

This project implements the following RAG workflow:

```text

PDF Document
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Embedding Generation
     ↓
FAISS Vector Store
     ↓
User Query
     ↓
Query Embedding
     ↓
Similarity Search
     ↓
Relevant Chunks
     ↓
    LLM
     ↓
Grounded Answer

```


# Features

1. Load text from a PDF document
2. Split the document into smaller chunks
3. Generate vector embeddings using Sentence Transformers
4. Store embeddings using FAISS
5. Perform similarity-based retrieval
6. Send retrieved context to an LLM
7. Generate answers grounded in the uploaded document
8. Prevent the LLM from answering from unrelated information when the answer isn't found in the document



# Technologies Used

```text

 Technology                         Purpose                  
 
 Python                             Programming language     
 PyPDF                              Extract text from PDF    
 Sentence Transformers              Generate text embeddings 
 FAISS                              Vector similarity search 
 GEMINI API                         Generate final answers   
 python-dotenv                      Load API key from `.env` 

```


# Project Structure

```text

rag-project/
│
├── document.pdf
├── rag.py
├── requirements.txt
├── .env
└── README.md

```


# Learning Objectives

This project helps demonstrate the fundamental concepts of:

1. Retrieval-Augmented Generation
2. Natural Language Processing
3. Text embeddings
4. Vector databases
5. Semantic search
6. Similarity search
7. Prompt engineering
8. LLM-based question answering