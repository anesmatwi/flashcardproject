# AI Pathology Flashcard Generator & Tutor

An end-to-end machine learning pipeline that processes medical textbooks into high-yield pathology flashcards, fine-tunes a local LLM for medical education, and utilizes an Adaptive Retrieval-Augmented Generation (RAG) system to provide clinical explanations.

## Overview
This project was developed to automate the creation of board-style pathology flashcards. It handles raw textbook ingestion, dataset generation, model fine-tuning, and evaluation. The final inference system uses a dual-model approach: a fine-tuned primary LLM for answering, and a lightweight CPU-based API for query rewriting and verification.

## Architecture & Features
* **Document Processing:** Automated extraction, OCR (PyTesseract), and chunking of PDFs and EPUBs into manageable data structures.
* **Dataset Generation:** Local generation of flashcards using BioMistral, with secondary validation and quality control via the Gemini 1.5 Pro API.
* **Model Fine-Tuning (LoRA):** Parameter-Efficient Fine-Tuning of `BioMistral-7B-SLERP` using 4-bit quantization to generate accurate, clinical explanations and handle negative samples.
* **Adaptive RAG Pipeline:** * Utilizes a local FAISS vector store with `PubMedBERT` embeddings.
    * Integrates external knowledge retrieval via PubMed (Entrez) and targeted clinical web searches (SerpAPI).
    * Implements confidence and repetition thresholds to dynamically trigger retrieval only when necessary.
* **Microservice Extractor (FastAPI):** A lightweight `Phi-3` server deployed on CPU to handle query rewriting, expansion, broadening, and context verification.
* **Evaluation:** Benchmarked using BLEU, ROUGE-L, BERTScore, and Flesch-Kincaid readability metrics.

## Tech Stack
* **Models:** BioMistral-7B, Phi-3-mini, PubMedBERT, Gemini 1.5 Pro
* **Libraries:** PyTorch, Hugging Face Transformers, PEFT, FAISS, LangChain, SentenceTransformers
* **Frameworks:** FastAPI, Jupyter

## Setup and Installation
[Provide instructions on creating a virtual environment, installing requirements.txt, and setting up the `.env` file for API keys.]

## Hardware Notes
This pipeline was designed to be orchestrated locally, taking advantage of a multi-GPU pool for parallel inference during dataset generation and model evaluation.
