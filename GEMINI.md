# Project Overview

This project is a collection of Python scripts and notebooks for experimenting with and building applications using Large Language Models (LLMs). It includes a variety-level of implementations of chatbots and Retrieval-Augmented Generation (RAG) systems.

The main components are:

*   **`ecom-bot`**: A chatbot for an e-commerce platform that can answer FAQs and check order statuses.
*   **`rag_system`**: A sophisticated RAG system with multiple retriever implementations (basic, hybrid, and advanced) for question answering over a PDF document.
*   **`study_process`**: A collection of scripts for learning and experimenting with basic LLM concepts, including a simple command-line chatbot.
*   **`study_rag`**: Scripts and notebooks for understanding and implementing RAG, including a detailed document preprocessing pipeline.

# Building and Running

## E-commerce Chatbot

To run the e-commerce chatbot, you need to have Python and the required dependencies installed.

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the chatbot:**
    ```bash
    python ecom-bot/app.py
    ```

## RAG System

To run the RAG system, you need to have Python and the required dependencies installed. You also need to have a Qdrant instance running.

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the RAG system:**
    ```bash
    python rag_system/basic_rag.py
    ```

## Study Process Chatbot

To run the simple command-line chatbot, you need to have Python and the required dependencies installed.

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the chatbot:**
    ```bash
    python study_process/cli_bot.py
    ```

# Development Conventions

*   The project uses Python for all the scripts.
*   The `ecom-bot` and `study_process` chatbots use `langchain_ollama` to interact with LLMs.
*   The `rag_system` uses `langchain-ollama`, `langchain-qdrant`, and `qdrant-client` for the RAG implementation.
*   The project uses `.env` files for managing environment variables.
*   The `ecom-bot` logs conversations to JSONL files in the `ecom-bot/logs` directory.
*   The `study_process` chatbot logs conversations to `chat_session.log`.
