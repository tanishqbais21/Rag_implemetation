# Local RAG (Retrieval-Augmented Generation) Pipeline

A robust, local-first Retrieval-Augmented Generation (RAG) pipeline built from scratch using Python, LangChain, ChromaDB, and HuggingFace Sentence Transformers. This project ingests various document formats, stores their embeddings in a local vector database, and retrieves highly relevant context to construct hallucination-resistant prompts for Large Language Models (LLMs).

## Features
- **Multi-Format Document Ingestion:** Supports automated loading of `.pdf`, `.txt`, and `.docx` files using specialized LangChain document loaders.
- **Intelligent Text Splitting:** Uses `RecursiveCharacterTextSplitter` with token-aware chunking to ensure semantic integrity when breaking down large documents.
- **Local Embedding Generation:** Generates document embeddings locally using the lightweight and fast `sentence-transformers/all-MiniLM-L6-v2` model, meaning no API costs or rate limits for embeddings.
- **Persistent Local Vector Store:** Integrates ChromaDB for storing embeddings persistently on disk (`data/vector_store`). It includes duplicate detection to prevent redundant data insertion.
- **Context-Aware Retrieval:** A custom Retriever module that performs semantic search, filters results by confidence threshold, and retrieves top-k relevant document chunks.
- **Structured Prompt Generation:** Formats retrieved contexts into a strictly structured, XML-style prompt that guides the LLM to rely strictly on the provided documents and avoid hallucinations.
- **Comprehensive Logging:** Centralized logging via a custom `AILogger` with rotating file handlers and console output to keep track of the pipeline's execution and performance.

## Project Structure
```text
Project_1/
├── main.py                     # Application entry point orchestrating ingestion and retrieval
├── test_main.py                # Unit tests for the main application
├── Data_ingestion/             
│   ├── load_data.py            # Document loading, text splitting, and embedding generation
│   └── Vector_db.py            # ChromaDB integration, document indexing, and similarity search
├── Retriver/                   
│   └── retrive.py              # Query embedding, semantic search, and prompt formatting
├── data/                       # Directory to place source documents (.pdf, .txt, .docx)
│   └── vector_store/           # Persistent ChromaDB storage (auto-generated)
├── logs/                       # Rotating logs output directory (auto-generated)
├── logging_config.py           # Logging utility configurations
└── requirements.txt            # Python dependencies
```

## Setup & Installation

1. **Clone the repository and set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Provide Data:**
   Create a folder named `data` in the root directory (if it doesn't exist) and place your `.pdf`, `.txt`, or `.docx` files inside.

## Usage

### 1. Data Ingestion & Querying Pipeline
You can run the entire pipeline directly from `main.py`. This script will initialize the vector store, configure the ingestion pipeline, and execute a query.

You can use command-line arguments to specify the data directory and your query:

```bash
# Run with default arguments
python main.py

# Run with custom arguments
python main.py --pdf_path "data" --query "What is the name of the three Main Character?"
```

**Note on Ingestion:** 
In `main.py`, the line `ingestion.run_pipeline(load_pdf_path)` is currently commented out to avoid re-ingesting documents if they are already stored. Uncomment it when you add new documents to the `data/` folder. The `VectorStore` automatically ignores exact duplicate entries.

## Testing

The project includes production-ready unit tests to verify the core application logic using mocked dependencies, ensuring it tests the flow without needing an actual database or data directory.

Run the tests using `pytest`:
```bash
pytest test_main.py
```

## Core Components Breakdown

- **`DataIngestionPipeline` (load_data.py)**: Scans a directory for supported files, utilizes appropriate loaders (`PyMuPDFLoader`, `TextLoader`, `Docx2txtLoader`), chunks the extracted text, and generates 384-dimensional embeddings.
- **`VectorStore` (Vector_db.py)**: Manages the connection to the local ChromaDB. Includes `_add_data()` which gracefully handles duplicates by comparing document IDs, and `get_scores()` which returns similarity scores and metadata for user queries.
- **`RAGRtriever` (retrive.py)**: Takes a raw user query, fetches its vector representation, retrieves matching snippets from the database, and creates an optimized prompt string for an LLM that explicitly demands citations and prevents external knowledge usage.
- **`AILogger` (logging_config.py)**: Provides clean, structured logging out-of-the-box. Logs are saved inside the `logs/` directory and capped at 1MB per file, rotating up to 3 backup files.

## Future Enhancements
- Integration of a local or cloud-based LLM instance (e.g., Llama.cpp, OpenAI, Gemini) to natively process the generated prompts.
- UI layer (e.g., Streamlit, Gradio) to upload files and ask questions interactively.

# Rag_implemetation
A robust, local-first Retrieval-Augmented Generation (RAG) pipeline built from scratch using Python, LangChain, ChromaDB, and HuggingFace Sentence Transformers.
