import argparse
from Data_ingestion import Vector_db
from Data_ingestion import load_data
from logging_config import AILogger
from Retriver import retrive

# Initialize the logger for the main application
logger = AILogger(name="MainLogger", log_dir="logs/MainLogger").logger

def run_ingestion(load_pdf_path: str, vector_store: Vector_db.VectorStore) -> load_data.DataIngestionPipeline:
    """
    Initializes the data ingestion pipeline and runs it on the given path.
    
    Args:
        load_pdf_path (str): The directory path where PDF data is stored.
        vector_store (VectorStore): The initialized vector database.
        
    Returns:
        DataIngestionPipeline: The initialized ingestion pipeline object.
    """
    try:
        # Initialize pipeline with the vector store
        ingestion = load_data.DataIngestionPipeline(vector_store=vector_store)
        
        # Run the pipeline (can be uncommented or conditionally executed based on arguments)
        # ingestion.run_pipeline(load_pdf_path)
        
        logger.info("Data ingestion pipeline configured/completed successfully.")
        return ingestion
    except Exception as e:
        logger.error(f"Failed during data ingestion setup: {e}")
        raise

def run_retrieval(query: str, vector_store: Vector_db.VectorStore, ingestion_object: load_data.DataIngestionPipeline) -> str:
    """
    Initializes the RAG retriever and fetches the answer for the given query.
    
    Args:
        query (str): The user's question to ask the RAG system.
        vector_store (VectorStore): The initialized vector database.
        ingestion_object (DataIngestionPipeline): The pipeline used for ingestion.
        
    Returns:
        str: The retrieved response/answer.
    """
    try:
        # Note: maintaining the original typo "injestion_object" as it's defined in retrive.py
        retriever = retrive.RAGRtriever(vector_store=vector_store, injestion_object=ingestion_object)
        
        # Retrieve the answer
        response = retriever.get_answer(query)
        return response
    except Exception as e:
        logger.error(f"Failed during retrieval: {e}")
        raise

def main(pdf_path: str, query: str):
    """
    Main application entry point. Coordinates ingestion and retrieval.
    """
    logger.info("Starting Application...")
    
    # Step 1: Initialize the vector store
    vector_store = Vector_db.VectorStore()
    
    # Step 2: Configure/Run Data Ingestion
    ingestion = run_ingestion(pdf_path, vector_store)
    
    # Step 3: Run Retrieval to get the answer for the query
    response = run_retrieval(query, vector_store, ingestion)
    
    print(f"\n--- Results ---")
    print(f"Query: {query}")
    print(f"Response:\n{response}")
    print(f"---------------\n")
    
    return response

if __name__ == "__main__":
    # Use argparse to allow dynamic inputs instead of hardcoding values
    parser = argparse.ArgumentParser(description="Run RAG Pipeline")
    parser.add_argument(
        "--pdf_path", 
        type=str, 
        default="data", 
        help="Path to the PDF data directory"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        default="What is the name of the three Main Character?", 
        help="Query to ask the RAG model"
    )
    
    # Parse arguments and run main
    args = parser.parse_args()
    main(args.pdf_path, args.query)
