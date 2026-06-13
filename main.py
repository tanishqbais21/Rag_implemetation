from Data_ingestion import load_data
from logging_config import AILogger
logger = AILogger(name="MainLogger")
load_pdf_path = "data"
ingestion = load_data.DataIngestionPipeline()
embeddings = ingestion.run_pipeline(load_pdf_path)
logger.info(f"Generated {len(embeddings)} embeddings from PDF directory.")
