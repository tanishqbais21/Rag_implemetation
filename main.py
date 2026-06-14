from Data_ingestion import load_data
from logging_config import AILogger
from Data_ingestion import Vector_db
logger = AILogger(name="MainLogger",log_dir="logs/MainLogger").logger
load_pdf_path = "data"
ingestion = load_data.DataIngestionPipeline()
embeddings,chunks= ingestion.run_pipeline(load_pdf_path)
logger.info(f"Generated {len(embeddings)} embeddings from PDF directory.")
logger.info(f"Generated {len(chunks)} from from PDF directory.")
vectorstore=Vector_db.VectorStore()
vectorstore._add_data(embeddings=embeddings,documents=chunks)
