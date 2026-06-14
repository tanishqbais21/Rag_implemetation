from Data_ingestion import Vector_db
from Data_ingestion import load_data
from logging_config import AILogger
from Data_ingestion import Vector_db
from Retriver import retrive

logger = AILogger(name="MainLogger",log_dir="logs/MainLogger").logger
vectorstore=Vector_db.VectorStore()
ingestion = load_data.DataIngestionPipeline(vector_store=vectorstore)
load_pdf_path = "data"
#ingestion.run_pipeline(load_pdf_path)
logger.info(f"Data ingestion pipeline completed.")
retriver=retrive.RAGRtriever(vector_store=vectorstore,injestion_object=ingestion)
query="Tell me the time when harry reached hogwarts?"
response=retriver.get_answer(query)
print(response)


