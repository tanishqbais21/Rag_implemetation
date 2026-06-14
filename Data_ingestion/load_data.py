from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from logging_config import AILogger
from transformers import AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import os
import numpy as np
from pathlib import Path
class DataIngestionPipeline:
    def __init__(self):
        self.log = AILogger(name="DataIngestionLogger",log_dir="logs/DataIngestionLogger").logger
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model=None
        self.model_name="all-MiniLM-L6-v2"

    #load the pre-trained model using sentence-transformers
    def _load_model(self, model_name: str="all-MiniLM-L6-v2"):

        try:
            self.log.info(f"Loading pre-trained model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_embedding_dimension() 
            self.log.info(f"Successfully loaded model: {model_name} with embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            self.log.error(f"Error loading model {model_name}: {e}")
            raise e
        
    def generate_embeddings(self, text: str) -> np.ndarray:
        """
        Generate embeddings for the given text using the loaded model."""
        try:
            if self.model is None:
                raise ValueError("Model is not loaded. Please load the model before generating embeddings.")
            self.log.info(f"Generating embeddings for text of length {len(text)}")
            embeddings = self.model.encode(text)
            return embeddings
        except Exception as e:
            self.log.error(f"Error generating embeddings: {e}")
            raise e     
    
    #count tokens in a text using the tokenizer
    def _count_tokens(self, text: str) -> int:

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    def _load_pdf(self, file_path: str) -> list[Document]:
        """
        Load a PDF directory and return its content as a Document object.
        
        Args:
            file_path (str): The path to the PDF file.
        
        Returns:
            list[Document]: The loaded documents.
        """
        all_documents=[]
        try:
            pdf_dir=Path(file_path)
            self.log.info(f"Loading PDF files from directory: {file_path}")
            pdf_files=list(pdf_dir.glob("**/*.pdf"))
            self.log.info(f"Found {len(pdf_files)} PDF files in directory: {file_path}")
            for pdf_file in pdf_files:
                self.log.info(f"Loading PDF file: {pdf_file}")
                loader = PyMuPDFLoader(str(pdf_file))
                documents = loader.load()
                all_documents.extend(documents)
                return all_documents
        except Exception as e:
            self.log.error(f"Error loading PDF files: {e}")
    
    def _split_documents(self, documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
        """
        Split documents into smaller chunks using RecursiveCharacterTextSplitter.
        
        Args:
            documents (list[Document]): The list of documents to split.
            chunk_size (int): The size of each chunk in characters.
            chunk_overlap (int): The number of overlapping characters between chunks.
        
        Returns:
            list[Document]: The list of split documents.
        """
        try:
            self.log.info(f"Splitting {len(documents)} documents into chunks.")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
                length_function=self._count_tokens
            )
            split_docs = text_splitter.split_documents(documents)
            self.log.info(f"Completed splitting documents. Total chunks created: {len(split_docs)}")
            return split_docs
        except Exception as e:
            self.log.error(f"Error splitting documents: {e}")
            raise e
    
    def run_pipeline(self, file_path: str)->tuple[list[np.ndarray],list[Document]]:
        """
        Run the data ingestion pipeline to load PDF files,generate embeddings and splitting documents.
        Args:
            file_path (str): The path to the PDF file.
        
        Returns:
            list[np.ndarray]: The generated embeddings.
        """
        self.log.info("Starting data ingestion pipeline.")
        try:
            documents = self._load_pdf(file_path)
            self.log.info(f"Data ingestion pipeline completed. Loaded {len(documents)} documents.")
            split_documents = self._split_documents(documents)
            self._load_model(self.model_name)
            numpy_embeddings = [self.generate_embeddings(doc.page_content) for doc in split_documents]
            return numpy_embeddings,split_documents
        except Exception as e:
            self.log.error(f"Error running data ingestion pipeline: {e}")