from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader,DirectoryLoader,TextLoader,Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from logging_config import AILogger
from Data_ingestion import Vector_db
from transformers import AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
import os
import numpy as np
from pathlib import Path
class DataIngestionPipeline:
    def __init__(self,vector_store:Vector_db.VectorStore):
        self.log = AILogger(name="DataIngestionLogger",log_dir="logs/DataIngestionLogger").logger
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model=None
        self.model_name="all-MiniLM-L6-v2"
        self.vector_store=vector_store
        self._load_model()

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
            embeddings = self.model.encode(text)
            return embeddings
        except Exception as e:
            self.log.error(f"Error generating embeddings: {e}")
            raise e     
    
    #count tokens in a text using the tokenizer
    def _count_tokens(self, text: str) -> int:

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    def _load_documents(self, directory_path: str) -> list[Document]:
        """
        Load multiple document types (PDF, TXT, DOCX) from a directory.
        
        Args:
            directory_path (str): The path to the directory containing documents.
        
        Returns:
            list[Document]: The loaded documents.
        """
        all_documents = []
        try:
            doc_dir = Path(directory_path)
            self.log.info(f"Scanning directory: {directory_path} for supported documents.")
            
            # 1. The Loader Mapping (The "Factory")
            loaders_mapping = {
                ".pdf": PyMuPDFLoader,
                ".txt": TextLoader,
                ".docx": Docx2txtLoader
            }
            
            # 2. Grab all files in the directory and subdirectories
            all_files = list(doc_dir.rglob("*"))
            
            for file_path in all_files:
                if not file_path.is_file():
                    continue # Skip directories

                ext = file_path.suffix.lower()
                
                # 3. Process supported files
                if ext in loaders_mapping:
                    self.log.info(f"Loading {ext} file: {file_path.name}")
                    try:
                        loader_class = loaders_mapping[ext]
                        loader = loader_class(str(file_path))
                        documents = loader.load()
                        
                        # Add helpful metadata so your VectorDB knows the format
                        for doc in documents:
                            doc.metadata["file_type"] = ext
                            
                        all_documents.extend(documents)
                    except Exception as load_err:
                        self.log.error(f"Failed to load {file_path.name}: {load_err}")
                
                # 4. Gracefully ignore unsupported files
                elif ext: 
                    self.log.warning(f"Skipping unsupported file type {ext}: {file_path.name}")

            self.log.info(f"Successfully loaded a total of {len(all_documents)} document parts/pages.")
            return all_documents

        except Exception as e:
            self.log.error(f"Error loading directory: {e}")
            return []
    
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
            length_split_docs=len(split_docs)
            #Removing the empty strings
            split_docs = [doc for doc in split_docs if doc.page_content.strip()]
            self.log.info(f"Empty chunks found: {length_split_docs-len(split_docs)} Removed:{length_split_docs-len(split_docs)} from the document")

            self.log.info(f"Completed splitting documents. Total chunks created: {len(split_docs)}")
            return split_docs
        except Exception as e:
            self.log.error(f"Error splitting documents: {e}")
            raise e
    
    def run_pipeline(self, file_path: str)->None:
        """
        Run the data ingestion pipeline to load PDF files,generate embeddings and splitting documents.
        Args:
            file_path (str): The path to the PDF file.
        
        Returns:
            list[np.ndarray]: The generated embeddings.
        """
        numpy_embeddings=[]
        self.log.info("Starting data ingestion pipeline.")
        try:
            documents = self._load_documents(file_path)
            self.log.info(f"Data Loading completed. Loaded {len(documents)} documents.")
            split_documents = self._split_documents(documents)
           
            for chunk in tqdm(split_documents, desc="Generating Embeddings", unit="chunk"):
                embeddings=self.generate_embeddings(chunk.page_content)
                numpy_embeddings.append(embeddings)
            self.vector_store._add_data(embeddings=numpy_embeddings,documents=split_documents)
            self.log.info(f"Data ingestion pipeline completed.")
        except Exception as e:
            self.log.error(f"Error running data ingestion pipeline: {e}")