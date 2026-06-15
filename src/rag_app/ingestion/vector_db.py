import os
import chromadb
from src.rag_app.utils.logging_config import AILogger
import numpy as np
from typing import List,Tuple,Optional,Dict
import uuid
class VectorStore:
    """Manages Document embedding in chroma DB  Vector store"""

    def __init__(self,collections_name:str="PDFDocuments",persistent_directory:str="data/vector_store"):
        self.collections_name=collections_name
        self.log=AILogger(name="Vector Logger",log_dir="logs/VectorDBLogger").logger
        self.directory=persistent_directory
        self.client=None
        self.collection=None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        try:
            #create persistent chroma db client
            os.makedirs(self.directory,exist_ok=True)
            self.log.info(f"Setting up ChromaDB client in{self.directory}")
            self.client=chromadb.PersistentClient(path=self.directory)
            self.log.info(f"Client Connected Sucessfully in :{self.directory}")
            #get collection
            self.log.info(f"Setting up collections{self.collections_name}")
            self.collection=self.client.get_or_create_collection(
                name=self.collections_name,
                metadata={"description":f"{self.collections_name} entry for RAG"}

            )
            self.log.info(f"Vector Store initialized Collection Name: {self.collections_name}")
            self.log.info(f"Existing Documents in collection: {self.collection.count()}")
        except Exception as e:
            self.log.error(f"Falied initializing vector store:{e}")
            raise e
    def _add_data(self,embeddings:list[np.ndarray],documents:List[any]):
            
            """Add embeddings to the vector store
            Args:
                documents: list of langchain documents
                embeddings: Numpy array of embeddings for documents
            """

            if len(documents)!=len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
                self.log.error(f"Mismatch in number of documents and embeddings Length of documents: {len(documents)} Length of Embeddings: {len(embeddings)}")

            #prepare data for chroma DB
            ids=[]
            metadatas=[]
            document_text=[]
            embedding_list=[]
            self.log.info(f"Preparing Data for Chroma DB")
            for i,(doc,embedding) in enumerate(zip(documents,embeddings)):

                file_path=doc.metadata.get("source")
                doc_id=f"{os.path.basename(file_path)}_chunk_{i}"
                ids.append(doc_id)

                #prepare metadata
                metadata=dict(doc.metadata)
                metadata['doc_index']=i
                metadata['content_length']=len(doc.page_content)
                metadatas.append(metadata)

                #document content
                document_text.append(doc.page_content)

                #Embedding append
                embedding_list.append(embedding.tolist())
            
            #checking if document already exists
            existing_data=self.collection.get(include=[])
            #create a set of document_ids for faster lookup
            existing_doc_ids=set(existing_data['ids'])

            #Filter out existing document
            filtered_ids = []
            filtered_metadatas = []
            filtered_documents = []
            filtered_embeddings = []
            
            for i,doc_id in enumerate(ids):
                if doc_id not in existing_doc_ids:
                    filtered_ids.append(doc_id)
                    filtered_metadatas.append(metadatas[i])
                    filtered_documents.append(document_text[i])
                    filtered_embeddings.append(embedding_list[i])
            duplicate_ids = len(ids) - len(filtered_ids)
            if duplicate_ids > 0:
                self.log.info(f"Existing documents in collection: {len(existing_doc_ids)} duplicate ids found {duplicate_ids}")
                self.log.info(f"Adding {len(filtered_ids)} new documents to the vector store")
            else:
                self.log.info(f"No duplicate ids found")
                self.log.info(f"Adding {len(filtered_ids)} new documents to the vector store")
            
            #all are duplicates then return    
            if not filtered_ids:
                self.log.info("No new documents to add to the vector store (all are duplicates).")
                self.log.info(f"Total documents in collection {self.collection.count()}")
                return

            try:
                self.collection.add(
                      ids=filtered_ids,
                      embeddings=filtered_embeddings,
                      metadatas=filtered_metadatas,
                      documents=filtered_documents
                 )
                self.log.info(f"Sucessfully added {len(filtered_ids)} documents to vector store")
                self.log.info(f"Total documents in collection {self.collection.count()}")
            except Exception as e:
                self.log.info(f"Errror Adding documents to vector store: {e}") 
                raise
    def get_scores(self,query_embedding:list[float],top_k:int=5,score_threshold:float=0.0)->tuple[list[float],list[str],list[dict]]:
        """
        Perform a similarity search in the vector store.
        
        Args:
            query_embedding: The embedding of the query.
            top_k: The number of top results to retrieve.
            score_threshold: The minimum similarity score to consider.
            
        Returns:
            A list of similarity scores.
        """
        self.log.info(f"Querying vector store with embedding")
        
        self.log.info(f"Top k: {top_k}, Score threshold: {score_threshold}")
        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=['distances','documents','metadatas']
            )
            self.log.info(f"Retrieved {len(results['ids'][0])} results from vector store")
            return results['distances'][0],results['documents'][0],results['metadatas'][0]
        except Exception as e:
            self.log.error(f"Error querying vector store: {str(e)}")
            return [],[],[]
    
                

        