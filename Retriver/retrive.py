from Data_ingestion import load_data
from Data_ingestion import Vector_db
from logging_config import AILogger
from typing import List,Dict,Any,Tuple
import numpy as np
import os
class RAGRtriever:
    def __init__(self,vector_store:Vector_db.VectorStore,injestion_object:load_data.DataIngestionPipeline):
        self.log=AILogger(name="RagRtriever",log_dir="logs/RagRtriever").logger
        self.vector_store=vector_store
        self.injestion_object=injestion_object
        
    def retrive(self,query:str,top_k:int=5,score_threshold:float=0.0)->list[Dict[str,Any]]:
        try:
            # Generate embeddings for the query
            self.log.info(f"Generating embeddings for query: {query}"   )
            query_embedding = self.injestion_object.generate_embeddings([query])
            self.log.info(f"Successfully generated embeddings for query: {query}")
            query_embedding_list=query_embedding.tolist()
            self.log.info(f"TYPE: {type(query_embedding_list)}")
            # Get similarity scores from vector store
            self.log.info(f"Getting similarity scores for query: {query}")
            scores,documents,metadata = self.vector_store.get_scores(query_embedding_list, top_k=top_k, score_threshold=score_threshold)
            self.log.info(f"Successfully retrieved similarity scores for query: {query}")

            # Apply score threshold and get top k results
            filtered_indices = [i for i, score in enumerate(scores) if score >= score_threshold][:top_k]
            self.log.info(f"Filtered indices: {filtered_indices}")
            # Retrieve the documents corresponding to the filtered indices
            retrieved_docs = []
            for idx in filtered_indices:
                retrieved_docs.append({
                    'text': documents[idx],
                    'score': float(scores[idx]),
                    'file_name':metadata[idx]['source']
                })

            self.log.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")
            return retrieved_docs

        except Exception as e:
            self.log.error(f"Error retrieving documents: {str(e)}")
            return [] 
    
    def format_prompt_for_llm(self,question: str, documents: list[str], metadatas: list[dict]) -> str:
        """
        Formats the retrieved documents and user question into a structured, 
        hallucination-resistant prompt for the LLM.
        """
        # 1. Build the Context String with XML-style boundaries
        context_parts = []
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            # Extract filename cleanly
            source_path = meta.get('source', 'Unknown_Source')
            file_name = os.path.basename(source_path)
            
            # Wrap each chunk in a clear document tag
            chunk_text = f"""<document index="{i+1}" source="{file_name}">\n{doc.strip()}\n</document>"""
            context_parts.append(chunk_text)
            
        full_context = "\n\n".join(context_parts)

        # 2. Assemble the Final Prompt 
    # Notice the strict separation of Instructions, Context, and Question
        prompt = f"""You are an expert analyst. Your task is to answer the user's question based strictly on the provided context.

        <instructions>
        1. Analyze the text within the <context> tags.
        2. Answer the question using ONLY the provided information. 
        3. If the answer is not present in the context, explicitly state: "I cannot find the answer in the provided documents." Do not use outside knowledge.
        4. Every time you state a fact, you MUST cite the source file inline using brackets. Example: "The magic bounced off him because of his giant blood [harrypotter.pdf]."
        </instructions>

        <context>
        {full_context}
        </context>

        <question>
        {question}
        </question>

        Answer:"""

        return prompt   
    def get_answer(self,query:str)->str:
        self.log.info(f"Generating answer for query: {query}")
        retrieved_docs = self.retrive(query)
        self.log.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")
        documents = [doc['text'] for doc in retrieved_docs]
        metadatas = [{'source':doc['file_name']} for doc in retrieved_docs]
        self.log.info(f"Generated context for query: {query}")
        prompt = self.format_prompt_for_llm(query,documents,metadatas)
        self.log.info(f"Generated prompt for query: {prompt}")
        #response = self.injestion_object.llm.invoke(prompt)
        #self.log.info(f"Generated prompt for query: {prompt}")
        return prompt