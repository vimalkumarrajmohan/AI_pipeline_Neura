from typing import List, Dict, Any, Optional
from src.vector_store import vector_store
from src.logger import logger
from src.config import Config


class RAGRetriever:
    def __init__(self, 
                 vector_store_instance=None,
                 top_k: int = None):
        self.vector_store = vector_store_instance or vector_store
        self.top_k = top_k or Config.TOP_K_RETRIEVAL
    
    def retrieve(self, 
                query_embedding: List[float],
                top_k: Optional[int] = None) -> List[Dict[str, Any]]:

        try:
            k = top_k or self.top_k
            logger.info(f"Retrieving top {k} documents")
            
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k,
                score_threshold=0.0
            )
            
            logger.info(f"Retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            context = f"[Document {i}] (Score: {doc.get('score', 0):.4f})\n"
            context += f"Source: {doc.get('source', 'Unknown')}\n"
            context += f"Content:\n{doc.get('document', '')}\n"
            context_parts.append(context)
        
        return "\n".join(context_parts)
    
    def get_context_for_query(self,
                             query_embedding: List[float],
                             top_k: Optional[int] = None,
                             include_scores: bool = True) -> Dict[str, Any]:
        try:
            documents = self.retrieve(query_embedding, top_k)
            context = self.format_context(documents)
            
            result = {
                "context": context,
                "num_documents": len(documents),
                "documents": documents if include_scores else [
                    {k: v for k, v in doc.items() if k != "score"}
                    for doc in documents
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            raise
    
    def summarize_context(self, context: str, max_length: int = 500) -> str:
        if len(context) <= max_length:
            return context
        
        sentences = context.split('.')
        summary = ""
        
        for sentence in sentences:
            if len(summary) + len(sentence) <= max_length:
                summary += sentence + "."
            else:
                break
        
        return summary.strip()


rag_retriever = RAGRetriever()
