from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.logger import logger
from src.config import Config


class VectorStore:
    def __init__(self, 
                 url: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 vector_size: int = 384):
        
        self.url = url or Config.QDRANT_URL
        self.collection_name = collection_name or Config.QDRANT_COLLECTION
        self.vector_size = vector_size
        
        try:
            self.client = QdrantClient(url=self.url)
            logger.info(f"Connected to Qdrant at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        
        self.id_counter = 0
    
    def create_collection(self, 
                         force_recreate: bool = False) -> bool:
        try:
            collections = self.client.get_collections()
            exists = any(col.name == self.collection_name for col in collections.collections)
            
            if exists:
                if force_recreate:
                    logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(collection_name=self.collection_name)
                else:
                    logger.info(f"Collection already exists: {self.collection_name}")
                    return True
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"Created collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def add_embeddings(self,
                      embeddings: List[List[float]],
                      documents: List[Dict[str, Any]],
                      metadata: Optional[List[Dict[str, Any]]] = None) -> bool:

        try:
            if len(embeddings) != len(documents):
                raise ValueError("Number of embeddings must match number of documents")
            
            points = []
            for i, (embedding, document) in enumerate(zip(embeddings, documents)):
                point_id = self.id_counter + i
                payload = {
                    "document": document.get("content", ""),
                    "source": document.get("source", ""),
                    "chunk_id": document.get("chunk_id", i),
                }
                
                if metadata and i < len(metadata):
                    payload.update(metadata[i])
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            self.id_counter += len(embeddings)
            logger.info(f"Added {len(embeddings)} embeddings to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            raise
    
    def search(self,
               query_embedding: List[float],
               top_k: int = 5,
               score_threshold: float = 0.0) -> List[Dict[str, Any]]:

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            documents = []
            for result in results.points:
                doc = {
                    "id": result.id,
                    "score": result.score,
                    "document": result.payload.get("document", ""),
                    "source": result.payload.get("source", ""),
                    "chunk_id": result.payload.get("chunk_id"),
                    "metadata": {k: v for k, v in result.payload.items() 
                               if k not in ["document", "source", "chunk_id"]}
                }
                documents.append(doc)
            
            logger.info(f"Search returned {len(documents)} results")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vector_count": collection.points_count,
                "vector_size": self.vector_size,
                "status": collection.status
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise


vector_store = VectorStore()
