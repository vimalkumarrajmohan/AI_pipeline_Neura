import unittest
from unittest.mock import patch, MagicMock
from src.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        with patch('src.vector_store.QdrantClient'):
            self.store = VectorStore(
                url="http://localhost:6333",
                collection_name="test_collection"
            )
    
    @patch('src.vector_store.QdrantClient')
    def test_vector_store_initialization(self, mock_client):
        store = VectorStore(
            url="http://localhost:6333",
            collection_name="test"
        )
        
        self.assertEqual(store.collection_name, "test")
        self.assertEqual(store.vector_size, 384)
    
    @patch('src.vector_store.QdrantClient')
    def test_add_embeddings(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        store = VectorStore(collection_name="test")
        
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        documents = [
            {"content": "Document 1", "source": "source1", "chunk_id": 0},
            {"content": "Document 2", "source": "source2", "chunk_id": 1},
        ]
        
        result = store.add_embeddings(embeddings, documents)
        
        self.assertTrue(result)
        self.assertEqual(store.id_counter, 2)
    
    @patch('src.vector_store.QdrantClient')
    def test_add_embeddings_mismatch(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        store = VectorStore(collection_name="test")
        
        embeddings = [[0.1, 0.2]]
        documents = [
            {"content": "Document 1", "source": "source1"},
            {"content": "Document 2", "source": "source2"},
        ]
        
        with self.assertRaises(ValueError):
            store.add_embeddings(embeddings, documents)
    
    @patch('src.vector_store.QdrantClient')
    def test_search(self, mock_client_class):
        mock_client = MagicMock()
        
        mock_result_point = MagicMock()
        mock_result_point.id = 1
        mock_result_point.score = 0.95
        mock_result_point.payload = {
            "document": "Retrieved document",
            "source": "source1",
            "chunk_id": 0
        }
        
        mock_query_results = MagicMock()
        mock_query_results.points = [mock_result_point]
        
        mock_client.query_points.return_value = mock_query_results
        mock_client_class.return_value = mock_client
        
        store = VectorStore(collection_name="test")
        
        query_embedding = [0.1, 0.2]
        results = store.search(query_embedding, top_k=5)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[0]["score"], 0.95)
        self.assertEqual(results[0]["document"], "Retrieved document")
    
    @patch('src.vector_store.QdrantClient')
    def test_get_stats(self, mock_client_class):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.points_count = 100
        mock_collection.status = "active"
        
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(collection_name="test")
        stats = store.get_stats()
        
        self.assertEqual(stats["vector_count"], 100)
        self.assertEqual(stats["vector_size"], 384)


if __name__ == "__main__":
    unittest.main()
