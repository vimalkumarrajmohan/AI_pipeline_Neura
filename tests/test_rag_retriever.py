import unittest
from unittest.mock import patch, MagicMock
from src.rag_retriever import RAGRetriever


class TestRAGRetriever(unittest.TestCase):
    def setUp(self):
        self.mock_vector_store = MagicMock()
        self.retriever = RAGRetriever(
            vector_store_instance=self.mock_vector_store,
            top_k=5
        )
    
    def test_retrieve(self):
        mock_documents = [
            {
                "id": 1,
                "score": 0.95,
                "document": "Document 1 content",
                "source": "source1",
                "chunk_id": 0
            },
            {
                "id": 2,
                "score": 0.85,
                "document": "Document 2 content",
                "source": "source2",
                "chunk_id": 1
            }
        ]
        
        self.mock_vector_store.search.return_value = mock_documents
        
        query_embedding = [0.1, 0.2, 0.3]
        results = self.retriever.retrieve(query_embedding)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["score"], 0.95)
        self.assertEqual(results[1]["score"], 0.85)
    
    def test_format_context(self):
        documents = [
            {
                "id": 1,
                "score": 0.95,
                "document": "Test document content",
                "source": "test.pdf",
                "chunk_id": 0
            }
        ]
        
        context = self.retriever.format_context(documents)
        
        self.assertIn("Document 1", context)
        self.assertIn("0.9500", context)
        self.assertIn("test.pdf", context)
        self.assertIn("Test document content", context)
    
    def test_format_context_empty(self):
        context = self.retriever.format_context([])
        
        self.assertEqual(context, "")
    
    def test_get_context_for_query(self):
        mock_documents = [
            {
                "id": 1,
                "score": 0.95,
                "document": "Document 1",
                "source": "source1",
                "chunk_id": 0
            }
        ]
        
        self.mock_vector_store.search.return_value = mock_documents
        
        query_embedding = [0.1, 0.2]
        result = self.retriever.get_context_for_query(query_embedding)
        
        self.assertIn("context", result)
        self.assertIn("num_documents", result)
        self.assertEqual(result["num_documents"], 1)
        self.assertIn("Document 1", result["context"])
    
    def test_get_context_without_scores(self):
        mock_documents = [
            {
                "id": 1,
                "score": 0.95,
                "document": "Document 1",
                "source": "source1",
                "chunk_id": 0
            }
        ]
        
        self.mock_vector_store.search.return_value = mock_documents
        
        query_embedding = [0.1, 0.2]
        result = self.retriever.get_context_for_query(
            query_embedding,
            include_scores=False
        )
        
        self.assertTrue(all("score" not in doc for doc in result["documents"]))
    
    def test_summarize_context_long(self):
        long_context = "This is a test. " * 100
        
        summary = self.retriever.summarize_context(long_context, max_length=100)
        
        self.assertLessEqual(len(summary), 100 + 10)
    
    def test_summarize_context_short(self):
        short_context = "This is a short context."
        
        summary = self.retriever.summarize_context(short_context, max_length=100)
        
        self.assertEqual(summary, short_context)


if __name__ == "__main__":
    unittest.main()
