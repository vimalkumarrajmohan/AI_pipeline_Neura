import unittest
from unittest.mock import patch, MagicMock
from src.agent import AIAgent


class TestAIAgent(unittest.TestCase):
    def setUp(self):
        with patch('src.agent.ChatGroq'):
            with patch('src.agent.HuggingFaceEmbeddings'):
                self.agent = AIAgent()
    
    def test_agent_initialization(self):
        with patch('src.agent.ChatGroq'):
            with patch('src.agent.HuggingFaceEmbeddings'):
                agent = AIAgent()
        
        self.assertIsNotNone(agent.llm)
    
    def test_classify_weather_query(self):
        state = {
            "query": "What is the weather in London?",
            "query_type": "",
            "weather_data": None,
            "context": "",
            "response": "",
            "messages": [],
            "metadata": {}
        }
        
        result = self.agent._classify_query(state)
        
        self.assertEqual(result["query_type"], "weather")
    
    def test_classify_pdf_query(self):
        with patch('src.agent.vector_store') as mock_store:
            mock_store.get_stats.return_value = {"vector_count": 5}
            
            state = {
                "query": "What are the main topics in the document?",
                "query_type": "",
                "weather_data": None,
                "context": "",
                "response": "",
                "messages": [],
                "metadata": {}
            }
            
            result = self.agent._classify_query(state)
            
            self.assertEqual(result["query_type"], "pdf")
    
    @patch('src.agent.weather_service')
    def test_fetch_weather(self, mock_weather):
        mock_weather.get_weather.return_value = {
            "city": "London",
            "temperature": 15.5,
            "humidity": 72
        }
        mock_weather.format_weather_text.return_value = "Weather: 15.5°C"
        
        state = {
            "query": "What is the weather in London?",
            "query_type": "weather",
            "weather_data": None,
            "context": "",
            "response": "",
            "messages": [],
            "metadata": {}
        }
        
        result = self.agent._fetch_weather(state)
        
        self.assertIsNotNone(result.get("weather_data"))
        self.assertIn("Weather", result["context"])
    
    def test_fetch_pdf_context_no_embeddings(self):
        self.agent.embeddings = None
        
        state = {
            "query": "What is in the document?",
            "query_type": "pdf",
            "weather_data": None,
            "context": "",
            "response": "",
            "messages": [],
            "metadata": {}
        }
        
        result = self.agent._fetch_pdf_context(state)
        
        self.assertIn("Embeddings not available", result["context"])
    
    @patch('src.agent.rag_retriever')
    def test_fetch_pdf_context_with_embeddings(self, mock_retriever):
        mock_retriever.get_context_for_query.return_value = {
            "context": "Retrieved context",
            "num_documents": 2
        }
        
        self.agent.embeddings = MagicMock()
        self.agent.embeddings.embed_query.return_value = [0.1, 0.2]
        
        state = {
            "query": "What is in the document?",
            "query_type": "pdf",
            "weather_data": None,
            "context": "",
            "response": "",
            "messages": [],
            "metadata": {}
        }
        
        result = self.agent._fetch_pdf_context(state)
        
        self.assertEqual(result["context"], "Retrieved context")
        self.assertEqual(result["metadata"]["retrieved_documents"], 2)
    
    @patch('src.agent.ChatGroq')
    def test_generate_response(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is a generated response."
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        with patch('src.agent.HuggingFaceEmbeddings'):
            agent = AIAgent()
        agent.llm = mock_llm
        
        state = {
            "query": "What is the weather?",
            "query_type": "weather",
            "weather_data": None,
            "context": "Weather data here",
            "response": "",
            "messages": [],
            "metadata": {}
        }
        
        result = agent._generate_response(state)
        
        self.assertEqual(result["response"], "This is a generated response.")
        self.assertGreater(len(result["messages"]), 0)
    
    @patch('src.agent.ChatGroq')
    def test_invoke_weather_query(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Clear skies today"
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        with patch('src.agent.HuggingFaceEmbeddings'):
            with patch('src.agent.weather_service') as mock_weather:
                agent = AIAgent()
                
                mock_weather.get_weather.return_value = {
                    "city": "London",
                    "temperature": 20,
                    "humidity": 60
                }
                mock_weather.format_weather_text.return_value = "Weather: 20°C"
                
                result = agent.invoke({"query": "Weather in London?"})
                
                self.assertEqual(result["query_type"], "weather")
                self.assertIsNotNone(result["response"])


if __name__ == "__main__":
    unittest.main()
