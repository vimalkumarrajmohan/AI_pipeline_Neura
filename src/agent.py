import os
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from src.logger import logger
from src.config import Config
from src.weather_service import weather_service
from src.pdf_processor import pdf_processor
from src.vector_store import vector_store
from src.rag_retriever import rag_retriever
from src.search_service import search_service
from langsmith import traceable


class AgentState:
    def __init__(self):
        self.query: str = ""
        self.query_type: str = ""
        self.weather_data: Optional[Dict[str, Any]] = None
        self.context: str = ""
        self.response: str = ""
        self.messages: List[Any] = []
        self.metadata: Dict[str, Any] = {}


class AIAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model=Config.GROQ_MODEL,
            temperature=Config.TEMPERATURE,
            api_key=Config.GROQ_API_KEY
        )
        
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
            self.embeddings = None
    
    def create_graph(self):
        graph = StateGraph(dict)
        
        graph.add_node("classify", self._classify_query)
        graph.add_node("fetch_weather", self._fetch_weather)
        graph.add_node("fetch_pdf_context", self._fetch_pdf_context)
        graph.add_node("fetch_general_context", self._fetch_general_context)
        graph.add_node("generate_response", self._generate_response)
        
        graph.add_edge(START, "classify")
        graph.add_edge("classify", "fetch_weather")
        graph.add_edge("classify", "fetch_pdf_context")
        graph.add_edge("classify", "fetch_general_context")
        graph.add_edge("fetch_weather", "generate_response")
        graph.add_edge("fetch_pdf_context", "generate_response")
        graph.add_edge("fetch_general_context", "generate_response")
        graph.add_edge("generate_response", END)
        
        return graph.compile()
    
    def _classify_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = state.get("query", "")
            query_lower = query.lower()
            
            weather_keywords = ["weather", "temperature", "rain", "snow", "wind", "forecast", "climate",
                                "humid", "pressure", "cloud", "storm", "thunder", "lightning", "fog",
                                "drizzle", "sleet", "hail", "degree", "celsius", "fahrenheit", "sunny",
                                "cloudy", "windy", "rainy", "snowy", "cold", "hot", "warm", "cool",
                                "humidity", "visibility", "sunset", "sunrise", "dew", "frost"]
            
            if any(keyword in query_lower for keyword in weather_keywords):
                state["query_type"] = "weather"
                logger.info("Query classified as weather query")
            else:
                try:
                    stats = vector_store.get_stats()
                    if stats.get("vector_count", 0) > 0:
                        state["query_type"] = "pdf"
                        logger.info("Query classified as PDF query (PDFs available)")
                    else:
                        state["query_type"] = "general"
                        logger.info("Query classified as general query (no PDFs available)")
                except:
                    state["query_type"] = "general"
                    logger.info("Query classified as general query (fallback)")
            
            return state
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            state["query_type"] = "general"
            return state

    def _fetch_weather(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if state.get("query_type") != "weather":
                return state
            
            query = state.get("query", "")
            
            extraction_prompt = f"""Extract the city name from this weather query. 
Return ONLY the city name, nothing else.
If no city is mentioned, return 'London'.

Query: {query}
City:"""
            
            extraction_messages = [HumanMessage(content=extraction_prompt)]
            
            logger.info("Using LLM to extract city name from query")
            city_response = self.llm.invoke(extraction_messages)
            city = city_response.content.strip()
            
            logger.info(f"Fetching weather for city: {city}")
            weather_data = weather_service.get_weather(city)
            
            state["weather_data"] = weather_data
            state["context"] = weather_service.format_weather_text(weather_data)
            
            return state
            
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            state["context"] = f"Error fetching weather: {str(e)}"
            return state
    
    def _fetch_pdf_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if state.get("query_type") != "pdf":
                return state
            
            query = state.get("query", "")
            
            if self.embeddings:
                query_embedding = self.embeddings.embed_query(query)
                
                logger.info("Retrieving context from vector store")
                context_result = rag_retriever.get_context_for_query(
                    query_embedding=query_embedding,
                    include_scores=True
                )
                
                state["context"] = context_result.get("context", "")
                state["metadata"]["retrieved_documents"] = context_result.get("num_documents", 0)
            else:
                state["context"] = "Embeddings not available"
            
            return state
            
        except Exception as e:
            logger.error(f"Error fetching PDF context: {e}")
            state["context"] = f"Error retrieving context: {str(e)}"
            return state
    
    def _fetch_general_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if state.get("query_type") != "general":
                return state
            
            query = state.get("query", "")
            
            logger.info("Fetching context from web search")
            search_results = search_service.search(query, max_results=5)
            context = search_service.format_search_context(search_results)
            
            state["context"] = context
            state["metadata"]["search_results_count"] = len(search_results.get("results", []))
            
            return state
            
        except ValueError as e:
            logger.warning(f"Search service not configured: {e}")
            state["context"] = "Unable to search. Please try asking about uploaded PDFs or weather instead."
            return state
        
        except Exception as e:
            logger.error(f"Error fetching general context: {e}")
            state["context"] = f"Error fetching information: {str(e)}"
            return state
    
    def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = state.get("query", "")
            context = state.get("context", "")
            query_type = state.get("query_type", "unknown")
          
            if query_type == "weather":
                system_prompt = """You are a helpful weather assistant. 
Provide accurate and clear weather information based on the data provided.
Format the response in a friendly and easy-to-read manner."""
                
            elif query_type == "pdf":
                system_prompt = """You are a helpful document assistant. 
Answer questions based on the provided context from the documents.
If the answer is not in the context, say so clearly.
Always cite the source document."""
                
            else:
                system_prompt = """You are a helpful AI assistant. 
Answer the user's question based on the provided search results and context.
If information comes from web sources, cite them appropriately.
Be accurate, clear, and concise."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
            ]
            
            logger.info("Generating response with LLM")
            response = self.llm.invoke(messages)
            
            state["response"] = response.content
            state["messages"] = messages + [response]
            
            logger.info("Response generated successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state["response"] = f"Error generating response: {str(e)}"
            return state
    
    @traceable
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            state = {
                "query": input_data.get("query", ""),
                "query_type": "",
                "weather_data": None,
                "context": "",
                "response": "",
                "messages": [],
                "metadata": {}
            }
            
            logger.info(f"Processing query: {state['query']}")
            
            state = self._classify_query(state)
            
            if state.get("query_type") == "weather":
                state = self._fetch_weather(state)
            elif state.get("query_type") == "pdf":
                state = self._fetch_pdf_context(state)
            else:
                state = self._fetch_general_context(state)
            
            state = self._generate_response(state)

            return state
            
        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            return {
                "query": input_data.get("query", ""),
                "response": f"Error: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def _process_query(self, query: str) -> str:
        result = self.invoke({"query": query})
        return result.get("response", "No response generated")


def create_agent() -> AIAgent:
    return AIAgent()
