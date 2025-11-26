from typing import Dict, Any, Optional
import requests
from src.logger import logger
from src.config import Config


class SearchService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(Config, 'TAVILY_API_KEY', None)
        self.base_url = "https://api.tavily.com/search"
        self.timeout = 10
        
        if not self.api_key:
            logger.warning("Tavily API key not configured. Search will be disabled.")
    
    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("Tavily API key not configured")
        
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True
            }
            
            logger.info(f"Searching for: {query}")
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            formatted_results = {
                "query": query,
                "answer": data.get("answer", ""),
                "results": []
            }
            
            for result in data.get("results", []):
                formatted_results["results"].append({
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", "")
                })
            
            logger.info(f"Search returned {len(formatted_results['results'])} results")
            return formatted_results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during search: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise
    
    def format_search_context(self, search_results: Dict[str, Any]) -> str:
        context = f"Search Query: {search_results.get('query', '')}\n\n"
        
        if search_results.get("answer"):
            context += f"Answer: {search_results['answer']}\n\n"
        
        context += "Sources:\n"
        for i, result in enumerate(search_results.get("results", []), 1):
            context += f"{i}. {result.get('title', 'No Title')}\n"
            context += f"   {result.get('content', 'No content')}\n"
            context += f"   URL: {result.get('url', 'No URL')}\n\n"
        
        return context


search_service = SearchService()
