# AI Pipeline with LangGraph, LangChain, and LangSmith

A comprehensive AI pipeline that combines weather API integration, PDF-based RAG, and agentic decision-making using LangGraph and Groq API.

## Features

- **Real-time Weather Data**: Fetch weather information using OpenWeatherMap API
- **PDF RAG System**: Retrieve and answer questions from PDF documents
- **Agentic Pipeline**: LangGraph-based decision making between weather and PDF queries
- **Vector Database**: Qdrant for efficient embedding storage and retrieval
- **LLM Evaluation**: LangSmith integration for response evaluation
- **Streamlit UI**: Interactive chat interface for easy interaction
- **Comprehensive Tests**: Unit tests for all major components

## Project Structure

```
Neura/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── weather_service.py
│   ├── pdf_processor.py
│   ├── vector_store.py
│   ├── agent.py
│   ├── rag_retriever.py
│   └── langsmith_evaluator.py
├── tests/
│   ├── __init__.py
│   ├── test_weather_service.py
│   ├── test_pdf_processor.py
│   ├── test_vector_store.py
│   ├── test_agent.py
│   └── test_rag_retriever.py
├── data/
│   └── sample.pdf (place your PDF files here)
├── logs/
│   └── app.log
├── app.py
├── requirements.txt
└── README.md
```


## Getting Started

### What You Need

- Python 3.10 or newer
- API keys for Groq, OpenWeatherMap, and LangSmith
- Qdrant running locally or in the cloud

### How to Set Up

1. **Download the project**
    - Clone this repo to your computer and open the folder:
      ```bash
      git clone https://github.com/vimalkumarrajmohan/AI_pipeline_Neura.git
      cd AI_pipeline_Neura
      ```

2. **Set up Python**
    - Make a virtual environment:
      ```bash
      python -m venv venv
      source venv/bin/activate
      ```

3. **Install the required packages**
    - Run:
      ```bash
      pip install -r requirements.txt
      ```
    - This will install all necessary dependencies including:
      - **LangChain ecosystem**: langchain, langchain-core, langchain-groq, langchain-community, langchain-qdrant, langchain-huggingface
      - **Agent & Orchestration**: langgraph, langsmith
      - **LLM & Embeddings**: groq, openai
      - **Vector Database**: qdrant-client
      - **Evaluation**: openevals
      - **UI & Utilities**: streamlit, requests, PyPDF2, pydantic, python-dotenv, httpx, IPython
      - **Testing**: pytest, pytest-asyncio

4. **Add your API keys**
    - Make a file called `.env` in the main folder. Put your keys in it like this:
      ```
      GROQ_API_KEY=your_groq_api_key
      OPENWEATHER_API_KEY=your_openweather_api_key
      LANGSMITH_API_KEY=your_langsmith_api_key
      LANGSMITH_PROJECT=your_project_name
      TAVILY_API_KEY=your_tavily_api_key
      QDRANT_URL=http://localhost:6333
      QDRANT_COLLECTION=documents
      GROQ_MODEL=mixtral-8x7b-32768
      TEMPERATURE=0.7
      LOG_LEVEL=INFO
      ```
    - **Required API Keys**:
      - `GROQ_API_KEY`: Get from [console.groq.com](https://console.groq.com)
      - `OPENWEATHER_API_KEY`: Get from [openweathermap.org](https://openweathermap.org/api)
      - `LANGSMITH_API_KEY`: Get from [smith.langchain.com](https://smith.langchain.com)
      - `TAVILY_API_KEY`: Get from [tavily.com](https://tavily.com) (for web search functionality)

5. **Start Qdrant (if you use it locally)**
    - Run this command:
      ```bash
      docker run -p 6333:6333 qdrant/qdrant
      ```

### How to Run

- **Streamlit App:**
  ```bash
  streamlit run app.py
  ```

- **Command Line Example:**
  ```bash
  python -c "from src.agent import create_agent; agent = create_agent(); result = agent.invoke({'query': 'What is the weather in London?'}); print(result)"
  ```

- **Run Tests:**
  ```bash
  pytest tests/ -v
  ```


## How It Works (In Simple Terms)

- **Weather Service** (`src/weather_service.py`):
    - Talks to the OpenWeatherMap API to get the latest weather for any city you ask about.

- **PDF Processing** (`src/pdf_processor.py`):
    - Reads PDF files, breaks them into smaller parts, and creates embeddings so the AI can search and answer questions about them.

- **Vector Store** (`src/vector_store.py`):
    - Uses Qdrant to store and search through the PDF chunks quickly and efficiently.

- **Agent** (`src/agent.py`):
    - Decides if your question is about the weather or about a PDF, and sends it to the right place. It connects all the parts together.

- **RAG Retriever** (`src/rag_retriever.py`):
    - Finds the most relevant information from your PDFs and summarizes it for you.

- **LangSmith Evaluator** (`src/langsmith_evaluator.py`):
    - Checks and logs how well the AI is answering your questions, so you can see how it’s doing.

## Usage Examples

### Query Weather
```python
from src.agent import create_agent

agent = create_agent()
result = agent.invoke({
    'query': 'What is the current weather in New York?',
    'city': 'New York'
})
print(result)
```

### Query PDF
```python
result = agent.invoke({
    'query': 'What are the main points about climate change?',
    'pdf_path': 'data/sample.pdf'
})
print(result)
```

### Chat Interface
Simply run the Streamlit app and use the chat interface to interact with both weather and PDF data.

## LangSmith Integration

The pipeline includes comprehensive LangSmith integration with automatic tracing and evaluation:

- **Automatic Tracing**: All LLM calls are automatically traced using the `@traceable` decorator
- **Metrics Tracking**: Response latency, token usage, and quality metrics
- **Custom Evaluators**: LLM-as-judge evaluation for response correctness using `openevals`
- **Experiment Tracking**: Compare different configurations and configurations via LangSmith
- **Live Evaluation**: Real-time evaluation of responses as they're generated

### Viewing LangSmith Results

1. Log into your [LangSmith dashboard](https://smith.langchain.com)
2. Navigate to the project specified in `LANGSMITH_PROJECT` environment variable
3. View traces, metrics, and evaluation results in the UI

### Evaluation Features

- **Correctness Evaluator**: Automatically evaluates if responses match reference answers
- **Custom Datasets**: Supports QA dataset evaluation on Transformer paper questions
- **Feedback Logging**: All evaluation results are logged to LangSmith for analysis

## Testing

The project includes comprehensive test coverage:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_agent.py -v

# Run with coverage
pytest tests/ --cov=src
```

### Test Coverage

- **Weather Service**: API integration, error handling
- **PDF Processing**: Document loading, chunking, embedding generation
- **Vector Store**: CRUD operations, similarity search
- **Agent**: Decision logic, pipeline orchestration
- **RAG Retriever**: Document retrieval, summarization

## Architecture

### Agent Flow

```
User Query
    ↓
Classify Node (LangGraph)
    ├→ Weather Query → Weather Service → LLM → Response
    ├→ PDF Query → RAG Retriever → LLM → Response
    └→ General Query → Web Search Service → LLM → Response
    ↓
LangSmith Evaluation & Logging
    ↓
Response to User
```

### Tech Stack

- **LLM Framework**: LangChain
- **Agent Orchestration**: LangGraph
- **LLM Provider**: Groq (primary), OpenAI (fallback)
- **Vector Database**: Qdrant
- **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **Web Search**: Tavily
- **Evaluation**: LangSmith + OpenEvals (LLM-as-judge)
- **UI**: Streamlit
- **Testing**: pytest, pytest-asyncio

## Performance Notes

- Average response time: ~2-5 seconds
- Vector similarity search: < 100ms
- PDF processing: ~1-2 seconds for typical documents

## Troubleshooting

### Connection Issues
- Ensure Qdrant is running on the specified URL: `QDRANT_URL`
- Check all API keys in `.env` file are correct and have sufficient quota
- Verify internet connection for external API calls

### PDF Loading Issues
- Verify PDF files are in the `data/pdfs/` directory
- Check file permissions and PDF format validity
- Use the `validate_pdf()` method in PDFProcessor to test

### LangSmith Not Logging
- Verify `LANGSMITH_API_KEY` is set correctly
- Check project name matches `LANGSMITH_PROJECT` environment variable
- Ensure API key has appropriate permissions in LangSmith workspace

### Embedding Generation Issues
- HuggingFace embedding model requires internet connection on first use
- Model is cached locally after first download
- Ensure sufficient disk space for model download (~100MB)

### Weather API Errors
- Verify city name is valid
- Check OpenWeatherMap API quota
- Ensure `OPENWEATHER_API_KEY` is active

## Contributing

Feel free to fork this project and submit pull requests for any improvements.



