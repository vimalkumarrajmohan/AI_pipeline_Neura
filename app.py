import os
from datetime import datetime

import streamlit as st

from src.agent import create_agent
from src.config import Config
from src.pdf_processor import pdf_processor
from src.vector_store import vector_store
from src.logger import logger
from langchain_huggingface import HuggingFaceEmbeddings

from src.langsmith_evaluator import evaluate_live_question_and_log 

st.set_page_config(page_title="Neura AI Pipeline - Simple", page_icon="🤖", layout="wide")


def process_and_store_pdf(pdf_path: str):
    try:
        vector_store.create_collection(force_recreate=False)
        
        documents = pdf_processor.process_pdf(pdf_path)
        
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        embeddings = []
        for doc in documents:
            embedding = embeddings_model.embed_query(doc["content"])
            embeddings.append(embedding)
        
        vector_store.add_embeddings(embeddings, documents)
        logger.info(f"Stored {len(documents)} embeddings in vector store")
        return len(documents)
        
    except Exception as e:
        logger.error(f"Error processing and storing PDF: {e}")
        raise


def init_state():
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = create_agent()
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            st.session_state.agent = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "graph_running" not in st.session_state:
        st.session_state.graph_running = True


def render_sidebar():
    st.sidebar.title("Upload and Files")
    st.sidebar.write("Upload PDF files to the knowledge base")

    uploaded = st.sidebar.file_uploader("Drag and drop or browse", type=["pdf"], key="sidebar_uploader")

    if uploaded is not None:
        os.makedirs(Config.PDF_DIR, exist_ok=True)
        path = os.path.join(Config.PDF_DIR, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.sidebar.success(f"File saved: {path}")

        if st.sidebar.button("Process latest PDF"):
            try:
                pdfs = pdf_processor.get_pdf_list()
                if not pdfs:
                    st.sidebar.warning("No PDFs to process")
                else:
                    latest = sorted(pdfs)[-1]
                    num_docs = process_and_store_pdf(latest)
                    st.sidebar.success(f"Processed {os.path.basename(latest)} into {num_docs} embeddings in vector store")
            except Exception as e:
                st.sidebar.error(f"Error processing PDF: {e}")

    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []


def render_main():
    st.header("AI Assistant")
    st.write("Ask about uploaded PDFs or request weather information.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask Anything"):
        st.session_state.messages.append({"role": "user",
                                            "content": prompt,
                                            "codes": '',
                                            "type": ''})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            agent = st.session_state.agent

            if agent:
                try:
                    result = agent.invoke({"query": prompt})
                    response = result.get("response", "No response")
                    for chunk in response.split():
                        full_response += chunk + " "
                        message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                    st.session_state.messages.append({"role": "assistant",
                                                    "content": full_response,
                                                    "ts": datetime.now().strftime("%H:%M:%S")})
                    
                    evaluation_result = evaluate_live_question_and_log(user_question=prompt,
                                                                        llm_response=full_response,)
                    
                   

                except Exception as e:
                    logger.error(f"Agent error: {e}")
                    error_msg = f"Error: {e}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant",
                                                    "content": error_msg,
                                                    "ts": datetime.now().strftime("%H:%M:%S")})
                    
            else:
                no_agent_msg = "Agent not available"
                message_placeholder.markdown(no_agent_msg)
                st.session_state.messages.append({"role": "assistant",
                                                    "content": no_agent_msg,
                                                    "ts": datetime.now().strftime("%H:%M:%S")})


def main():
    init_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
