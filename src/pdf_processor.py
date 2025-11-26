import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
from src.logger import logger
from src.config import Config


class PDFProcessor:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.pdf_dir = Config.PDF_DIR
        os.makedirs(self.pdf_dir, exist_ok=True)
    
    def load_pdf(self, file_path: str) -> str:
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            if not file_path.endswith('.pdf'):
                raise ValueError(f"File is not a PDF: {file_path}")
            
            logger.info(f"Loading PDF: {file_path}")
            
            text = ""
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                if len(pdf_reader.pages) == 0:
                    raise ValueError("PDF file is empty")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text += page.extract_text()
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
            
            logger.info(f"Successfully loaded PDF with {len(text)} characters")
            return text
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid PDF: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise
    
    def split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            text = self.load_pdf(file_path)
            chunks = self.split_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    "content": chunk,
                    "source": file_path,
                    "chunk_id": i,
                    "metadata": {
                        "source": Path(file_path).name,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                }
                documents.append(doc)
            
            logger.info(f"Processed PDF into {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def get_pdf_list(self, directory: Optional[str] = None) -> List[str]:
        search_dir = directory or self.pdf_dir
        
        if not os.path.exists(search_dir):
            logger.warning(f"Directory does not exist: {search_dir}")
            return []
        
        pdf_files = []
        for file in os.listdir(search_dir):
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(search_dir, file))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {search_dir}")
        return pdf_files
    
    def validate_pdf(self, file_path: str) -> bool:
        try:
            with open(file_path, 'rb') as f:
                PyPDF2.PdfReader(f)
            return True
        except Exception as e:
            logger.warning(f"Invalid PDF file {file_path}: {e}")
            return False


pdf_processor = PDFProcessor()
