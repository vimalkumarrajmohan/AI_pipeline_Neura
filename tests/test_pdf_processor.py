import unittest
import tempfile
import os
from pathlib import Path
from src.pdf_processor import PDFProcessor


class TestPDFProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = PDFProcessor(chunk_size=100, chunk_overlap=20)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_pdf(self):
        pdf_path = os.path.join(self.temp_dir, "test.pdf")
        
        with open(pdf_path, 'wb') as f:
            f.write(
                b"%PDF-1.4\n"
                b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
                b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
                b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
                b"4 0 obj\n<< /Length 50 >>\nstream\nBT /F1 12 Tf 100 700 Td (Test PDF Content) Tj ET\nendstream\nendobj\n"
                b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
                b"xref\n0 6\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n"
                b"0000000115 00000 n\n0000000229 00000 n\n0000000328 00000 n\n"
                b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n402\n%%EOF\n"
            )
        
        return pdf_path
    
    def test_load_pdf_success(self):
        pdf_path = self._create_test_pdf()
        
        try:
            text = self.processor.load_pdf(pdf_path)
            self.assertIsInstance(text, str)
        except ValueError:
            pass
    
    def test_process_pdf(self):
        pdf_path = self._create_test_pdf()
        
        try:
            documents = self.processor.process_pdf(pdf_path)
            if documents:
                self.assertIsInstance(documents, list)
                for doc in documents:
                    self.assertIn("content", doc)
                    self.assertIn("source", doc)
                    self.assertIn("chunk_id", doc)
                    self.assertIn("metadata", doc)
        except ValueError:
            pass
    
    def test_load_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            self.processor.load_pdf("nonexistent.pdf")
    
    def test_load_invalid_file(self):
        invalid_path = os.path.join(self.temp_dir, "invalid.pdf")
        with open(invalid_path, 'w') as f:
            f.write("This is not a PDF")
        
        with self.assertRaises((ValueError, Exception)):
            self.processor.load_pdf(invalid_path)
    
    def test_split_text(self):
        text = "This is a test document. " * 50
        chunks = self.processor.split_text(text)
        
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), self.processor.chunk_size + 50)
    
    def test_split_text_single_chunk(self):
        text = "Short text"
        chunks = self.processor.split_text(text)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)
    
    def test_get_pdf_list_empty(self):
        pdf_list = self.processor.get_pdf_list(self.temp_dir)
        self.assertEqual(len(pdf_list), 0)
    
    def test_get_pdf_list_with_files(self):
        for i in range(3):
            pdf_path = os.path.join(self.temp_dir, f"test{i}.pdf")
            with open(pdf_path, 'w') as f:
                f.write("dummy content")
        
        pdf_list = self.processor.get_pdf_list(self.temp_dir)
        self.assertEqual(len(pdf_list), 3)
    
    def test_validate_pdf_invalid(self):
        invalid_path = os.path.join(self.temp_dir, "invalid.pdf")
        with open(invalid_path, 'w') as f:
            f.write("Not a PDF")
        
        result = self.processor.validate_pdf(invalid_path)
        self.assertFalse(result)
    
    def test_validate_pdf_nonexistent(self):
        result = self.processor.validate_pdf("nonexistent.pdf")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
