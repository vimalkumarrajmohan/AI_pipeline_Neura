import pytest
import os
from pathlib import Path
from src.config import Config


@pytest.fixture
def temp_pdf_file(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    
    pdf_path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td (Test Content) Tj ET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n"
        b"0000000115 00000 n\n0000000229 00000 n\n0000000323 00000 n\n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n397\n%%EOF\n"
    )
    
    return str(pdf_path)


@pytest.fixture
def mock_weather_response():
    return {
        "city": "London",
        "country": "GB",
        "temperature": 15.5,
        "feels_like": 14.2,
        "humidity": 72,
        "pressure": 1013,
        "wind_speed": 4.5,
        "wind_direction": 230,
        "cloudiness": 60,
        "description": "Partly cloudy",
        "main_weather": "Clouds",
        "visibility": 10000,
        "sunrise": 1700000000,
        "sunset": 1700030000,
    }

