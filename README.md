# SiSah-AI
Sistem Analisis Keaslian Ijazah & Dokumen Pendidikan berbasis OCR + Forensik Digital + LLM.

## Fitur
- OCR ekstraksi teks dokumen
- Deteksi manipulasi (ELA)
- Validasi metadata & aturan dokumen
- Reasoning LLM (Ollama, Gemini, dsb)

## Cara Menjalankan
```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python3 main.py
python main.py contoh_ijazah.jpg
```

## install
```bash
sudo apt install tesseract-ocr tesseract-ocr-ind libtesseract-dev
```

## Struktur
```bash
.
├── 001.bmp
├── 001.jpg
├── debug_view_processed.jpg
├── main.py
├── README.md
├── requirements.txt
├── src
│   ├── core
│   │   ├── forensic.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── forensic.cpython-312.pyc
│   │   │   └── __init__.cpython-312.pyc
│   │   └── validator.py
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── __pycache__
│   │   └── __init__.cpython-312.pyc
│   ├── services
│   │   ├── __init__.py
│   │   ├── llm_client.py
│   │   ├── ocr_engine.py
│   │   └── __pycache__
│   │       ├── __init__.cpython-312.pyc
│   │       ├── llm_client.cpython-312.pyc
│   │       └── ocr_engine.cpython-312.pyc
│   └── utils
│       ├── image_proc.py
│       ├── __init__.py
│       └── __pycache__
│           ├── image_proc.cpython-312.pyc
│           └── __init__.cpython-312.pyc
└── tests
    └── test_flow.py

```