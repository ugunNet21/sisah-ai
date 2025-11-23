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