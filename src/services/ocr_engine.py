# src/services/ocr_engine.py

import pytesseract
from src.utils.image_proc import ImageProcessor

class OCREngine:
    def __init__(self, lang='ind+eng'):
        self.lang = lang

    def extract_text(self, image_path: str) -> str:
        try:
            # Load dan Preprocess
            img = ImageProcessor.load_image(image_path)
            processed_img = ImageProcessor.preprocess_for_ocr(img)
            
            # Konfigurasi Tesseract (Assume Block of text)
            custom_config = r'--oem 3 --psm 6'
            
            # Eksekusi OCR
            text = pytesseract.image_to_string(processed_img, lang=self.lang, config=custom_config)
            return text.strip()
            
        except Exception as e:
            print(f"[ERROR OCR] Gagal membaca gambar: {e}")
            return ""