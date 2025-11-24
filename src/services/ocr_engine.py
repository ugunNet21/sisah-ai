# src/services/ocr_engine.py

import pytesseract
import cv2
from src.utils.image_proc import ImageProcessor
import logging

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, lang='ind+eng'):
        self.lang = lang
        
    def extract_text(self, image_path: str, debug=False) -> str:
        """
        Extract text dengan multiple preprocessing dan OCR configs
        """
        try:
            # Load dan preprocessing
            logger.info(f"Loading image: {image_path}")
            img = ImageProcessor.load_image(image_path)
            
            # Auto-rotate jika perlu
            img = ImageProcessor.detect_document_orientation(img)
            
            # === Multiple Preprocessing Strategies ===
            processed_v1 = ImageProcessor.preprocess_for_ocr(img, debug=debug)
            processed_v2 = ImageProcessor.preprocess_variant_2(img)
            
            results = []
            
            # === STRATEGI 1: Best quality preprocessing + PSM 6 ===
            config_1 = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:/- '
            text_1 = pytesseract.image_to_string(processed_v1, lang=self.lang, config=config_1)
            results.append(text_1)
            
            # === STRATEGI 2: PSM 3 (Fully automatic) ===
            config_2 = r'--oem 3 --psm 3'
            text_2 = pytesseract.image_to_string(processed_v1, lang=self.lang, config=config_2)
            results.append(text_2)
            
            # === STRATEGI 3: PSM 4 dengan variant preprocessing ===
            config_3 = r'--oem 3 --psm 4'
            text_3 = pytesseract.image_to_string(processed_v2, lang=self.lang, config=config_3)
            results.append(text_3)
            
            # === STRATEGI 4: PSM 11 (Sparse text) untuk dokumen dengan layout kompleks ===
            config_4 = r'--oem 3 --psm 11'
            text_4 = pytesseract.image_to_string(processed_v1, lang=self.lang, config=config_4)
            results.append(text_4)
            
            # === STRATEGI 5: PSM 1 (Auto with OSD - Orientation and Script Detection) ===
            config_5 = r'--oem 3 --psm 1'
            text_5 = pytesseract.image_to_string(processed_v1, lang=self.lang, config=config_5)
            results.append(text_5)
            
            # Pilih hasil terpanjang dan paling banyak kata
            best_text = max(results, key=lambda x: len(x.split()))
            
            # Post-processing: bersihkan karakter aneh
            best_text = self._clean_ocr_text(best_text)
            
            logger.info(f"OCR completed. Text length: {len(best_text)} chars, Words: {len(best_text.split())}")
            
            if debug:
                logger.debug(f"OCR Result Preview:\n{best_text[:300]}")
                # Save all results for comparison
                with open("debug_ocr_all_results.txt", "w") as f:
                    for i, txt in enumerate(results, 1):
                        f.write(f"\n{'='*50}\nSTRATEGY {i}:\n{'='*50}\n{txt}\n")
            
            return best_text.strip()
            
        except Exception as e:
            logger.error(f"OCR Error: {e}", exc_info=True)
            return ""
    
    def _clean_ocr_text(self, text: str) -> str:
        """Bersihkan hasil OCR dari karakter aneh"""
        import re
        
        # Hapus karakter non-printable kecuali newline dan space
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t ')
        
        # Hapus karakter ASCII art dan symbols berlebihan
        text = re.sub(r'[|=\-_]{3,}', ' ', text)  # Remove lines
        text = re.sub(r'[^\w\s.,:/()\'"\-]', ' ', text)  # Keep only alphanumeric + basic punctuation
        
        # Hapus multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Hapus multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Perbaiki common OCR errors
        replacements = {
            # Institusi
            'l<embar': 'Lembar', 'l<ementerian': 'Kementerian',
            'Unlversitas': 'Universitas', 'Universttas': 'Universitas',
            'UNIVERSITAS': 'Universitas', 'INSTITUT': 'Institut',
            'SEKOLAH TINGGI': 'Sekolah Tinggi', 'STMIK': 'STMIK',
            'Pollteknik': 'Politeknik',
            
            # Pendidikan
            'Dlploma': 'Diploma', 'Sar]ana': 'Sarjana', 'SARJANA': 'Sarjana',
            'Managemen': 'Manajemen', 'Managemcn': 'Manajemen',
            'lnformatika': 'Informatika', 'Informatilca': 'Informatika',
            'Komp uter': 'Komputer', 'Kompter': 'Komputer',
            'Tcknik': 'Teknik', 'Teknlk': 'Teknik',
            
            # Umum
            'lndones1a': 'Indonesia', 'Negcri': 'Negeri',
            'Pemer1ntah': 'Pemerintah', 'Republ1k': 'Republik',
            
            # SMK specific
            'KEJURUAN': 'Kejuruan', 'MENENGAH': 'Menengah',
            'Komp etensi': 'Kompetensi', 'Keahl1an': 'Keahlian'
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        
        # Remove isolated single characters (OCR noise)
        text = re.sub(r'\b[^aAiI\d]\b', ' ', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def get_confidence_data(self, image_path: str) -> dict:
        """
        Mendapatkan confidence score per word dari OCR
        """
        try:
            img = ImageProcessor.load_image(image_path)
            processed_img = ImageProcessor.preprocess_for_ocr(img)
            
            data = pytesseract.image_to_data(processed_img, lang=self.lang, output_type=pytesseract.Output.DICT)
            
            # Hitung rata-rata confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'average_confidence': avg_confidence,
                'total_words': len(confidences),
                'low_confidence_words': sum(1 for c in confidences if c < 60)
            }
        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return {'average_confidence': 0, 'total_words': 0, 'low_confidence_words': 0}