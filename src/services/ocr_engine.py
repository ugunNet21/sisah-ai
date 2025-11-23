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
        Extract text dengan multiple OCR configs untuk hasil terbaik
        """
        try:
            # Load dan preprocessing
            logger.info(f"Loading image: {image_path}")
            img = ImageProcessor.load_image(image_path)
            
            # Auto-rotate jika perlu
            img = ImageProcessor.detect_document_orientation(img)
            
            # Preprocessing
            processed_img = ImageProcessor.preprocess_for_ocr(img, debug=debug)
            
            # === STRATEGI 1: PSM 6 (Assume uniform block of text) ===
            config_1 = r'--oem 3 --psm 6'
            text_1 = pytesseract.image_to_string(processed_img, lang=self.lang, config=config_1)
            
            # === STRATEGI 2: PSM 3 (Fully automatic page segmentation) ===
            config_2 = r'--oem 3 --psm 3'
            text_2 = pytesseract.image_to_string(processed_img, lang=self.lang, config=config_2)
            
            # === STRATEGI 3: PSM 4 (Single column text) ===
            config_3 = r'--oem 3 --psm 4'
            text_3 = pytesseract.image_to_string(processed_img, lang=self.lang, config=config_3)
            
            # Pilih hasil terpanjang (biasanya paling akurat)
            results = [text_1, text_2, text_3]
            best_text = max(results, key=len)
            
            logger.info(f"OCR completed. Text length: {len(best_text)} chars")
            
            if debug:
                logger.debug(f"OCR Result Preview:\n{best_text[:200]}")
            
            return best_text.strip()
            
        except Exception as e:
            logger.error(f"OCR Error: {e}", exc_info=True)
            return ""
    
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