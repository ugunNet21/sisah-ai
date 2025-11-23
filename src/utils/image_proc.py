# src/utils/image_proc.py

import cv2
import numpy as np
import os

class ImageProcessor:
    @staticmethod
    def load_image(image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File tidak ditemukan: {image_path}")
        return cv2.imread(image_path)

    @staticmethod
    def preprocess_for_ocr(image):
        """
        Mengubah gambar menjadi hitam putih (biner) agar huruf lebih tegas.
        """
        # 1. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Denoising (hilangkan bintik-bintik)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 3. Thresholding (Otomatis menyesuaikan pencahayaan / Otsu)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary