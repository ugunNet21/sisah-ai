# src/utils/image_proc.py

import cv2
import numpy as np
import os
from PIL import Image

class ImageProcessor:
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    @staticmethod
    def load_image(image_path: str):
        """Load image dengan support berbagai format"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File tidak ditemukan: {image_path}")
        
        # Cek format
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ImageProcessor.SUPPORTED_FORMATS:
            raise ValueError(f"Format tidak didukung: {ext}")
        
        # Gunakan PIL dulu untuk konversi format yang bermasalah
        try:
            pil_image = Image.open(image_path)
            # Konversi ke RGB jika perlu
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Konversi ke numpy array untuk OpenCV
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            # Fallback ke cv2 langsung
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Gagal membaca gambar: {e}")
            return img

    @staticmethod
    def preprocess_for_ocr(image, debug=False):
        """
        Multi-stage preprocessing untuk meningkatkan akurasi OCR
        Menghasilkan multiple versions untuk strategi OCR berbeda
        """
        # 1. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize jika terlalu kecil (min width 1500px untuk OCR optimal pada ijazah)
        height, width = gray.shape
        if width < 1500:
            scale = 1500 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 3. Denoising (lebih agresif)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 4. Contrast Enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 5. Sharpening untuk teks yang blur
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        
        # 6. Adaptive Thresholding
        binary = cv2.adaptiveThreshold(
            sharpened, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # 7. Morphological operations untuk cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Debug: Simpan semua tahapan
        if debug:
            cv2.imwrite("debug_1_original_gray.jpg", gray)
            cv2.imwrite("debug_2_denoised.jpg", denoised)
            cv2.imwrite("debug_3_enhanced.jpg", enhanced)
            cv2.imwrite("debug_4_sharpened.jpg", sharpened)
            cv2.imwrite("debug_5_final_binary.jpg", binary)
        
        return binary
    
    @staticmethod
    def preprocess_variant_2(image):
        """Varian preprocessing kedua - untuk teks yang sangat kecil"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize lebih besar
        height, width = gray.shape
        if width < 2000:
            scale = 2000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Otsu thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    @staticmethod
    def detect_document_orientation(image):
        """Deteksi orientasi dokumen dan rotate jika perlu"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Deteksi edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Deteksi garis
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                angles.append(angle)
            
            # Ambil median angle
            median_angle = np.median(angles)
            
            # Rotate jika miring
            if abs(median_angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), 
                                        flags=cv2.INTER_CUBIC, 
                                        borderMode=cv2.BORDER_REPLICATE)
                return rotated
        
        return image