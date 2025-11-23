# src/core/forensic.py

import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class ForensicAnalyzer:
    def analyze_ela(self, image_path: str) -> dict:
        """
        Error Level Analysis untuk deteksi manipulasi gambar.
        Return: dict dengan score dan interpretasi
        """
        try:
            orig = cv2.imread(image_path)
            if orig is None:
                return {"score": 0.0, "status": "ERROR", "detail": "Cannot read image"}

            # 1. Simpan ulang dengan kompresi JPEG kualitas 90%
            temp_file = "temp_ela_analysis.jpg"
            cv2.imwrite(temp_file, orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # 2. Baca gambar hasil kompresi
            resaved = cv2.imread(temp_file)
            
            # 3. Hitung selisih absolut
            diff = cv2.absdiff(orig, resaved)
            
            # 4. Konversi ke grayscale untuk analisis
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # 5. Hitung statistik
            max_val = np.max(gray_diff)
            mean_val = np.mean(gray_diff)
            std_val = np.std(gray_diff)
            
            # 6. Hitung persentase pixel dengan anomali tinggi
            threshold = 20
            anomaly_pixels = np.sum(gray_diff > threshold)
            total_pixels = gray_diff.shape[0] * gray_diff.shape[1]
            anomaly_percentage = (anomaly_pixels / total_pixels) * 100
            
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Interpretasi
            status = "ORIGINAL"
            confidence = "High"
            
            if max_val > 40 or anomaly_percentage > 5:
                status = "HIGHLY_SUSPICIOUS"
                confidence = "High"
            elif max_val > 25 or anomaly_percentage > 2:
                status = "SUSPICIOUS"
                confidence = "Medium"
            elif max_val > 15:
                status = "MINOR_EDIT"
                confidence = "Low"
            
            logger.info(f"ELA Analysis - Max: {max_val:.2f}, Mean: {mean_val:.2f}, Anomaly: {anomaly_percentage:.2f}%")
            
            return {
                "score": float(max_val),
                "mean_score": float(mean_val),
                "std_score": float(std_val),
                "anomaly_percentage": float(anomaly_percentage),
                "status": status,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Forensic analysis error: {e}", exc_info=True)
            return {"score": 0.0, "status": "ERROR", "detail": str(e)}
    
    def check_metadata(self, image_path: str) -> dict:
        """
        Ekstrak metadata EXIF untuk analisis tambahan
        """
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            img = Image.open(image_path)
            exif_data = img._getexif()
            
            if exif_data is None:
                return {"has_exif": False, "warning": "No EXIF data (possibly edited)"}
            
            metadata = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                metadata[tag] = str(value)
            
            return {
                "has_exif": True,
                "software": metadata.get("Software", "Unknown"),
                "datetime": metadata.get("DateTime", "Unknown"),
                "camera_model": metadata.get("Model", "Unknown")
            }
            
        except Exception as e:
            logger.warning(f"Metadata extraction error: {e}")
            return {"has_exif": False, "error": str(e)}