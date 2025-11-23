# src/core/forensic.py

import cv2
import numpy as np
import os

class ForensicAnalyzer:
    def analyze_ela(self, image_path: str) -> float:
        """
        Menghasilkan skor ELA. 
        Makin TINGGI skornya, makin besar kemungkinan gambar tersebut hasil editan/tempelan.
        """
        try:
            orig = cv2.imread(image_path)
            if orig is None:
                return 0.0

            # 1. Simpan ulang gambar dengan kompresi kualitas rendah (90%)
            temp_file = "temp_ela_analysis.jpg"
            cv2.imwrite(temp_file, orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # 2. Baca gambar yang sudah dikompresi
            resaved = cv2.imread(temp_file)
            
            # 3. Hitung selisih pixel absolut (Original vs Resaved)
            diff = cv2.absdiff(orig, resaved)
            
            # 4. Cari nilai ekstrem (Scale amplifikasi)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            max_val = np.max(gray_diff)
            
            # Bersihkan file temp
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            return float(max_val)
            
        except Exception as e:
            print(f"[ERROR FORENSIC] {e}")
            return 0.0