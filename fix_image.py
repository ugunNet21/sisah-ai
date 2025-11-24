#!/usr/bin/env python3
"""
Quick image fixer untuk ijazah yang sulit dibaca
Usage: python fix_image.py input.jpg output.jpg
"""
import sys
import cv2
import numpy as np
from pathlib import Path

def fix_image(input_path: str, output_path: str, mode='auto'):
    """
    Fix image quality untuk OCR
    Modes: auto, aggressive, light
    """
    print(f"Loading: {input_path}")
    img = cv2.imread(input_path)
    
    if img is None:
        print("Error: Cannot read image")
        return False
    
    h, w = img.shape[:2]
    print(f"Original size: {w}x{h}")
    
    # 1. Resize if too small
    if w < 2000:
        scale = 2000 / w
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"Resized to: {new_w}x{new_h}")
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if mode == 'aggressive':
        # For very old/damaged documents
        print("Mode: AGGRESSIVE")
        
        # Heavy denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
        
        # Auto level
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Heavy sharpening
        kernel_sharp = np.array([[-1,-1,-1,-1,-1],
                                 [-1, 2, 2, 2,-1],
                                 [-1, 2, 8, 2,-1],
                                 [-1, 2, 2, 2,-1],
                                 [-1,-1,-1,-1,-1]]) / 8.0
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        
        # Binary threshold
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        
    elif mode == 'light':
        # For good quality but need slight improvement
        print("Mode: LIGHT")
        
        # Light denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
        
        # Sharpening
        kernel_sharp = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharp)
        
        # Adaptive threshold
        result = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
    else:  # auto
        print("Mode: AUTO")
        
        # Standard preprocessing
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Sharpening
        kernel_sharp = np.array([[-1,-1,-1],[-1,10,-1],[-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        
        # Adaptive threshold
        result = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 15, 3)
    
    # Save
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python fix_image.py input.jpg [output.jpg] [mode]")
        print("  mode: auto (default), aggressive, light")
        print("\nExamples:")
        print("  python fix_image.py 001.jpg 001_fixed.jpg")
        print("  python fix_image.py 001.jpg 001_fixed.jpg aggressive")
        return
    
    input_path = sys.argv[1]
    
    # Auto generate output name if not provided
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_fixed{p.suffix}")
    
    mode = sys.argv[3] if len(sys.argv) >= 4 else 'auto'
    
    if mode not in ['auto', 'aggressive', 'light']:
        print(f"Invalid mode: {mode}")
        return
    
    success = fix_image(input_path, output_path, mode)
    
    if success:
        print("\n✅ Done!")
        print(f"\nNow run:")
        print(f"  python main.py {output_path}")

if __name__ == "__main__":
    main()