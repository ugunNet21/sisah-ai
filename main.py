# main.py

import os
import sys
from dotenv import load_dotenv
from src.services.ocr_engine import OCREngine
from src.services.llm_client import OllamaClient
from src.core.forensic import ForensicAnalyzer

# Load env
load_dotenv()

def main():
    print(f"\n=== {os.getenv('APP_NAME')} ===")
    
    # 1. Ambil input path gambar dari argument command line
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_image>")
        print("Example: python main.py ijazah_sample.jpg")
        return

    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print("❌ Error: File gambar tidak ditemukan!")
        return

    # Inisialisasi Service
    ocr = OCREngine()
    llm = OllamaClient()
    forensics = ForensicAnalyzer()

    # --- STEP 1: FORENSIK ---
    print("\n🔍 [1/3] Menganalisis Integritas Digital...")
    ela_score = forensics.analyze_ela(image_path)
    threshold = float(os.getenv("ELA_THRESHOLD", 15.0))
    
    forensic_status = "✅ AMAN (Original)"
    if ela_score > threshold:
        forensic_status = f"⚠️ WARNING (Terindikasi Editan - Score: {ela_score})"
    else:
        forensic_status += f" (Score: {ela_score})"
        
    print(f"   -> Status: {forensic_status}")

    # --- STEP 2: OCR ---
    print("\n📖 [2/3] Mengekstrak Teks (OCR)...")
    raw_text = ocr.extract_text(image_path)
    if not raw_text:
        print("❌ Gagal membaca teks. Pastikan gambar jelas.")
        return
    print(f"   -> Teks Mentah (Preview): {raw_text[:80].replace(chr(10), ' ')}...")

    # --- STEP 3: LLM PARSING ---
    print("\n🧠 [3/3] Mengolah Data dengan AI...")
    parsed_data = llm.parse_ijazah_text(raw_text)

    # --- HASIL AKHIR ---
    print("\n" + "="*50)
    print("   LAPORAN VERIFIKASI IJAZAH   ")
    print("="*50)
    print(f"INTEGRITAS FILE : {forensic_status}")
    print("-" * 50)
    print(f"Nama Mahasiswa  : {parsed_data.get('nama_lengkap', '-')}")
    print(f"No. Ijazah      : {parsed_data.get('nomor_ijazah', '-')}")
    print(f"Perguruan Tinggi: {parsed_data.get('universitas', '-')}")
    print(f"Program Studi   : {parsed_data.get('prodi', '-')}")
    print(f"IPK             : {parsed_data.get('ipk', '-')}")
    print(f"Tgl Lulus       : {parsed_data.get('tanggal_lulus', '-')}")
    print("="*50)

if __name__ == "__main__":
    main()