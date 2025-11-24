# main.py

import os
import sys
from dotenv import load_dotenv
from src.services.ocr_engine import OCREngine
from src.services.llm_client import OllamaClient
from src.core.forensic import ForensicAnalyzer
from src.utils.logger import setup_logger
from src.utils.csv_exporter import CSVExporter
import logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)

def print_header():
    """Print application header"""
    print("\n" + "=" * 60)
    print(f"  {os.getenv('APP_NAME', 'SiSah-AI v2.0')}  ".center(60))
    print("  Sistem Verifikasi Keaslian Dokumen Pendidikan  ".center(60))
    print("=" * 60)

def print_forensic_result(forensic_data: dict):
    """Print forensic analysis results"""
    status = forensic_data.get('status', 'UNKNOWN')
    score = forensic_data.get('score', 0)
    anomaly = forensic_data.get('anomaly_percentage', 0)
    
    # Color-coded output
    if status == "ORIGINAL":
        icon = "✅"
        status_text = f"{icon} AMAN (Original)"
    elif status == "MINOR_EDIT":
        icon = "⚠️"
        status_text = f"{icon} MINOR EDIT (Score: {score:.1f})"
    elif status == "SUSPICIOUS":
        icon = "⚠️"
        status_text = f"{icon} SUSPICIOUS (Score: {score:.1f}, Anomaly: {anomaly:.2f}%)"
    else:
        icon = "🚨"
        status_text = f"{icon} HIGHLY SUSPICIOUS (Score: {score:.1f}, Anomaly: {anomaly:.2f}%)"
    
    print(f"   Status: {status_text}")
    print(f"   Detail: Max Error={score:.2f}, Mean={forensic_data.get('mean_score', 0):.2f}, Anomaly={anomaly:.2f}%")

def print_document_data(data: dict, doc_type: str):
    """Print extracted document data"""
    print("\n" + "=" * 60)
    print(f"  LAPORAN VERIFIKASI - {doc_type}  ".center(60))
    print("=" * 60)
    
    # SD/SMP/SMA/SMK
    if doc_type in ["IJAZAH_SD", "IJAZAH_SMP", "IJAZAH_SMA", "IJAZAH_SMK"]:
        print(f"Nama Lengkap     : {data.get('nama_lengkap') or '-'}")
        print(f"NISN             : {data.get('nisn') or '-'}")
        print(f"No. Ijazah       : {data.get('nomor_ijazah') or '-'}")
        print(f"Nama Sekolah     : {data.get('nama_sekolah') or '-'}")
        print(f"Tempat Lahir     : {data.get('tempat_lahir') or '-'}")
        print(f"Tanggal Lahir    : {data.get('tanggal_lahir') or '-'}")
        print(f"Nama Orang Tua   : {data.get('nama_ortu') or '-'}")
        print(f"Tanggal Lulus    : {data.get('tanggal_lulus') or '-'}")
        
        if doc_type == "IJAZAH_SMK" and data.get('kompetensi_keahlian'):
            print(f"Kompetensi       : {data.get('kompetensi_keahlian')}")
    
    # Perguruan Tinggi
    elif "IJAZAH" in doc_type and any(x in doc_type for x in ["D3", "S1", "S2", "S3"]):
        print(f"Nama Mahasiswa   : {data.get('nama_lengkap') or '-'}")
        print(f"NIM              : {data.get('nim') or '-'}")
        print(f"No. Ijazah       : {data.get('nomor_ijazah') or '-'}")
        print(f"Perguruan Tinggi : {data.get('nama_pt') or '-'}")
        print(f"Program Studi    : {data.get('program_studi') or '-'}")
        print(f"Jenjang          : {data.get('jenjang') or '-'}")
        print(f"IPK              : {data.get('ipk') or '-'}")
        print(f"Gelar            : {data.get('gelar') or '-'}")
        print(f"Tempat Lahir     : {data.get('tempat_lahir') or '-'}")
        print(f"Tanggal Lahir    : {data.get('tanggal_lahir') or '-'}")
        print(f"Tanggal Lulus    : {data.get('tanggal_lulus') or '-'}")
    else:
        print("Data ekstraksi untuk tipe dokumen ini masih dalam pengembangan.")
    
    print("=" * 60)

def main():
    print_header()
    
    # Validate arguments
    if len(sys.argv) < 2:
        print("\n❌ Error: File gambar tidak ditemukan!")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <path_to_image>")
        print("\nExample:")
        print(f"  python {sys.argv[0]} ijazah_sample.jpg")
        print(f"  python {sys.argv[0]} 001.bmp")
        return

    image_path = sys.argv[1]
    filename = os.path.basename(image_path)
    
    # Validate file exists
    if not os.path.exists(image_path):
        print(f"\n❌ Error: File tidak ditemukan: {image_path}")
        return
    
    logger.info(f"Processing file: {image_path}")
    
    # Initialize services
    ocr = OCREngine()
    llm = OllamaClient()
    forensics = ForensicAnalyzer()
    csv_exporter = CSVExporter()
    
    # Container untuk hasil
    scan_result = {
        'filename': filename,
        'status': 'FAILED'
    }
    
    try:
        # === STEP 1: FORENSIC ANALYSIS ===
        print("\n🔍 [1/4] Menganalisis Integritas Digital...")
        forensic_data = forensics.analyze_ela(image_path)
        scan_result['forensic'] = forensic_data
        print_forensic_result(forensic_data)
        
        # Metadata check
        metadata = forensics.check_metadata(image_path)
        if not metadata.get('has_exif', False):
            print("   ⚠️  Warning: Tidak ada metadata EXIF (kemungkinan sudah diedit)")
        
        # === STEP 2: OCR EXTRACTION ===
        print("\n📖 [2/4] Mengekstrak Teks (OCR)...")
        debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"
        raw_text = ocr.extract_text(image_path, debug=debug_mode)
        
        if not raw_text or len(raw_text) < 20:
            print("❌ Gagal membaca teks. Pastikan gambar jelas dan tidak terbalik.")
            logger.warning(f"OCR failed or returned insufficient text: {len(raw_text)} chars")
            scan_result['status'] = 'OCR_FAILED'
            csv_exporter.export_result(scan_result)
            return
        
        # Get OCR confidence
        confidence_data = ocr.get_confidence_data(image_path)
        scan_result['ocr_confidence'] = confidence_data
        
        print(f"   ✓ Teks berhasil diekstrak ({len(raw_text)} karakter)")
        print(f"   Confidence Score: {confidence_data['average_confidence']:.1f}%")
        
        if debug_mode:
            print(f"   Preview: {raw_text[:150].replace(chr(10), ' ')}...")
        
        # === STEP 3: DOCUMENT TYPE DETECTION ===
        print("\n🔎 [3/4] Mendeteksi Jenis Dokumen...")
        doc_type = llm.detect_document_type(raw_text)
        scan_result['document_type'] = doc_type
        print(f"   ✓ Jenis Dokumen: {doc_type}")
        
        # === STEP 4: LLM PARSING ===
        print("\n🧠 [4/4] Mengolah Data dengan AI...")
        
        if "IJAZAH" in doc_type:
            parsed_data = llm.parse_document(raw_text, doc_type)
            scan_result['data'] = parsed_data
            scan_result['status'] = 'COMPLETED'
            
            # === DISPLAY RESULTS ===
            print("\n✅ Proses Selesai!")
            print_document_data(parsed_data, doc_type)
            
        else:
            print(f"   ⚠️  Dokumen jenis '{doc_type}' belum didukung untuk parsing detail.")
            scan_result['data'] = {}
            scan_result['status'] = 'UNSUPPORTED_TYPE'
        
        # === EXPORT TO CSV ===
        csv_exporter.export_result(scan_result)
        print(f"\n💾 Hasil disimpan ke: {csv_exporter.csv_path}")
        
        # Show statistics
        stats = csv_exporter.get_statistics()
        print(f"📊 Total Scan: {stats.get('total_scans', 0)} | Suspicious: {stats.get('suspicious_count', 0)}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Proses dibatalkan oleh user.")
        logger.info("Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Main process error: {e}", exc_info=True)
        scan_result['status'] = 'ERROR'
        csv_exporter.export_result(scan_result)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()