# src/services/llm_client.py
import requests
import json
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL")
        self.model = os.getenv("OLLAMA_MODEL", "llama3")
    
    def detect_document_type(self, raw_text: str) -> str:
        """
        Deteksi jenis dokumen dari teks OCR
        """
        prompt = f"""
        Analisis teks berikut dan tentukan JENIS DOKUMEN-nya.
        
        Teks:
        {raw_text[:500]}
        
        Pilihan jenis dokumen:
        - IJAZAH_S1 (Ijazah Sarjana/S1)
        - IJAZAH_D3 (Ijazah Diploma)
        - IJAZAH_S2 (Ijazah Magister)
        - IJAZAH_S3 (Ijazah Doktor)
        - TRANSKRIP_NILAI
        - SERTIFIKAT
        - SURAT_KETERANGAN
        - KTP
        - TIDAK_DIKENALI
        
        Jawab HANYA dengan salah satu pilihan di atas, tanpa penjelasan tambahan.
        """
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            doc_type = result['response'].strip().upper()
            
            logger.info(f"Detected document type: {doc_type}")
            return doc_type
            
        except Exception as e:
            logger.error(f"Document type detection error: {e}")
            return "TIDAK_DIKENALI"

    def parse_ijazah_text(self, raw_text: str) -> dict:
        """
        Parse teks ijazah menjadi structured data
        """
        prompt = f"""
            Anda adalah sistem AI untuk ekstraksi data ijazah perguruan tinggi Indonesia.

            TEKS OCR (mungkin berantakan):
            {raw_text}

            TUGAS ANDA:
            1. Ekstrak data berikut dari teks di atas:
            - nama_lengkap: Nama mahasiswa (HURUF KAPITAL biasanya)
            - nomor_ijazah: Nomor Ijazah/Nomor Seri (kombinasi angka dan huruf, contoh: 107x722, S82012022000216)
            - nama_pt: Nama Universitas/Institut/Sekolah Tinggi LENGKAP
            - program_studi: Nama program studi/jurusan
            - ipk: IPK dalam format angka desimal (contoh: 3.45)
            - tanggal_lulus: Tanggal kelulusan format YYYY-MM-DD
            - gelar: Gelar akademik (S.Kom, S.T., S.E., dll)

            2. ATURAN PENTING:
            - Nomor ijazah BUKAN nama orang
            - Perbaiki typo umum (Universttas → Universitas, Managemen → Manajemen)
            - Jika data tidak ditemukan, isi dengan null (bukan string kosong)
            - Nama PT harus lengkap (jangan singkat)
            - IPK antara 0.00 - 4.00

            3. OUTPUT FORMAT:
            Harus JSON murni tanpa backticks atau teks pembuka/penutup:
            {{
                "nama_lengkap": "NAMA LENGKAP",
                "nomor_ijazah": "123456789",
                "nama_pt": "Universitas/Institut/STMIK ...",
                "program_studi": "Teknik Informatika",
                "ipk": 3.45,
                "tanggal_lulus": "2022-10-31",
                "gelar": "S.Kom"
            }}
        """

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "temperature": 0.1  # Rendah untuk konsistensi
            }
            
            logger.info("Sending request to Ollama...")
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            raw_response = result['response']
            
            logger.debug(f"LLM Raw Response: {raw_response[:200]}")
            
            # Parse JSON
            parsed = json.loads(raw_response)
            
            # Validasi dan normalisasi
            parsed = self._normalize_data(parsed)
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return self._empty_result()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection error: {e}")
            return self._empty_result()
        except Exception as e:
            logger.error(f"LLM parsing error: {e}", exc_info=True)
            return self._empty_result()
    
    def _normalize_data(self, data: dict) -> dict:
        """Normalisasi dan validasi data hasil parsing"""
        # IPK validation
        if 'ipk' in data and data['ipk']:
            try:
                ipk = float(data['ipk'])
                if ipk < 0 or ipk > 4:
                    data['ipk'] = None
                else:
                    data['ipk'] = round(ipk, 2)
            except:
                data['ipk'] = None
        
        # Tanggal validation
        if 'tanggal_lulus' in data and data['tanggal_lulus']:
            if len(str(data['tanggal_lulus'])) < 8:
                data['tanggal_lulus'] = None
        
        # Normalize None values
        for key in ['nama_lengkap', 'nomor_ijazah', 'nama_pt', 'program_studi', 'gelar']:
            if key not in data or data[key] == "" or data[key] == "-":
                data[key] = None
        
        return data
    
    def _empty_result(self) -> dict:
        """Template hasil kosong"""
        return {
            "nama_lengkap": None,
            "nomor_ijazah": None,
            "nama_pt": None,
            "program_studi": None,
            "ipk": None,
            "tanggal_lulus": None,
            "gelar": None
        }