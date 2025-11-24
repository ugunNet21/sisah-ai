# src/services/llm_client.py
import requests
import json
import os
from dotenv import load_dotenv
import logging
import re

load_dotenv()
logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL")
        self.model = os.getenv("OLLAMA_MODEL", "llama3")
    
    def detect_document_type(self, raw_text: str) -> str:
        """
        Deteksi jenis dokumen dan level pendidikan
        """
        text_lower = raw_text.lower()
        text_clean = ' '.join(raw_text.split())
        
        # PT keywords check first (prioritas tertinggi)
        pt_keywords = ['universitas', 'institut teknologi', 'sekolah tinggi', 'stmik', 'stie', 'stkip', 'politeknik', 'akademi']
        has_pt = any(keyword in text_lower for keyword in pt_keywords)
        
        # Specific check untuk menghindari salah deteksi SMK vs STMIK
        if 'stmik' in text_lower or 'amik' in text_lower:
            has_pt = True
        
        # Gelar keywords untuk confirm PT
        gelar_pt = ['sarjana', 's.kom', 's.t', 's.e', 's.pd', 's.si', 'magister', 'm.kom', 'doktor', 'diploma', 'd3', 'ahli madya', 'a.md']
        has_gelar = any(keyword in text_lower for keyword in gelar_pt)
        
        if has_pt or has_gelar:
            # Check jenjang dari gelar atau keyword
            if any(keyword in text_lower for keyword in ['diploma', 'd3', 'd-3', 'd-iii', 'ahli madya', 'a.md', 'amd']):
                return "IJAZAH_D3"
            elif any(keyword in text_lower for keyword in ['magister', 's2', 's-2', 's-ii', 'm.kom', 'm.t', 'm.pd', 'm.si', 'master']):
                return "IJAZAH_S2"
            elif any(keyword in text_lower for keyword in ['doktor', 's3', 's-3', 's-iii', 'dr.', 'ph.d']):
                return "IJAZAH_S3"
            else:
                # Default S1 untuk PT
                return "IJAZAH_S1"
        
        # SMK - Check after PT untuk avoid confusion
        smk_keywords = ['sekolah menengah kejuruan', 'smk negeri', 'smk swasta', 
                        'kompetensi keahlian', 'bidang keahlian']
        if any(keyword in text_lower for keyword in smk_keywords):
            if 'transkrip' not in text_lower:
                return "IJAZAH_SMK"
        
        # SD/SMP/SMA
        if any(keyword in text_lower for keyword in ['sekolah dasar', 'sd negeri', 'sd swasta', 'madrasah ibtidaiyah']):
            return "IJAZAH_SD"
        if any(keyword in text_lower for keyword in ['sekolah menengah pertama', 'smp negeri', 'smp swasta', 'madrasah tsanawiyah']):
            return "IJAZAH_SMP"
        if any(keyword in text_lower for keyword in ['sekolah menengah atas', 'sma negeri', 'sma swasta', 'madrasah aliyah']):
            return "IJAZAH_SMA"
        
        # Transkrip
        if any(keyword in text_lower for keyword in ['transkrip', 'daftar nilai', 'kartu hasil studi', 'khs']):
            return "TRANSKRIP_NILAI"
        
        # Sertifikat
        if any(keyword in text_lower for keyword in ['sertifikat', 'certificate', 'pelatihan']):
            return "SERTIFIKAT"
        
        # Fallback
        return self._detect_via_llm(raw_text)
    
    def _detect_via_llm(self, raw_text: str) -> str:
        """Fallback detection menggunakan LLM"""
        prompt = f"""
        Analisis teks dan tentukan JENIS DOKUMEN.
        
        Teks:
        {raw_text[:500]}
        
        Pilihan: IJAZAH_SD, IJAZAH_SMP, IJAZAH_SMA, IJAZAH_SMK, IJAZAH_D3, IJAZAH_S1, IJAZAH_S2, IJAZAH_S3, TRANSKRIP_NILAI, SERTIFIKAT, TIDAK_DIKENALI
        
        Jawab HANYA dengan salah satu pilihan di atas.
        """
        
        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            response.raise_for_status()
            doc_type = response.json()['response'].strip().upper()
            logger.info(f"LLM detected document type: {doc_type}")
            return doc_type
        except Exception as e:
            logger.error(f"LLM detection error: {e}")
            return "TIDAK_DIKENALI"

    def parse_document(self, raw_text: str, doc_type: str) -> dict:
        """
        Parse dokumen sesuai tipe-nya
        """
        if "IJAZAH_SD" in doc_type or "IJAZAH_SMP" in doc_type or "IJAZAH_SMA" in doc_type or "IJAZAH_SMK" in doc_type:
            return self.parse_ijazah_menengah(raw_text, doc_type)
        elif "IJAZAH" in doc_type:  # S1, S2, S3, D3
            return self.parse_ijazah_tinggi(raw_text, doc_type)
        else:
            return self._empty_result()

    def parse_ijazah_menengah(self, raw_text: str, doc_type: str) -> dict:
        """
        Parser khusus untuk ijazah SD/SMP/SMA/SMK
        """
        prompt = f"""
Ekstrak data ijazah pendidikan menengah ({doc_type}) dari teks OCR berantakan ini.

TEKS OCR:
{raw_text}

CARI dengan fleksibel (teks mungkin terpotong/salah):
1. nama_lengkap: Nama siswa (biasanya HURUF KAPITAL, cari setelah "Nama" atau "yang bernama")
2. nomor_ijazah: Nomor Seri (format: DN-XX Aa XXXXXX atau angka panjang, cari "Nomor" atau "Seri")
3. nisn: 10 digit angka (biasanya ada label "NISN")
4. nama_sekolah: Nama sekolah lengkap (SD/SMP/SMA/SMK + Negeri/Swasta)
5. tempat_lahir: Kota kelahiran
6. tanggal_lahir: Tanggal lahir (format apapun, convert ke YYYY-MM-DD)
7. tanggal_lulus: Tanggal lulus (cari "lulus" atau "kelulusan")
8. nama_ortu: Nama ayah/orang tua

ATURAN:
- Teks OCR berantakan, cari pattern yang mirip
- Jika tidak ada, isi null
- Tanggal format YYYY-MM-DD

OUTPUT JSON:
{{
    "nama_lengkap": null,
    "nomor_ijazah": null,
    "nisn": null,
    "nama_sekolah": null,
    "tempat_lahir": null,
    "tanggal_lahir": null,
    "tanggal_lulus": null,
    "nama_ortu": null,
    "kompetensi_keahlian": null
}}
"""
        return self._send_to_llm(prompt, is_higher_ed=False)

    def parse_ijazah_tinggi(self, raw_text: str, doc_type: str) -> dict:
        """
        Parser untuk ijazah perguruan tinggi (D3/S1/S2/S3)
        """
        prompt = f"""
Anda ekstraksi data ijazah perguruan tinggi Indonesia ({doc_type}).

TEKS OCR:
{raw_text}

EKSTRAK DENGAN TELITI (jangan gunakan placeholder/contoh):

1. nama_lengkap: Cari nama mahasiswa (HURUF KAPITAL, bukan nomor)
2. nomor_ijazah: Cari "Nomor Ijazah" atau "Nomor Seri" (format: huruf+angka panjang)
3. nim: Nomor Induk Mahasiswa (biasanya 10 digit)
4. nama_pt: Cari nama institusi LENGKAP DAN SPESIFIK
   - Contoh BENAR: "STMIK Budi Darma Medan", "Universitas Indonesia"
   - Contoh SALAH: "Universitas/Institut/STMIK ...", "Sekolah Tinggi"
5. program_studi: Nama jurusan lengkap
6. jenjang: {doc_type.replace('IJAZAH_', '')}
7. ipk: Angka 0.00-4.00 (cari "IPK" atau "Indeks Prestasi")
8. gelar: S.Kom, S.T., M.Kom, dll
9. tempat_lahir: Kota kelahiran
10. tanggal_lahir: Format YYYY-MM-DD
11. tanggal_lulus: Format YYYY-MM-DD (cari "Lulus" atau "Wisuda")

ATURAN KRITIS:
- JANGAN ISI DENGAN "..." ATAU PLACEHOLDER
- JANGAN ISI DENGAN CONTOH DARI PROMPT INI
- Jika benar-benar tidak ada di teks, isi null
- Nomor ijazah BUKAN nama orang
- Nama PT harus spesifik dan lengkap

OUTPUT JSON (tanpa backticks):
{{
    "nama_lengkap": "cari di teks asli",
    "nomor_ijazah": "cari di teks asli",
    "nim": "cari di teks asli",
    "nama_pt": "cari nama institusi LENGKAP di teks",
    "program_studi": "cari di teks asli",
    "jenjang": "{doc_type.replace('IJAZAH_', '')}",
    "ipk": 0.00,
    "tanggal_lulus": "YYYY-MM-DD",
    "gelar": "cari di teks",
    "tempat_lahir": "cari di teks",
    "tanggal_lahir": "YYYY-MM-DD"
}}
"""
        return self._send_to_llm(prompt, is_higher_ed=True)

    def _send_to_llm(self, prompt: str, is_higher_ed: bool = False) -> dict:
        """Send prompt to Ollama and parse response"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "temperature": 0.05,  # Very low for consistency
                "top_p": 0.9,
                "num_predict": 500
            }
            
            logger.info("Sending request to Ollama...")
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=90)
            response.raise_for_status()
            
            result = response.json()
            raw_response = result['response']
            
            logger.debug(f"LLM Raw Response: {raw_response[:300]}")
            
            # Parse JSON
            parsed = json.loads(raw_response)
            
            # CRITICAL: Remove placeholder values
            parsed = self._remove_placeholders(parsed)
            
            # Post-processing validation
            parsed = self._validate_and_fix(parsed, is_higher_ed)
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Raw response: {raw_response}")
            return self._empty_result()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection error: {e}")
            return self._empty_result()
        except Exception as e:
            logger.error(f"LLM parsing error: {e}", exc_info=True)
            return self._empty_result()
    
    def _remove_placeholders(self, data: dict) -> dict:
        """Remove placeholder values yang tidak valid"""
        placeholder_patterns = [
            'nama lengkap', 'nama mahasiswa', 'cari di teks',
            '123456789', '1234567890', '...', 
            'universitas/institut', 'sekolah tinggi',
            'teknik informatika',  # Generic prodi
            's.kom', 's.t', 's.e',  # Generic gelar tanpa context
            '2022-10-31', 'yyyy-mm-dd',  # Template dates
            'jakarta'  # Generic location
        ]
        
        for key, value in data.items():
            if value and isinstance(value, str):
                val_lower = str(value).lower().strip()
                # Check if value is placeholder
                if any(pattern in val_lower for pattern in placeholder_patterns):
                    # Hanya null jika benar-benar placeholder murni
                    if val_lower in placeholder_patterns or len(val_lower) < 3:
                        logger.warning(f"Removing placeholder value for {key}: {value}")
                        data[key] = None
        
        return data
    
    def _validate_and_fix(self, data: dict, is_higher_ed: bool) -> dict:
        """
        Validasi dan perbaiki hasil parsing
        """
        # Fix 1: Deteksi jika nama dan nomor tertukar
        if is_higher_ed and 'nama_lengkap' in data and 'nomor_ijazah' in data:
            nama = str(data.get('nama_lengkap', '') or '')
            nomor = str(data.get('nomor_ijazah', '') or '')
            
            # Jika nama berisi banyak angka, kemungkinan salah
            if nama and len(re.findall(r'\d', nama)) > len(nama) * 0.5:
                logger.warning(f"Detected swapped name-number. Swapping: '{nama}' <-> '{nomor}'")
                data['nama_lengkap'], data['nomor_ijazah'] = nomor, nama
        
        # Fix 2: Validasi IPK
        if 'ipk' in data and data['ipk']:
            try:
                ipk = float(data['ipk'])
                if ipk < 0 or ipk > 4:
                    logger.warning(f"Invalid IPK range: {ipk}")
                    data['ipk'] = None
                else:
                    data['ipk'] = round(ipk, 2)
            except:
                data['ipk'] = None
        
        # Fix 3: Validasi NISN (harus 10 digit)
        if 'nisn' in data and data['nisn']:
            nisn = str(data['nisn']).replace('-', '').replace(' ', '')
            if not nisn.isdigit() or len(nisn) != 10:
                logger.warning(f"Invalid NISN: {data['nisn']}")
                data['nisn'] = None
            else:
                data['nisn'] = nisn
        
        # Fix 4: Validasi nama PT
        if 'nama_pt' in data and data['nama_pt']:
            pt_name = str(data['nama_pt'])
            # Check for generic/placeholder
            if any(x in pt_name for x in ['...', 'Universitas/Institut', 'STMIK ...', 'Sekolah Tinggi ...']):
                logger.warning(f"Generic PT name detected: {pt_name}")
                data['nama_pt'] = None
            # Check if too short (likely error)
            elif len(pt_name) < 10:
                logger.warning(f"PT name too short: {pt_name}")
                data['nama_pt'] = None
        
        # Fix 5: Normalize None values
        for key in data.keys():
            if data[key] in ["", "-", "null", "None", "N/A"]:
                data[key] = None
        
        return data
    
    def _empty_result(self) -> dict:
        """Template hasil kosong"""
        return {
            "nama_lengkap": None,
            "nomor_ijazah": None,
            "nama_sekolah": None,
            "nama_pt": None,
            "program_studi": None,
            "ipk": None,
            "tanggal_lulus": None
        }