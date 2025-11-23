# src/services/llm_client.py
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

class OllamaClient:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL")
        self.model = os.getenv("OLLAMA_MODEL", "llama3")

    def parse_ijazah_text(self, raw_text: str) -> dict:
        """
        Mengirim raw text dari OCR ke LLM untuk diekstrak menjadi JSON.
        """
        prompt = f"""
        Anda adalah asisten AI untuk verifikasi dokumen negara.
        Tugas: Ekstrak informasi dari teks Ijazah berikut yang mungkin berantakan (hasil OCR).
        
        Teks OCR:
        {raw_text}
        
        Instruksi Khusus:
        1. Cari Nama Lengkap, Nomor Ijazah (PIN), Nama Kampus, Prodi, dan IPK.
        2. Perbaiki typo minor (misal: 'Universttas' -> 'Universitas').
        3. JANGAN mengarang data jika tidak ada di teks. Isi dengan null.
        4. Output WAJIB hanya JSON murni tanpa teks pembuka/penutup.
        
        Format JSON Target:
        {{
            "nama_lengkap": "...",
            "nomor_ijazah": "...",
            "nama_pt": "...",
            "program_studi": "...",
            "ipk": 0.00,
            "tanggal_lulus": "YYYY-MM-DD"
        }}
        """

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"  # Fitur baru Ollama untuk memaksakan JSON
        }

        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            return json.loads(result['response'])
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return {}