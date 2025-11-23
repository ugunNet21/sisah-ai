# src/models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List

class IjazahData(BaseModel):
    nama_lengkap: Optional[str] = Field(None, description="Nama mahasiswa")
    nomor_ijazah: Optional[str] = Field(None, description="Nomor Ijazah Nasional/PIN")
    nama_pt: Optional[str] = Field(None, description="Nama Perguruan Tinggi")
    program_studi: Optional[str] = Field(None, description="Jurusan/Prodi")
    ipk: Optional[float] = Field(None, description="Indeks Prestasi Kumulatif")
    tanggal_lulus: Optional[str] = Field(None, description="Tanggal kelulusan format YYYY-MM-DD")

class ValidationReport(BaseModel):
    is_valid_structure: bool
    forensic_check: str
    extracted_data: IjazahData
    llm_analysis: str