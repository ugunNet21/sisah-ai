# src/utils/csv_exporter.py

import pandas as pd
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CSVExporter:
    def __init__(self, csv_path: str = "./results/scan_history.csv"):
        self.csv_path = csv_path
        
        # Create results directory if not exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Initialize CSV with headers if not exists
        if not os.path.exists(csv_path):
            self._create_csv()
    
    def _create_csv(self):
        """Create CSV with headers"""
        df = pd.DataFrame(columns=[
            'timestamp',
            'filename',
            'document_type',
            'forensic_status',
            'forensic_score',
            'anomaly_percentage',
            'ocr_confidence',
            'nama_lengkap',
            'nomor_ijazah',
            'nama_pt',
            'program_studi',
            'ipk',
            'tanggal_lulus',
            'gelar',
            'processing_status'
        ])
        df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"CSV file created: {self.csv_path}")
    
    def export_result(self, scan_result: dict):
        """
        Export hasil scan ke CSV
        
        Args:
            scan_result: dict containing all scan data
        """
        try:
            # Read existing CSV
            df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            
            # Prepare new row
            new_row = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'filename': scan_result.get('filename', ''),
                'document_type': scan_result.get('document_type', ''),
                'forensic_status': scan_result.get('forensic', {}).get('status', ''),
                'forensic_score': scan_result.get('forensic', {}).get('score', 0),
                'anomaly_percentage': scan_result.get('forensic', {}).get('anomaly_percentage', 0),
                'ocr_confidence': scan_result.get('ocr_confidence', {}).get('average_confidence', 0),
                'nama_lengkap': scan_result.get('data', {}).get('nama_lengkap', ''),
                'nomor_ijazah': scan_result.get('data', {}).get('nomor_ijazah', ''),
                'nama_pt': scan_result.get('data', {}).get('nama_pt', ''),
                'program_studi': scan_result.get('data', {}).get('program_studi', ''),
                'ipk': scan_result.get('data', {}).get('ipk', ''),
                'tanggal_lulus': scan_result.get('data', {}).get('tanggal_lulus', ''),
                'gelar': scan_result.get('data', {}).get('gelar', ''),
                'processing_status': scan_result.get('status', 'COMPLETED')
            }
            
            # Append to dataframe
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save to CSV
            df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"Result exported to CSV: {self.csv_path}")
            
        except Exception as e:
            logger.error(f"CSV export error: {e}", exc_info=True)
    
    def get_statistics(self) -> dict:
        """Get statistics from CSV"""
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            
            return {
                'total_scans': len(df),
                'suspicious_count': len(df[df['forensic_status'].str.contains('SUSPICIOUS', na=False)]),
                'avg_confidence': df['ocr_confidence'].mean(),
                'document_types': df['document_type'].value_counts().to_dict()
            }
        except Exception as e:
            logger.error(f"Statistics error: {e}")
            return {}