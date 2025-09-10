import psutil
import gc
from typing import Dict, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class MemoryInfo:
    """Struttura dati per info memoria"""
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    critical: bool
    
class MemoryMonitor:
    """Monitor memoria sistema ottimizzato per M4"""
    
    def __init__(self, warning_threshold: float = 85.0, critical_threshold: float = 95.0):
        """
        Inizializza monitor memoria.
        
        Args:
            warning_threshold: Soglia warning % memoria
            critical_threshold: Soglia critica % memoria
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
    def get_memory_info(self) -> MemoryInfo:
        """Ottieni informazioni dettagliate memoria"""
        memory = psutil.virtual_memory()
        
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        used_gb = memory.used / (1024**3)
        percent_used = memory.percent
        
        return MemoryInfo(
            total_gb=round(total_gb, 1),
            available_gb=round(available_gb, 1),
            used_gb=round(used_gb, 1),
            percent_used=round(percent_used, 1),
            critical=percent_used > self.critical_threshold
        )
    
    def check_memory_for_dataframe(self, df_size_mb: float) -> Dict[str, any]:
        """
        Controlla se c'è abbastanza memoria per caricare un DataFrame.
        
        Args:
            df_size_mb: Dimensione stimata DataFrame in MB
            
        Returns:
            Dict con risultato controllo e suggerimenti
        """
        memory_info = self.get_memory_info()
        available_mb = memory_info.available_gb * 1024
        
        # Considera overhead: DataFrame + agente + Streamlit + LM Studio
        overhead_factor = 2.5  # Fattore sicurezza per M4
        required_mb = df_size_mb * overhead_factor
        
        can_load = available_mb > required_mb
        
        result = {
            "can_load": can_load,
            "required_mb": round(required_mb, 1),
            "available_mb": round(available_mb, 1),
            "overhead_factor": overhead_factor,
            "memory_info": memory_info
        }
        
        # Aggiungi suggerimenti se memoria insufficiente
        if not can_load:
            result["suggestions"] = [
                "Chiudi altre applicazioni per liberare memoria",
                "Prova con un file CSV più piccolo",
                "Riavvia LM Studio se ha memory leak",
                f"File ottimale: max {available_mb // overhead_factor:.0f}MB"
            ]
        
        return result
    
    def cleanup_memory(self):
        """Pulizia memoria Python"""
        gc.collect()  # Garbage collection manuale
        
    def get_process_memory(self, process_name: str) -> Optional[float]:
        """Ottieni memoria usata da processo specifico (es. LM Studio)"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                if process_name.lower() in proc.info['name'].lower():
                    return proc.info['memory_info'].rss / (1024**2)  # MB
        except:
            pass
        return None

# Istanza globale
memory_monitor = MemoryMonitor()
