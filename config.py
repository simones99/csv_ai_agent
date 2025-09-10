import os
from typing import Dict, Any

class Config:
    """Configurazione principale dell'applicazione"""
    
    # Configurazioni LM Studio
    LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
    LM_STUDIO_MODEL_NAME = "local-model"
    
    # Limiti per MacBook M4 Base (16GB RAM)
    MAX_FILE_SIZE_MB = 500  # Massimo 500MB per file CSV
    MAX_DATAFRAME_MEMORY_MB = 200  # Limite memoria DataFrame
    MAX_CONVERSATION_HISTORY = 10  # Mantieni solo 10 conversazioni
    
    # Configurazioni LLM
    LLM_TEMPERATURE = 0.1  # Bassa per analisi dati consistenti
    LLM_MAX_TOKENS = 1500  # Ottimale per M4 base
    LLM_TIMEOUT = 120  # 2 minuti timeout
    
    # Configurazioni Streamlit
    PAGE_TITLE = "🤖 CSV AI Agent"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    
    # Percorsi
    DATA_UPLOAD_PATH = "data/uploads"
    SAMPLE_DATA_PATH = "data/samples"
    ASSETS_PATH = "assets"
    
    # Configurazioni sicurezza
    ALLOWED_FILE_TYPES = ["csv"]
    SAFE_MODE = True  # Abilita controlli sicurezza extra
    
    @classmethod
    def get_system_info(cls) -> Dict[str, Any]:
        """Ottieni informazioni sistema per debugging"""
        import psutil
        import platform
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 1)
        }

# Istanza globale configurazione
config = Config()
