import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import warnings

class DataFrameOptimizer:
    """Ottimizza DataFrame per ridurre uso memoria"""
    
    def __init__(self):
        self.optimization_log: List[str] = []
        
    def optimize_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Ottimizza DataFrame riducendo uso memoria.
        
        Args:
            df: DataFrame originale
            
        Returns:
            Tuple di (DataFrame ottimizzato, statistiche ottimizzazione)
        """
        original_memory = df.memory_usage(deep=True).sum()
        self.optimization_log = []
        optimized_df = df.copy()
        
        # 1. Ottimizza colonne numeriche intere
        optimized_df = self._optimize_integers(optimized_df)
        
        # 2. Ottimizza colonne float
        optimized_df = self._optimize_floats(optimized_df)
        
        # 3. Ottimizza colonne object/string
        optimized_df = self._optimize_objects(optimized_df)
        
        # 4. Converti a category dove appropriato
        optimized_df = self._optimize_categories(optimized_df)
        
        final_memory = optimized_df.memory_usage(deep=True).sum()
        reduction_percent = ((original_memory - final_memory) / original_memory) * 100
        
        stats = {
            "original_memory_mb": round(original_memory / (1024**2), 2),
            "optimized_memory_mb": round(final_memory / (1024**2), 2),
            "reduction_mb": round((original_memory - final_memory) / (1024**2), 2),
            "reduction_percent": round(reduction_percent, 1),
            "optimization_log": self.optimization_log.copy()
        }
        
        return optimized_df, stats
    
    def _optimize_integers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ottimizza colonne integer scegliendo tipo più piccolo possibile"""
        int_cols = df.select_dtypes(include=['int64', 'int32']).columns
        
        for col in int_cols:
            col_min, col_max = df[col].min(), df[col].max()
            
            # Prova tipi unsigned se tutti valori >= 0
            if col_min >= 0:
                if col_max <= 255:
                    df[col] = df[col].astype('uint8')
                    self.optimization_log.append(f"{col}: int64 → uint8")
                elif col_max <= 65535:
                    df[col] = df[col].astype('uint16')
                    self.optimization_log.append(f"{col}: int64 → uint16")
                elif col_max <= 4294967295:
                    df[col] = df[col].astype('uint32')
                    self.optimization_log.append(f"{col}: int64 → uint32")
            
            # Prova tipi signed
            else:
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype('int8')
                    self.optimization_log.append(f"{col}: int64 → int8")
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype('int16')
                    self.optimization_log.append(f"{col}: int64 → int16")
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype('int32')
                    self.optimization_log.append(f"{col}: int64 → int32")
        
        return df
    
    def _optimize_floats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ottimizza colonne float convertendo a float32 dove possibile"""
        float_cols = df.select_dtypes(include=['float64']).columns
        
        for col in float_cols:
            # Controlla se la precisione float32 è sufficiente
            original_values = df[col].dropna()
            if len(original_values) > 0:
                # Converti a float32 e riconverti per check precisione
                float32_values = original_values.astype('float32').astype('float64')
                
                # Se la differenza è trascurabile, usa float32
                max_diff = np.abs(original_values - float32_values).max()
                if max_diff < 1e-6:  # Soglia tolleranza
                    df[col] = df[col].astype('float32')
                    self.optimization_log.append(f"{col}: float64 → float32")
        
        return df
    
    def _optimize_objects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ottimizza colonne object convertendo stringhe corte"""
        obj_cols = df.select_dtypes(include=['object']).columns
        
        for col in obj_cols:
            # Se tutti i valori sono stringhe corte, prova string dtype
            if df[col].dtype == 'object':
                try:
                    # Calcola lunghezza media stringa
                    str_lengths = df[col].dropna().astype(str).str.len()
                    avg_length = str_lengths.mean() if len(str_lengths) > 0 else 0
                    
                    # Se stringhe corte, usa string dtype più efficiente
                    if avg_length < 50:
                        df[col] = df[col].astype('string')
                        self.optimization_log.append(f"{col}: object → string")
                except:
                    pass
        
        return df
    
    def _optimize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converti colonne con pochi valori unici a category"""
        # Considera solo colonne object/string
        candidate_cols = df.select_dtypes(include=['object', 'string']).columns
        
        for col in candidate_cols:
            unique_ratio = df[col].nunique() / len(df)
            
            # Se < 50% valori unici, converti a category
            if unique_ratio < 0.5:
                original_memory = df[col].memory_usage(deep=True)
                df[col] = df[col].astype('category')
                new_memory = df[col].memory_usage(deep=True)
                
                if new_memory < original_memory:
                    reduction = ((original_memory - new_memory) / original_memory) * 100
                    self.optimization_log.append(f"{col}: string → category (-{reduction:.1f}%)")
                else:
                    # Riporta indietro se non conviene
                    df[col] = df[col].astype('string')
        
        return df
    
    def get_memory_report(self, df: pd.DataFrame) -> Dict[str, any]:
        """Genera report dettagliato memoria DataFrame"""
        memory_usage = df.memory_usage(deep=True)
        
        return {
            "total_memory_mb": round(memory_usage.sum() / (1024**2), 2),
            "index_memory_mb": round(memory_usage.iloc[0] / (1024**2), 2),
            "columns_memory": {
                col: round(memory_usage[col] / (1024**2), 2) 
                for col in df.columns
            },
            "dtypes": dict(df.dtypes),
            "shape": df.shape,
            "null_counts": dict(df.isnull().sum())
        }

# Istanza globale
optimizer = DataFrameOptimizer()
