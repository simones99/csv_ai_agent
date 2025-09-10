import pandas as pd
import requests
import json
from typing import Optional, Tuple, Dict, Any, List
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

from config import config
from utils.memory_monitor import memory_monitor
from utils.data_optimizer import optimizer
from agents.tools import CSVAnalysisTools
from agents.prompts import PromptTemplates

class StreamlitCallbackHandler(BaseCallbackHandler):
    """
    Callback handler per mostrare progresso in Streamlit.
    """
    
    def __init__(self, container):
        self.container = container
        self.step_count = 0
        
    def on_agent_action(self, action, **kwargs):
        """Chiamato quando l'agente esegue un'azione"""
        self.step_count += 1
        self.container.write(f"🔄 Step {self.step_count}: {action.tool}")
        
    def on_tool_end(self, output, **kwargs):
        """Chiamato quando un tool finisce"""
        self.container.write(f"✅ Done")

class CSVAgent:
    """Agente AI principale per analisi CSV con LM Studio locale"""
    
    def __init__(self):
        """Inizializza agente con configurazioni ottimizzate"""
        self.llm = self._setup_llm()
        self.agent = None
        self.df = None
        self.conversation_history: List[Dict] = []
        self.tools = CSVAnalysisTools()
        self.prompt_templates = PromptTemplates()
        
        # Verifica connessione all'avvio
        self.connection_status = self._test_connection()
        
    def _setup_llm(self) -> ChatOpenAI:
        """
        Configura LLM con LM Studio backend.
        
        Configurazioni ottimizzate per Qwen3 4B su M4:
        - Temperature bassa per analisi dati consistenti
        - Max tokens bilanciato per performance/qualità
        - Timeout generoso per hardware entry-level
        """
        return ChatOpenAI(
            base_url=config.LM_STUDIO_BASE_URL,
            api_key="not-needed",  # LM Studio non richiede API key
            model=config.LM_STUDIO_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            timeout=config.LLM_TIMEOUT,
            streaming=True,  # Abilita streaming per feedback real-time
            callbacks=[]  # Aggiunto dinamicamente quando necessario
        )
    
    def _test_connection(self) -> Dict[str, Any]:
        """
        Testa connessione a LM Studio con diagnostica avanzata.
        
        Returns:
            Dict con status connessione e info diagnostiche
        """
        try:
            # Test semplice prima
            response = requests.get(f"{config.LM_STUDIO_BASE_URL.replace('/v1', '')}/health", timeout=5)
            if response.status_code in (200, 404):  # Accetta anche 200
                return {"connected": True, "status": "healthy"}
            else:
                # Fallback: testa endpoint chat
                test_payload = {
                    "model": config.LM_STUDIO_MODEL_NAME,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                }
                
                response = requests.post(
                    f"{config.LM_STUDIO_BASE_URL}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=test_payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    return {
                        "connected": True,
                        "status": "healthy",
                        "model_loaded": True,
                        "response_time_ms": response.elapsed.total_seconds() * 1000
                    }
                else:
                    return {
                        "connected": False,
                        "status": "error",
                        "error_code": response.status_code,
                        "error_message": response.text
                    }
        except requests.exceptions.ConnectionError:
            return {
                "connected": False,
                "status": "connection_refused",
                "error_message": "LM Studio non raggiungibile. Verificare che sia avviato."
            }
        except requests.exceptions.Timeout:
            return {
                "connected": False,
                "status": "timeout", 
                "error_message": "LM Studio risponde troppo lentamente."
            }
        except Exception as e:
            return {
                "connected": False,
                "status": "unknown_error",
                "error_message": str(e)
            }
    
    def load_csv(self, file_path: str, progress_callback=None) -> Tuple[bool, str, Optional[Dict]]:
        """
        Carica e ottimizza CSV con controlli memoria avanzati.
        
        Args:
            file_path: Percorso file CSV
            progress_callback: Callback per aggiornamenti progresso
            
        Returns:
            Tuple di (successo, messaggio, statistiche)
        """
        try:
            if progress_callback:
                progress_callback("📁 Loading CSV file...")
            
            # 1. Carica CSV con chunking per file grandi
            try:
                # Prima, leggi solo le prime righe per stima dimensioni
                sample_df = pd.read_csv(file_path, nrows=1000)
                estimated_size_mb = (sample_df.memory_usage(deep=True).sum() / len(sample_df)) * \
                                  sum(1 for _ in open(file_path)) / (1024**2)
                
                # Controllo memoria preliminare
                memory_check = memory_monitor.check_memory_for_dataframe(estimated_size_mb)
                if not memory_check["can_load"]:
                    return False, f"❌ File too large ({estimated_size_mb:.1f}MB). {memory_check['suggestions'][0]}", None
                
                # Carica file completo
                if progress_callback:
                    progress_callback("🔄 Reading complete data...")
                
                self.df = pd.read_csv(file_path)
                
            except pd.errors.EmptyDataError:
                return False, "❌ Empty CSV file", None
            except pd.errors.ParserError as e:
                return False, f"❌ CSV parsing error: {str(e)}", None
            
            if progress_callback:
                progress_callback("⚡ Optimizing memory...")
            
            # 2. Ottimizza DataFrame
            optimized_df, optimization_stats = optimizer.optimize_dataframe(self.df)
            self.df = optimized_df
            
            # 3. Validazioni base
            if len(self.df) == 0:
                return False, "❌ Dataset empty after loading", None
            
            if len(self.df.columns) == 0:
                return False, "❌ No columns found", None
            
            if progress_callback:
                progress_callback("🤖 Creating AI agent...")
            
            # 4. Crea agente pandas specializzato
            self.agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=self.df,
                verbose=False,  # Disabilita per output pulito
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Più stabile di OPENAI_FUNCTIONS
                allow_dangerous_code=True,  # Necessario per operazioni pandas
                prefix=self.prompt_templates.get_agent_prefix(self.df),
                max_iterations=3,  # Limita iterazioni per evitare loop
                early_stopping_method="generate",
                handle_parsing_errors=True  # Nuovo: gestisce errori parsing output
            )
            
            # 5. Reset cronologia
            self.conversation_history = []
            
            # 6. Genera statistiche complete
            stats = {
                "dataset_info": {
                    "rows": len(self.df),
                    "columns": len(self.df.columns),
                    "memory_mb": optimization_stats["optimized_memory_mb"],
                    "null_values": int(self.df.isnull().sum().sum()),
                    "duplicate_rows": int(self.df.duplicated().sum())
                },
                "optimization": optimization_stats,
                "memory_check": memory_check,
                "column_info": {
                    "numeric": list(self.df.select_dtypes(include=['number']).columns),
                    "categorical": list(self.df.select_dtypes(include=['object', 'category']).columns),
                    "datetime": list(self.df.select_dtypes(include=['datetime']).columns)
                }
            }
            
            if progress_callback:
                progress_callback("✅ Upload complete!")
            
            success_msg = (
                f"✅ Dataset uploaded successfully!\n"
                f"📊 {stats['dataset_info']['rows']:,} rows × {stats['dataset_info']['columns']} columns\n"
                f"💾 Optimized memory: {stats['optimization']['reduction_percent']:.1f}% reduction"
            )
            
            return True, success_msg, stats
            
        except Exception as e:
            return False, f"❌ Unexpected error: {str(e)}", None
    
    def analyze(self, query: str, progress_callback=None) -> Dict[str, Any]:
        """
        Analyze data based on user query.
        
        Args:
            query: Domanda/richiesta dell'utente
            progress_callback: Callback per aggiornamenti progresso
            
        Returns:
            Dict con risultato analisi e metadati
        """
        if not self.agent or self.df is None:
            return {
                "success": False,
                "result": "❌ No dataset loaded. Please upload a CSV file first.",
                "metadata": {}
            }
        
        try:
            if progress_callback:
                progress_callback("🧠 The AI agent is thinking...")
            
            # Prepara query con contesto
            enhanced_query = self._enhance_query(query)
            
            if progress_callback:
                progress_callback("⚙️ Running analysis...")
            
            # Esegui analisi con callback per streaming
            response = self.agent.invoke({
                "input": enhanced_query
            })
            
            result = response.get("output", "No response generated")
            
            # Salva nella cronologia con limite
            self._add_to_history(query, result)
            
            if progress_callback:
                progress_callback("✅ Analysis complete!")
            
            return {
                "success": True,
                "result": result,
                "metadata": {
                    "query_length": len(query),
                    "response_length": len(result),
                    "conversation_count": len(self.conversation_history),
                    "dataset_shape": self.df.shape
                }
            }
            
        except Exception as e:
            error_msg = f"❌ Analysis error: {str(e)}"
            
            # Log errore per debugging
            print(f"Agent error: {e}")
            
            return {
                "success": False,
                "result": error_msg,
                "metadata": {"error_type": type(e).__name__}
            }
    
    def _enhance_query(self, query: str) -> str:
        """
        Enhance query with useful context (ENGLISH).
        
        Strategia:
        - Aggiunge info dataset per orientare l'agente
        - Include cronologia recente per contesto
        - Guida l'agente verso risposte strutturate
        """
        # Info dataset essenziali
        dataset_context = (
            f"Current dataset: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns.\n"
            f"Columns: {', '.join(self.df.columns[:8])}"
        )
        
        if len(self.df.columns) > 8:
            dataset_context += f"... (and {len(self.df.columns) - 8} more columns)"
        
        # Cronologia recente (ultime 2 conversazioni)
        recent_context = ""
        if len(self.conversation_history) > 0:
            recent_context = "\n\nRecent conversation context:\n"
            for item in self.conversation_history[-2:]:
                recent_context += f"Q: {item['query'][:100]}...\n"
                recent_context += f"A: {item['response'][:150]}...\n"
        
        # Combina tutto
        enhanced = f"""
{dataset_context}

{recent_context}

User question: {query}

Instructions:
- Analyze the data in the DataFrame `df`
- Provide precise and concise answers
- If you show numbers, format them clearly
- If there are data errors, report them
- Respond in English only
"""
        
        return enhanced.strip()
    
    def _add_to_history(self, query: str, response: str):
        """Aggiungi alla cronologia con gestione limite memoria"""
        self.conversation_history.append({
            "query": query,
            "response": response,
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
        })
        
        # Mantieni solo le ultime N conversazioni
        if len(self.conversation_history) > config.MAX_CONVERSATION_HISTORY:
            self.conversation_history = self.conversation_history[-config.MAX_CONVERSATION_HISTORY:]
            
            # Cleanup memoria Python
            memory_monitor.cleanup_memory()
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Genera sommario completo dataset per dashboard"""
        if self.df is None:
            return {"error": "No dataset loaded"}
        
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        summary = {
            "basic_info": {
                "shape": self.df.shape,
                "memory_mb": round(self.df.memory_usage(deep=True).sum() / (1024**2), 2),
                "columns": list(self.df.columns),
                "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            },
            "data_quality": {
                "null_counts": dict(self.df.isnull().sum()),
                "null_percentage": dict((self.df.isnull().sum() / len(self.df) * 100).round(1)),
                "duplicates": int(self.df.duplicated().sum()),
                "duplicate_percentage": round(self.df.duplicated().sum() / len(self.df) * 100, 1)
            },
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Statistiche numeriche
        if len(numeric_cols) > 0:
            numeric_stats = self.df[numeric_cols].describe()
            summary["numeric_summary"] = {
                col: {
                    "mean": round(numeric_stats.loc["mean", col], 2),
                    "std": round(numeric_stats.loc["std", col], 2),
                    "min": numeric_stats.loc["min", col],
                    "max": numeric_stats.loc["max", col],
                    "median": round(self.df[col].median(), 2)
                }
                for col in numeric_cols
            }
        
        # Statistiche categoriche
        if len(categorical_cols) > 0:
            summary["categorical_summary"] = {
                col: {
                    "unique_count": self.df[col].nunique(),
                    "top_value": self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else None,
                    "top_count": self.df[col].value_counts().iloc[0] if len(self.df[col].value_counts()) > 0 else 0
                }
                for col in categorical_cols
            }
        
        return summary
    
    def get_suggested_queries(self) -> List[str]:
        """Genera query suggerite basate sul dataset corrente"""
        if self.df is None:
            return ["Please upload a dataset to see personalized suggestions"]
        
        suggestions = [
            "Show the first 10 rows of the dataset",
            "Which columns have null values?",
            "Generate descriptive statistics for all numeric columns"
        ]
        
        # Suggerimenti basati sulle colonne
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            suggestions.append(f"What is the distribution of the column '{col}'?")
            
            if len(numeric_cols) > 1:
                col2 = numeric_cols[1]
                suggestions.append(f"Is there a correlation between '{col}' and '{col2}'?")
        
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            suggestions.append(f"Show the most frequent values in the column '{col}'")
            
            if len(numeric_cols) > 0:
                num_col = numeric_cols[0]
                suggestions.append(f"Group the data by '{col}' and calculate the mean of '{num_col}'")
        
        return suggestions
    
    def export_conversation_history(self) -> pd.DataFrame:
        """Esporta cronologia conversazioni come DataFrame"""
        if not self.conversation_history:
            return pd.DataFrame({"message": ["No conversation found"]})
        
        return pd.DataFrame(self.conversation_history)

# Istanza globale agente
csv_agent = CSVAgent()
