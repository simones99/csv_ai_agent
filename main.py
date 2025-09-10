"""
Applicazione Streamlit principale per CSV AI Agent.

Architettura UI:
- Sidebar per configurazione e stato sistema
- Tab multiple per organizzare funzionalità
- Progress feedback per operazioni lunghe
- Dashboard completa con metriche
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import os
import io

# Import moduli progetto
from config import config
from agents.csv_agent import csv_agent
from utils.memory_monitor import memory_monitor
from utils.data_optimizer import optimizer

# Configurazione pagina Streamlit
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/simones99/csv_ai_agent',
        'Report a bug': ' https://github.com/simones99/csv_ai_agent/issues',
        'About': "# CSV AI Agent\nAgente AI locale per analisi dati CSV\n\nCreato con ❤️ usando LangChain e LM Studio"
    }
)

# CSS personalizzato per UI migliore
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}

.success-message {
    padding: 0.5rem;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 0.25rem;
    color: #155724;
}

.error-message {
    padding: 0.5rem;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 0.25rem;
    color: #721c24;
}

.warning-message {
    padding: 0.5rem;
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.25rem;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR (ENGLISH) ---
def render_sidebar():
    """Minimal sidebar: logo/title, RAM, LM Studio, project info, useful links"""
    with st.sidebar:
        if Path("assets/logo.png").exists():
            st.image("assets/logo.png", width=180)
        else:
            st.title("🤖 CSV AI Agent")
        st.markdown("---")
        # RAM
        memory_info = memory_monitor.get_memory_info()
        st.metric("Free RAM", f"{memory_info.available_gb:.1f}GB", help="Available system RAM")
        if memory_info.critical:
            st.error("⚠️ Critical RAM! Please close other apps.")
        elif memory_info.percent_used > 85:
            st.warning("RAM almost full.")
        st.markdown("---")
        # LM Studio
        st.subheader("LM Studio")
        connection_status = csv_agent.connection_status
        if connection_status["connected"]:
            st.success("Connected")
        else:
            st.error("Not connected")
            with st.expander("Troubleshooting"):
                st.markdown("""
                1. Start LM Studio
                2. Load model (Qwen3 4B)
                3. Start local server (port 1234)
                """)
        if st.button("Test Connection"):
            csv_agent.connection_status = csv_agent._test_connection()
            st.rerun()
        st.markdown("---")
        # Project info
        st.caption("CSV AI Agent • LangChain • LM Studio • Streamlit")
        with st.expander("Useful Links"):
            st.markdown("""
            - [GitHub](https://github.com/simones99/csv_ai_agent)
            - [LM Studio](https://lmstudio.ai/)
            - [Qwen3 Model](https://huggingface.co/Qwen/Qwen3-4B-Instruct)
            """)

# --- UPLOAD (ENGLISH) ---
def render_upload_section():
    """Upload file or sample, basic validation"""
    st.header("Upload a CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type=config.ALLOWED_FILE_TYPES)
    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024**2)
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            st.error(f"File too large ({file_size_mb:.1f}MB). Max {config.MAX_FILE_SIZE_MB}MB.")
            return None
        memory_check = memory_monitor.check_memory_for_dataframe(file_size_mb)
        if not memory_check["can_load"]:
            st.error("Insufficient memory!")
            return None
        if st.button("Upload and analyze", type="primary"):
            return process_file_upload(uploaded_file)
    st.markdown("---")
    st.markdown("**Or use a sample dataset:**")
    sample_files = create_sample_datasets()
    for filename, description in sample_files.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{filename}**: {description}")
        with col2:
            if st.button(f"Load", key=f"sample_{filename}"):
                return load_sample_dataset(filename)
    return None

def process_file_upload(uploaded_file):
    """Processa caricamento file con progress tracking"""
    
    # Salva temporaneamente
    temp_path = Path(config.DATA_UPLOAD_PATH) / uploaded_file.name
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Progress container
    progress_container = st.empty()
    progress_bar = st.progress(0)
    
    def progress_callback(message):
        progress_container.write(f"🔄 {message}")
        time.sleep(0.1)  # Visual feedback
    
    try:
        # Carica con l'agente
        success, message, stats = csv_agent.load_csv(str(temp_path), progress_callback)
        
        progress_bar.progress(100)
        
        if success:
            progress_container.success(message)
            
            # Salva in sessione per persistenza
            st.session_state.dataset_loaded = True
            st.session_state.dataset_stats = stats
            st.session_state.current_file = uploaded_file.name
            
            return True
        else:
            progress_container.error(message)
            return False
            
    finally:
        # Cleanup file temporaneo
        try:
            temp_path.unlink()
        except:
            pass

def create_sample_datasets():
    """Crea dataset di esempio per testing"""
    samples_dir = Path(config.SAMPLE_DATA_PATH)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset vendite
    sales_path = samples_dir / "sales_data.csv"
    if not sales_path.exists():
        import numpy as np
        np.random.seed(42)
        
        data = {
            'Data': pd.date_range('2023-01-01', periods=1000, freq='D'),
            'Prodotto': np.random.choice(['Laptop', 'Mouse', 'Tastiera', 'Monitor', 'Cuffie'], 1000),
            'Categoria': np.random.choice(['Elettronica', 'Accessori', 'Gaming'], 1000),
            'Prezzo': np.random.normal(200, 100, 1000).round(2),
            'Quantità': np.random.randint(1, 20, 1000),
            'Venditore': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana'], 1000),
            'Regione': np.random.choice(['Nord', 'Sud', 'Centro', 'Isole'], 1000)
        }
        
        df = pd.DataFrame(data)
        df['Ricavi'] = (df['Prezzo'] * df['Quantità']).round(2)
        df.to_csv(sales_path, index=False)
    
    # Dataset clienti  
    customers_path = samples_dir / "customer_data.csv"
    if not customers_path.exists():
        np.random.seed(123)
        
        data = {
            'ID_Cliente': range(1, 501),
            'Nome': [f'Cliente_{i}' for i in range(1, 501)],
            'Età': np.random.randint(18, 80, 500),
            'Genere': np.random.choice(['M', 'F'], 500),
            'Città': np.random.choice(['Milano', 'Roma', 'Napoli', 'Torino', 'Firenze'], 500),
            'Spesa_Totale': np.random.exponential(500, 500).round(2),
            'Ordini_Count': np.random.randint(1, 50, 500),
            'Ultima_Visita': pd.date_range('2023-01-01', periods=500, freq='D')
        }
        
        df = pd.DataFrame(data)
        df.to_csv(customers_path, index=False)
    
    return {
        "sales_data.csv": "Dati vendite e-commerce (1000 righe) - Prodotti, prezzi, ricavi",
        "customer_data.csv": "Dati clienti (500 righe) - Demografia e comportamento acquisto"
    }

def load_sample_dataset(filename):
    """Carica dataset di esempio"""
    file_path = Path(config.SAMPLE_DATA_PATH) / filename
    
    if file_path.exists():
        progress_container = st.empty()
        
        def progress_callback(message):
            progress_container.write(f"🔄 {message}")
        
        success, message, stats = csv_agent.load_csv(str(file_path), progress_callback)
        
        if success:
            progress_container.success(f"✅ {filename} caricato!")
            st.session_state.dataset_loaded = True
            st.session_state.dataset_stats = stats
            st.session_state.current_file = filename
            return True
        else:
            progress_container.error(message)
    else:
        st.error(f"File {filename} non trovato")
    
    return False

# --- OVERVIEW (ENGLISH) ---
def render_dataset_overview():
    if not csv_agent.df is not None:
        st.info("Upload a dataset to see the overview")
        return
    st.header("Dataset Overview")
    summary = csv_agent.get_dataset_summary()
    st.write(f"**Rows:** {summary['basic_info']['shape'][0]:,}  |  **Columns:** {summary['basic_info']['shape'][1]}")
    st.write(f"**Memory:** {summary['basic_info']['memory_mb']}MB  |  **Nulls:** {sum(summary['data_quality']['null_counts'].values())}")
    st.dataframe(csv_agent.df.head(5), use_container_width=True)

# --- CHAT (ENGLISH) ---
def render_chat_interface():
    if csv_agent.df is None:
        st.info("Upload a dataset first to chat")
        return
    st.header("Chat with the AI Agent")
    user_query = st.text_area("Question:", value=st.session_state.get('chat_input', ''), height=80)
    if st.button("Send", type="primary") and user_query.strip():
        with st.spinner("Analyzing..."):
            result = csv_agent.analyze(user_query, lambda m: None)
        if result['success']:
            st.success(result['result'])
        else:
            st.error(result['result'])
    if csv_agent.conversation_history:
        st.markdown("---")
        st.markdown("**History:**")
        for c in reversed(csv_agent.conversation_history[-5:]):
            st.markdown(f"- **Q:** {c['query']}\n- **A:** {c['response']}")

# --- INSIGHTS (ENGLISH) ---
def render_insights_dashboard():
    if csv_agent.df is None:
        st.info("Upload a dataset to generate insights")
        return
    st.header("Automatic Insights")
    if st.button("Generate Insights", type="primary"):
        queries = [
            "Columns with most nulls",
            "Correlations between numeric variables",
            "Distribution of categorical variables",
            "Outliers in numeric data",
            "3 interesting analyses for this dataset"
        ]
        for q in queries:
            with st.spinner(f"{q}..."):
                result = csv_agent.analyze(q, lambda m: None)
            if result['success']:
                st.success(f"{q}: {result['result']}")
            else:
                st.error(result['result'])

# --- EXPORT (ENGLISH) ---
def render_export():
    if csv_agent.df is not None:
        st.header("Export")
        if st.button("Download optimized CSV"):
            csv_data = csv_agent.df.to_csv(index=False)
            st.download_button("Download CSV", csv_data, file_name=f"optimized_{st.session_state.get('current_file', 'dataset')}.csv", mime="text/csv")
        if st.button("Download Excel report"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                csv_agent.df.to_excel(writer, sheet_name='Data', index=False)
            st.download_button("Download Excel", buffer.getvalue(), file_name=f"report_{st.session_state.get('current_file', 'dataset')}.xlsx", mime="application/vnd.ms-excel")
        if csv_agent.conversation_history:
            conversations_df = csv_agent.export_conversation_history()
            csv_conv = conversations_df.to_csv(index=False)
            st.download_button("Download Chat CSV", csv_conv, file_name="ai_agent_conversations.csv", mime="text/csv")
    else:
        st.info("Upload a dataset to enable export")

# --- MAIN (ENGLISH) ---
def main():
    st.title("CSV AI Agent")
    st.caption("Local CSV analysis with AI • LangChain + LM Studio")
    render_sidebar()
    if not csv_agent.connection_status["connected"]:
        st.error("LM Studio not connected! Configure in the sidebar.")
        st.stop()
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Load Data", "Overview", "Chat", "Insights", "Export"
    ])
    with tab1:
        upload_result = render_upload_section()
        if upload_result:
            st.success("Dataset uploaded! Go to the other tabs.")
    with tab2:
        render_dataset_overview()
    with tab3:
        render_chat_interface()
    with tab4:
        render_insights_dashboard()
    with tab5:
        render_export()
    st.markdown("---")
    st.caption("CSV AI Agent • 2025 • GitHub: https://github.com/simones99/csv_ai_agent")

if __name__ == "__main__":
    # Inizializza sessione se necessario
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    
    main()
