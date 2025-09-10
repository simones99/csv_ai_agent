# 🤖 CSV AI Agent

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Local AI Agent for intelligent CSV data analysis using LM Studio and Qwen3 4B**

A complete project demonstrating integration of modern AI technologies for data analysis, optimized for MacBook Pro M4 and fully local (privacy-first approach).

![CSV AI Agent Demo](assets/screenshots/dashboard.png)

## 🌟 Features

### ✨ Core Capabilities
- **🧠 Conversational AI Agent**: Query your data using natural language
- **🏠 Fully Local**: No external APIs, privacy guaranteed
- **⚡ M4 Optimized**: Performance tuning specific for Apple Silicon
- **📊 Complete Dashboard**: Overview, automatic insights, visualizations
- **💾 Smart Memory Management**: Intelligent handling of large datasets
- **🔄 Real-time Feedback**: Progress tracking for long operations

### 🛡️ Privacy & Security
- **Zero Cloud**: All data stays on your computer
- **No API Keys**: No external paid services required
- **Open Source**: Fully transparent and auditable code

### 📈 Supported Analysis
- Advanced descriptive statistics
- Correlation and pattern analysis
- Outlier and anomaly identification
- Data quality assessment
- Interactive visualizations
- Intelligent analysis suggestions

## 🛠️ Tech Stack

| Component | Technology | Version | Role |
|-----------|------------|---------|------|
| **AI Framework** | LangChain | 0.2+ | AI agent orchestration |
| **LLM Backend** | LM Studio + Qwen3 4B | Latest | Local language model |
| **Interface** | Streamlit | 1.38+ | Interactive web UI |
| **Data Processing** | Pandas | 2.2+ | Data manipulation |
| **Visualizations** | Plotly | 5.24+ | Interactive charts |
| **Memory Monitoring** | psutil | Latest | System resource monitoring |

## 🚀 Quick Start

### Prerequisites
- **Hardware**: MacBook Pro M4 (16GB+ RAM recommended)
- **Software**: Python 3.8+, LM Studio installed
- **AI Model**: Qwen3 4B 2507 (downloadable from LM Studio)

### 5-Minute Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/csv-ai-agent.git
cd csv-ai-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure LM Studio (see detailed section)
# - Download Qwen3 4B 2507
# - Start local server on port 1234

# 5. Launch application
streamlit run main.py
```

🎉 **Your AI agent is ready!** Open browser at `http://localhost:8501`

## 📋 Detailed Installation

### 1. Python Environment Setup

```bash
# Check Python version (3.8+ required)
python --version

# Clone repository
git clone https://github.com/yourusername/csv-ai-agent.git
cd csv-ai-agent

# Create isolated virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### 2. LM Studio Setup

#### Download & Installation
1. **Download LM Studio**: [lmstudio.ai](https://lmstudio.ai/)
2. **Install** following macOS instructions
3. **Launch LM Studio**

#### Model Configuration
1. **Search "Qwen3 4B 2507"** in Models section
2. **Download** the `Q4_K_M` version (~2.5GB)
3. **Load model** in Chat section
4. **Start Local Server**:
   - Go to "Local Server"
   - Click "Start Server"
   - Verify URL: `http://localhost:1234`

#### Connection Test
```bash
# Quick connection test
curl http://localhost:1234/v1/models

# Should return loaded model information
```

## 💻 Usage

### Typical Workflow

#### 1. **System Check**
- Check LM Studio status in sidebar
- Verify available RAM
- Test AI model connection

#### 2. **Load Dataset**
```
📁 "Load Data" tab → Upload CSV or use sample dataset
```

#### 3. **Explore Data**
```
📊 "Overview" tab → View statistics and data quality
```

#### 4. **Interactive Analysis**
```
💬 "Chat Analysis" tab → Ask questions in natural language
```

#### 5. **Automatic Insights**
```
🎯 "Insights" tab → Generate advanced automatic analysis
```

### Example Queries

#### Basic Queries
```
- "How many rows does the dataset have?"
- "Are there any null values?"
- "Show statistics for numeric columns"
```

#### Analytical Queries
```
- "What's the correlation between price and sales?"
- "Identify outliers in the revenue column"
- "Group by category and calculate averages"
```

#### Exploratory Queries
```
- "Find interesting patterns in the data"
- "Suggest 3 useful analyses for this dataset"
- "Are there any anomalies I should investigate?"
```

## 📁 Project Structure

```
csv-ai-agent/
├── 📄 README.md                    # Main documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 config.py                   # Central configuration
├── 📄 main.py                     # Main Streamlit app
│
├── 🧠 agents/                     # Core AI Agent
│   ├── 📄 __init__.py
│   ├── 📄 csv_agent.py            # Main agent logic
│   ├── 📄 tools.py                # Custom tools
│   └── 📄 prompts.py              # Prompt templates
│
├── 🛠️ utils/                      # Utilities and helpers
│   ├── 📄 __init__.py
│   ├── 📄 data_optimizer.py       # DataFrame optimization
│   ├── 📄 validation.py           # Data validation
│   └── 📄 memory_monitor.py       # Memory monitoring
│
├── 📊 data/                       # Data management
│   ├── 📁 samples/                # Sample datasets
│   └── 📁 uploads/                # User files (auto-created)
│
├── 🎨 assets/                     # Graphic resources
│   ├── 📄 logo.png               # Application logo
│   └── 📁 screenshots/           # README screenshots
│
└── 🧪 tests/                      # Automated tests
    ├── 📄 __init__.py
    ├── 📄 test_agent.py          # AI agent tests
    └── 📄 test_utils.py          # Utility tests
```

## ⚙️ Configuration

### Hardware Limits (MacBook M4 16GB)
- **CSV Files**: Maximum 500MB
- **Dataset Memory**: Max 200MB after optimization
- **Conversations**: Limit 10 in history

### Performance Tips
- **Close other apps** during heavy analysis
- **Use sample datasets** for quick testing
- **Monitor RAM** in sidebar
- **Restart LM Studio** if memory leak occurs

## 🔧 Troubleshooting

### Common Issues

#### ❌ "LM Studio not connected"
**Solutions**:
```bash
# 1. Verify LM Studio is open
# 2. Go to "Local Server" → "Start Server"
# 3. Verify model loaded in "Chat"
# 4. Test connection:
curl http://localhost:1234/v1/models
```

#### ❌ "Insufficient memory"
**Solutions**:
```bash
# 1. Close other applications
# 2. Restart LM Studio
# 3. Use smaller CSV files (<100MB)
# 4. Restart Streamlit app
```

#### ❌ "Model loading error"
**Solutions**:
```bash
# 1. Re-download Qwen3 4B model
# 2. Update LM Studio to latest version
# 3. Verify 10GB+ free disk space
```

## 🤝 Contributing

Contributions welcome! This project is open-source and aims to grow with the community.

### Development Setup
```bash
# Fork on GitHub and clone your fork
git clone https://github.com/your-username/csv-ai-agent.git
cd csv-ai-agent

# Create feature branch
git checkout -b feature/feature-name

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Future Features Ideas
- **Multi-agent Architecture**: Specialized agents for specific tasks
- **Advanced Visualizations**: ML plots, heatmaps, 3D charts
- **Database Integration**: PostgreSQL, SQLite, MongoDB support
- **Report Generation**: Automatic PDF exports with insights
- **API REST**: Endpoints for external integrations

## 📄 License

This project is released under [MIT License](LICENSE).

### Commercial Use
✅ **Permitted**: Use this code in your commercial projects
✅ **Modify**: Adapt the code to your needs
✅ **Distribute**: Redistribute with attribution
✅ **Sell**: Include in paid products

## 🙏 Acknowledgments

### Open Source Technologies
- **[LangChain](https://langchain.com)**: Framework for LLM applications
- **[Streamlit](https://streamlit.io)**: Rapid web app prototyping
- **[Pandas](https://pandas.pydata.org)**: Data manipulation library
- **[Plotly](https://plotly.com)**: Interactive visualizations

### AI Models
- **[Qwen Team](https://qwenlm.github.io)**: Qwen3 4B model
- **[LM Studio](https://lmstudio.ai)**: Local LLM runtime

---

**Made with ❤️ | Optimized for 🍎 Apple Silicon**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/csv-ai-agent.svg?style=social)](https://github.com/yourusername/csv-ai-agent/stargazers)

> 💡 **Tip**: Join [GitHub Discussions](https://github.com/yourusername/csv-ai-agent/discussions) for questions, ideas and showcase!