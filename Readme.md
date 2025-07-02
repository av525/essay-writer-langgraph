# 🤖 LangGraph Multi-Agent Essay Writer

A proof-of-concept demonstrating **Multi-Agent AI (Agentic AI)** using LangGraph for automated essay writing with research, planning, writing, and reflection capabilities.

## 🎯 Overview

This project showcases a **multi-agent system** where specialized AI agents collaborate to produce high-quality essays through an iterative process of planning, research, writing, and reflection.

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Planner   │───▶│   Researcher │───▶│   Writer    │
│   Agent     │    │    Agent     │    │   Agent     │
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌──────────────┐          │
│  Critic     │◀───│   Research   │◀─────────┘
│  Agent      │    │   Agent      │
└─────────────┘    └──────────────┘
```

## 🔧 Key Components

- **Planning Agent**: Creates essay outlines and structure
- **Research Agent**: Gathers relevant information using Tavily search
- **Writer Agent**: Generates essay content based on plan and research
- **Critic Agent**: Reviews and provides feedback for improvements
- **Research Critic Agent**: Conducts additional research based on critique

## 🚀 Features

- **Multi-Agent Workflow**: Specialized agents for different tasks
- **Iterative Improvement**: Essays refined through multiple revision cycles
- **Real-time Tracking**: Live execution steps in Streamlit UI
- **Web Research Integration**: Dynamic content gathering via Tavily API
- **State Management**: Persistent workflow state using LangGraph

## 📱 User Interface

Interactive Streamlit web app with:
- Essay topic input
- Configurable revision cycles
- Real-time execution monitoring
- Essay output with download capability

## 🛠️ Tech Stack

- **LangGraph**: Multi-agent orchestration framework
- **Azure OpenAI**: Large language model for agents
- **Tavily API**: Web search for research
- **Streamlit**: Web interface
- **Python**: Core implementation

## ⚡ Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/langgraph-essay-writer
cd langgraph-essay-writer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Add your API keys to .env
```

4. **Run the application**
```bash
streamlit run streamlit_app.py
```

## 📋 Requirements

- Azure OpenAI API access
- Tavily API key for web search
- Python 3.8+

## 🎯 Use Cases

- **Content Creation**: Automated essay and article generation
- **Research Assistance**: Multi-source information gathering
- **Educational Tools**: Writing assistance with iterative improvement
- **Business Documentation**: Structured content generation

## 🔮 Future Enhancements

- Support for multiple essay formats
- Integration with additional research sources
- Advanced critique mechanisms
- Export to various formats (PDF, Word, etc.)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built with ❤️ using LangGraph Multi-Agent Framework**