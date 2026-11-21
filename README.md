# ⚖️ LexAI — AI-Powered Legal Assistant for Indian Law

> An intelligent, production-grade legal intelligence assistant for Indian law — combining **LangGraph multi-stage DAG workflow**, **Pinecone cluster-aware RAG**, **SerpAPI live web fallback**, and **Groq LLM inference**, wrapped in a modern, interactive **Streamlit** user experience.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-orange)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-LLM-green)](https://groq.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-purple)](https://www.pinecone.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🌟 Overview & Key Capabilities

LexAI bridges the gap between complex Indian statutes (Bharatiya Nyaya Sanhita / IPC, Bharatiya Nagarik Suraksha Sanhita / CrPC, Consumer Protection, Labour Codes) and ordinary citizens.

### 🎯 Features

1. **🏛️ Multi-Stage Pipeline Orchestration (LangGraph)**
   - **Domain Classifier**: Routes legal vs. general inquiries.
   - **Query Refiner**: Injects Indian statutory keywords and jurisdiction context.
   - **Risk Classifier**: Categorizes risk into 🟢 Low, 🟡 Medium, or 🔴 High with reasoning.
   - **Cluster-Aware RAG Engine**: 2-stage retrieval over Pinecone vector clusters.
   - **SerpAPI Fallback**: Gated by similarity confidence score threshold.
   - **Legal Expert Synthesis**: Structured advice under Indian law with rights and citations.

2. **💬 Rich Interactive Chat Interface**
   - **Visual Risk Badges & Confidence Meters**: Color-coded risk pills and RAG vs. Web source badges.
   - **Structured Section Tabs**: Summary | Relevant Laws | Action Roadmap | Rights & Notes | Sources.
   - **Interactive Action Roadmap**: Vertical stepper with clickable checklist boxes saved in SQLite.
   - **"Lawyer Needed?" Card**: Required / Optional / Not Needed badge with clear justification.
   - **Follow-up Suggestions**: Contextual questions generated after each response.
   - **Pipeline Transparency Expander**: "How I Answered This" inspection of DAG execution and latency.

3. **💾 Local SQLite History & Persistence**
   - Conversations categorized into **Today**, **Yesterday**, **Last 7 days**, and **Older**.
   - Pin, rename, delete, and search across consultations.
   - Auto-generated chat titles from the opening inquiry.
   - Guest mode default + optional user registration and authentication.

4. **📝 Legal Document & Notice Generator**
   - Instant drafts for **RTI Applications**, **Police Complaints for FIR**, **Consumer Complaints**, and **Legal Notices**.
   - Form-based parameter input with downloadable `.txt` and `.md` outputs.

5. **📄 Contract & Document Analyzer**
   - Upload agreements, FIR copies, or notices (PDF/TXT).
   - Automated plain-language breakdown of key parties, risky clauses, red flags, and recommendations.

6. **📊 Analytics Dashboard**
   - Built with **Plotly**: Risk severity donut, RAG vs. Web fallback ratio, query volume timeline, and confidence score trends.

7. **🚨 Emergency Helplines & Glossary**
   - Direct access to national helplines: Police (112), Women in Distress (181), Cyber Fraud (1930), Free Legal Aid (15100), Consumer Grievance (1915).
   - Searchable legal glossary with plain-language definitions for legal jargon.

8. **📤 Export & Sharing**
   - Export consultation reports as branded **PDF** or formatted **Markdown**.

---

## 🏗️ Architecture & Project Structure

```
LAW_chat_bot-main/
├── app.py                   # Master Streamlit application
├── requirements.txt         # Core dependencies
├── .env.example             # Environment variable template
├── README.md                # Documentation & setup guide
│
├── core/                    # Refactored pipeline & inference logic
│   ├── config.py            # Global settings & constants
│   ├── prompts.py           # Structured prompts & draft templates
│   ├── llm.py               # Groq LLM integration & streaming
│   ├── rag.py               # Pinecone cluster retrieval & SentenceTransformers
│   ├── search.py            # SerpAPI India-restricted search
│   └── pipeline.py          # LangGraph DAG & event-driven run_query()
│
├── db/                      # SQLite persistence layer
│   └── database.py          # Thread-safe tables: chats, messages, bookmarks, progress
│
├── ui/                      # Streamlit UI & visual styling
│   ├── components.py        # Structured answer cards, hero, badges, roadmaps
│   ├── analytics.py         # Plotly analytics dashboard
│   └── styles.css           # Custom theme tokens & glassmorphism styling
│
├── utils/                   # Helpers & document generators
│   ├── helpers.py           # Parsing, date grouping, risk formatters
│   └── export.py            # PDF & Markdown export generators
│
├── data/                    # Local storage (lexai.db)
│   └── .gitkeep
│
└── agent.ipynb              # Original research notebook (preserved)
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10+
- Groq API Key ([console.groq.com](https://console.groq.com))
- Pinecone API Key ([pinecone.io](https://www.pinecone.io)) *(optional if testing in fallback mode)*
- SerpAPI Key ([serpapi.com](https://serpapi.com)) *(optional)*

### 2. Installation

```bash
# Clone or navigate to the repository
cd LAW_chat_bot-main

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your API credentials:

```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
SERPAPI_API_KEY=your_serpapi_api_key_here

GROQ_MODEL=llama-3.3-70b-versatile
RAG_THRESHOLD=0.5
RAG_TOP_K=3
```

*(Note: You can also enter and update your API keys directly inside the **Settings** tab in the running app!)*

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🛡️ Trust & Safety Guardrails

- **Mandatory Disclaimer**: Every response includes an explicit notice that LexAI provides general legal information and not formal legal advice.
- **Strict Jurisdiction Restriction**: The LLM prompt explicitly prevents citations of foreign (US/UK) laws and restricts legal summaries to Indian statutes.
- **Zero Third-Party Trackers**: All conversation history, user preferences, and feedback are saved strictly in your local `data/lexai.db` SQLite database.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
