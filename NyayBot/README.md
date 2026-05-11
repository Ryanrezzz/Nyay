<p align="center">
  <img src="assets/banner.png" alt="NyayBot Banner" width="100%"/>
</p>

<h1 align="center">⚖️ NyayBot</h1>

<p align="center">
  <strong>An AI-powered Legal Assistant for Indian Criminal Law</strong><br/>
  <em>Covering 893 sections across BNS 2023 &amp; IPC 1860 — with verified bailable/cognizable classifications</em>
</p>

<p align="center">
  <a href="https://g36knctqiqnwoqigkpqdjf.streamlit.app"><img src="https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge" alt="Streamlit App"/></a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LangChain-🦜-1C3C3C?style=for-the-badge" alt="LangChain"/>
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-00599C?style=for-the-badge" alt="FAISS"/>
  <img src="https://img.shields.io/badge/Gemini-Embeddings-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/BNS_2023-358_Sections-10B981?style=flat-square" alt="BNS Sections"/>
  <img src="https://img.shields.io/badge/IPC_1860-535_Sections-3B82F6?style=flat-square" alt="IPC Sections"/>
  <img src="https://img.shields.io/badge/Classifications-659_Verified-F59E0B?style=flat-square" alt="Classifications"/>
  <img src="https://img.shields.io/badge/Embeddings-3072d_Vectors-8B5CF6?style=flat-square" alt="Embedding Dimensions"/>
</p>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [File-by-File Breakdown](#-file-by-file-breakdown)
  - [Root Files](#root-files)
  - [Data Pipeline (`src/data_pipeline/`)](#data-pipeline-srcdata_pipeline)
  - [Embedding Layer (`src/embedding/`)](#embedding-layer-srcembedding)
  - [RAG Pipeline (`src/rag/`)](#rag-pipeline-srcrag)
  - [Data Files (`data/`)](#data-files-data)
  - [Vector Store (`embeddings/`)](#vector-store-embeddings)
- [How It Works — End to End](#-how-it-works--end-to-end)
- [Setup & Installation](#-setup--installation)
- [Environment Variables](#-environment-variables)
- [Running Locally](#-running-locally)
- [Deployment](#-deployment)
- [Disclaimer](#️-disclaimer)

---

## 🔍 Overview

**NyayBot** (_"Nyay" = Justice in Hindi_) is a Retrieval-Augmented Generation (RAG) chatbot that provides accurate, hallucination-free legal information on Indian criminal law. It covers two statutes:

| Statute | Sections | Status |
|---------|----------|--------|
| **Bharatiya Nyaya Sanhita (BNS) 2023** | 358 sections | 🟢 Current law (from 1 July 2024) |
| **Indian Penal Code (IPC) 1860** | 535 sections | 🔵 Old law (until 30 June 2024) |

Users can ask questions in everyday language — _"What happens if someone steals my phone?"_ — and NyayBot translates that into legal terminology, retrieves the most relevant sections from a FAISS vector store, and generates a warm, conversational response grounded **only** in the retrieved legal text.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔎 **Multi-Query Expansion** | Converts casual language into 3 legal search phrases using Llama 3.3 70B, then searches FAISS with all queries for broader recall |
| 🧠 **RAG with Zero Hallucination** | Strict prompt engineering ensures responses use **only** retrieved legal text — never fabricated sections |
| ⚖️ **Dual-Law Coverage** | Always shows BNS (current) first, then IPC (old), so users understand both the active and legacy law |
| 📋 **Verified Classifications** | 659 sections have verified bailable/cognizable/triable-by metadata parsed from official BNSS & CrPC schedules |
| 💬 **Multi-Turn Memory** | Full conversation history via LangChain `RunnableWithMessageHistory` — follow-up questions work naturally |
| 🌐 **Streamlit Cloud Deployment** | One-click deploy with secrets management; no infrastructure required |

---

## 🏗 System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                          USER QUERY                                │
│                  "What if someone steals my phone?"                 │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │   QUERY EXPANSION     │
               │  (Groq / Llama 3.3)   │
               │                       │
               │  → "theft of mobile"  │
               │  → "larceny movable"  │
               │  → "dishonest taking" │
               └───────────┬───────────┘
                           │  4 queries (original + 3 expanded)
                           ▼
               ┌───────────────────────┐
               │   FAISS RETRIEVER     │
               │  (893 sections,       │
               │   3072-d vectors)     │
               │                       │
               │  Top-k=5 per query    │
               │  Deduplicate → Top 7  │
               └───────────┬───────────┘
                           │  7 unique legal sections
                           ▼
               ┌───────────────────────┐
               │   LLM GENERATION      │
               │  (Cerebras / Qwen 3   │
               │   235B-A22B)          │
               │                       │
               │  Strict prompt:       │
               │  • Only use <database> │
               │  • BNS before IPC     │
               │  • No hallucination   │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │   STREAMLIT UI        │
               │                       │
               │  Chat interface with  │
               │  session memory       │
               └───────────────────────┘
```

---

## 🛠 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Interactive chat UI with session state |
| **Orchestration** | LangChain | RAG chain, prompt templates, memory management |
| **Primary LLM** | Qwen 3 235B-A22B (via Cerebras) | Response generation with legal grounding |
| **Expansion LLM** | Llama 3.3 70B (via Groq) | Query expansion — casual → legal terminology |
| **Embeddings** | Google Gemini `gemini-embedding-001` | 3072-dimensional dense vectors |
| **Vector Store** | FAISS (CPU) | Similarity search over 893 section embeddings |
| **PDF Parsing** | pdfplumber / PyMuPDF | Text extraction from official gazette PDFs |
| **Deployment** | Streamlit Cloud | Hosted at `nyay.streamlit.app` |

---

## 📂 Project Structure

```
NyayBot/
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python dependencies
├── .env                            # API keys (git-ignored)
├── .gitignore                      # Git exclusion rules
├── assets/
│   └── banner.png                  # README banner image
│
├── src/
│   ├── data_pipeline/              # PDF → Structured JSON pipeline
│   │   ├── parse_pdf.py            # Step 1: Extract raw text from IPC PDF
│   │   ├── clean_text.py           # Step 2: Split IPC text into sections
│   │   ├── structure_section.py    # Step 3: Structure IPC sections to schema
│   │   ├── patch_sections.py       # Step 4: Fix missing/wrong IPC definitions
│   │   ├── parse_bns.py            # Steps 1-3 combined for BNS
│   │   ├── parse_bnss_schedule.py  # Parse BNS bailable/cognizable from BNSS
│   │   ├── parse_crpc_schedule.py  # Parse IPC bailable/cognizable from CrPC
│   │   └── merge_classification.py # Merge classifications into structured JSON
│   │
│   ├── embedding/                  # Vector embedding generation
│   │   ├── build_embeddings.py     # Generate Gemini embeddings for all sections
│   │   └── build_faiss_index.py    # Build & save FAISS index from embeddings
│   │
│   └── rag/                        # Retrieval-Augmented Generation
│       └── rag_pipeline.py         # Core RAG chain with memory & query expansion
│
├── data/
│   ├── raw_pdfs/                   # Source PDF files (git-ignored)
│   │   ├── ipc_1860.pdf            # Indian Penal Code 1860
│   │   ├── bns_2023.pdf            # Bharatiya Nyaya Sanhita 2023
│   │   ├── bnss_first_schedule.pdf # BNSS First Schedule (classifications)
│   │   └── crpc.pdf                # CrPC First Schedule (classifications)
│   │
│   ├── extracted_text/             # Raw extracted text (git-ignored)
│   │   ├── ipc_1860_raw.txt        # ~483K chars of IPC text
│   │   └── bns_2023_raw.txt        # ~398K chars of BNS text
│   │
│   ├── structured_json/            # Processed section data
│   │   ├── ipc_sections_raw.json   # IPC sections (raw, pre-structuring)
│   │   ├── ipc_structured.json     # IPC sections (structured schema, ~639KB)
│   │   └── bns_structured.json     # BNS sections (structured schema, ~550KB)
│   │
│   ├── section_classification.json # Merged bailable/cognizable data (659 entries)
│   └── bnss_classification_parsed.json  # Parsed BNSS schedule data
│
└── embeddings/
    ├── section_embeddings.npy      # Pre-computed 893×3072 embedding matrix (~11MB)
    ├── section_metadata.json       # Metadata for each embedding vector
    └── faiss_index/
        ├── index.faiss             # FAISS vector index (~11MB)
        └── index.pkl               # Serialized metadata for FAISS
```

---

## 📝 File-by-File Breakdown

### Root Files

#### `app.py` — Streamlit Application Entry Point

The main application file that creates the web-based chat interface.

| Aspect | Detail |
|--------|--------|
| **Framework** | Streamlit with `page_title='NyayBot'`, `page_icon='⚖️'` |
| **Chain Loading** | Uses `@st.cache_resource` to load the RAG chain once and reuse across sessions |
| **Session State** | Maintains `messages` list and `session_id` for multi-turn conversation |
| **Secrets Injection** | Iterates over `st.secrets` and injects them as environment variables for cloud deployment |
| **Sidebar** | Displays database coverage stats, features list, disclaimer, and a "Clear Chat" button |

```python
# Key pattern: Secrets → Environment Variables (for Streamlit Cloud)
for key in st.secrets:
    os.environ[key] = st.secrets[key]
```

---

#### `requirements.txt` — Dependencies

Organized into logical groups:

| Group | Packages |
|-------|----------|
| **Data Pipeline** | `pdfplumber`, `PyMuPDF` |
| **Vector DB** | `faiss-cpu` |
| **RAG Pipeline** | `langchain`, `langchain-core`, `langchain-community`, `langchain-groq`, `langchain-openai`, `langchain-google-genai` |
| **Frontend** | `streamlit` |
| **Utilities** | `python-dotenv` |

---

#### `.gitignore` — Version Control Exclusions

Excludes `.env` (API keys), `__pycache__/`, `data/raw_pdfs/` (large PDFs), `data/extracted_text/` (regenerable), `.DS_Store`, and IDE config directories.

---

### Data Pipeline (`src/data_pipeline/`)

The data pipeline converts official Indian law PDFs into structured, searchable JSON. It runs as a sequence of scripts executed in order.

---

#### `parse_pdf.py` — Step 1: IPC PDF Text Extraction

| Aspect | Detail |
|--------|--------|
| **Input** | `data/raw_pdfs/ipc_1860.pdf` |
| **Output** | `data/extracted_text/ipc_1860_raw.txt` |
| **Library** | `pdfplumber` |
| **Method** | Iterates every page, extracts text, inserts page markers (`--- PAGE X ---`), warns on empty/scanned pages |

---

#### `clean_text.py` — Step 2: IPC Section Splitting

| Aspect | Detail |
|--------|--------|
| **Input** | `data/extracted_text/ipc_1860_raw.txt` |
| **Output** | `data/structured_json/ipc_sections_raw.json` |
| **Key Logic** | Regex `\n(\d{1,3}[A-Z]?)\.\s+[A-Z"\[]` identifies section boundaries |
| **TOC Skip** | Finds `"ACT NO. 45 OF 1860"` and starts parsing from there |
| **Deduplication** | When multiple matches exist for the same section number, keeps the one with the longest `raw_text` |

---

#### `structure_section.py` — Step 3: IPC Schema Structuring

Transforms raw section text into a normalized schema.

| Aspect | Detail |
|--------|--------|
| **Input** | `data/structured_json/ipc_sections_raw.json` |
| **Output** | `data/structured_json/ipc_structured.json` |
| **Parsing** | Splits on em dash (`—`) to separate title from description |
| **Schema Fields** | `section_id`, `act`, `section_number`, `title`, `description`, `punishment`, `bailable`, `cognizable`, `triable_by`, `ipc_equivalent`, `bns_equivalent`, `valid_from`, `valid_until`, `superseded_by`, `last_verified`, `source_url`, `version` |

**Output Schema Example:**
```json
{
  "section_id": "IPC_302",
  "act": "IPC",
  "section_number": "302",
  "title": "Punishment for murder",
  "description": "Whoever commits murder shall be punished with death...",
  "bailable": "No",
  "cognizable": "Yes",
  "triable_by": "Court of Session",
  "valid_from": "1860-10-06",
  "valid_until": "2024-06-30"
}
```

---

#### `patch_sections.py` — Step 4: IPC Definition Patching

A one-time fix script that corrects ~30 short definition sections (IPC §17–§52) that were missed or corrupted during PDF extraction.

| Aspect | Detail |
|--------|--------|
| **Problem** | Short definition sections (e.g., `"Government"`, `"Judge"`, `"Offence"`) were either missed by the regex or replaced with footnote text from the PDF |
| **Solution** | Contains a `MANUAL_SECTIONS` dictionary with the correct text for 30 sections, patches them into `ipc_sections_raw.json` |
| **Logic** | Adds missing sections; replaces existing ones if their text starts with `"Ins. by"`, `"Subs. by"`, or `"The words"` (footnote indicators) |

---

#### `parse_bns.py` — BNS Combined Pipeline (Steps 1–3)

Handles the entire BNS pipeline in one file — extraction, splitting, and structuring.

| Aspect | Detail |
|--------|--------|
| **Input** | `data/raw_pdfs/bns_2023.pdf` |
| **Outputs** | `data/extracted_text/bns_2023_raw.txt` + `data/structured_json/bns_structured.json` |
| **TOC Handling** | Skips to the **second** occurrence of `"CHAPTER I"` (first is in the table of contents) |
| **Structuring** | Same em-dash splitting and schema as IPC, but with `valid_from: "2024-07-01"` and `valid_until: null` |

---

#### `parse_bnss_schedule.py` — BNS Classification Parser

Parses the **BNSS First Schedule PDF** to extract bailable/cognizable/triable-by metadata for BNS sections.

| Aspect | Detail |
|--------|--------|
| **Input** | `data/raw_pdfs/bnss_first_schedule.pdf` |
| **Output** | Merged into `data/section_classification.json` |
| **Challenges** | PDF text has spaced-out words (`"C o g n i z a b l e"`) — handled with regex cleanup |
| **Merge Strategy** | Parsed data is merged with existing manual classifications; **manual data takes priority** |
| **Coverage** | Extracts classifications for ~311 BNS sections |

---

#### `parse_crpc_schedule.py` — IPC Classification Parser

Parses the **CrPC First Schedule** (pages 167–204) for IPC section classifications.

| Aspect | Detail |
|--------|--------|
| **Input** | `data/raw_pdfs/crpc.pdf` (pages 167–204 only) |
| **Output** | Merged into `data/section_classification.json` |
| **"Ditto" Handling** | CrPC schedule uses `"ditto"` to repeat the previous row's values — this script carries forward the last seen cognizable/bailable/triable values |
| **Coverage** | Extracts classifications for ~348 IPC sections |

---

#### `merge_classification.py` — Classification → Structured JSON Merger

Merges the parsed classification data into the final structured JSON files.

| Aspect | Detail |
|--------|--------|
| **Inputs** | `data/section_classification.json` + `data/structured_json/bns_structured.json` + `data/structured_json/ipc_structured.json` |
| **Outputs** | Updated `bns_structured.json` and `ipc_structured.json` with `bailable`, `cognizable`, and `triable_by` fields populated |
| **Value Mapping** | Boolean `true/false` → `"Yes"/"No"` strings for human readability in the chatbot |

---

### Embedding Layer (`src/embedding/`)

---

#### `build_embeddings.py` — Vector Embedding Generation

Generates dense vector representations for all 893 law sections.

| Aspect | Detail |
|--------|--------|
| **Model** | Google `gemini-embedding-001` (3072 dimensions) |
| **Input** | Combined `ipc_structured.json` + `bns_structured.json` |
| **Text Preparation** | Concatenates `act + section_number + title + description + classification` into one searchable string per section |
| **Rate Limiting** | Processes in batches of 90, waits 61 seconds between batches to stay under the 100/min API limit |
| **Outputs** | `embeddings/section_embeddings.npy` (893 × 3072 float32 matrix) + `embeddings/section_metadata.json` |

---

#### `build_faiss_index.py` — FAISS Index Construction

Builds the FAISS similarity search index from the pre-computed embeddings.

| Aspect | Detail |
|--------|--------|
| **Input** | `section_embeddings.npy` + all structured JSONs |
| **Output** | `embeddings/faiss_index/index.faiss` + `embeddings/faiss_index/index.pkl` |
| **Method** | Uses LangChain's `FAISS.from_embeddings()` with text-embedding pairs and metadata |
| **Metadata** | Each vector stores `section_id`, `act`, `section_number`, and `title` for retrieval-time filtering |
| **Validation** | Runs a test query `"punishment for murder"` and prints top-5 results |

---

### RAG Pipeline (`src/rag/`)

---

#### `rag_pipeline.py` — Core RAG Chain with Memory

The heart of NyayBot — a multi-stage retrieval and generation pipeline.

**Component Breakdown:**

| Function | Purpose |
|----------|---------|
| `format_docs()` | Formats retrieved FAISS documents with act/section headers for the LLM context window |
| `expand_query()` | Uses Llama 3.3 70B to convert casual user language into 3 formal legal search phrases |
| `multi_retrieve()` | Searches FAISS with all 4 queries (original + 3 expanded), deduplicates by section ID, sorts BNS first, returns top 7 |
| `build_rag_chain()` | Assembles the full LangChain pipeline with retriever, prompt, LLM, memory, and output parser |

**Prompt Engineering Highlights:**

- **Personality**: Warm, conversational, empathetic — not robotic
- **Strict Grounding**: Only uses text inside `<database>` tags; never fabricates sections
- **Law Whitelist**: Only BNS and IPC exist — no IT Act, POCSO, Consumer Protection, etc.
- **Forbidden Content**: No websites, helplines, phone numbers, or civil remedies
- **Output Format**: Structured tables with Section / Punishment / Bailable / Cognizable / Triable by
- **Action Field**: Always exactly `"Consult a qualified lawyer for personalized legal advice."`

**LLM Configuration:**

| LLM | Provider | Model | Purpose | Temperature |
|-----|----------|-------|---------|-------------|
| Primary | Cerebras | `qwen-3-235b-a22b-instruct-2507` | Response generation | 0 |
| Expansion | Groq | `llama-3.3-70b-versatile` | Query expansion | 0.1 |

**Memory System:**
- Uses `ChatMessageHistory` with `RunnableWithMessageHistory`
- Session-based (keyed by `session_id`)
- Enables multi-turn follow-up questions

---

### Data Files (`data/`)

| File / Directory | Size | Description |
|-----------------|------|-------------|
| `raw_pdfs/` | ~4 MB | Source PDFs from the official Indian Code repository (git-ignored) |
| `extracted_text/` | ~880 KB | Raw text extracted from PDFs (regenerable, git-ignored) |
| `structured_json/ipc_structured.json` | 639 KB | 535 IPC sections in normalized schema |
| `structured_json/bns_structured.json` | 550 KB | 358 BNS sections in normalized schema |
| `structured_json/ipc_sections_raw.json` | 401 KB | Intermediate IPC sections (pre-structuring) |
| `section_classification.json` | 92 KB | 659 bailable/cognizable classifications |
| `bnss_classification_parsed.json` | 14 KB | Raw parsed BNSS schedule data |

---

### Vector Store (`embeddings/`)

| File | Size | Description |
|------|------|-------------|
| `section_embeddings.npy` | ~11 MB | 893 × 3072 float32 numpy array |
| `section_metadata.json` | 149 KB | Metadata mapping for each vector |
| `faiss_index/index.faiss` | ~11 MB | FAISS flat index for similarity search |
| `faiss_index/index.pkl` | 973 KB | Pickled LangChain FAISS metadata |

---

## 🔄 How It Works — End to End

### Phase 1 — Data Ingestion (Offline, Run Once)

```bash
# 1. Extract text from IPC PDF
python src/data_pipeline/parse_pdf.py

# 2. Split IPC into individual sections
python src/data_pipeline/clean_text.py

# 3. Patch missing IPC definition sections
python src/data_pipeline/patch_sections.py

# 4. Structure IPC sections into schema
python src/data_pipeline/structure_section.py

# 5. Parse BNS PDF → structured JSON (all-in-one)
python src/data_pipeline/parse_bns.py

# 6. Parse BNSS schedule for BNS classifications
python src/data_pipeline/parse_bnss_schedule.py

# 7. Parse CrPC schedule for IPC classifications
python src/data_pipeline/parse_crpc_schedule.py

# 8. Merge classifications into structured JSONs
python src/data_pipeline/merge_classification.py
```

### Phase 2 — Embedding & Indexing (Offline, Run Once)

```bash
# 9. Generate Gemini embeddings (takes ~10 min due to rate limits)
python src/embedding/build_embeddings.py

# 10. Build FAISS index from embeddings
python src/embedding/build_faiss_index.py
```

### Phase 3 — Serving (Runtime)

```bash
# 11. Launch Streamlit app
streamlit run app.py
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- API keys for Google Gemini, Cerebras, and Groq

### Install

```bash
# Clone the repository
git clone https://github.com/Ryanrezzz/Nyay.git
cd Nyay

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY="your-google-gemini-api-key"
GROQ_API_KEY="your-groq-api-key"
CEREBRAS_API_KEY="your-cerebras-api-key"
```

For **Streamlit Cloud**, add these same keys in the app's **Secrets** section (Settings → Secrets).

---

## 🏃 Running Locally

```bash
# Terminal mode (no UI)
python src/rag/rag_pipeline.py

# Streamlit UI
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## ☁️ Deployment

NyayBot is deployed on **Streamlit Cloud** at **[nyay.streamlit.app](https://g36knctqiqnwoqigkpqdjf.streamlit.app)**.

The deployment requires:
1. All files in the `embeddings/faiss_index/` directory committed to Git
2. API keys configured in Streamlit Cloud Secrets
3. `app.py` injects secrets into environment variables automatically via:
   ```python
   for key in st.secrets:
       os.environ[key] = st.secrets[key]
   ```

---

## ⚠️ Disclaimer

> **NyayBot provides legal information, NOT legal advice.**
> Always consult a qualified lawyer for personalized legal guidance. The classifications and section data are parsed from official government documents but may contain parsing artifacts. This tool is intended for educational and informational purposes only.

---

<p align="center">
  <sub>Built with ❤️ using LangChain, FAISS, and Streamlit</sub>
</p>
