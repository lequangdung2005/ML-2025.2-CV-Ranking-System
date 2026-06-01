# TECHNICAL PROPOSAL: AI-POWERED RESUME SCREENING & EVALUATION PIPELINE

## 1. Executive Summary & Project Objectives
Traditional recruitment workflows suffer from extreme inefficiencies due to the manual overhead of parsing resumes, bias in human filtering, and the inability to effectively capture semantic alignment between a candidate's profile and a Job Description (JD). 

This proposal outlines the technical architecture and implementation roadmap for an **Intelligent Resume Screening and Evaluation Pipeline**. Utilizing cutting-edge Vision-Language Models (VLMs), Large Language Models (LLMs), dense vector retrieval, and cross-encoder re-ranking, this system automates candidate evaluation from raw ingestion to a final qualitative and quantitative judgment.

### Core Objectives:
* **Zero-Loss Text Ingestion:** Parse complex, multi-column, and graphical CV layouts flawlessly.
* **Deterministic Schema Enforcement:** Convert unstructured resume text into highly reliable, standardized JSON objects.
* **Scalable Semantic Retrieval:** Sub-second retrieval of the top candidates matching a given JD using high-dimensional vector spaces.
* **Explainable AI Judgment:** Provide transparent, criteria-based matching scores along with localized strengths and weaknesses analysis.

---

## 2. System Architecture Breakdown

The proposed pipeline consists of four sequential, loosely-coupled modules designed for high scalability and throughput.

```
[Candidate CV (PDF/Img)] ──> 1. Data Ingestion & OCR (VLM API)
                                         │
                                         ▼ (Markdown / Text)
                               2. Information Extraction (LLM API + Instructor/Pydantic)
                                         │
                                         ▼ (Structured CV JSON)
                              3. Semantic Matching & Retrieval (BAAI/bge-m3 + Milvus/Qdrant) <── [Job Description (JD)]
                                         │                                                             │
                                         ▼ (Top-K Re-ranked CVs)                                       │
                              4. LLM-as-a-Judge Evaluation (LLM Evaluator) <───────────────────────────┘
                                         │
                                         ▼
                              [Final Score (0-100) & Strengths/Weaknesses]
```

### Module 1: Data Ingestion & Multi-Modal OCR
* **Input:** Candidate CVs in unstructured formats (Scanned PDF, Native PDF, JPEG, PNG).
* **Core Processing:** * Traditional OCR engines fail when encountering creative layouts, multi-column resumes, timeline graphics, or embedded charts.
    * This module leverages **Qwen2-VL** or **LLaVA** (Vision-Language Models) to visually analyze page topology. The model processes the document as an image-text hybrid, recognizing reading order, headers, and section boundaries natively.
* **Tech Stack:** Hosted VLM API (e.g., Qwen2-VL / LLaVA-class vision model), or equivalent provider-supported vision model.
* **Output:** Structural, cleanly organized Markdown or clean text retaining structural intent.

### Module 2: High-Fidelity Information Extraction
* **Input:** Raw Markdown text from Module 1.
* **Core Processing:**
    * To make downstream data queryable, unstructured text must be schema-bound.
    * We call a hosted **LLM API** (e.g., Llama-3-class instruct model) for extraction and normalization.
    * To enforce a strict JSON output schema without brittle regex parsing, we integrate the **Instructor** library. This uses Pydantic to guide the model's output formatting via schema-constrained generation.
    * API hardening: implement retries with exponential backoff, request timeouts, rate-limit handling, and optional response caching for repeat runs.
* **Extracted Entities:** * *Personal Details:* Name, Contact, Location.
    * *Skills:* Categorized (Hard skills, Soft skills, Frameworks, Languages).
    * *Experience:* Chronological list containing company name, role, duration, and structured bullet points of achievements.
    * *Education:* Degrees, institutions, graduation dates, certifications.
* **Tech Stack:** LLM API (provider-hosted), Instructor, Pydantic.
* **Output:** Validated `CV_Structured_Data.json`.

### Module 3: Hybrid Retrieval & Semantic Re-ranking
* **Input:** `CV_Structured_Data.json` and raw Job Description (JD) text.
* **Core Processing:**
    1.  **Vector Embedding:** The JD text and parsed CV profiles are mapped into a unified vector space using the **BAAI/bge-m3** embedding model, chosen for its excellent multilingual support and long-context handling.
    2.  **Dense Vector Search:** Embeddings are indexed into a high-performance Vector Database (**Milvus** or **Qdrant**). A Top-K approximate nearest neighbor (ANN) search retrieves candidates with strong semantic overlap.
    3.  **Cross-Encoder Re-ranking:** Bi-encoders (like bge-m3) calculate embeddings independently, losing deep cross-context nuances. To eliminate false positives, the retrieved Top-K CVs are passed through a **Cross-Encoder model**. The Cross-Encoder performs simultaneous attention over both the JD and the CV, generating a highly refined relevance score.
* **Tech Stack:** BAAI/bge-m3, Milvus / Qdrant, HuggingFace Cross-Encoder.
* **Output:** Ranked shortlist of Top-K candidates.

### Module 4: LLM-as-a-Judge Evaluation
* **Input:** Job Description (JD) text and the target candidate's `CV_Structured_Data.json`.
* **Core Processing:**
    * While vector databases rank candidates based on similarity, they cannot perform multi-step critical reasoning. This module acts as the digital recruitment expert.
    * An LLM Evaluator is provided a specialized, zero-shot/few-shot evaluation prompt containing the explicit JD requirements and the candidate's JSON profile.
    * The LLM systematically cross-examines requirements (e.g., "Must have 3+ years in Kubernetes") against the JSON profile and generates an analytical synthesis.
* **Tech Stack:** Hosted LLM API (reasoning-capable model).
* **Output:**
    * **Quantitative Score:** A calibrated matching index from 0 to 100.
    * **Qualitative Analysis:** Bulleted insights detailing **Strengths** (gaps filled) and **Weaknesses** (missing tech stacks or experience gaps).

---

## 3. Technology Stack Matrix

| Component | Technology / Model Selected | Justification |
| :--- | :--- | :--- |
| **Ingestion & OCR** | Hosted VLM API (Qwen2-VL / LLaVA-class) | Layout-aware extraction without maintaining GPU OCR/vision infrastructure. |
| **LLM/VLM Access** | Hosted API (OpenAI/Anthropic/Groq/Together/etc.) | Eliminates model deployment/ops; leverages provider scaling, reliability, and managed updates. |
| **LLM Model Class** | Llama-3-class Instruct (or equivalent) | High performance-to-cost ratio; strong structured extraction and evaluation prompting. |
| **Schema Validation** | Instructor (with Pydantic) | Eliminates runtime JSON syntax errors via constrained decoding. |
| **Embedding Model** | BAAI/bge-m3 | Supports dense, sparse (BM25-like), and multi-vector search modes natively. |
| **Vector Database** | Milvus / Qdrant | Enterprise-grade scaling, partition-key support, and ultra-low search latency. |
| **Re-ranker** | BGE-Reranker-Large | High discriminatory accuracy for complex semantic matching tasks. |

---

## 4. Implementation Strategy & Roadmap

The project is structured across four distinct implementation phases over a targeted timeline:

### Phase 1: Foundations, API Integration, and Data Ingestion (Weeks 1-2)
* Stand up core services: (optional) Milvus/Qdrant and object storage for raw CVs + extracted artifacts.
* Integrate LLM/VLM API providers: secrets management, environment configuration, request timeouts, retries/backoff, rate-limit handling, and basic cost tracking (token usage).
* Build ingestion pipeline: PDF/image normalization, page rendering, and a VLM/OCR step producing clean Markdown/text.
* Add observability: structured logging, request/trace IDs, and error dashboards for extraction failures.

### Phase 2: Schema Extraction, Validation, and Quality Benchmarks (Weeks 3-4)
* Define comprehensive Pydantic schemas for the resume structure.
* Implement Instructor-based extraction using the LLM API, with deterministic validation and repair loops for malformed outputs.
* Create a benchmark set (e.g., 100 diverse resumes) and track: JSON validity rate, field-level accuracy, and processing latency/cost.
* Introduce caching for repeat runs and idempotent processing (same input yields same stored artifacts).

### Phase 3: Retrieval, Re-ranking, and Offline Evaluation (Weeks 5-6)
* Configure embedding pipelines and write custom vector upload scripts.
* Integrate Cross-Encoder modules into the query pipeline.
* Optimize indexing hyper-parameters (e.g., HNSW configuration) for fast, accurate retrieval.
* Build an offline evaluation harness for ranking quality (e.g., Precision@K / nDCG) using labeled JD-CV relevance pairs.

### Phase 4: LLM Judge Calibration, API Productization, and Release Readiness (Weeks 7-8)
* Engineer robust prompts for the LLM-as-a-Judge module.
* Introduce prompt-guard Rails to mitigate hallucinated scoring metrics.
* Develop an API layer (FastAPI) to expose the full pipeline endpoints for application integration.
* Add security and compliance controls: PII handling/redaction policies, data retention, and access controls.
* Load test end-to-end throughput and finalize operational runbooks (alerts, on-call signals, provider failover strategy if needed).
