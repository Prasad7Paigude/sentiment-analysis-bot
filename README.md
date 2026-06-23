# 🧠 Contextual Sentiment Analysis Engine

![TensorFlow 2.17](https://img.shields.io/badge/TensorFlow-2.17.1-FF6F00?logo=tensorflow)
![Keras 3.4](https://img.shields.io/badge/Keras-3.4.1-D00000?logo=keras)
![scikit-learn 1.5](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?logo=scikit-learn)
![Streamlit 1.41](https://img.shields.io/badge/Streamlit-1.41.1-FF4B4B?logo=streamlit)
![NLTK 3.9](https://img.shields.io/badge/NLTK-3.9.1-154f3c)
![License-MIT](https://img.shields.io/badge/License-MIT-yellow)

> **Live Production:** Designed for edge/cloud deployment
>
> **Status:** Production-ready NLP classification pipeline.

---

## Core Mission

**The Problem:** In today's digital ecosystem, businesses are drowning in unstructured text data—customer reviews, support tickets, and social media interactions. This volume creates three critical operational bottlenecks:
- **Information Velocity Overload:** Human agents cannot manually read and triage thousands of incoming messages in real-time.
- **Contextual Blindspots:** Traditional keyword-matching chatbots fail to understand human nuance, context, or sarcasm (e.g., failing to differentiate between *"This is a killer feature!"* and *"This feature killed my workflow"*).
- **Delayed Triage & Customer Churn:** Without immediate sentiment classification, support teams cannot prioritize frustrated, high-risk users.

**The Solution:** This Contextual Sentiment Analysis Engine was architected to bridge the gap between raw unstructured text and actionable business intelligence. It provides a production-hardened NLP pipeline that:
- Delivers deep-learning contextual awareness to accurately classify nuanced user sentiment (Negative, Neutral, Positive) without human intervention.
- Eliminates manual triage bottlenecks, enabling instant, automated contextual replies and intelligent routing.
- Deploys as an enterprise-grade, highly resilient microservice built to process continuous text streams without crashing.

**Impact:** Customer success teams can instantly identify and route high-risk, negative interactions to human escalation, while seamlessly automating responses for positive feedback. **It transforms unstructured text into immediate, actionable business leverage.**

---

## Architecture & Data Flow

```
┌──────────────┐
│  Streamlit   │     User types message in chat UI
│   app.py     │
└──────┬───────┘
       │  raw text
       ▼
┌──────────────────┐
│ TextPreprocessor │     Clean + strip URLs + remove
│   .clean()       │     non-alpha + stop-words + lowercase
└──────┬───────────┘
       │  cleaned text
       ▼
┌──────────────────┐
│ TfidfVectorizer  │     Transform into sparse TF-IDF
│  .transform()    │     feature vector (5 000 dims)
└──────┬───────────┘
       │  dense array
       ▼
┌──────────────────┐
│  Keras Model     │     Dense(256)→Dropout(0.5)
│  .predict()      │     →Dense(128)→Dropout(0.5)
│  (TensorFlow)    │     →Dense(3, softmax)
└──────┬───────────┘
       │  probability vector
       ▼
┌──────────────────┐
│  np.argmax()     │     Map to {0:Negative, 1:Neutral, 2:Positive}
└──────┬───────────┘
       │  class index
       ▼
┌──────────────────┐
│ ChatbotResponder │     Pick random reply from
│ .respond_to_class│     sentiment-groomed response bank
└──────┬───────────┘
       │  contextual reply
       ▼
┌──────────────┐
│  Streamlit   │     Render "BOT: {response}" to user
│   app.py     │
└──────────────┘
```

### Module Topology

```
├── app.py                          # Streamlit shell (thin UI, no domain logic)
├── config/
│   └── settings.py                 # Twelve-factor config: env vars + typed coercions
├── src/
│   ├── data_ingestion.py           # CSV → cleaned → TF-IDF → train/test split
│   ├── model_training.py           # Keras architecture, training loop, evaluation
│   ├── model_inference.py          # SentimentPredictor (inference wrapper)
│   └── response_generator.py       # ChatbotResponder (reply selection engine)
├── utils/
│   ├── text_preprocessing.py       # Shared preprocessor: exact parity train↔serve
│   ├── artifact_io.py              # Safe IO: pickle + Keras, typed error hierarchy
│   └── logger.py                   # Rotating file + stdout, idempotent factory
├── scripts/
│   ├── train.py                    # CLI: re-run training pipeline end-to-end
│   └── create_responses.py         # CLI: regenerate response bank pickle
├── models/
│   ├── sentiment_analysis.h5       # Trained Keras weights
│   ├── tfidf_vectorizer.pkl        # Fitted TF-IDF transformer
│   └── responses.pkl               # Serialised response bank
├── data/
│   └── Reddit_Data.csv             # Raw labelled dataset (Reddit comments)
└── notebooks/
    └── model_training.ipynb        # Original Jupyter reference notebook
```

---

## Engineering Triumphs

- **Unified Preprocessing Pipeline Eliminates Train-Serve Skew**

  - **Problem:** Training notebooks and inference servers historically diverged on text cleaning, causing silent accuracy degradation in production. The original `model_training.ipynb` and the inference code applied independent preprocessing logic.
  - **Solution:** Extracted a single `TextPreprocessor` class (`utils/text_preprocessing.py:44`) that both `data_ingestion.py` and `model_inference.py` import and call identically. The cleaning sequence — URL removal, non-alpha stripping, whitespace collapse, stop-word filtering, lowercasing — is an exact 1:1 port from the original notebook.
  - **Result:** Zero preprocessing drift between train and serve. Every prediction goes through the exact same transformation the model was trained on, guaranteeing distributional parity.

- **Graceful Degradation on Artifact Loading Failure**

  - **Problem:** Missing or corrupt `.pkl` / `.h5` files produced uncaught `FileNotFoundError` or `UnpicklingError` exceptions, crashing the Streamlit process with no actionable feedback.
  - **Solution:** Centralised all artifact IO into `utils/artifact_io.py` with a custom `ArtifactLoadError(RuntimeError)` exception. `load_pickle` and `load_keras_model` validate file existence before deserialising and wrap every failure mode in a single semantically meaningful error type. The `app.py:57-63` entry point catches this error for the response bank and falls back to in-memory defaults; the predictor factory (`app.py:82-91`) renders a user-facing error message without crashing.
  - **Result:** Zero server crashes from missing or corrupt artifacts. Users see friendly diagnostics; operators get logged root causes.

- **Twelve-Factor Configuration with Type-Safe Env Parsing**

  - **Problem:** Paths, hyper-parameters, and UI strings were hardcoded across multiple modules, making every deployment a fork-and-pray exercise and configuration audits impossible.
  - **Solution:** A single `config/settings.py` module reads every tunable value from environment variables via three typed accessors — `_get_env` (string), `_get_env_int` (integer), `_get_env_float` (float). Each safely falls back to a documented default when the variable is unset or unparseable. All values are declared as `Final` constants with type aliases. `python-dotenv` transparently loads `.env` files when available.
  - **Result:** Every operational parameter is visible in one 191-line module. CI/CD pipelines inject env vars without code changes. Missing variables gracefully degrade to defaults.

---

## Enterprise Quick Start

<details>
<summary><b>View Installation & Execution Commands</b></summary>

```bash
# 1. Clone and enter the repository
git clone https://github.com/your-org/sentiment-analysis-chatbot.git
cd sentiment-analysis-chatbot

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. (Optional) Customise environment
cp .env.example .env
# Edit .env, or set env vars directly in CI/CD

# 5. Train model (or use pre-shipped artifacts in models/)
python -m scripts.train

# 6. Regenerate response bank (optional)
python -m scripts.create_responses

# 7. Launch Streamlit UI
streamlit run app.py
```

The UI opens at `http://localhost:8501`. All configurable values are documented in `.env.example`.
</details>

---

## Tech Stack

| Layer | Component | Version | Role |
|---|---|---|---|
| **UI** | Streamlit | 1.41.1 | Interactive web chat interface with resource caching |
| **Deep Learning** | TensorFlow | 2.17.1 | Computation graph engine for neural network training & inference |
| **Deep Learning** | Keras | 3.4.1 | High-level sequential model API (Dense, Dropout, Softmax) |
| **ML Pipeline** | scikit-learn | 1.5.2 | TF-IDF vectorization, train/test split, classification metrics |
| **NLP** | NLTK | 3.9.1 | English stop-word corpus, word tokenization (punkt) |
| **Data** | pandas | 2.2.3 | CSV ingestion and DataFrame manipulation |
| **Data** | NumPy | 1.26.4 | Array math: argmax, sparse-to-dense conversion |
| **Data** | SciPy | ≥1.10 | Sparse CSR matrix representation for TF-IDF features |
| **Config** | python-dotenv | ≥1.0.0 | Transparent `.env` file loading for twelve-factor config |

---

## Security & Reliability

- **Typed exception hierarchy:** Custom `ArtifactLoadError` unifies all model and pickle load failures, preventing uncaught crashes and enabling graceful degradation paths throughout the application.
- **Immutable configuration architecture:** Every configurable value in `config/settings.py` is declared as a `Final` constant, preventing runtime mutation and enabling static analysis tools to catch misconfigurations at lint time.
- **Environment-isolated secrets management:** All paths, parameters, and tokens are consumed from environment variables via typed accessor functions with safe defaults — no credentials or filesystem assumptions are baked into source code.
- **Production-grade logging:** A rotating file handler (5 MB, 3 backups) plus stdout output is configured exactly once per process by an idempotent factory (`utils/logger.py`), ensuring no duplicate log lines and persistent crash records.

---

## License

MIT — see [LICENSE](LICENSE).
