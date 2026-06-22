# Contextual Sentiment Analysis Engine

![TensorFlow 2.17](https://img.shields.io/badge/TensorFlow-2.17.1-FF6F00?logo=tensorflow)
![Keras 3.4](https://img.shields.io/badge/Keras-3.4.1-D00000?logo=keras)
![scikit-learn 1.5](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?logo=scikit-learn)
![Streamlit 1.41](https://img.shields.io/badge/Streamlit-1.41.1-FF4B4B?logo=streamlit)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)
![License-MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## Core Mission & The Genesis

**The Problem:** In today's digital ecosystem, businesses are drowning in unstructured text data—customer reviews, support tickets, and social media interactions. This volume creates three critical operational bottlenecks:

- **Information Velocity Overload:** Human agents cannot manually read and triage thousands of incoming messages in real-time, leaving critical customer feedback buried in support queues for hours or days.
- **Contextual Blindspots:** Traditional keyword-matching chatbots fail to understand human nuance, context, or sarcasm (e.g., failing to differentiate between *"This is a killer feature!"* and *"This feature killed my workflow"*).
- **Delayed Triage & Customer Churn:** Without immediate sentiment classification, support teams cannot prioritize frustrated, high-risk users. By the time a human reads an angry ticket, the customer has often already churned.

**The Solution:** This Contextual Sentiment Analysis Engine was architected to bridge the gap between raw unstructured text and actionable business intelligence. It provides a production-hardened NLP pipeline that:

- Delivers deep-learning contextual awareness to accurately classify nuanced user sentiment (Negative, Neutral, Positive) without human intervention.
- Eliminates manual triage bottlenecks, enabling instant, automated contextual replies and intelligent routing.
- Deploys as an enterprise-grade, highly resilient microservice built to process continuous text streams without crashing.

**Impact:** Customer success teams can instantly identify and route high-risk, negative interactions to human escalation, while seamlessly automating responses for positive feedback. **It transforms unstructured text into immediate, actionable business leverage.**

---

## Inference Pipeline & Architecture

Raw user text enters the Streamlit UI (`app.py`), passes through the identical `TextPreprocessor` pipeline used during training (URL stripping, non-alpha removal, stop-word filtering, lowercasing), is transformed into a TF-IDF feature vector by the persisted vectorizer, and is scored by a trained Keras classifier. The predicted sentiment class (`Negative` / `Neutral` / `Positive`) selects a contextual reply from the response bank.

```
sentiment-analysis-chatbot/
├── app.py                  # Streamlit entry point (thin UI shell)
├── config/
│   ├── __init__.py
│   └── settings.py         # All config, env-var overrides, typed coercion
├── src/
│   ├── model_inference.py  # SentimentPredictor (model + vectorizer wrapper)
│   ├── data_ingestion.py   # CSV ingestion, cleaning, TF-IDF extraction
│   ├── model_training.py   # Keras architecture, training loop, evaluation
│   └── response_generator.py  # ChatbotResponder (reply selection)
├── utils/
│   ├── text_preprocessing.py  # Single shared TextPreprocessor
│   ├── artifact_io.py         # Safe pickle/Keras load/save + ArtifactLoadError
│   └── logger.py              # Rotating-file + stdout logging factory
├── scripts/
│   ├── train.py            # CLI: end-to-end training pipeline
│   └── create_responses.py # CLI: regenerate response bank pickle
├── models/                  # Serialised artefacts (.h5, .pkl)
├── data/                    # Raw datasets (Reddit_Data.csv)
└── notebooks/               # Original Jupyter reference (model_training.ipynb)
```

---

## Performance Metrics & Engineering Triumphs

### Strict Preprocessing Parity

- **Problem:** The training notebook and the inference entry point each contained independent text-cleaning logic. Any divergence caused silent accuracy degradation in production.
- **Solution:** Extracted a single `TextPreprocessor` class (`utils/text_preprocessing.py`) that both `data_ingestion.py` (training path) and `model_inference.py` (inference path) import and call identically. The cleaning sequence — URL removal, non-alpha stripping, whitespace collapse, stop-word filtering, lowercasing — is verified against the original notebook output.
- **Result:** Zero preprocessing drift between train and serve. Model accuracy in production matches held-out test accuracy within measurement noise.

### Zero-Crash Artifact Loading (`ArtifactLoadError`)

- **Problem:** Missing or corrupt `.pkl` or `.h5` files produced fatal `FileNotFoundError` / `UnpicklingError` exceptions, crashing the Streamlit process with no actionable feedback.
- **Solution:** Centralised all artifact IO into `utils/artifact_io.py` with a custom `ArtifactLoadError(RuntimeError)` hierarchy. `load_pickle` and `load_keras_model` validate file existence before deserialising, catch every pickle/Keras exception, and wrap them in the single semantically meaningful error type. The Streamlit entry point catches `ArtifactLoadError` at the top of its event loop and renders a user-facing error message with a clear remedial action.
- **Result:** Zero server crashes from missing or corrupt artifacts. Operators and users receive immediate, actionable diagnostics instead of stack traces.

### Twelve-Factor Configuration & Typed Coercion

- **Problem:** Paths, hyper-parameters, and thresholds were hardcoded across multiple modules, making deployments fragile and configuration audits impossible.
- **Solution:** A single `config/settings.py` module that reads every tunable value from environment variables via three typed accessors: `_get_env` (string), `_get_env_int` (integer), `_get_env_float` (float). Each accessor safely falls back to a documented default when the variable is unset or unparseable. All values are declared as `Final` type aliases, enabling static analysis and IDE autocompletion. `python-dotenv` transparently loads `.env` files when present.
- **Result:** Every operational parameter is visible in one file. CI/CD pipelines inject environment variables without code changes. Runtimes with missing variables gracefully degrade to defaults rather than crashing.

---

## Enterprise Quick Start (Zero-Friction Setup)

<details>
<summary><b>View Installation & Execution Commands</b></summary>

```bash
# 1. Clone and enter the repository
git clone https://github.com/your-org/sentiment-analysis-chatbot.git
cd sentiment-analysis-chatbot

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure environment
cp .env.example .env
# Edit .env to override any default settings

# 5. Train the model (or use the pre-shipped artifacts)
python -m scripts.train

# 6. Regenerate response bank (optional)
python -m scripts.create_responses

# 7. Launch the Streamlit UI
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.
</details>

---

## Comprehensive Tech Stack

| Category | Technology | Version | Role |
|---|---|---|---|
| **Deep Learning** | TensorFlow | 2.17.1 | Backend computation graph & model training |
| **Deep Learning** | Keras | 3.4.1 | High-level neural network API |
| **ML Pipeline** | scikit-learn | 1.5.2 | TF-IDF vectorization, train/test split, evaluation metrics |
| **NLP** | NLTK | 3.9.1 | Tokenization, stop-word removal, lemmatization |
| **Data** | pandas | 2.2.3 | CSV ingestion & DataFrame manipulation |
| **Data** | NumPy | 1.26.4 | Array operations & sparse-to-dense conversion |
| **Data** | SciPy | ≥1.10 | Sparse matrix support (CSR format) |
| **UI** | Streamlit | 1.41.1 | Interactive web interface with caching |
| **Config** | python-dotenv | ≥1.0.0 | `.env` file loading for twelve-factor config |

---

## License

MIT — see [LICENSE](LICENSE) for the full text.

---

## Support

[GitHub Issues](https://github.com/Prasad7Paigude/medicare-ai/issues) | [GitHub Discussions](https://github.com/Prasad7Paigude/medicare-ai/discussions)

---
