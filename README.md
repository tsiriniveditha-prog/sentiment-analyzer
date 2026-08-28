#  SentimentIQ — Sentiment Analysis Web App




> A hybrid NLP sentiment analyzer built with Python, Flask, and a custom rule-based + lexicon ML model — no external ML libraries required.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-black?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen?style=flat-square)

---

##  Live Application
Access the deployed web application:
 **[https://sentiment-analyzer-dahc.onrender.com/](https://sentiment-analyzer-dahc.onrender.com/)**


---

##  Overview

**SentimentIQ** is a web application that classifies text as **Positive**, **Negative**, or **Neutral** using a custom hybrid NLP model. It is built entirely from scratch without relying on heavy ML frameworks — demonstrating core machine learning concepts like feature extraction, scoring functions, normalization, and confidence estimation.

This project was developed as part of the **Amazon ML Summer School 2026** program, targeting key ML topics including:
- Natural Language Processing (NLP)
- Feature Engineering
- Rule-based vs. Learning-based models
- Model evaluation and testing

---

##  Features

| Feature | Description |
|---|---|
|  Real-time Analysis | Analyze any text instantly via a clean web UI |
|  Confidence Score | Normalized confidence metric [0–100%] per prediction |
|  Word Highlights | Visual breakdown of positive/negative keywords |
|  Emoji Support | Handles emoji-based sentiment signals |
|  Negation Handling | Detects phrases like "not bad", "didn't hate" |
|  Intensifiers | Boosts scores for words like "very", "extremely" |
|  Batch API | `/batch` endpoint for analyzing multiple texts |
|  History Panel | Tracks last 5 analyses in the UI |
|  Full Test Suite | 15+ unit tests with pytest |

---

##  Project Structure

```
sentiment-analyzer/
│
├── app.py                    # Flask web server + API routes
├── requirements.txt          # Python dependencies
├── pytest.ini                # Pytest configuration
├── .gitignore
│
├── model/
│   ├── __init__.py
│   └── sentiment_model.py    # Core ML model (hybrid NLP)
│
├── templates/
│   └── index.html            # Frontend (HTML + CSS + JS)
│
└── tests/
    └── test_sentiment.py     # Unit test suite (pytest)
```

---

##  How the Model Works

The model uses a **3-layer hybrid scoring approach**:

```
Input Text
    │
    ▼
[1] Tokenization + Cleaning
    │
    ▼
[2] Lexicon Scoring
    ├─ Positive word hits  (+1.0 each)
    ├─ Negative word hits  (-1.0 each)
    ├─ Negation window     (flips sign within 3 tokens)
    └─ Intensifier boost   (×1.5 multiplier)
    │
    ▼
[3] Emoji Scoring
    ├─ Positive emojis     (+0.5 each)
    └─ Negative emojis     (-0.5 each)
    │
    ▼
[4] Normalization
    └─ score / √(token_count) → clamped to [-1.0, +1.0]
    │
    ▼
[5] Label Assignment
    ├─ score > 0.1  → Positive
    ├─ score < -0.1 → Negative
    └─ otherwise    → Neutral
```

### Why This Approach?

| Concept | Implementation |
|---|---|
| Feature Engineering | Words, emojis, negations, intensifiers as features |
| Normalization | Length-invariant scoring via √n dampening |
| Confidence | Distance from decision boundary, scaled to [0, 1] |
| Negation Handling | Context window of 3 tokens before each word |

---

##  Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/sentiment-analyzer.git
cd sentiment-analyzer

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Then open your browser at: **http://127.0.0.1:5000**

---

## 🔌 API Reference

### `POST /analyze`

Analyze a single text string.

**Request:**
```json
{
  "text": "This product is absolutely amazing!"
}
```

**Response:**
```json
{
  "label": "Positive",
  "score": 0.7071,
  "confidence": 0.91,
  "word_count": 5,
  "highlights": [
    { "word": "amazing", "type": "positive" }
  ],
  "text_preview": "This product is absolutely amazing!"
}
```

---

### `POST /batch`

Analyze multiple texts in one request.

**Request:**
```json
{
  "texts": [
    "Great product!",
    "Terrible service.",
    "It arrived today."
  ]
}
```

**Response:**
```json
{
  "results": [
    { "label": "Positive", "score": 0.707, ... },
    { "label": "Negative", "score": -0.707, ... },
    { "label": "Neutral",  "score": 0.0, ... }
  ]
}
```

---

##  Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage (optional)
pip install pytest-cov
python -m pytest tests/ --cov=model -v
```

Expected output:
```
tests/test_sentiment.py::TestSentimentLabels::test_positive_text    PASSED
tests/test_sentiment.py::TestSentimentLabels::test_negative_text    PASSED
tests/test_sentiment.py::TestSentimentLabels::test_neutral_text     PASSED
tests/test_sentiment.py::TestSentimentLabels::test_negation         PASSED
tests/test_sentiment.py::TestSentimentLabels::test_intensifier      PASSED
... (15+ tests total)
```

---

##  ML Concepts Demonstrated

This project intentionally demonstrates foundational ML concepts that are central to the Amazon ML Summer School curriculum:

1. **Natural Language Processing** — Tokenization, lexicon lookup, feature extraction from text
2. **Feature Engineering** — Transforming raw text into numerical signals (score)
3. **Normalization** — Making predictions length-invariant using √n dampening
4. **Confidence Estimation** — Converting raw scores to interpretable confidence values
5. **Evaluation** — Writing unit tests that validate model behavior on edge cases
6. **API Design** — Exposing ML models via RESTful endpoints (real-world deployment pattern)
7. **Negation & Context Windows** — Understanding that meaning is context-dependent

---

##  Future Improvements

- [ ] Integrate a trained ML classifier (Naive Bayes / Logistic Regression) using scikit-learn
- [ ] Add CSV upload for bulk batch processing
- [ ] Export analysis history as JSON/CSV
- [ ] Add language detection and multi-language support
- [ ] Deploy on AWS (EC2 / Elastic Beanstalk) — aligned with Amazon ecosystem
- [ ] Add a training pipeline with labeled dataset (IMDb / SST-2)

---

##  Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

##  License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---


- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [your-profile](https://linkedin.com/in/your-profile)

---

> *Built for Amazon ML Summer School 2026 — an initiative to equip students with industry-level ML skills through hands-on projects and mentorship from Amazon Scientists.*
