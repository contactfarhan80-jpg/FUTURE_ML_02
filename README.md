# Task 2 — Support Ticket Classification

Build a system to automatically classify customer support tickets and assign priority levels.

## Features
- ✅ Text cleaning & tokenization (lowercasing, punctuation/URL/number stripping, stopword removal)
- ✅ Ticket category classification (Account / Billing / Technical / General)
- ✅ Priority tagging (High / Medium / Low) with a rule-based urgency-keyword booster
- ✅ Model performance evaluation (accuracy, weighted F1)

## Tech Stack
- **Backend:** Python, Flask, scikit-learn (TF-IDF + calibrated LinearSVC), NLTK (optional), joblib
- **Frontend:** HTML, CSS, vanilla JavaScript

## Project Structure
```
task2_ticket_classification/
├── backend/
│   ├── app.py                  # Flask API
│   ├── classifier.py           # Text preprocessing + training + inference
│   ├── requirements.txt
│   ├── sample_data/
│   │   └── tickets.csv         # 40 labeled sample support tickets
│   └── trained_models/         # created at runtime (joblib model files)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

## Getting Started

### 1. Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
The API runs at `http://localhost:5001`.

> NLTK stopwords are optional — if you want to use them instead of the
> built-in fallback list, run once: `python -m nltk.downloader stopwords`.

### 2. Frontend
Open `frontend/index.html` in your browser. Click **Train Models** first,
then paste (or click a sample) ticket text and click **Classify Ticket**.

> Update `API_BASE` in `frontend/script.js` if your backend runs elsewhere.

## API Reference

| Method | Endpoint          | Description                                                     |
|--------|--------------------|-------------------------------------------------------------------|
| GET    | `/api/health`      | Health check + whether models are trained                        |
| GET    | `/api/sample-data` | Returns the bundled sample tickets                                |
| POST   | `/api/train`       | Trains category & priority classifiers. Optional custom `data`.  |
| POST   | `/api/classify`    | Body: `{"text": "..."}` → category + priority + confidence       |

### Example response (`/api/classify`)
```json
{
  "category": {"label": "Technical", "confidence": 0.87, "distribution": {"Account": 0.03, "Billing": 0.02, "General": 0.08, "Technical": 0.87}},
  "priority": {"label": "High", "confidence": 0.91, "distribution": {"High": 0.91, "Low": 0.02, "Medium": 0.07}, "urgency_keywords_detected": ["down", "urgent"]},
  "cleaned_text": "production server customers checkout urgent"
}
```

## How It Works
1. **Preprocessing:** lowercase → strip URLs/punctuation/numbers → remove stopwords.
2. **Feature extraction:** TF-IDF over unigrams + bigrams.
3. **Modeling:** two independent `LinearSVC` classifiers (category, priority)
   wrapped in `CalibratedClassifierCV` for probability-style confidence scores.
4. **Business logic layer:** a small keyword list (`urgent`, `down`, `cannot`,
   `crash`, etc.) boosts priority to High when detected, even if the model
   itself is unsure — mirroring how real support teams triage.

## Possible Extensions
- Swap TF-IDF + SVM for a fine-tuned transformer (e.g. DistilBERT) for higher accuracy.
- Add a feedback loop where agents can correct predictions to continuously retrain.
- Persist tickets + predictions in a database and add a `/api/tickets` history endpoint.
- Add authentication and a support-agent dashboard view.

## License
MIT — free to use and modify for learning or portfolio purposes.
