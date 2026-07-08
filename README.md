# NSSF Age Eligibility Checker

A machine learning system that determines whether an applicant qualifies for NSSF (National Social Security Fund) age-based benefits — combining hard mandatory checks with a trained classifier for the final confidence score.

## How eligibility works

Eligibility isn't decided by the ML model alone. It's a two-stage gate:

1. **Mandatory requirements (hard stop).** An applicant must have a valid NSSF number, be 55 or older, pass biometric verification, have bank details on file, a valid ID document, a photograph, matching name and parent-name records, and a statement cleanliness score of at least 75%. Fail any one of these and the application is rejected immediately — no ML involved, no exceptions.
2. **ML confidence score.** If every mandatory check passes, a Random Forest classifier scores the application. Approval requires a predicted probability of at least 80% — a deliberately conservative bar given this feeds financial decisions.

This split matters: it means the model can never override a hard rule (e.g., approve someone under 55), it just adds judgment on top of applicants who already meet every baseline requirement.

## Components

| File | What it does |
|---|---|
| `nssf_eligibility_model.py` | Defines `ProductionNSSFEligibilityModel` — generates synthetic training data, trains the Random Forest, validates applicants, saves/loads the model with `joblib` |
| `nssf_api.py` | Flask REST API that loads the trained model and serves predictions |
| `nssf_production_model.pkl` | Pre-trained model artifact, ready to load without retraining |
| `form.html` | Standalone HTML form UI for submitting an applicant's details |
| `cli.py` | Interactive command-line client for the API |
| `requirements.txt` | Python dependencies |

## Model

- **Algorithm:** Random Forest (100 trees, max depth 10, balanced class weights)
- **Features:** the 9 critical fields listed above
- **Training data:** synthetically generated (5,000 samples by default, 60% eligible / 40% ineligible, seeded for reproducibility) — there's no real applicant data in this repo
- **Decision threshold:** 0.8 predicted probability for approval

Because training data is synthetic, this is best framed as a demonstration of the eligibility logic and pipeline architecture rather than a model trained on real outcomes.

## API endpoints

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | API info and available endpoints |
| GET | `/health` | Health check, confirms model is loaded |
| GET | `/test` | Runs a prediction against a built-in sample applicant |
| POST | `/predict` | Single applicant prediction |
| POST | `/predict/batch` | Batch predictions — send `{"claims": [...]}` |
| GET | `/model/info` | Model metadata and the list of mandatory requirements |
| POST | `/model/retrain` | Retrains the model from scratch on fresh synthetic data |

### Example request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "nssf_number": "123456789",
    "age": 58,
    "biometric_match": 1,
    "has_bank_details": 1,
    "has_id_document": 1,
    "has_photograph": 1,
    "name_match_score": 1,
    "parent_name_match": 1,
    "statement_cleanliness": 85
  }'
```

## Setup

```bash
pip install -r requirements.txt
python nssf_api.py
```

The API starts on `http://0.0.0.0:5000`. On first run, if `nssf_production_model.pkl` isn't found it trains a fresh model automatically; otherwise it loads the one already included in this repo.

## Known gaps (things to fix before calling this production-ready)

- `cli.py` imports from a module called `nssf_api_client`, which doesn't exist in this project yet — the CLI won't run until that client module (wrapping calls to the Flask endpoints) is written.
- `form.html` is a standalone file — it isn't served by `nssf_api.py` (no route or `render_template` wired up), so right now it's a disconnected frontend mockup rather than a working UI.
- `app.run(debug=True, ...)` in `nssf_api.py` should be turned off before any real deployment — debug mode exposes a lot more than you want in production.

## Possible next steps

- Write `nssf_api_client.py` so the CLI actually works
- Wire `form.html` up to the API, either via Flask routes or a small separate frontend
- Swap the synthetic training data for anonymized real outcomes if this ever moves past demo stage
- Add authentication to the API before exposing it beyond localhost

## Author

Edgar Katuramu
