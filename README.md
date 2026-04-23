# W30D4 — CI/CD for ML Products (Demo Repo)

**AISE 2026 · Week 30, Day 4**

This repo is used during the W30D4 live class demo to show GitHub Actions running a CI pipeline for an ML product.

---

## What's in this repo

```
.
├── .github/workflows/
│   ├── ml-prod.yaml              # Main CI pipeline (referenced on slide 16)
│   └── ml-prod-additional.yaml   # ML-specific extensions (referenced on slide 17)
├── app.py                        # Minimal FastAPI app with /health, /predict, /metadata
├── requirements.txt              # Python dependencies
├── scripts/
│   ├── evaluate_model.py         # Stub model evaluation (called by ml-prod.yaml)
│   └── validate_data.py          # Stub data validation (called by ml-prod-additional.yaml)
└── tests/
    ├── __init__.py
    └── test_api.py               # Pytest suite covering the API
```

---

## Running locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run ML evaluation
python scripts/evaluate_model.py --threshold 0.85

# Start the API
uvicorn app:app --reload
curl http://localhost:8000/health
```

---

## Demo workflow during class

See `DEMO_GUIDE.md` for the step-by-step demo script the instructor follows during class.

**Quick version:**

1. Show the green baseline run in the Actions tab
2. Make a small edit (change the version string or add a comment) → commit → push → show pipeline running
3. Make an intentionally breaking edit (e.g., lower the model accuracy below threshold) → commit → push → show red X in Actions
4. Fix the break → commit → push → show green again

---

## Intentional simplifications

- The model is a stub (no real training)
- `evaluate_model.py` returns a hardcoded accuracy
- `validate_data.py` returns a hardcoded pass

These are intentional — they let the pipeline run end-to-end in under 30 seconds without real training data or GPU resources, so the demo stays focused on the *pipeline mechanics*, not ML.

---

## Starter kit for fellows

Fellows will build their own version of these files in their team capstone repos during the hands-on lab. This repo is a reference implementation — not a template to fork.
