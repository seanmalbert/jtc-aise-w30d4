# W30D4 Reference — Data Validation & Model Performance Testing

**Companion document for W30D4 CI/CD for ML Products**

This reference covers two of the main questions for Breakout 1:

1. *"What kinds of tests would actually validate training data schema?"*
2. *"How do you check model performance against a threshold?"*

---

## Part 1 — Validating Training Data Schema

Data schema validation tests fall into roughly six categories, from cheap/fast to more sophisticated. Teams working on their MVP tonight should focus on categories 1–3. Categories 4–6 are stretch goals and real-world patterns fellows will encounter in production.

### 1. Structural tests — the easiest and most common

These check that the dataset has the right *shape* before you let it anywhere near a model.

- Column names match the expected schema (no typos, no missing columns, no unexpected extras)
- Column data types are correct (the `age` column is actually numeric, not string)
- Required columns aren't entirely null
- Row count is within expected bounds (not zero, not 10x larger than usual)

**Example:**

```python
import pandas as pd

EXPECTED_COLUMNS = ["text", "label"]
EXPECTED_DTYPES = {"text": "object", "label": "object"}

def test_schema_structure(df: pd.DataFrame):
    assert list(df.columns) == EXPECTED_COLUMNS, f"Columns mismatch: {df.columns}"
    for col, expected_dtype in EXPECTED_DTYPES.items():
        actual_dtype = df[col].dtype.name
        assert actual_dtype == expected_dtype, f"{col} has dtype {actual_dtype}, expected {expected_dtype}"
    assert len(df) > 0, "Dataset is empty"
    assert len(df) < 10_000_000, "Dataset is suspiciously large"
```

### 2. Value-range tests — catch corrupted or impossible values

- Numeric columns fall within valid ranges (age between 0 and 120, probability between 0 and 1)
- Categorical columns only contain known labels (a `country` column shouldn't suddenly have "Atlantis")
- Date columns fall within plausible time ranges (no timestamps from 1970 or 2099)
- No negative values where they're nonsensical (negative prices, negative counts)

**Example:**

```python
ALLOWED_LABELS = {"positive", "negative"}

def test_label_values(df: pd.DataFrame):
    invalid_labels = set(df["label"].unique()) - ALLOWED_LABELS
    assert not invalid_labels, f"Unexpected labels found: {invalid_labels}"
```

### 3. Null and completeness tests — missing data has real consequences for models

- Null rate per column stays below a threshold (e.g., "less than 5% of rows are missing `user_id`")
- Required fields are never null
- Null patterns aren't concentrated in specific segments (if all the nulls are from one demographic, you have a sampling bias problem, not a data quality problem)

**Example:**

```python
REQUIRED_COLUMNS = ["text", "label"]
MAX_NULL_RATE = 0.05  # 5%

def test_completeness(df: pd.DataFrame):
    for col in REQUIRED_COLUMNS:
        null_rate = df[col].isnull().mean()
        assert null_rate <= MAX_NULL_RATE, \
            f"{col} has {null_rate:.1%} nulls, exceeds {MAX_NULL_RATE:.0%} threshold"
```

### 4. Uniqueness and duplicate tests — important for training data integrity

- Primary keys are actually unique (no duplicate user IDs)
- Training and test sets don't overlap (a huge source of silent bugs that inflate accuracy scores)
- No duplicate rows where duplicates would bias training

**Example:**

```python
def test_no_train_test_leakage(train_df, test_df):
    train_ids = set(train_df["id"])
    test_ids = set(test_df["id"])
    overlap = train_ids & test_ids
    assert not overlap, f"Train/test leakage: {len(overlap)} overlapping IDs"
```

### 5. Distribution tests — catch data drift, the silent killer of ML products

- Feature distributions today match the training distribution (means, medians, variances within tolerance)
- Class balance hasn't shifted dramatically (if your training data was 50/50 positive/negative and today's data is 90/10, the model's assumptions no longer hold)
- Statistical tests like Kolmogorov-Smirnov or Population Stability Index (PSI) for numeric features, Chi-square for categorical features

**Example:**

```python
from scipy import stats

def test_distribution_stability(reference_df, current_df, column, p_threshold=0.05):
    statistic, p_value = stats.ks_2samp(reference_df[column], current_df[column])
    assert p_value > p_threshold, \
        f"{column} distribution shifted (p={p_value:.4f})"
```

Distribution tests usually run on a schedule (nightly or weekly) rather than per-commit, because they need a reference baseline to compare against.

### 6. Relational and business-logic tests — the hardest to write but most valuable

- Cross-column constraints (shipping date must be after order date)
- Referential integrity (every `order_id` in the orders table exists in the products table)
- Domain-specific rules (a medical record with `pregnancy=true` and `gender=male` is a data error, not a valid edge case)

**Example:**

```python
def test_business_logic(df):
    # Shipping date must be after order date
    invalid = df[df["ship_date"] < df["order_date"]]
    assert len(invalid) == 0, f"{len(invalid)} rows have ship_date before order_date"
```

### Real-world tools teams use

- **Great Expectations** — a popular Python library for data validation. You write "expectations" like `expect_column_values_to_be_between("age", 0, 120)` and get automated reports
- **Pandera** — schema validation for pandas DataFrames with a more code-first style
- **Deequ** — Amazon's data quality library, used in big-data pipelines
- **TensorFlow Data Validation (TFDV)** — Google's library, especially good at drift detection
- **dbt tests** — for teams whose data lives in a warehouse, dbt has built-in tests for uniqueness, null checks, and accepted values

### How this plays in a CI pipeline

A typical `validate_data.py` step in CI does three things:

1. Loads a sample of the training data (or the full dataset if small)
2. Runs a predefined suite of structural, value-range, and completeness checks
3. Exits with code 1 and a clear error message if any check fails

The cheap tests (categories 1–4) run on every commit because they're fast. The expensive tests (category 5, distribution drift) often run nightly or weekly rather than per-commit.

### Guidance for Breakout 1

For their data validation tests, fellows should focus on categories 1–3 — structural, value-range, and null checks. Those are fast, cheap, and catch 80% of real problems. Distribution testing and cross-column business logic are great stretch goals but not required for a working MVP.

**Starter example for `testing_strategy.md`:**

> **Test: Training data schema validation**
> - **What it checks:** The training CSV has columns `[text, label]` in that order, with `text` as string and `label` as one of `{positive, negative}`, and no null values in either column
> - **When it runs:** On every commit that touches `data/` or `scripts/train_model.py`
> - **What failure looks like:** Pipeline exits 1 with a message identifying which column failed and how many rows were affected
> - **Severity:** 🛑 Blocker — training on malformed data produces a broken model

---

## Part 2 — Checking Model Performance Against Thresholds

Model performance tests answer the question CI/CD for ML exists to answer: *does the model still work well enough to ship?* These tests are the ML-specific layer on top of standard unit tests.

Like data validation, performance tests fall into categories from cheap/fast to more sophisticated. Fellows should implement at least one from categories 1–2 tonight.

### 1. Accuracy threshold tests — the baseline check

The simplest and most common ML test: *does the model meet a minimum accuracy threshold on a held-out test set?*

**Example (this is what your `scripts/evaluate_model.py` does):**

```python
def test_model_accuracy(model, X_test, y_test, min_accuracy=0.85):
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()
    assert accuracy >= min_accuracy, \
        f"Accuracy {accuracy:.4f} below threshold {min_accuracy:.4f}"
```

**Key design choices:**

- **Use a held-out test set.** Never evaluate on training data — your model has already seen it and will look better than it really is
- **Pick a threshold that's meaningful, not just achievable.** Setting the bar at 0.50 for a binary classifier is meaningless — random guessing hits that. Set it based on what your product actually needs
- **Lock the test set.** If the test data changes every run, your threshold loses meaning. The same test set should be used across runs for comparable results

### 2. Multiple metric tests — accuracy alone can lie

Accuracy alone is misleading when classes are imbalanced. A fraud detector that predicts "not fraud" for every transaction might hit 99% accuracy while being completely useless. Use precision and recall together, or F1 as a balance metric.

**Example:**

```python
from sklearn.metrics import precision_score, recall_score, f1_score

def test_model_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)

    precision = precision_score(y_test, predictions, average="binary")
    recall = recall_score(y_test, predictions, average="binary")
    f1 = f1_score(y_test, predictions, average="binary")

    assert precision >= 0.80, f"Precision {precision:.4f} below 0.80"
    assert recall >= 0.75, f"Recall {recall:.4f} below 0.75"
    assert f1 >= 0.77, f"F1 {f1:.4f} below 0.77"
```

**Which metric for which problem:**

| Metric | When it matters |
|---|---|
| **Accuracy** | Balanced classes, roughly equal cost for both error types |
| **Precision** | False positives are costly (spam filter marking real email as spam, fraud alert bothering good customers) |
| **Recall** | False negatives are costly (cancer screening missing real cases, security alert missing a real attack) |
| **F1** | You need balance between precision and recall |
| **AUC-ROC** | You care about ranking predictions by confidence, not just final classification |
| **MAE / RMSE** | Regression problems (predicting a number, not a class) |

### 3. Performance regression tests — did we get worse?

A subtler and more important check: *compared to the last deployed model, did we regress?* A new model might still hit 0.85 accuracy but drop from the previous 0.92 — that's a regression you want to catch.

**Example:**

```python
import json

def test_no_regression(model, X_test, y_test, baseline_path="models/baseline_metrics.json"):
    predictions = model.predict(X_test)
    new_accuracy = (predictions == y_test).mean()

    with open(baseline_path) as f:
        baseline = json.load(f)
    baseline_accuracy = baseline["accuracy"]

    # Allow some tolerance for noise (e.g., 1% degradation is acceptable)
    TOLERANCE = 0.01
    assert new_accuracy >= baseline_accuracy - TOLERANCE, \
        f"Regression: {new_accuracy:.4f} vs baseline {baseline_accuracy:.4f}"
```

**Key design choice:** Build a baseline metrics file into your model registry or repo. Every time you promote a model to production, update the baseline. Every CI run compares the new candidate against that baseline.

### 4. Latency and size thresholds — performance is more than accuracy

A 99% accurate model that takes 30 seconds to respond is broken in practice. CI should catch performance regressions in response time and model artifact size.

**Example (this is what the `--max-latency-ms` flag does in your `evaluate_model.py`):**

```python
import time

def test_inference_latency(model, sample_inputs, max_latency_ms=200):
    latencies = []
    for inp in sample_inputs:
        start = time.perf_counter()
        _ = model.predict([inp])
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    assert p95 <= max_latency_ms, \
        f"P95 latency {p95:.1f}ms exceeds {max_latency_ms:.1f}ms"


def test_model_size(model_path, max_mb=100):
    import os
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    assert size_mb <= max_mb, \
        f"Model size {size_mb:.1f}MB exceeds {max_mb:.1f}MB"
```

**Key design choice:** Use percentiles (p50, p95, p99) rather than averages. A model with a 100ms average but a 5-second p99 will still feel broken to some users. *This directly previews the performance optimization work fellows will do in W32D2.*

### 5. Slice-based tests — does the model work for everyone?

Overall accuracy can hide that the model performs great for 80% of users and terribly for the other 20%. Slice-based tests evaluate performance across important subgroups.

**Example:**

```python
def test_performance_by_slice(model, X_test, y_test, slice_column):
    slices = X_test[slice_column].unique()
    for slice_value in slices:
        mask = X_test[slice_column] == slice_value
        X_slice = X_test[mask]
        y_slice = y_test[mask]

        if len(X_slice) < 50:
            continue  # Skip small slices for statistical reliability

        predictions = model.predict(X_slice)
        slice_accuracy = (predictions == y_slice).mean()

        assert slice_accuracy >= 0.80, \
            f"Accuracy for {slice_column}={slice_value}: {slice_accuracy:.4f} below 0.80"
```

**When to use this:** Any time your product serves a diverse user base. This is how you catch fairness issues, demographic bias, and edge-case failures before they reach users.

### 6. Behavioral tests — does the model reason correctly?

Not just "is the prediction right" but "does it change appropriately when inputs change?" These are sometimes called invariance tests or directional expectation tests.

**Example:**

```python
def test_invariance_to_capitalization(model):
    pred_lower = model.predict(["this is a great product"])[0]
    pred_upper = model.predict(["THIS IS A GREAT PRODUCT"])[0]
    assert pred_lower == pred_upper, \
        "Prediction should be invariant to capitalization"


def test_sentiment_direction(model):
    positive_pred = model.predict(["I loved this"])[0]
    negative_pred = model.predict(["I hated this"])[0]
    assert positive_pred != negative_pred, \
        "Model should distinguish sentiment direction"
```

**Reference:** Ribeiro et al., *"Beyond Accuracy: Behavioral Testing of NLP Models with CheckList"* (2020) — the canonical paper on this approach.

### Real-world tools teams use

- **MLflow** — tracks metrics across runs and makes baseline comparisons easy
- **Weights & Biases** — experiment tracking with automated performance comparison
- **scikit-learn's `metrics` module** — covers most standard classification and regression metrics
- **DeepChecks** — comprehensive ML validation library with built-in performance, drift, and fairness tests
- **CheckList** — the library from the Ribeiro et al. paper for behavioral testing

### How this plays in a CI pipeline

A typical `evaluate_model.py` step runs five checks:

1. Load the trained model and a locked test dataset
2. Generate predictions and compute accuracy, precision, recall, F1
3. Compare against minimum thresholds (blocker if below)
4. Compare against baseline metrics from the last deployed model (blocker if regressed beyond tolerance)
5. Exit 0 if all pass, 1 with specific error messages if any fail

The key design principle: **thresholds should be product-driven, not model-driven.** Don't set the threshold at whatever your current model achieves. Set it at what your product actually needs, then iterate the model to meet it.

### Guidance for Breakout 1

For their model performance tests tonight, fellows should implement at least:

- One **accuracy threshold test** (category 1) — the baseline check
- One **additional metric** (category 2) — precision, recall, or F1 depending on their product

Latency tests (category 4) are a nice stretch goal and a natural bridge to W32D2. Regression tests and slice-based tests are real-world patterns worth mentioning but not required tonight.

**Starter example for `testing_strategy.md`:**

> **Test: Model accuracy threshold**
> - **What it checks:** The trained model achieves at least 0.85 accuracy on the held-out test set
> - **When it runs:** On every commit that touches `scripts/train_model.py` or `data/train.csv`
> - **What failure looks like:** Pipeline exits 1 with a message showing actual vs. expected accuracy
> - **Severity:** 🛑 Blocker — deploying a model below threshold means shipping broken product behavior

---

## Common Questions During Breakout 1

**"What accuracy threshold should we pick?"**

Think product, not model. Ask: *what happens if the model is wrong 15% of the time? Is that acceptable for our users?* If yes, 0.85 is your threshold. If no (e.g., medical diagnosis), set it higher. If you can tolerate more errors (e.g., casual recommendations), set it lower. The threshold should force the model to meet the product's needs, not the other way around.

**"Our dataset is tiny — can we still do data validation?"**

Yes, and it matters more for small datasets because one bad row is a bigger percentage of your data. Structural and null checks are still fast and cheap. You might skip distribution tests (not enough statistical power) but everything else still applies.

**"Should we test on training data or a separate test set?"**

Always a separate test set. Testing on training data is like grading your own homework — the model has already seen those examples and will look better than it really is. If you don't have a test set yet, carve off 20% of your data and lock it away before training.

**"How often should these tests run?"**

- Structural, value-range, completeness, accuracy threshold → **on every commit** (fast, cheap)
- Distribution drift, slice-based performance → **nightly or weekly** (more expensive, needs baseline)
- Behavioral tests → **on every commit** if they're fast, **nightly** if they're slow

**"What if a test fails but we think the model is actually fine?"**

First, don't silence the test. A failing test is telling you something. Your options are:

1. **Fix the model** — the usual and correct path
2. **Adjust the threshold** — if the original threshold was wrong (with documented justification)
3. **Mark the test as a warning instead of a blocker** — if the failure isn't product-critical

What you should not do: delete the test, ignore the failure, or set the threshold to whatever the model currently achieves. That defeats the purpose of having tests.

---

## Quick Reference — What to Build Tonight

A minimal but complete `testing_strategy.md` for a typical capstone MVP has:

**Data validation (categories 1–3):**
- Schema test — correct columns, correct types
- Value range test — labels are in the allowed set, numerics in expected ranges
- Completeness test — required columns have acceptable null rates

**Model performance (categories 1–2):**
- Accuracy threshold test — model meets minimum on held-out test set
- Precision or recall test (pick based on product risk) — or F1 if balanced matters

**API tests (from the main lesson):**
- `/health` returns 200
- `/predict` accepts valid input and returns a prediction
- `/predict` rejects invalid input with a 422

Seven tests, each labeled 🛑 Blocker / 💡 Warning / 📊 Monitoring. That's a complete starter CI pipeline for an ML product MVP.

---

## Further Reading

For fellows who want to go deeper after class:

- Ribeiro et al., *"Beyond Accuracy: Behavioral Testing of NLP Models with CheckList"* (2020) — [arxiv.org/abs/2005.04118](https://arxiv.org/abs/2005.04118)
- Breck et al., *"The ML Test Score: A Rubric for ML Production Readiness"* (Google, 2017) — [research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)
- Great Expectations documentation — [greatexpectations.io](https://greatexpectations.io)
- scikit-learn metrics guide — [scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
- Google Cloud Architecture — *MLOps: Continuous delivery and automation pipelines in ML* — [cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
