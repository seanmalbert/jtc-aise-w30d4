"""
Model evaluation script for the W30D4 demo CI pipeline.

Usage:
    python scripts/evaluate_model.py --threshold 0.85

Exits with code 0 if accuracy >= threshold, else exits 1 (fails the CI step).

This is a stub for demo purposes. In a real ML product, this script would:
  1. Load your trained model artifact (e.g., from models/latest.pkl)
  2. Load a held-out test dataset
  3. Run predictions and compute accuracy, precision, recall, F1
  4. Fail the pipeline if any metric falls below its threshold
"""

import argparse
import sys


def load_and_evaluate_model():
    """
    Stub evaluation — returns a hardcoded accuracy.
    Replace with real model loading and evaluation in production.

    Example real implementation:
        import joblib
        import pandas as pd
        model = joblib.load("models/latest.pkl")
        test_data = pd.read_csv("data/test.csv")
        X = test_data.drop(columns=["label"])
        y = test_data["label"]
        predictions = model.predict(X)
        accuracy = (predictions == y).mean()
        return accuracy
    """
    # Stub: always returns 0.90 so the demo pipeline passes baseline.
    # To simulate a failure during class, change this to 0.75 and push.
    return 0.90


def main():
    parser = argparse.ArgumentParser(description="Evaluate model against threshold.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum acceptable accuracy (pipeline fails below this).",
    )
    args = parser.parse_args()

    accuracy = load_and_evaluate_model()

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Threshold:      {args.threshold:.4f}")

    if accuracy < args.threshold:
        print(f"FAIL: accuracy {accuracy:.4f} is below threshold {args.threshold:.4f}")
        sys.exit(1)

    print("PASS: accuracy meets threshold")
    sys.exit(0)


if __name__ == "__main__":
    main()
