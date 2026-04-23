"""
Model evaluation script for the W30D4 demo CI pipeline.

Usage:
    python scripts/evaluate_model.py --threshold 0.85
    python scripts/evaluate_model.py --model models/latest.pkl --test-data data/test.csv --min-accuracy 0.85 --max-latency-ms 200

Supports both simple and extended invocations:
  - Simple (used by ml-prod.yaml):              --threshold
  - Extended (used by ml-prod-additional.yaml): --model, --test-data, --min-accuracy, --max-latency-ms

Exits with code 0 if all checks pass, 1 if any check fails.

This is a stub for demo purposes. In a real ML product, this script would:
  1. Load your trained model artifact from the --model path
  2. Load a held-out test dataset from --test-data
  3. Run predictions and compute accuracy, precision, recall, F1
  4. Measure inference latency on a sample of prompts
  5. Fail the pipeline if any metric falls below its threshold
"""

import argparse
import sys


def load_and_evaluate_model(model_path: str, test_data_path: str):
    """
    Stub evaluation — returns hardcoded accuracy and latency.
    Replace with real model loading and evaluation in production.

    Example real implementation:
        import joblib
        import pandas as pd
        import time

        model = joblib.load(model_path)
        test_data = pd.read_csv(test_data_path)
        X = test_data.drop(columns=["label"])
        y = test_data["label"]

        start = time.perf_counter()
        predictions = model.predict(X)
        elapsed_ms = (time.perf_counter() - start) * 1000 / len(X)

        accuracy = (predictions == y).mean()
        return accuracy, elapsed_ms
    """
    # Stub: always returns 0.90 accuracy and 120ms latency so the demo pipeline passes baseline.
    # To simulate a failure during class, change accuracy to 0.75 and push.
    accuracy = 0.75
    latency_ms = 120.0
    return accuracy, latency_ms


def main():
    parser = argparse.ArgumentParser(description="Evaluate model against thresholds.")
    parser.add_argument(
        "--model",
        default="models/latest.pkl",
        help="Path to the trained model artifact.",
    )
    parser.add_argument(
        "--test-data",
        default="data/test.csv",
        help="Path to the test dataset.",
    )
    # Accept either --threshold OR --min-accuracy (both mean the same thing)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum acceptable accuracy (alias for --min-accuracy).",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="Minimum acceptable accuracy. Pipeline fails if actual accuracy is below this.",
    )
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=None,
        help="Maximum acceptable inference latency in milliseconds. Pipeline fails if exceeded.",
    )
    args = parser.parse_args()

    # Resolve which accuracy threshold to use (prefer --min-accuracy, fall back to --threshold, default 0.85)
    min_accuracy = (
        args.min_accuracy if args.min_accuracy is not None else args.threshold
    )
    if min_accuracy is None:
        min_accuracy = 0.85

    accuracy, latency_ms = load_and_evaluate_model(args.model, args.test_data)

    print(f"Model:          {args.model}")
    print(f"Test data:      {args.test_data}")
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Min accuracy:   {min_accuracy:.4f}")
    print(f"Latency:        {latency_ms:.1f}ms")
    if args.max_latency_ms is not None:
        print(f"Max latency:    {args.max_latency_ms:.1f}ms")

    failed = False

    if accuracy < min_accuracy:
        print(f"FAIL: accuracy {accuracy:.4f} is below minimum {min_accuracy:.4f}")
        failed = True

    if args.max_latency_ms is not None and latency_ms > args.max_latency_ms:
        print(
            f"FAIL: latency {latency_ms:.1f}ms exceeds maximum {args.max_latency_ms:.1f}ms"
        )
        failed = True

    if failed:
        sys.exit(1)

    print("PASS: all checks met thresholds")
    sys.exit(0)


if __name__ == "__main__":
    main()
