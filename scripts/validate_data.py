"""
Data validation script for the W30D4 demo CI pipeline.

Usage:
    python scripts/validate_data.py

Checks that the training/test dataset conforms to expected schema.
Exits 0 if valid, 1 if the schema check fails.

This is a stub for demo purposes. In a real ML product, this script would:
  1. Load the dataset (CSV, Parquet, etc.)
  2. Check column names match the expected schema
  3. Check for null values, out-of-range values, and distribution shifts
  4. Fail the pipeline if any check fails
"""

import sys


EXPECTED_COLUMNS = ["text", "label"]
ALLOWED_LABELS = {"positive", "negative"}


def validate_data():
    """
    Stub validation — pretends to check a dataset and returns results.
    Replace with real pandas-based checks in production.
    """
    # Stub: simulate a clean dataset.
    # To simulate a failure during class, change issues_found to a non-empty list.
    issues_found = []

    print(f"Expected columns: {EXPECTED_COLUMNS}")
    print(f"Allowed labels:   {sorted(ALLOWED_LABELS)}")

    if issues_found:
        print("FAIL: data validation found issues:")
        for issue in issues_found:
            print(f"  - {issue}")
        return False

    print("PASS: data schema is valid")
    return True


def main():
    if not validate_data():
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
