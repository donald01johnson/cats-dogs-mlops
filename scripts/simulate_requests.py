"""
scripts/simulate_requests.py
=============================
Post-deployment model performance simulation script.

- Picks a sample of real test images from data/processed/test/
- Sends each image to the deployed API via POST /predict
- Compares predicted label with true label (from folder name)
- Computes: accuracy, precision, recall, F1-score
- Saves results to artifacts/post_deploy_metrics.json
- Prints a formatted performance report to console

Usage:
    # Make sure the API is running first:
    #   docker compose -f deploy/docker-compose.yml up -d
    # Then run:
    PYTHONPATH=. python scripts/simulate_requests.py

    # To test against a different URL:
    API_URL=http://localhost:8000 python scripts/simulate_requests.py

    # To change number of samples per class:
    NUM_SAMPLES=50 python scripts/simulate_requests.py
"""

import os
import sys
import json
import time
import random
from pathlib import Path

import requests
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_URL", "http://localhost:8000")
PROCESSED_ROOT   = Path("data/processed/test")
ARTIFACTS_DIR    = Path("artifacts")
OUTPUT_JSON      = ARTIFACTS_DIR / "post_deploy_metrics.json"
NUM_SAMPLES      = int(os.getenv("NUM_SAMPLES", "30"))   # per class
RANDOM_SEED      = 42

CLASS_MAP = {"cats": "cat", "dogs": "dog"}   # folder name → API label


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_api_health() -> bool:
    """Verify the API is reachable and model is loaded before running tests."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if resp.status_code == 200:
            body = resp.json()
            if body.get("model_loaded"):
                return True
            print(f"[ERROR] API is up but model_loaded=False: {body}")
        else:
            print(f"[ERROR] /health returned HTTP {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot connect to API at {API_BASE_URL}")
        print("        Make sure the container is running:")
        print("        docker compose -f deploy/docker-compose.yml up -d")
    return False


def predict_image(image_path: Path) -> dict:
    """
    Send a single image to POST /predict and return the full response dict.
    Returns None on any error.
    """
    try:
        with open(image_path, "rb") as f:
            files    = {"file": (image_path.name, f, "image/jpeg")}
            t0       = time.time()
            response = requests.post(
                f"{API_BASE_URL}/predict",
                files=files,
                timeout=30
            )
            latency_ms = (time.time() - t0) * 1000

        if response.status_code == 200:
            result = response.json()
            result["latency_ms"] = round(latency_ms, 2)
            return result
        else:
            print(f"  [WARN] {image_path.name} → HTTP {response.status_code}")
            return None

    except Exception as e:
        print(f"  [WARN] {image_path.name} → Error: {e}")
        return None


def collect_samples() -> list:
    """
    Collect up to NUM_SAMPLES images per class from the test set.
    Returns a list of (image_path, true_label) tuples.
    """
    samples = []
    random.seed(RANDOM_SEED)

    for folder, label in CLASS_MAP.items():
        class_dir = PROCESSED_ROOT / folder
        if not class_dir.exists():
            print(f"[WARN] Test directory not found: {class_dir} — skipping {folder}")
            continue

        all_images = sorted(class_dir.glob("*.jpg"))
        selected   = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))
        for img in selected:
            samples.append((img, label))

    random.shuffle(samples)
    return samples


# ── Main ───────────────────────────────────────────────────────────────────────

def run_simulation():
    print("=" * 60)
    print("  Cats vs Dogs — Post-Deployment Performance Simulation")
    print("=" * 60)
    print(f"  API URL     : {API_BASE_URL}")
    print(f"  Test set    : {PROCESSED_ROOT}")
    print(f"  Samples     : {NUM_SAMPLES} per class")
    print("=" * 60)

    # ── Step 1: Health check ───────────────────────────────────────────────
    print("\n[1] Checking API health...")
    if not check_api_health():
        sys.exit(1)
    print("    API is healthy and model is loaded ✅")

    # ── Step 2: Collect samples ────────────────────────────────────────────
    print(f"\n[2] Collecting test samples...")
    samples = collect_samples()
    if not samples:
        print("[ERROR] No test images found. Run preprocess.py first.")
        sys.exit(1)
    print(f"    Collected {len(samples)} images "
          f"({sum(1 for _, l in samples if l == 'cat')} cats, "
          f"{sum(1 for _, l in samples if l == 'dog')} dogs)")

    # ── Step 3: Run predictions ────────────────────────────────────────────
    print(f"\n[3] Running predictions against {API_BASE_URL}/predict ...")
    print(f"    {'Image':<30} {'True':>6} {'Pred':>6} {'Conf':>8} {'Latency':>10} {'Result':>8}")
    print(f"    {'-'*30} {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")

    y_true      = []
    y_pred      = []
    latencies   = []
    results_log = []
    errors      = 0

    for idx, (img_path, true_label) in enumerate(samples):
        response = predict_image(img_path)

        if response is None:
            errors += 1
            continue

        pred_label  = response["label"]
        confidence  = response["confidence"]
        latency_ms  = response["latency_ms"]
        correct     = "✅" if pred_label == true_label else "❌"

        y_true.append(true_label)
        y_pred.append(pred_label)
        latencies.append(latency_ms)

        results_log.append({
            "image":       img_path.name,
            "true_label":  true_label,
            "pred_label":  pred_label,
            "confidence":  confidence,
            "latency_ms":  latency_ms,
            "correct":     pred_label == true_label,
        })

        print(f"    {img_path.name:<30} {true_label:>6} {pred_label:>6} "
              f"{confidence:>8.4f} {latency_ms:>9.1f}ms {correct:>8}")

    # ── Step 4: Compute metrics ────────────────────────────────────────────
    if not y_true:
        print("\n[ERROR] No successful predictions — cannot compute metrics.")
        sys.exit(1)

    labels    = ["cat", "dog"]
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="dog", average="binary")
    recall    = recall_score(y_true, y_pred,    pos_label="dog", average="binary")
    f1        = f1_score(y_true, y_pred,        pos_label="dog", average="binary")
    cm        = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    report = classification_report(y_true, y_pred, target_names=labels)

    # ── Step 5: Print report ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  POST-DEPLOYMENT PERFORMANCE REPORT")
    print("=" * 60)
    print(f"\n  Samples evaluated : {len(y_true)}")
    print(f"  Errors / skipped  : {errors}")
    print(f"\n  ── Classification Metrics ──")
    print(f"  Accuracy          : {accuracy*100:.2f}%")
    print(f"  Precision (dog)   : {precision*100:.2f}%")
    print(f"  Recall    (dog)   : {recall*100:.2f}%")
    print(f"  F1-Score  (dog)   : {f1*100:.2f}%")
    print(f"\n  ── Confusion Matrix (rows=true, cols=pred) ──")
    print(f"                  Pred:cat  Pred:dog")
    print(f"  True:cat         {cm[0][0]:>6}    {cm[0][1]:>6}")
    print(f"  True:dog         {cm[1][0]:>6}    {cm[1][1]:>6}")
    print(f"\n  ── Per-Class Report ──")
    print(report)
    print(f"  ── Latency (ms) ──")
    print(f"  Avg : {avg_latency:.1f}ms")
    print(f"  Min : {min_latency:.1f}ms")
    print(f"  Max : {max_latency:.1f}ms")
    print(f"  P95 : {p95_latency:.1f}ms")
    print("=" * 60)

    # ── Step 6: Save JSON ──────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    output = {
        "simulation_config": {
            "api_url":        API_BASE_URL,
            "num_samples":    NUM_SAMPLES,
            "total_evaluated": len(y_true),
            "errors":         errors,
            "random_seed":    RANDOM_SEED,
        },
        "classification_metrics": {
            "accuracy":       round(accuracy,  4),
            "precision_dog":  round(precision, 4),
            "recall_dog":     round(recall,    4),
            "f1_score_dog":   round(f1,        4),
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm,
        },
        "latency_ms": {
            "avg": round(avg_latency, 2),
            "min": round(min_latency, 2),
            "max": round(max_latency, 2),
            "p95": round(p95_latency, 2),
        },
        "per_image_results": results_log,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved → {OUTPUT_JSON}")
    print("\n  Done! Simulation complete. ✅")


if __name__ == "__main__":
    run_simulation()
