"""
src/model/train.py
==================
Training loop for the SimpleCNN Cats vs Dogs classifier.

Features:
  - MLflow experiment tracking (params, metrics, artifacts)
  - Best model checkpoint saved to artifacts/model.pt
  - Confusion matrix + loss curve plots logged as MLflow artifacts
  - Early-stop friendly (tracks best val accuracy)

Usage:
    PYTHONPATH=. python src/model/train.py
"""

import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib
matplotlib.use("Agg")            # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import mlflow
import mlflow.pytorch

from src.data.dataset import get_dataloader, IDX_TO_CLASS
from src.model.cnn import SimpleCNN

# ── Hyperparameters ────────────────────────────────────────────────────────────
CONFIG = {
    "epochs":        10,
    "batch_size":    32,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "dropout":       0.5,
    "num_workers":   2,
    "scheduler_step": 5,
    "scheduler_gamma": 0.5,
    "model_arch":    "SimpleCNN",
    "optimizer":     "Adam",
    "image_size":    224,
    "num_classes":   2,
}

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH    = ARTIFACTS_DIR / "model.pt"
CM_PATH       = ARTIFACTS_DIR / "confusion_matrix.png"
LOSS_PATH     = ARTIFACTS_DIR / "loss_curve.png"
METRICS_PATH  = ARTIFACTS_DIR / "train_metrics.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_loss_curve(train_losses, val_losses, save_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses,   "r-o", label="Val Loss",   linewidth=2)
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Loss curve saved -> {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    cm     = confusion_matrix(y_true, y_pred)
    labels = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix (Validation Set)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {save_path}")


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds    = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = total = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds    = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc, all_labels, all_preds


# ── Main ───────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  Cats vs Dogs — Training Pipeline")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    print(f"  Config : {CONFIG}")
    print("=" * 60)

    # ── DataLoaders ────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_loader = get_dataloader("train", batch_size=CONFIG["batch_size"],
                                  num_workers=CONFIG["num_workers"])
    val_loader   = get_dataloader("val",   batch_size=CONFIG["batch_size"],
                                  num_workers=CONFIG["num_workers"])

    print(f"  Train batches : {len(train_loader)}  "
          f"({len(train_loader.dataset)} images)")
    print(f"  Val   batches : {len(val_loader)}  "
          f"({len(val_loader.dataset)} images)")

    # ── Model, loss, optimizer ─────────────────────────────────────────────
    model     = SimpleCNN(num_classes=CONFIG["num_classes"],
                          dropout=CONFIG["dropout"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),
                     lr=CONFIG["lr"],
                     weight_decay=CONFIG["weight_decay"])
    scheduler = StepLR(optimizer,
                       step_size=CONFIG["scheduler_step"],
                       gamma=CONFIG["scheduler_gamma"])

    print(f"\nModel parameters: {model.count_parameters():,}")

    # ── MLflow setup ───────────────────────────────────────────────────────
    mlflow.set_experiment("cats-dogs-classification")

    with mlflow.start_run(run_name="SimpleCNN-baseline") as run:
        print(f"\nMLflow Run ID : {run.info.run_id}")

        # Log all hyperparameters
        mlflow.log_params(CONFIG)

        best_val_acc  = 0.0
        train_losses  = []
        val_losses    = []
        history       = []

        # ── Epoch loop ─────────────────────────────────────────────────────
        for epoch in range(1, CONFIG["epochs"] + 1):
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, y_true, y_pred = evaluate(
                model, val_loader, criterion, DEVICE)

            scheduler.step()
            elapsed = time.time() - t0

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Log metrics to MLflow per epoch
            mlflow.log_metric("train_loss",    train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc,  step=epoch)
            mlflow.log_metric("val_loss",      val_loss,   step=epoch)
            mlflow.log_metric("val_accuracy",  val_acc,    step=epoch)

            print(f"\nEpoch [{epoch:02d}/{CONFIG['epochs']}]  "
                  f"Time: {elapsed:.1f}s")
            print(f"  Train  Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}%")
            print(f"  Val    Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%")

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc":  round(train_acc,  4),
                "val_loss":   round(val_loss,   4),
                "val_acc":    round(val_acc,    4),
            })

            # ── Save best model ────────────────────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  *** New best model saved! val_accuracy = {val_acc*100:.2f}%")

        # ── Post-training artifacts ────────────────────────────────────────
        print("\n--- Generating artifacts ---")

        # Final val pass for confusion matrix
        _, _, y_true, y_pred = evaluate(model, val_loader, criterion, DEVICE)

        plot_loss_curve(train_losses, val_losses, LOSS_PATH)
        plot_confusion_matrix(y_true, y_pred, CM_PATH)

        # Classification report
        labels = [IDX_TO_CLASS[i] for i in range(2)]
        report = classification_report(y_true, y_pred, target_names=labels)
        print(f"\nClassification Report:\n{report}")

        # Save training history JSON
        with open(METRICS_PATH, "w") as f:
            json.dump(history, f, indent=2)

        # Log artifacts to MLflow
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(CM_PATH))
        mlflow.log_artifact(str(LOSS_PATH))
        mlflow.log_artifact(str(METRICS_PATH))

        # Log best metric summary
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Best Val Accuracy : {best_val_acc*100:.2f}%")
        print(f"  Model saved       : {MODEL_PATH}")
        print(f"  MLflow Run ID     : {run.info.run_id}")
        print("=" * 60)
        print("\nTo view MLflow UI:  mlflow ui  (then open http://localhost:5000)")


if __name__ == "__main__":
    # Install seaborn if not present
    try:
        import seaborn
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
        import seaborn as sns

    train()
