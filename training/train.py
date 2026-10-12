"""
AI Exercise Form & Posture Intelligence Platform
Multi-Model Training Pipeline & Academic Benchmark Engine
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from app.config import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR
from models.classifier import ExerciseFormClassifier
from models.preprocessing import DatasetPreprocessor, EXERCISE_CLASSES
from models.temporal_model import PyTorchTemporalModel
from training.collect_data import SyntheticDatasetGenerator
from training.preprocess import session_aware_train_val_test_split


def create_sequence_batches(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 15,
    session_col: str = "session_id",
    label_col: str = "exercise_encoded"
) -> Tuple[np.ndarray, np.ndarray]:
    """Chunks grouped session frames into 3D sequence tensors for recurrent LSTM training."""
    X_seqs = []
    y_seqs = []

    for session_id, group in df.groupby(session_col):
        feat_matrix = group[feature_cols].values
        labels = group[label_col].values
        if len(feat_matrix) < seq_len:
            continue

        for i in range(0, len(feat_matrix) - seq_len + 1, 5):  # stride 5
            X_seqs.append(feat_matrix[i:i + seq_len])
            # Use mode label of window
            y_seqs.append(labels[i + seq_len - 1])

    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.int64)


def run_training_pipeline() -> pd.DataFrame:
    """
    Executes full multi-model training, session-aware benchmarking,
    and academic artifact generation.
    """
    dataset_path = DATA_DIR / "workout_biomechanics_dataset.csv"
    if not dataset_path.exists():
        print("Dataset not found. Synthesizing multi-subject benchmark dataset...")
        generator = SyntheticDatasetGenerator()
        df = generator.generate_full_dataset(output_csv=dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    print(f"Loaded dataset: {len(df)} frames across {df['session_id'].nunique()} unique sessions.")

    # 1. Feature columns (feat_00 to feat_67)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]

    # 2. Fit Preprocessor
    preprocessor = DatasetPreprocessor()
    df["exercise_encoded"] = preprocessor.encode_exercise(df["exercise"].tolist())
    preprocessor.save(str(ARTIFACTS_DIR / "preprocessor.joblib"))

    # 3. Session-Aware Train / Val / Test Split (Zero Leakage!)
    train_df, val_df, test_df = session_aware_train_val_test_split(
        df,
        session_col="session_id",
        val_ratio=0.15,
        test_ratio=0.20,
        random_seed=42
    )

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")

    # Scale features using training statistics only
    X_train_raw = train_df[feat_cols].values
    X_train = preprocessor.fit_transform(X_train_raw)
    y_train = train_df["exercise_encoded"].values

    X_test_raw = test_df[feat_cols].values
    X_test = preprocessor.transform(X_test_raw)
    y_test = test_df["exercise_encoded"].values

    # Models to benchmark
    models_to_evaluate = [
        ("Random Forest", "random_forest"),
        ("SVM (RBF Kernel)", "svm"),
        ("MLP Neural Net", "mlp"),
        ("Gradient Boosting", "gradient_boosting"),
    ]

    benchmark_records = []

    # 4. Train Scikit-Learn Classifiers
    for model_display_name, model_key in models_to_evaluate:
        print(f"--> Training {model_display_name}...")
        clf = ExerciseFormClassifier(model_type=model_key)
        train_time = clf.train(X_train, y_train)

        # Predictions on held-out session-aware test set
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Real latency & throughput benchmark
        latency_ms, fps = clf.benchmark_inference(X_test[:50])

        clf.save(str(MODELS_DIR / f"{model_key}_classifier.joblib"))

        # Save classification report
        report_str = classification_report(
            y_test, y_pred, labels=list(range(len(EXERCISE_CLASSES))), target_names=EXERCISE_CLASSES, zero_division=0
        )
        with open(ARTIFACTS_DIR / f"classification_report_{model_key}.txt", "w") as f:
            f.write(f"=== Model: {model_display_name} ===\n\n" + report_str)

        # Plot & save Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(EXERCISE_CLASSES))))
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap="Blues")
        fig.colorbar(cax)
        ax.set_xticks(range(len(EXERCISE_CLASSES)))
        ax.set_yticks(range(len(EXERCISE_CLASSES)))
        ax.set_xticklabels(EXERCISE_CLASSES, rotation=45, ha="left")
        ax.set_yticklabels(EXERCISE_CLASSES)
        for i in range(len(EXERCISE_CLASSES)):
            for j in range(len(EXERCISE_CLASSES)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red" if cm[i, j] < 20 else "white")
        ax.set_title(f"Confusion Matrix: {model_display_name}", pad=20)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True Ground Truth")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / f"confusion_matrix_{model_key}.png", dpi=150)
        plt.close()

        benchmark_records.append({
            "Model": model_display_name,
            "Accuracy (%)": round(acc * 100.0, 2),
            "Precision (%)": round(prec * 100.0, 2),
            "Recall (%)": round(rec * 100.0, 2),
            "F1-Score (%)": round(f1 * 100.0, 2),
            "Inference Latency (ms)": latency_ms,
            "Throughput (FPS)": fps,
            "Training Time (s)": round(train_time, 2),
        })

    # 5. Train Temporal PyTorch LSTM Model
    print("--> Training PyTorch LSTM Temporal Sequence Model...")
    # Prepare scaled sequence tensors
    train_df_scaled = train_df.copy()
    train_df_scaled[feat_cols] = X_train
    test_df_scaled = test_df.copy()
    test_df_scaled[feat_cols] = X_test

    X_train_seq, y_train_seq = create_sequence_batches(train_df_scaled, feat_cols, seq_len=15)
    X_test_seq, y_test_seq = create_sequence_batches(test_df_scaled, feat_cols, seq_len=15)

    if len(X_train_seq) > 0 and len(X_test_seq) > 0:
        lstm_model = PyTorchTemporalModel(input_dim=len(feat_cols), num_classes=len(EXERCISE_CLASSES))
        lstm_train_time = lstm_model.train(X_train_seq, y_train_seq, epochs=20, batch_size=32)

        y_pred_lstm = lstm_model.predict(X_test_seq)
        acc_lstm = accuracy_score(y_test_seq, y_pred_lstm)
        prec_lstm = precision_score(y_test_seq, y_pred_lstm, average="weighted", zero_division=0)
        rec_lstm = recall_score(y_test_seq, y_pred_lstm, average="weighted", zero_division=0)
        f1_lstm = f1_score(y_test_seq, y_pred_lstm, average="weighted", zero_division=0)

        latency_lstm, fps_lstm = lstm_model.benchmark_inference(X_test_seq[:30])
        lstm_model.save(str(MODELS_DIR / "lstm_temporal_model.pt"))

        # Save LSTM classification report
        report_str_lstm = classification_report(
            y_test_seq, y_pred_lstm, labels=list(range(len(EXERCISE_CLASSES))), target_names=EXERCISE_CLASSES, zero_division=0
        )
        with open(ARTIFACTS_DIR / "classification_report_lstm.txt", "w") as f:
            f.write("=== Model: Temporal LSTM Network ===\n\n" + report_str_lstm)

        # Plot & save LSTM Confusion Matrix
        cm_lstm = confusion_matrix(y_test_seq, y_pred_lstm, labels=list(range(len(EXERCISE_CLASSES))))
        fig_l, ax_l = plt.subplots(figsize=(6, 5))
        cax_l = ax_l.matshow(cm_lstm, cmap="Blues")
        fig_l.colorbar(cax_l)
        ax_l.set_xticks(range(len(EXERCISE_CLASSES)))
        ax_l.set_yticks(range(len(EXERCISE_CLASSES)))
        ax_l.set_xticklabels(EXERCISE_CLASSES, rotation=45, ha="left")
        ax_l.set_yticklabels(EXERCISE_CLASSES)
        for i in range(len(EXERCISE_CLASSES)):
            for j in range(len(EXERCISE_CLASSES)):
                ax_l.text(j, i, str(cm_lstm[i, j]), ha="center", va="center", color="red" if cm_lstm[i, j] < 20 else "white")
        ax_l.set_title("Confusion Matrix: Temporal LSTM Network", pad=20)
        ax_l.set_xlabel("Predicted")
        ax_l.set_ylabel("True Ground Truth")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "confusion_matrix_lstm.png", dpi=150)
        plt.close()

        benchmark_records.append({
            "Model": "Temporal LSTM Network",
            "Accuracy (%)": round(acc_lstm * 100.0, 2),
            "Precision (%)": round(prec_lstm * 100.0, 2),
            "Recall (%)": round(rec_lstm * 100.0, 2),
            "F1-Score (%)": round(f1_lstm * 100.0, 2),
            "Inference Latency (ms)": latency_lstm,
            "Throughput (FPS)": fps_lstm,
            "Training Time (s)": round(lstm_train_time, 2),
        })

    # 6. Save Comparison CSV and Charts
    benchmark_df = pd.DataFrame(benchmark_records)
    benchmark_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
    print("\n=== MODEL COMPARISON BENCHMARK RESULTS ===")
    print(benchmark_df.to_string(index=False))

    # Plot Accuracy vs Latency comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(benchmark_df["Model"], benchmark_df["Accuracy (%)"], color="#38BDF8")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Model Accuracy Comparison (Leakage-Free Test Set)")
    ax1.tick_params(axis="x", rotation=30)
    for i, v in enumerate(benchmark_df["Accuracy (%)"]):
        ax1.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")

    ax2.bar(benchmark_df["Model"], benchmark_df["Inference Latency (ms)"], color="#818CF8")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Inference Latency Benchmark (Lower is Better)")
    ax2.tick_params(axis="x", rotation=30)
    for i, v in enumerate(benchmark_df["Inference Latency (ms)"]):
        ax2.text(i, v + 0.2, f"{v:.1f}ms", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "model_comparison.png", dpi=150)
    plt.close()

    return benchmark_df


if __name__ == "__main__":
    run_training_pipeline()
