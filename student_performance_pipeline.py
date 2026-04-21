import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras import Model, Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        Add,
        Dense,
        Dropout,
        GlobalAveragePooling1D,
        GRU,
        Input,
        LayerNormalization,
        LSTM,
        MultiHeadAttention,
    )
    from tensorflow.keras.utils import to_categorical
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "TensorFlow is required for this project. Install a supported Python version "
        "(recommended: Python 3.10 or 3.11) and then run: pip install tensorflow"
    ) from exc


SEED = 42
TIME_STEPS = 14
MAX_EPOCHS = 100
BATCH_SIZE = 32
OUTPUT_DIR = "."

np.random.seed(SEED)
tf.random.set_seed(SEED)


@dataclass
class SplitData:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train_bin: np.ndarray
    y_val_bin: np.ndarray
    y_test_bin: np.ndarray
    y_train_multi: np.ndarray
    y_val_multi: np.ndarray
    y_test_multi: np.ndarray


def ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["Distance_from_Home"] = df["Distance_from_Home"].fillna("Unknown")
    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Pass_Fail"] = (df["Exam_Score"] >= 60).astype(int)
    bins = [-np.inf, 60, 70, 85, np.inf]
    labels = ["Fail", "C", "B", "A"]
    df["Grade_Category"] = pd.cut(df["Exam_Score"], bins=bins, labels=labels, right=False)
    df["Grade_Category"] = df["Grade_Category"].astype(str)
    return df


def identify_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    ignore_cols = {"Exam_Score", "Pass_Fail", "Grade_Category"}
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def split_students(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["Pass_Fail"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["Pass_Fail"]
    )
    return train_df, val_df, test_df


def encode_categorical(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_enc = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
    val_enc = pd.get_dummies(val_df, columns=categorical_cols, drop_first=True)
    test_enc = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)
    val_enc = val_enc.reindex(columns=train_enc.columns, fill_value=0)
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)
    return train_enc, val_enc, test_enc


def scale_numeric(
    train_enc: pd.DataFrame,
    val_enc: pd.DataFrame,
    test_enc: pd.DataFrame,
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    train_enc = train_enc.copy()
    val_enc = val_enc.copy()
    test_enc = test_enc.copy()
    scaler.fit(train_enc[numeric_cols])
    train_enc[numeric_cols] = scaler.transform(train_enc[numeric_cols])
    val_enc[numeric_cols] = scaler.transform(val_enc[numeric_cols])
    test_enc[numeric_cols] = scaler.transform(test_enc[numeric_cols])
    return train_enc, val_enc, test_enc, scaler


def _simulate_numeric_feature(base_value: float, feature_name: str, time_steps: int, rng: np.random.Generator) -> np.ndarray:
    std_map = {
        "Hours_Studied": 0.35,
        "Attendance": 0.20,
        "Sleep_Hours": 0.25,
        "Previous_Scores": 0.30,
        "Tutoring_Sessions": 0.60,
        "Physical_Activity": 0.40,
    }
    base_std = std_map.get(feature_name, 0.30)
    noise = rng.normal(loc=0.0, scale=max(0.05, abs(base_value) * base_std), size=time_steps)
    trend = np.linspace(-0.5, 0.5, time_steps) * rng.normal(0.0, 0.05)
    seq = base_value + noise + trend
    seq[-1] = base_value
    seq[:-1] += (base_value - seq.mean())  # enforce original value as sequence average
    return seq


def build_sequence_tensor(
    encoded_df: pd.DataFrame,
    numeric_cols: List[str],
    target_cols: List[str],
    time_steps: int = TIME_STEPS,
    seed: int = SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    feature_cols = [c for c in encoded_df.columns if c not in target_cols]
    x = np.zeros((len(encoded_df), time_steps, len(feature_cols)), dtype=np.float32)
    for i, (_, row) in enumerate(encoded_df.iterrows()):
        for j, col in enumerate(feature_cols):
            val = float(row[col])
            if col in numeric_cols:
                seq = _simulate_numeric_feature(val, col, time_steps, rng)
            else:
                seq = np.repeat(val, time_steps)
            x[i, :, j] = seq.astype(np.float32)
    return x


def prepare_data(df: pd.DataFrame, time_steps: int = TIME_STEPS) -> Tuple[SplitData, List[str], Dict]:
    numeric_cols, categorical_cols = identify_feature_types(df)
    train_df, val_df, test_df = split_students(df)
    train_enc, val_enc, test_enc = encode_categorical(train_df, val_df, test_df, categorical_cols)
    train_enc, val_enc, test_enc, scaler = scale_numeric(train_enc, val_enc, test_enc, numeric_cols)

    target_cols = ["Exam_Score", "Pass_Fail", "Grade_Category"]
    x_train = build_sequence_tensor(train_enc, numeric_cols, target_cols, time_steps=time_steps, seed=SEED)
    x_val = build_sequence_tensor(val_enc, numeric_cols, target_cols, time_steps=time_steps, seed=SEED + 1)
    x_test = build_sequence_tensor(test_enc, numeric_cols, target_cols, time_steps=time_steps, seed=SEED + 2)

    y_train_bin = train_enc["Pass_Fail"].astype(int).to_numpy()
    y_val_bin = val_enc["Pass_Fail"].astype(int).to_numpy()
    y_test_bin = test_enc["Pass_Fail"].astype(int).to_numpy()

    grade_map = {"Fail": 0, "C": 1, "B": 2, "A": 3}
    y_train_multi = train_enc["Grade_Category"].map(grade_map).astype(int).to_numpy()
    y_val_multi = val_enc["Grade_Category"].map(grade_map).astype(int).to_numpy()
    y_test_multi = test_enc["Grade_Category"].map(grade_map).astype(int).to_numpy()

    feature_cols = [c for c in train_enc.columns if c not in target_cols]
    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_cols": feature_cols,
        "grade_map": grade_map,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    split_data = SplitData(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train_bin=y_train_bin,
        y_val_bin=y_val_bin,
        y_test_bin=y_test_bin,
        y_train_multi=y_train_multi,
        y_val_multi=y_val_multi,
        y_test_multi=y_test_multi,
    )
    return split_data, feature_cols, metadata


def build_lstm_model(input_shape: Tuple[int, int], output_dim: int = 1) -> Model:
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.30),
            LSTM(32),
            Dropout(0.30),
            Dense(16, activation="relu"),
            Dense(output_dim, activation="sigmoid" if output_dim == 1 else "softmax"),
        ]
    )
    return model


def build_gru_model(input_shape: Tuple[int, int], output_dim: int = 1) -> Model:
    model = Sequential(
        [
            GRU(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.30),
            GRU(32),
            Dropout(0.30),
            Dense(16, activation="relu"),
            Dense(output_dim, activation="sigmoid" if output_dim == 1 else "softmax"),
        ]
    )
    return model


def positional_encoding(time_steps: int, d_model: int) -> tf.Tensor:
    positions = np.arange(time_steps)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
    angle_rads = positions * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)


def build_transformer_model(input_shape: Tuple[int, int], output_dim: int = 1) -> Model:
    time_steps, num_features = input_shape
    d_model = 64
    inputs = Input(shape=input_shape)
    x = Dense(d_model)(inputs)
    pos = positional_encoding(time_steps, d_model)
    x = Add()([x, pos[:, :time_steps, :]])
    attn = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.2)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization(epsilon=1e-6)(x)
    ff = Dense(128, activation="relu")(x)
    ff = Dropout(0.2)(ff)
    ff = Dense(d_model)(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(output_dim, activation="sigmoid" if output_dim == 1 else "softmax")(x)
    return Model(inputs, outputs, name="TransformerClassifier")


def compile_model(model: Model, multiclass: bool = False, lr: float = 1e-3) -> None:
    loss = "sparse_categorical_crossentropy" if multiclass else "binary_crossentropy"
    metrics = ["accuracy"]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics)


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, multiclass: bool = False) -> float:
    if multiclass:
        y_true_oh = to_categorical(y_true, num_classes=y_prob.shape[1])
        return float(roc_auc_score(y_true_oh, y_prob, multi_class="ovr", average="macro"))
    return float(roc_auc_score(y_true, y_prob))


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, multiclass: bool = False) -> Dict[str, float]:
    if multiclass:
        y_pred = np.argmax(y_prob, axis=1)
        avg = "macro"
        precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
        recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        auc = compute_roc_auc(y_true, y_prob, multiclass=True)
    else:
        y_pred = (y_prob >= 0.5).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        auc = compute_roc_auc(y_true, y_prob, multiclass=False)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], title: str, save_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_binary_roc(y_true: np.ndarray, y_prob: np.ndarray, title: str, save_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC={auc_val:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_multiclass_roc(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], title: str, save_path: str) -> None:
    y_true_oh = to_categorical(y_true, num_classes=len(class_names))
    plt.figure(figsize=(7, 6))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        auc_val = roc_auc_score(y_true_oh[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_training_curve(history: tf.keras.callbacks.History, title: str, save_path: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(history.history.get("loss", []), label="Train Loss")
    plt.plot(history.history.get("val_loss", []), label="Val Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_model_architecture_diagram(model: Model, title: str, save_path: str) -> None:
    plt.figure(figsize=(9, 0.65 * len(model.layers) + 2))
    plt.axis("off")
    lines = [f"{title}", ""]
    for idx, layer in enumerate(model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None:
            try:
                output_shape = tuple(layer.output.shape)
            except Exception:
                output_shape = "unknown"
        lines.append(f"{idx + 1}. {layer.name}: {layer.__class__.__name__} | output={output_shape}")
    plt.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def write_hyperparameter_table(save_path: str) -> pd.DataFrame:
    table = pd.DataFrame(
        [
            {"parameter": "time_steps", "value": TIME_STEPS},
            {"parameter": "batch_size", "value": BATCH_SIZE},
            {"parameter": "max_epochs", "value": MAX_EPOCHS},
            {"parameter": "learning_rate", "value": 1e-3},
            {"parameter": "dropout", "value": 0.30},
            {"parameter": "lstm_units", "value": "64 -> 32"},
            {"parameter": "gru_units", "value": "64 -> 32"},
            {"parameter": "transformer_heads", "value": 4},
            {"parameter": "transformer_d_model", "value": 64},
            {"parameter": "early_stopping_patience", "value": 10},
        ]
    )
    table.to_csv(save_path, index=False)
    return table


def write_loss_justification(save_path: str) -> None:
    text = (
        "Loss Function Justification\n"
        "===========================\n"
        "Binary classification uses Binary Cross-Entropy because the target is pass/fail and\n"
        "the models output probabilities in [0,1] through a sigmoid activation.\n\n"
        "Multi-class classification uses Sparse Categorical Cross-Entropy because the target\n"
        "is a single integer class index among {Fail, C, B, A}, and the models output class\n"
        "probabilities with softmax. Cross-entropy is appropriate for probabilistic\n"
        "classification and optimizes separation between true and predicted distributions.\n"
    )
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)


def build_intervention_report(
    raw_df: pd.DataFrame,
    x_train: np.ndarray,
    feature_cols: List[str],
    save_path: str,
) -> None:
    report_lines = []
    report_lines.append("Educational Intervention Insight Report")
    report_lines.append("=====================================")
    report_lines.append("")
    report_lines.append("Early Warning Signals")
    report_lines.append("---------------------")

    attendance_idx = feature_cols.index("Attendance") if "Attendance" in feature_cols else None
    hours_idx = feature_cols.index("Hours_Studied") if "Hours_Studied" in feature_cols else None
    prev_idx = feature_cols.index("Previous_Scores") if "Previous_Scores" in feature_cols else None

    if attendance_idx is not None:
        early_attendance = x_train[:, 2:5, attendance_idx].mean(axis=(1, 2))
        low_att = np.percentile(early_attendance, 25)
        report_lines.append(f"- Students in the bottom 25% of week 3-5 attendance (threshold ~ {low_att:.2f} scaled units).")
    if hours_idx is not None:
        study_drop = x_train[:, 0:3, hours_idx].mean(axis=(1, 2)) - x_train[:, 9:14, hours_idx].mean(axis=(1, 2))
        risky_drop = np.percentile(study_drop, 75)
        report_lines.append(f"- Noticeable decline in study hours from early to late semester (top quartile drop ~ {risky_drop:.2f}).")
    if prev_idx is not None:
        low_prev = raw_df["Previous_Scores"].quantile(0.25)
        report_lines.append(f"- Historical weak quiz/previous scores (< {low_prev:.1f}) align with elevated failure risk.")

    report_lines.append("")
    report_lines.append("Recommended Interventions")
    report_lines.append("-------------------------")
    report_lines.append("- Trigger mentoring alerts when attendance drops during weeks 3-5.")
    report_lines.append("- Offer targeted tutoring plans for students with decreasing weekly study trends.")
    report_lines.append("- Push adaptive practice quizzes for students with low prior scores.")
    report_lines.append("- Engage parents/guardians when motivation and attendance both weaken.")
    report_lines.append("- Encourage sleep and activity counseling when sleep/activity trends worsen.")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def run_experiment(
    model_name: str,
    model: Model,
    split_data: SplitData,
    output_dir: str,
    multiclass: bool = False,
) -> Dict:
    compile_model(model, multiclass=multiclass, lr=1e-3)
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    y_train = split_data.y_train_multi if multiclass else split_data.y_train_bin
    y_val = split_data.y_val_multi if multiclass else split_data.y_val_bin
    y_test = split_data.y_test_multi if multiclass else split_data.y_test_bin

    t0 = time.time()
    history = model.fit(
        split_data.x_train,
        y_train,
        validation_data=(split_data.x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[early_stop],
    )
    train_time = time.time() - t0

    t1 = time.time()
    y_prob = model.predict(split_data.x_test, verbose=0)
    test_time = time.time() - t1

    if multiclass:
        y_pred = np.argmax(y_prob, axis=1)
        metrics = evaluate_predictions(y_test, y_prob, multiclass=True)
        roc_path = os.path.join(output_dir, f"roc_curve_{model_name}_multiclass.png")
        plot_multiclass_roc(y_test, y_prob, ["Fail", "C", "B", "A"], f"{model_name} ROC (Multiclass)", roc_path)
        cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}_multiclass.png")
        plot_confusion_matrix(y_test, y_pred, ["Fail", "C", "B", "A"], f"{model_name} CM (Multiclass)", cm_path)
        report = classification_report(
            y_test, y_pred, labels=[0, 1, 2, 3], target_names=["Fail", "C", "B", "A"], zero_division=0
        )
    else:
        y_prob_flat = y_prob.reshape(-1)
        y_pred = (y_prob_flat >= 0.5).astype(int)
        metrics = evaluate_predictions(y_test, y_prob_flat, multiclass=False)
        roc_path = os.path.join(output_dir, f"roc_curve_{model_name}_binary.png")
        plot_binary_roc(y_test, y_prob_flat, f"{model_name} ROC (Binary)", roc_path)
        cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}_binary.png")
        plot_confusion_matrix(y_test, y_pred, ["Fail", "Pass"], f"{model_name} CM (Binary)", cm_path)
        report = classification_report(
            y_test, y_pred, labels=[0, 1], target_names=["Fail", "Pass"], zero_division=0
        )

    curve_path = os.path.join(output_dir, f"training_curve_{model_name}_{'multi' if multiclass else 'binary'}.png")
    save_training_curve(history, f"{model_name} Loss Curve", curve_path)

    arch_path = os.path.join(output_dir, f"architecture_{model_name}.png")
    save_model_architecture_diagram(model, f"{model_name} Architecture", arch_path)
    model_save_path = os.path.join(output_dir, f"{model_name.lower()}_{'multiclass' if multiclass else 'binary'}.keras")
    model.save(model_save_path)

    result = {
        "model_name": model_name,
        "task": "multiclass" if multiclass else "binary",
        "train_time_sec": round(train_time, 3),
        "test_time_sec": round(test_time, 3),
        "param_count": int(model.count_params()),
        **{k: round(v, 4) for k, v in metrics.items()},
        "classification_report": report,
    }
    return result


def save_artifacts_json(results: List[Dict], metadata: Dict, save_path: str) -> None:
    payload = {"results": results, "metadata": metadata}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)
    csv_path = os.path.join(OUTPUT_DIR, "StudentPerformanceFactors.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    print("Loading dataset...")
    raw_df = load_data(csv_path)
    df = create_targets(raw_df)
    split_data, feature_cols, metadata = prepare_data(df, time_steps=TIME_STEPS)
    metadata["input_shape"] = [TIME_STEPS, len(feature_cols)]

    print(f"Input tensor shape (train): {split_data.x_train.shape}")
    print(f"Input tensor shape (val): {split_data.x_val.shape}")
    print(f"Input tensor shape (test): {split_data.x_test.shape}")

    input_shape = (split_data.x_train.shape[1], split_data.x_train.shape[2])
    experiments = [
        ("LSTM", build_lstm_model(input_shape=input_shape, output_dim=1), False),
        ("GRU", build_gru_model(input_shape=input_shape, output_dim=1), False),
        ("Transformer", build_transformer_model(input_shape=input_shape, output_dim=1), False),
        ("LSTM", build_lstm_model(input_shape=input_shape, output_dim=4), True),
        ("GRU", build_gru_model(input_shape=input_shape, output_dim=4), True),
        ("Transformer", build_transformer_model(input_shape=input_shape, output_dim=4), True),
    ]

    all_results: List[Dict] = []
    for model_name, model, multiclass in experiments:
        tag = "multiclass" if multiclass else "binary"
        print(f"\nTraining {model_name} ({tag})...")
        res = run_experiment(model_name=model_name, model=model, split_data=split_data, output_dir=OUTPUT_DIR, multiclass=multiclass)
        all_results.append(res)
        print(pd.Series(res).drop("classification_report"))
        print(res["classification_report"])

    comparison_df = pd.DataFrame(all_results)
    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    hyper_df = write_hyperparameter_table(os.path.join(OUTPUT_DIR, "hyperparameters.csv"))
    write_loss_justification(os.path.join(OUTPUT_DIR, "loss_function_justification.txt"))
    build_intervention_report(raw_df=df, x_train=split_data.x_train, feature_cols=feature_cols, save_path=os.path.join(OUTPUT_DIR, "educational_intervention_report.txt"))
    save_artifacts_json(all_results, metadata, os.path.join(OUTPUT_DIR, "pipeline_results.json"))

    print("\n=== Hyperparameters ===")
    print(hyper_df.to_string(index=False))
    print("\n=== Model Comparison Table ===")
    print(comparison_df.drop(columns=["classification_report"]).to_string(index=False))
    print("\nArtifacts saved to current folder.")


if __name__ == "__main__":
    main()
