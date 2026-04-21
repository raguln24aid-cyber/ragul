import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ModuleNotFoundError:
    tf = None
    TF_AVAILABLE = False


DATA_PATH = "StudentPerformanceFactors.csv"
RESULTS_PATH = "pipeline_results.json"
COMPARISON_PATH = "model_comparison.csv"


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Distance_from_Home"] = df["Distance_from_Home"].fillna("Unknown")
    return df


def build_feature_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    ignore_cols = {"Exam_Score"}
    cols = [c for c in df.columns if c not in ignore_cols]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if c not in numeric_cols]
    categories = {}
    for c in categorical_cols:
        # Guard against mixed dtypes and missing values in object columns.
        values = df[c].fillna("Unknown").map(str).unique().tolist()
        categories[c] = sorted(values, key=lambda x: x.lower())
    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "categories": categories,
    }


def make_sequence_row(
    row_features: pd.DataFrame,
    feature_cols: List[str],
    numeric_cols: List[str],
    time_steps: int = 14,
) -> np.ndarray:
    x = np.zeros((1, time_steps, len(feature_cols)), dtype=np.float32)
    for j, c in enumerate(feature_cols):
        val = float(row_features.iloc[0][c])
        if c in numeric_cols:
            seq = np.repeat(val, time_steps)
        else:
            seq = np.repeat(val, time_steps)
        x[0, :, j] = seq.astype(np.float32)
    return x


def get_dummy_columns(df: pd.DataFrame, categorical_cols: List[str]) -> List[str]:
    sample = pd.get_dummies(df.drop(columns=["Exam_Score"]), columns=categorical_cols, drop_first=True)
    return sample.columns.tolist()


def encode_user_input(
    user_df: pd.DataFrame,
    categorical_cols: List[str],
    expected_cols: List[str],
) -> pd.DataFrame:
    encoded = pd.get_dummies(user_df, columns=categorical_cols, drop_first=True)
    encoded = encoded.reindex(columns=expected_cols, fill_value=0)
    return encoded


def pretty_metric(value: float) -> str:
    return f"{value:.4f}" if isinstance(value, (float, int, np.floating)) else str(value)


def main() -> None:
    st.set_page_config(page_title="Student Performance Sequential DL", page_icon="🎓", layout="wide")
    st.markdown(
        """
        <style>
            .main-title { font-size: 2.1rem; font-weight: 700; color: #0E7490; }
            .card {
                border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px;
                background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="main-title">Student Performance Prediction - Sequential Deep Learning</div>', unsafe_allow_html=True)
    st.caption("LSTM | GRU | Transformer | Simulated 14-week student trajectories")
    if not TF_AVAILABLE:
        st.warning(
            "TensorFlow not found in this environment. "
            "Dashboard features work, but prediction is disabled until TensorFlow is installed."
        )

    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found: {DATA_PATH}")
        return

    df = load_dataset(DATA_PATH)
    schema = build_feature_schema(df)

    with st.sidebar:
        st.header("Controls")
        show_rows = st.slider("Preview rows", min_value=5, max_value=50, value=10, step=5)
        selected_model = st.selectbox("Binary model for prediction", ["LSTM", "GRU", "Transformer"])
        st.markdown("---")
        st.write("Run training first:")
        st.code("python student_performance_pipeline.py")

    tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Model Results", "Interactive Prediction", "Saved Plots"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Students", len(df))
        c2.metric("Numeric Features", len(schema["numeric_cols"]))
        c3.metric("Categorical Features", len(schema["categorical_cols"]))
        st.dataframe(df.head(show_rows), use_container_width=True)

    with tab2:
        if os.path.exists(COMPARISON_PATH):
            cmp_df = pd.read_csv(COMPARISON_PATH)
            st.markdown("### Model Comparison")
            st.dataframe(cmp_df, use_container_width=True)
        else:
            st.warning("`model_comparison.csv` not found. Run the training script first.")

        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            st.markdown("### Key Metrics")
            metrics_rows = []
            for r in payload.get("results", []):
                metrics_rows.append(
                    {
                        "Model": r["model_name"],
                        "Task": r["task"],
                        "Accuracy": pretty_metric(r["accuracy"]),
                        "Precision": pretty_metric(r["precision"]),
                        "Recall": pretty_metric(r["recall"]),
                        "F1": pretty_metric(r["f1_score"]),
                        "ROC-AUC": pretty_metric(r["roc_auc"]),
                        "Params": int(r["param_count"]),
                    }
                )
            st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)
        else:
            st.info("`pipeline_results.json` not found.")

    with tab3:
        st.markdown("### Single Student Prediction (Pass/Fail)")
        st.markdown('<div class="card">Input student profile to estimate passing probability.</div>', unsafe_allow_html=True)

        input_dict = {}
        col_a, col_b = st.columns(2)

        for idx, nc in enumerate(schema["numeric_cols"]):
            default_val = float(df[nc].median())
            container = col_a if idx % 2 == 0 else col_b
            input_dict[nc] = container.number_input(nc, value=default_val)

        for cc in schema["categorical_cols"]:
            options = schema["categories"][cc]
            default_option = options[0] if options else ""
            input_dict[cc] = st.selectbox(cc, options, index=options.index(default_option) if default_option in options else 0)

        if st.button("Predict", use_container_width=True):
            if not TF_AVAILABLE:
                st.error("Prediction unavailable: install TensorFlow in a supported Python environment (3.10 or 3.11).")
                return
            user_df = pd.DataFrame([input_dict])
            expected_cols = get_dummy_columns(df, schema["categorical_cols"])
            encoded_user = encode_user_input(user_df, schema["categorical_cols"], expected_cols)

            # Use dataset-level scaler approximation for a robust fallback in UI.
            scaler = StandardScaler()
            encoded_all = pd.get_dummies(df.drop(columns=["Exam_Score"]), columns=schema["categorical_cols"], drop_first=True)
            encoded_all = encoded_all.reindex(columns=expected_cols, fill_value=0)
            scaler.fit(encoded_all[schema["numeric_cols"]])
            encoded_user[schema["numeric_cols"]] = scaler.transform(encoded_user[schema["numeric_cols"]])

            x_seq = make_sequence_row(encoded_user, expected_cols, schema["numeric_cols"], time_steps=14)
            model_path = f"{selected_model.lower()}_binary.keras"
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}. Save model files after training.")
            else:
                model = tf.keras.models.load_model(model_path)
                prob = float(model.predict(x_seq, verbose=0).reshape(-1)[0])
                pred = "Pass" if prob >= 0.5 else "Fail"
                st.success(f"Prediction: {pred}")
                st.progress(min(max(prob, 0.0), 1.0))
                st.write(f"Pass Probability: **{prob:.4f}**")

    with tab4:
        st.markdown("### Generated Figures")
        figure_files = [
            "confusion_matrix_LSTM_binary.png",
            "confusion_matrix_GRU_binary.png",
            "confusion_matrix_Transformer_binary.png",
            "roc_curve_LSTM_binary.png",
            "roc_curve_GRU_binary.png",
            "roc_curve_Transformer_binary.png",
            "architecture_LSTM.png",
            "architecture_GRU.png",
            "architecture_Transformer.png",
        ]
        cols = st.columns(3)
        for i, fp in enumerate(figure_files):
            if os.path.exists(fp):
                cols[i % 3].image(fp, caption=fp, width="stretch")
            else:
                cols[i % 3].info(f"Missing: {fp}")


if __name__ == "__main__":
    main()
