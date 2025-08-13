import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="AUB Online Learning – Predictive App", layout="wide")
st.title("AUB Online Learning – Predictive App")
st.caption("EDA • Final Status Prediction • Major Grouping Prediction")

# --------------------------------------------------
# Robust artifact path handling
# --------------------------------------------------
# Prefer artifacts/; fall back to repo root if not present
CANDIDATE_DIRS = [Path("artifacts"), Path(".")]

def find_base_dir() -> Path:
    for d in CANDIDATE_DIRS:
        if (d / "clean_all_enrolled3.csv").exists() or (d / "preprocess_fs.pkl").exists():
            return d
    return Path("artifacts")  # default (will trigger missing-files error)

ART = find_base_dir()

def pick(*candidates: Path) -> Path:
    """Return the first existing path among candidates, else the first (to show in error)."""
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

# Required files (try artifacts/ first, then root)
EDA_CSV = pick(ART / "clean_all_enrolled3.csv", Path("clean_all_enrolled3.csv"))

PREP_FS = pick(ART / "preprocess_fs.pkl", Path("preprocess_fs.pkl"))
LE_FS   = pick(ART / "le_final_status.pkl", Path("le_final_status.pkl"))
# Prefer RF model; fall back to XGB or LogReg if needed
MODEL_FS = pick(
    ART / "final_status_rf_tuned.pkl", Path("final_status_rf_tuned.pkl"),
    ART / "final_status_xgb_tuned.pkl", Path("final_status_xgb_tuned.pkl"),
    ART / "final_status_logreg_tuned.pkl", Path("final_status_logreg_tuned.pkl"),
)

PREP_MG = pick(ART / "preprocess_mg_feat.pkl", Path("preprocess_mg_feat.pkl"))
LE_MG   = pick(ART / "le_major_group.pkl", Path("le_major_group.pkl"))
MODEL_MG = pick(ART / "major_group_xgb_tuned.pkl", Path("major_group_xgb_tuned.pkl"))

# --------------------------------------------------
# Fail fast with a clear message if anything is missing
# --------------------------------------------------
required_paths = [EDA_CSV, PREP_FS, LE_FS, MODEL_FS, PREP_MG, LE_MG, MODEL_MG]
missing = [str(p) for p in required_paths if not Path(p).exists()]
if missing:
    st.error("Missing required files:\n\n" + "\n".join(f"- {m}" for m in missing))
    st.info("Fix: Upload the files to an 'artifacts/' folder (preferred) or the repo root. "
            "The app searches artifacts/ first, then the root.")
    st.stop()

# --------------------------------------------------
# Cache loaders (faster app startup)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_fs_artifacts(prep_path: Path, le_path: Path, model_path: Path):
    prep = joblib.load(prep_path)
    le = joblib.load(le_path)
    model = joblib.load(model_path)
    # feature columns expected by the preprocessor
    feat = list(prep.transformers_[0][2]) + list(prep.transformers_[1][2])
    return prep, le, model, feat

@st.cache_resource(show_spinner=False)
def load_mg_artifacts(prep_path: Path, le_path: Path, model_path: Path):
    prep = joblib.load(prep_path)
    le = joblib.load(le_path)
    model = joblib.load(model_path)
    feat = list(prep.transformers_[0][2]) + list(prep.transformers_[1][2])
    return prep, le, model, feat

# --------------------------------------------------
# Load everything
# --------------------------------------------------
df_eda = load_csv(EDA_CSV)
preprocess_fs, le_fs, model_fs, feat_fs = load_fs_artifacts(PREP_FS, LE_FS, MODEL_FS)
preprocess_mg, le_mg, model_mg, feat_mg = load_mg_artifacts(PREP_MG, LE_MG, MODEL_MG)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def preprocess_inputs(df_raw: pd.DataFrame, preprocess, feat_list):
    """Ensure required columns exist, keep only expected feature order, then transform."""
    data = df_raw.copy()
    for col in feat_list:
        if col not in data.columns:
            data[col] = np.nan
    data = data[feat_list]
    return preprocess.transform(data)

def predict_batch(df_upload: pd.DataFrame, preprocess, feat_list, model, label_encoder, top2=False):
    X = preprocess_inputs(df_upload, preprocess, feat_list)
    proba = model.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)
    preds = label_encoder.inverse_transform(pred_idx)

    out = df_upload.copy()
    out["Prediction"] = preds
    out["Confidence"] = proba[np.arange(len(pred_idx)), pred_idx]

    if top2:
        top2_col = []
        for row in proba:
            idx = np.argsort(row)[-2:][::-1]
            top2_col.append(" | ".join([f"{label_encoder.classes_[i]}:{row[i]:.2f}" for i in idx]))
        out["Top2"] = top2_col

    agg = out["Prediction"].value_counts().rename_axis("Class").reset_index(name="Count")
    return out, agg

def csv_download(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def show_expected_columns(cols):
    st.write("**Expected columns (case-sensitive):**")
    st.code(", ".join(cols), language="text")

def template_button(cols, filename):
    tpl = pd.DataFrame(columns=cols)
    csv = tpl.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV template", data=csv, file_name=filename, mime="text/csv")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_eda, tab_fs, tab_mg = st.tabs(["EDA", "Final Status", "Major Grouping"])

# ============================ EDA ============================
with tab_eda:
    st.subheader("Quick Data Overview (Cleaned)")
    st.dataframe(df_eda.head(50), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if "Final Status" in df_eda.columns:
            st.bar_chart(df_eda["Final Status"].value_counts())
    with c2:
        if "Major Grouping" in df_eda.columns:
            st.bar_chart(df_eda["Major Grouping"].value_counts())
    with c3:
        if "Age" in df_eda.columns:
            st.bar_chart(df_eda["Age"].dropna())

# ===================== Final Status (batch + single) =====================
with tab_fs:
    st.subheader("Predict Final Status (Active / Inactive / Dropped / Graduated)")

    st.markdown("### Batch Prediction (many students)")
    show_expected_columns(feat_fs)
    template_button(feat_fs, "template_final_status.csv")
    fs_file = st.file_uploader("Upload CSV for Final Status", type=["csv"], key="fs_csv")

    if fs_file is not None:
        df_up = pd.read_csv(fs_file)
        res, agg = predict_batch(df_up, preprocess_fs, feat_fs, model_fs, le_fs, top2=False)

        st.markdown("#### Per-student predictions")
        st.dataframe(res, use_container_width=True)
        csv_download(res, "final_status_predictions.csv", "Download predictions CSV")

        st.markdown("#### Aggregate counts")
        st.dataframe(agg, use_container_width=True)

    st.divider()
    st.markdown("### Single Student (optional)")
    with st.form("fs_single"):
        inputs = {}
        for col in feat_fs:
            if col == "Age":
                inputs[col] = st.number_input("Age", min_value=10, max_value=100, value=30)
            else:
                inputs[col] = st.text_input(col, value="")
        go = st.form_submit_button("Predict Final Status")
        if go:
            X = preprocess_inputs(pd.DataFrame([inputs]), preprocess_fs, feat_fs)
            proba = model_fs.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            pred = le_fs.inverse_transform([pred_idx])[0]
            st.success(f"Prediction: **{pred}** (confidence {proba[pred_idx]:.2f})")
            st.write("Probabilities:")
            st.json({le_fs.classes_[i]: float(p) for i, p in enumerate(proba)})

# ===================== Major Grouping (batch + single) =====================
with tab_mg:
    st.subheader("Predict Major Grouping (e.g., Engineering, CS/IT/MIS, Health)")

    st.markdown("### Batch Prediction (many students)")
    show_expected_columns(feat_mg)
    template_button(feat_mg, "template_major_grouping.csv")
    mg_file = st.file_uploader("Upload CSV for Major Grouping", type=["csv"], key="mg_csv")

    if mg_file is not None:
        df_up = pd.read_csv(mg_file)
        res, agg = predict_batch(df_up, preprocess_mg, feat_mg, model_mg, le_mg, top2=True)

        st.markdown("#### Per-student predictions")
        st.dataframe(res, use_container_width=True)
        csv_download(res, "major_grouping_predictions.csv", "Download predictions CSV")

        st.markdown("#### Aggregate counts")
        st.dataframe(agg, use_container_width=True)

    st.divider()
    st.markdown("### Single Student (optional)")
    with st.form("mg_single"):
        inputs = {}
        for col in feat_mg:
            if col == "Age":
                inputs[col] = st.number_input("Age", min_value=10, max_value=100, value=30, key=f"age_{col}")
            else:
                inputs[col] = st.text_input(col, value="", key=f"text_{col}")
        go = st.form_submit_button("Predict Major Grouping")
        if go:
            X = preprocess_inputs(pd.DataFrame([inputs]), preprocess_mg, feat_mg)
            proba = model_mg.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            pred = le_mg.inverse_transform([pred_idx])[0]
            top2_idx = np.argsort(proba)[-2:][::-1]
            top2 = [(le_mg.classes_[i], float(proba[i])) for i in top2_idx]

            st.success(f"Prediction: **{pred}**")
            st.write("Top-2 suggestions:")
            st.json([{k: v} for k, v in top2])
            st.write("Probabilities:")
            st.json({le_mg.classes_[i]: float(p) for i, p in enumerate(proba)})
