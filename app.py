"""
Diabetes Risk Predictor — Streamlit app (v2 UI)

Flow:
  1. User picks which kind of data they have: Clinical (lab values) or
     Survey (lifestyle/health-indicator questionnaire) — shown as two
     selectable cards.
  2. App shows a sectioned form with only the fields that model needs.
  3. App loads the best model that was trained for that dataset
     (models/saved/<name>_best_model.joblib, produced by
     models/train_clinical.py or models/train_survey.py).
  4. App predicts probability of diabetes and shows a gauge + risk verdict.

Run locally:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from src.preprocessing import (CLINICAL_GENDER_OPTIONS, CLINICAL_SMOKING_OPTIONS,
                                SURVEY_AGE_BUCKETS, SURVEY_EDUCATION,
                                SURVEY_GENHLTH, SURVEY_INCOME)
from src.utils import load_model

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺",
                    layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}

.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px;}

.hero {
    padding: 1.75rem 2rem;
    border-radius: 16px;
    background: linear-gradient(120deg, #1E88E5 0%, #43A047 100%);
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 0.95rem;}

.section-card {
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    background: #F8FAFC;
    border: 1px solid #E5E9F0;
    margin-bottom: 1.2rem;
}
.section-card h4 {margin-top: 0;}

div[data-testid="stForm"] {
    border: 1px solid #E5E9F0;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    background: #FFFFFF;
}

.stButton>button, .stFormSubmitButton>button {
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
}
.stFormSubmitButton>button {
    background: linear-gradient(120deg, #1E88E5, #43A047);
    color: white;
    border: none;
    width: 100%;
}

.risk-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
}
.risk-low {background: #E6F4EA; color: #1E7E34;}
.risk-moderate {background: #FFF4E0; color: #B8720B;}
.risk-high {background: #FDE8E8; color: #C62828;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🩺 Diabetes Risk Predictor</h1>
    <p>Powered by a thesis comparing clinical vs. survey-based ML models
    (Logistic Regression, Random Forest, Gradient Boosting).
    <b>This is an educational demo, not a medical diagnosis</b> —
    please talk to a healthcare professional for real advice.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Step 1 — choose data type, as two clickable cards
# ---------------------------------------------------------------------------
if "data_type" not in st.session_state:
    st.session_state.data_type = None

st.markdown("#### Step 1 — What kind of information do you have?")
c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("### 🧪 Clinical")
        st.caption("Lab values from a blood test: age, BMI, HbA1c, glucose, "
                   "gender, smoking history, hypertension, heart disease.")
        if st.button("Use clinical data", use_container_width=True, key="pick_clinical"):
            st.session_state.data_type = "clinical"
with c2:
    with st.container(border=True):
        st.markdown("### 📋 Survey")
        st.caption("Lifestyle & health questionnaire: general health, activity, "
                   "diet, income, education, and other BRFSS-style indicators.")
        if st.button("Use survey data", use_container_width=True, key="pick_survey"):
            st.session_state.data_type = "survey"

if st.session_state.data_type is None:
    st.info("👆 Pick a card above to continue.")
    st.stop()

is_clinical = st.session_state.data_type == "clinical"
model_key = "clinical" if is_clinical else "survey"
st.success(f"Selected: **{'Clinical' if is_clinical else 'Survey'}** data — "
           "you can click the other card above any time to switch.")

# ---------------------------------------------------------------------------
# Step 2 — load the matching model (trained ahead of time by models/train_*.py)
# ---------------------------------------------------------------------------
try:
    pipeline, metadata = load_model(model_key)
except FileNotFoundError as e:
    st.error(str(e))
    st.markdown(
        "Train it first:\n\n"
        f"```bash\npython models/train_{model_key}.py --data data/"
        f"{'DS1' if is_clinical else 'DS2'}.csv\n```"
    )
    st.stop()

with st.sidebar:
    st.markdown("### 🤖 Model in use")
    st.metric("Deployed model", metadata["best_model_name"])
    st.metric("Test AUC-ROC", metadata["test_metrics"].get("roc_auc", "n/a"))
    st.metric("Test F1 @ 0.50", metadata["test_metrics"].get("f1", "n/a"))
    st.metric("Test Recall @ 0.50", metadata["test_metrics"].get("recall", "n/a"))
    with st.expander("Cross-validation leaderboard"):
        st.dataframe(pd.DataFrame(metadata["cv_leaderboard"]).T,
                     use_container_width=True)
    if metadata.get("threshold_presets"):
        with st.expander("Recall vs. precision by threshold", expanded=True):
            st.caption("Lower thresholds catch more diabetics (higher recall) "
                       "at the cost of more false alarms (lower precision) — "
                       "useful for a screening tool where missing a case is "
                       "worse than a false positive.")
            presets_df = pd.DataFrame(metadata["threshold_presets"])
            cols = [c for c in ["recall", "precision", "f1", "note"]
                    if c in presets_df.columns]
            st.dataframe(presets_df.set_index("threshold")[cols],
                         use_container_width=True)
    st.divider()
    st.caption("Compares Logistic Regression, Random Forest, and XGBoost — "
               "each with and without ADASYN oversampling — plus a voting "
               "ensemble, selected by 5-fold cross-validated recall.")

# ---------------------------------------------------------------------------
# Step 3 — sensitivity / threshold picker (kept OUTSIDE the form so switching
# presets updates the slider immediately — widgets inside st.form only
# refresh on submit)
# ---------------------------------------------------------------------------
st.markdown("#### Step 2 — Choose sensitivity")

presets = sorted(metadata.get("threshold_presets", []), key=lambda p: p["threshold"])
recommended = metadata.get("decision_threshold", 0.5)

if presets:
    min_thr = presets[0]["threshold"]

    def _label(p):
        if p["threshold"] == recommended:
            icon = "🎯"
        elif p["threshold"] == min_thr:
            icon = "🚨"
        elif p["threshold"] == 0.50:
            icon = "⚖️"
        else:
            icon = "🔹"
        return f"{icon} {p['threshold']:.2f}"

    label_to_preset = {_label(p): p for p in presets}
    options = list(label_to_preset.keys()) + ["🔧 Custom"]
    default_index = next(
        (i for i, p in enumerate(presets) if p["threshold"] == recommended), 0
    )

    choice = st.radio("Mode", options, horizontal=True,
                       label_visibility="collapsed", index=default_index)

    if choice == "🔧 Custom":
        threshold = st.slider(
            "Decision threshold — lower catches more at-risk people (higher "
            "recall), higher is more conservative (fewer false alarms)",
            0.05, 0.95, float(recommended), 0.05,
        )
    else:
        p = label_to_preset[choice]
        threshold = p["threshold"]
        note = p.get("note") or "Threshold"
        st.caption(f"**{note}** — Recall {p['recall']:.1%} · "
                   f"Precision {p['precision']:.1%} on the held-out test set.")
        st.slider(
            "Decision threshold — lower catches more at-risk people (higher "
            "recall), higher is more conservative (fewer false alarms)",
            0.05, 0.95, threshold, 0.05, disabled=True,
        )
else:
    # Fallback for models trained before threshold sweeps were added —
    # retrain with models/train_*.py to get the full preset picker.
    threshold = st.slider(
        "Decision threshold — lower catches more at-risk people (higher "
        "recall), higher is more conservative (fewer false alarms)",
        0.05, 0.95, float(recommended), 0.05,
    )

# ---------------------------------------------------------------------------
# Step 4 — sectioned input form
# ---------------------------------------------------------------------------
st.markdown("#### Step 3 — Enter your details")

with st.form("prediction_form"):
    row = {}

    if is_clinical:
        st.markdown('<div class="section-card"><h4>🧬 Lab values</h4></div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        row["age"] = c1.number_input("Age", min_value=0, max_value=120, value=35)
        row["bmi"] = c2.number_input("BMI", min_value=10.0, max_value=70.0,
                                      value=25.0, step=0.1)
        row["HbA1c_level"] = c3.number_input("HbA1c level (%)", min_value=3.0,
                                               max_value=15.0, value=5.5, step=0.1)
        row["blood_glucose_level"] = c4.number_input(
            "Blood glucose (mg/dL)", min_value=50, max_value=400, value=100)

        st.markdown('<div class="section-card"><h4>👤 Background & history</h4></div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        row["gender"] = c1.selectbox("Gender", CLINICAL_GENDER_OPTIONS)
        row["smoking_history"] = c2.selectbox("Smoking history", CLINICAL_SMOKING_OPTIONS)
        row["hypertension"] = 1 if c3.toggle("Hypertension") else 0
        row["heart_disease"] = 1 if c4.toggle("Heart disease") else 0

    else:
        st.markdown('<div class="section-card"><h4>📊 General profile</h4></div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        row["BMI"] = c1.number_input("BMI", min_value=10.0, max_value=70.0,
                                      value=25.0, step=0.1)
        age_label = c2.selectbox("Age range", list(SURVEY_AGE_BUCKETS.values()))
        row["Age"] = [k for k, v in SURVEY_AGE_BUCKETS.items() if v == age_label][0]
        row["Sex"] = 1 if c3.radio("Sex", ["Female", "Male"], horizontal=True) == "Male" else 0

        genhlth_label = st.select_slider("General health", list(SURVEY_GENHLTH.values()),
                                          value="Good")
        row["GenHlth"] = [k for k, v in SURVEY_GENHLTH.items() if v == genhlth_label][0]

        c1, c2 = st.columns(2)
        row["MentHlth"] = c1.slider("Poor mental health days (last 30 days)", 0, 30, 0)
        row["PhysHlth"] = c2.slider("Poor physical health days (last 30 days)", 0, 30, 0)

        c1, c2 = st.columns(2)
        edu_label = c1.selectbox("Education level", list(SURVEY_EDUCATION.values()))
        row["Education"] = [k for k, v in SURVEY_EDUCATION.items() if v == edu_label][0]
        income_label = c2.selectbox("Household income", list(SURVEY_INCOME.values()))
        row["Income"] = [k for k, v in SURVEY_INCOME.items() if v == income_label][0]

        st.markdown('<div class="section-card"><h4>🩺 Health history</h4></div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            row["HighBP"] = int(st.toggle("High blood pressure"))
            row["HighChol"] = int(st.toggle("High cholesterol"))
            row["CholCheck"] = int(st.toggle("Cholesterol checked (last 5y)", value=True))
        with c2:
            row["Stroke"] = int(st.toggle("Ever had a stroke"))
            row["HeartDiseaseorAttack"] = int(st.toggle("Heart disease / attack"))
            row["DiffWalk"] = int(st.toggle("Difficulty walking/climbing stairs"))
        with c3:
            row["AnyHealthcare"] = int(st.toggle("Has healthcare coverage", value=True))
            row["NoDocbcCost"] = int(st.toggle("Skipped doctor due to cost"))
            row["HvyAlcoholConsump"] = int(st.toggle("Heavy alcohol consumption"))

        st.markdown('<div class="section-card"><h4>🥗 Lifestyle</h4></div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        row["PhysActivity"] = int(c1.toggle("Physically active (last 30 days)", value=True))
        row["Fruits"] = int(c2.toggle("Eats fruit 1+ times/day", value=True))
        row["Veggies"] = int(c3.toggle("Eats vegetables 1+ times/day", value=True))
        row["Smoker"] = int(st.toggle("Smoked 100+ cigarettes in lifetime"))

    submitted = st.form_submit_button("🔍  Predict my diabetes risk")

# ---------------------------------------------------------------------------
# Step 5 — predict + visual result
# ---------------------------------------------------------------------------
if submitted:
    X = pd.DataFrame([row])[metadata["feature_order"]]
    proba = float(pipeline.predict_proba(X)[0, 1])
    prediction = int(proba >= threshold)

    if proba < 0.33:
        level, css_class, color = "Low", "risk-low", "#1E7E34"
    elif proba < 0.66:
        level, css_class, color = "Moderate", "risk-moderate", "#B8720B"
    else:
        level, css_class, color = "High", "risk-high", "#C62828"

    st.markdown("#### Result")
    res_col, gauge_col = st.columns([1, 1])

    with res_col:
        st.markdown(f'<span class="risk-badge {css_class}">{level} risk</span>',
                    unsafe_allow_html=True)
        st.markdown(f"## {proba:.1%}")
        st.caption("Estimated probability of diabetes")

        if prediction == 1:
            st.error("⚠️ The model flags this as **higher risk**. This is not "
                      "a diagnosis — please consider speaking with a "
                      "healthcare provider.")
        else:
            st.success("✅ The model flags this as **lower risk**, based on "
                        "the inputs given.")

        st.caption(
            f"Model: **{metadata['best_model_name']}** · "
            f"Threshold: **{threshold:.2f}** · "
            f"Test AUC: **{metadata['test_metrics'].get('roc_auc', 'n/a')}**"
        )

    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 33], "color": "#E6F4EA"},
                    {"range": [33, 66], "color": "#FFF4E0"},
                    {"range": [66, 100], "color": "#FDE8E8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
        ))
        fig.update_layout(height=260, margin=dict(t=10, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)