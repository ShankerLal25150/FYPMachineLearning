"""
Feature schemas for the two diabetes datasets used in the thesis
"Diabetes Prediction Using Machine Learning".

Each schema drives THREE things from a single source of truth:
  1. The Streamlit input form (widget type, label, min/max, help text)
  2. The column order the trained sklearn Pipeline expects
  3. Basic client-side validation

DS1 = Clinical dataset  (numeric labs + demographics, binary target)
DS4 = BRFSS 2015 survey dataset (21 self-reported health indicators,
      multi-class target collapsed to binary: Diabetes vs No/Pre-Diabetes)
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Field_:
    name: str                      # column name expected by the pipeline
    label: str                     # human label shown in the Streamlit form
    kind: Literal["number", "select", "binary"]
    help: str = ""
    # number
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    default: Any = None
    # select
    options: list[Any] = field(default_factory=list)
    # binary -> rendered as Yes/No toggle, stored as 0/1
    yes_label: str = "Yes"
    no_label: str = "No"


# ---------------------------------------------------------------------------
# DS1 — Clinical dataset (Kaggle "diabetes_prediction_dataset", 100k rows)
# Target: diabetes (0/1)
# ---------------------------------------------------------------------------
CLINICAL_FIELDS: list[Field_] = [
    Field_("age", "Age (years)", "number", min_value=0.0, max_value=120.0, step=1.0, default=40.0),
    Field_("gender", "Gender", "select", options=["Female", "Male", "Other"], default="Female"),
    Field_("bmi", "BMI (Body Mass Index)", "number", min_value=10.0, max_value=70.0, step=0.1, default=25.0,
           help="Weight (kg) / Height (m)^2"),
    Field_("HbA1c_level", "HbA1c Level (%)", "number", min_value=3.0, max_value=15.0, step=0.1, default=5.5,
           help="Average blood sugar over the last 2-3 months"),
    Field_("blood_glucose_level", "Blood Glucose Level (mg/dL)", "number",
           min_value=50.0, max_value=400.0, step=1.0, default=120.0),
    Field_("hypertension", "Hypertension", "binary"),
    Field_("heart_disease", "Heart Disease", "binary"),
    Field_("smoking_history", "Smoking History", "select",
           options=["never", "No Info", "current", "former", "ever", "not current"],
           default="never"),
]
CLINICAL_TARGET_LABELS = {0: "No Diabetes", 1: "Diabetes"}
CLINICAL_COLUMN_ORDER = ["gender", "age", "hypertension", "heart_disease",
                          "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]

# ---------------------------------------------------------------------------
# DS4 — BRFSS 2015 survey dataset (self-reported, 21 features)
# Original target Diabetes_012 collapsed to binary Diabetes_binary
#   0 = No Diabetes / Pre-Diabetes (merged)   1 = Diabetes
# ---------------------------------------------------------------------------
SURVEY_FIELDS: list[Field_] = [
    Field_("HighBP", "High Blood Pressure", "binary"),
    Field_("HighChol", "High Cholesterol", "binary"),
    Field_("CholCheck", "Cholesterol Check in Last 5 Years", "binary"),
    Field_("BMI", "BMI (Body Mass Index)", "number", min_value=10.0, max_value=70.0, step=0.1, default=25.0),
    Field_("Smoker", "Smoked 100+ Cigarettes in Lifetime", "binary"),
    Field_("Stroke", "Ever Had a Stroke", "binary"),
    Field_("HeartDiseaseorAttack", "Coronary Heart Disease or Heart Attack", "binary"),
    Field_("PhysActivity", "Physical Activity in Last 30 Days", "binary"),
    Field_("Fruits", "Consumes Fruit 1+ Times/Day", "binary"),
    Field_("Veggies", "Consumes Vegetables 1+ Times/Day", "binary"),
    Field_("HvyAlcoholConsump", "Heavy Alcohol Consumption", "binary",
           help="Adult men >14 drinks/week, adult women >7 drinks/week"),
    Field_("AnyHealthcare", "Has Any Healthcare Coverage", "binary"),
    Field_("NoDocbcCost", "Skipped Doctor Due to Cost (Last 12 Months)", "binary"),
    Field_("GenHlth", "General Health", "select", options=[1, 2, 3, 4, 5], default=3,
           help="1 = Excellent ... 5 = Poor"),
    Field_("MentHlth", "Poor Mental Health Days (Last 30 Days)", "number",
           min_value=0.0, max_value=30.0, step=1.0, default=0.0),
    Field_("PhysHlth", "Poor Physical Health Days (Last 30 Days)", "number",
           min_value=0.0, max_value=30.0, step=1.0, default=0.0),
    Field_("DiffWalk", "Serious Difficulty Walking / Climbing Stairs", "binary"),
    Field_("Sex", "Sex", "select", options=["Female", "Male"], default="Female"),
    Field_("Age", "Age Category (BRFSS 13-level code)", "select",
           options=list(range(1, 14)), default=7,
           help="1=18-24 ... 9=60-64 ... 13=80+. See BRFSS codebook for exact bands."),
    Field_("Education", "Education Level (BRFSS code)", "select",
           options=list(range(1, 7)), default=4,
           help="1=None/Kindergarten ... 6=College graduate"),
    Field_("Income", "Income Level (BRFSS code)", "select",
           options=list(range(1, 9)), default=5,
           help="1=<$10k ... 8=$75k+"),
]
SURVEY_TARGET_LABELS = {0: "No Diabetes / Pre-Diabetes", 1: "Diabetes"}
SURVEY_COLUMN_ORDER = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
                        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
                        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]

SEX_MAP = {"Female": 0, "Male": 1}