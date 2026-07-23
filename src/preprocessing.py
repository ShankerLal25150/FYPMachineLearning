"""
Shared column definitions and preprocessing builders.

Keeping this in one place means the training scripts (models/train_clinical.py,
models/train_survey.py) and the Streamlit app (app.py) can never drift apart on
column names / order — the #1 cause of "it worked in Colab but not in the app"
bugs when you move a notebook into a real project.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# DS1 — Clinical dataset (e.g. Kaggle "Diabetes Prediction Dataset")
# ---------------------------------------------------------------------------
CLINICAL_TARGET = "diabetes"

CLINICAL_NUMERIC_COLS = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
CLINICAL_CATEGORICAL_COLS = ["gender", "smoking_history"]
CLINICAL_BINARY_COLS = ["hypertension", "heart_disease"]

CLINICAL_FEATURE_ORDER = (
    CLINICAL_NUMERIC_COLS + CLINICAL_CATEGORICAL_COLS + CLINICAL_BINARY_COLS
)

# Options shown in the Streamlit form. Edit these if your CSV uses different labels.
CLINICAL_GENDER_OPTIONS = ["Female", "Male", "Other"]
CLINICAL_SMOKING_OPTIONS = [
    "never", "No Info", "former", "current", "not current", "ever"
]


def build_clinical_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, CLINICAL_NUMERIC_COLS),
        ("cat", categorical_pipe, CLINICAL_CATEGORICAL_COLS),
        ("bin", "passthrough", CLINICAL_BINARY_COLS),
    ])


# ---------------------------------------------------------------------------
# DS2 — Survey dataset (BRFSS-style health indicators)
# ---------------------------------------------------------------------------
SURVEY_RAW_TARGET = "Diabetes_012"      # 0 = no, 1 = pre-diabetes, 2 = diabetes
SURVEY_BINARY_TARGET = "Diabetes_binary"  # 0 = no/pre (merged), 1 = diabetes

SURVEY_CONTINUOUS_COLS = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age",
                           "Education", "Income"]
SURVEY_BINARY_COLS = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
                       "HeartDiseaseorAttack", "PhysActivity", "Fruits",
                       "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
                       "NoDocbcCost", "DiffWalk", "Sex"]

SURVEY_FEATURE_ORDER = SURVEY_CONTINUOUS_COLS + SURVEY_BINARY_COLS

# BRFSS-style coded lookups used to build friendly dropdowns in the app.
SURVEY_AGE_BUCKETS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+",
}
SURVEY_GENHLTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
SURVEY_EDUCATION = {
    1: "Never attended / kindergarten only", 2: "Grades 1-8",
    3: "Grades 9-11", 4: "High school graduate",
    5: "Some college / technical school", 6: "College graduate",
}
SURVEY_INCOME = {
    1: "< $10,000", 2: "$10,000-$15,000", 3: "$15,000-$20,000",
    4: "$20,000-$25,000", 5: "$25,000-$35,000", 6: "$35,000-$50,000",
    7: "$50,000-$75,000", 8: "> $75,000",
}


def build_survey_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("scale", StandardScaler(), SURVEY_CONTINUOUS_COLS),
        ("passthrough", "passthrough", SURVEY_BINARY_COLS),
    ])


def merge_survey_target(df):
    """Reproduces the notebook's binary merge: pre-diabetes (1) folded into
    no-diabetes (0), diabetes (2) becomes the positive class (1)."""
    df = df.copy()
    df[SURVEY_BINARY_TARGET] = df[SURVEY_RAW_TARGET].map({0: 0, 1: 0, 2: 1})
    return df
