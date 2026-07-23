# Diabetes Risk Predictor

Streamlit app for the thesis *"Diabetes Prediction Using Machine Learning."*
Lets a user pick which kind of data they have — **Clinical** (lab
measurements: HbA1c, glucose, BMI, etc.) or **Survey** (self-reported
BRFSS-style questions, no labs needed) — fills in a form, and gets a
prediction from the matching trained pipeline.

```
diabetes-predictor/
├── app.py                    # Streamlit app (the webpage)
├── requirements.txt          # pip deps — Streamlit Community Cloud reads this
├── environment.yml           # conda env for local dev (optional)
├── .streamlit/config.toml    # theme
├── src/
│   ├── schemas.py            # single source of truth for both forms
│   ├── train_clinical.py     # trains + saves the DS1 pipeline
│   ├── train_survey.py       # trains + saves the DS4 pipeline
│   └── predict_utils.py      # loads pipelines, builds rows, predicts
├── models/                   # generated .joblib + .json (see step 2)
└── data/                     # put your DS1.csv / DS4.csv here (gitignored)
```

## Why two separate scripts and not the notebooks directly

Your Colab notebooks do the EDA/training/tuning but never call
`joblib.dump(...)` — they only `files.download()` result CSVs and PNGs.
Streamlit needs an actual saved, reloadable model object. `train_clinical.py`
and `train_survey.py` reproduce your winning configuration from each
notebook (same preprocessing, same tuned model, same ADASYN + threshold
logic for the survey set) and save a single deployable artifact per
dataset:

- `models/clinical_pipeline.joblib` + `clinical_metadata.json`
- `models/survey_pipeline.joblib` + `survey_metadata.json`

The app never re-trains anything — it just loads these two files.

## 1. Local setup

```bash
git clone <your-new-repo-url>
cd diabetes-predictor

# option A: conda
conda env create -f environment.yml
conda activate diabetes-predictor

# option B: plain venv + pip
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train and save both pipelines

Export your two datasets from Kaggle/Colab as CSVs and drop them in `data/`:

```bash
mkdir -p data
# data/DS1.csv  — clinical dataset (target column: diabetes)
# data/DS4.csv  — BRFSS dataset (target column: Diabetes_012, raw multi-class
#                 export is fine — train_survey.py collapses it to binary)
```

```bash
python src/train_clinical.py --data data/DS1.csv --out models
python src/train_survey.py   --data data/DS4.csv --out models --tune
```

`train_survey.py` runs a real `RandomizedSearchCV` on ADASYN-resampled
data by default (a couple of minutes on a laptop). It also re-derives your
recall-floor (≥0.70) threshold-selection rule on a held-out validation
split rather than hardcoding a number, so it stays correct if you swap
in a slightly different CSV split.

Check `models/` now — you should see 4 files:
`clinical_pipeline.joblib`, `clinical_metadata.json`,
`survey_pipeline.joblib`, `survey_metadata.json`.

## 3. Run it locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Pick Clinical or Survey at the top,
fill the form, hit Predict.

## 4. Push to GitHub

```bash
git init
git add .
git commit -m "Diabetes risk predictor: clinical + survey pipelines"
git branch -M main
git remote add origin https://github.com/<you>/diabetes-predictor.git
git push -u origin main
```

**Note on model file size:** XGBoost pipelines are usually small (a few
MB), but if `git add` complains about a file over 100 MB (can happen with
a heavily tuned Random Forest), use [Git LFS](https://git-lfs.com/):
```bash
git lfs install
git lfs track "models/*.joblib"
git add .gitattributes models/*.joblib
git commit -m "Track model files with LFS"
```

Since `data/*.csv` is gitignored, your raw datasets never get pushed —
only the trained pipelines in `models/` do, which is what the app needs.

## 5. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → pick your `diabetes-predictor` repo, branch `main`, main file `app.py`.
3. Streamlit Cloud auto-detects `requirements.txt` and builds the environment —
   you don't need to upload `environment.yml` (that one's just for local conda use).
4. Deploy. First build takes a couple of minutes (installing xgboost/sklearn).
5. Your app is live at `https://<your-app-name>.streamlit.app`.

Any time you `git push` to `main`, Streamlit Cloud redeploys automatically.

## Notes / things worth knowing before you defend this

- **Clinical model** uses your thesis's tuned XGBoost config
  (Test AUC-ROC 0.9797) at a fixed 0.5 decision threshold — matching
  `TrainClinical.py` as already built. If you also want the recall-floor
  threshold logic applied to DS1 (you mentioned sweeping τ 0.10–0.90
  elsewhere), say the word and I'll add the same validation-split
  threshold search here for consistency across both models.
- **Survey model** uses ADASYN + tuned XGBoost + your recall-floor
  (≥0.70) threshold rule, reproduced faithfully from notebook cells
  62–75 rather than guessed.
- The app shows the probability, the *threshold actually used* (not a
  silent 0.5), and the model's test AUC — good for anticipating jury
  questions about why threshold choice matters more than raw accuracy.
- This is a demo/thesis artifact, not a certified medical tool — the
  app says so on every prediction, and you may want to keep that
  disclaimer for the defense too.
