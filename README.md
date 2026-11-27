# Diabetes Risk Screening — Machine Learning Classification

**Author:** Oru Stanley Enowru — Junior Data Scientist (Healthcare Data Analytics)

## Problem statement

Diabetes can be carried for a long time without a person knowing. By the time
symptoms finally show, it may be too late for easy treatment. This project
screens patients who are likely to have diabetes, so that medical intervention
can happen faster — a screening aid for clinicians, not a diagnosis.

## Data

**Source:** Pima Indians Diabetes dataset — a standard public dataset
(originally from the U.S. National Institute of Diabetes and Digestive and
Kidney Diseases; this copy fetched from
`https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`).

- **768 patients**, all female, of Pima Indian heritage, aged 21+
- **268 positive (34.9%)** — diabetes present; 500 negative (65.1%)
- **Features (8):** Pregnancies, Glucose, BloodPressure, SkinThickness,
  Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target:** Outcome (1 = diabetes, 0 = no diabetes)

> Limitation: this dataset covers one specific population. Results do **not**
> generalise to men, other age groups, or other populations without
> re-training and validation on local data.

## Pipeline

1. **Sourcing** — downloaded the public CSV (768 rows, no header).
2. **Cleaning** — in this dataset a value of `0` is biologically impossible
   for Glucose, BloodPressure, SkinThickness, Insulin and BMI, so those zeros
   were treated as missing and imputed with the column median:

   | Column | Zeros → missing |
   |---|---:|
   | Glucose | 5 |
   | BloodPressure | 35 |
   | SkinThickness | 227 |
   | Insulin | 374 |
   | BMI | 11 |
   | **Total missing values imputed** | **652** |

   Imputation medians: Glucose 117.0, BloodPressure 72.0, SkinThickness 29.0,
   Insulin 125.0, BMI 32.3. No rows were dropped; after imputation there were
   0 missing values.
3. **Split** — train / validation / test, 60 / 20 / 20, stratified so each
   split keeps the ~35% positive rate, seed 42:
   train 460, validation 154, test 154.
4. **Models** — Logistic Regression (with standardisation), Random Forest
   (300 trees), Gradient Boosting.
5. **Evaluation** — accuracy, precision, recall, F1 and ROC-AUC on both the
   validation and the held-out test set. Recall matters most for screening:
   a missed diabetic case is more costly than a false alarm that leads to a
   follow-up test.

## Results

### Validation set (154 patients)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.753 | 0.660 | 0.611 | 0.635 | 0.861 |
| Random Forest | 0.727 | 0.620 | 0.574 | 0.596 | 0.822 |
| **Gradient Boosting** | **0.773** | **0.673** | **0.685** | **0.679** | 0.843 |

### Test set (154 patients, held out)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.740** | **0.659** | 0.509 | **0.574** | **0.789** |
| Random Forest | 0.708 | 0.580 | 0.547 | 0.563 | 0.786 |
| Gradient Boosting | 0.701 | 0.571 | 0.528 | 0.549 | 0.772 |

### What was chosen, and why

Gradient Boosting was best on the validation set (F1 0.679, recall 0.685),
but its test scores fell back (F1 0.549) — with only 154 test patients,
scores move around a lot between splits, and the three models are close on
ROC-AUC (0.77–0.79). Logistic Regression is reported as the final model:
best test F1 (0.574) and ROC-AUC (0.789), and — importantly for a clinical
setting — its coefficients can be explained to a clinician, unlike a
black-box ensemble.

Its test confusion matrix: **TN 87, FP 14, FN 26, TP 27** — out of 53
diabetic patients in the test set, it caught 27 and missed 26.

## Honest limitations

- **Recall is only ~0.51 on the test set.** This model misses about half of
  diabetic cases at the default 0.5 threshold. It is a screening *starting
  point*, not deployment-ready: lowering the decision threshold, class
  weighting, or more data would be the next steps to raise recall.
- Scores differ between validation and test, so the ranking of the three
  models is not stable — a larger dataset would be needed for a firm choice.
- The dataset is one specific population (see Data above); it must not be
  used to screen other populations as-is.
- This is a portfolio/learning project. It is not a medical device and must
  not be used to diagnose patients.

## Charts

- `charts/roc_curves.png` — ROC curves for all three models (test set)
- `charts/confusion_matrix.png` — confusion matrix, Logistic Regression (test set)

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python analysis.py
```

Outputs `results_summary.txt` and both charts. Seed is fixed (42), so the
numbers above reproduce exactly.
