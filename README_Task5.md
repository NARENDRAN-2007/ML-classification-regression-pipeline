# Task 5 — Machine Learning Modeling

Complete ML pipeline: raw data → preprocessing → training → evaluation →
interpretation, for both a classification problem (Titanic survival) and
a regression problem (California housing prices), using the datasets
recommended in the brief.

## Files

| File | Description |
|---|---|
| `task5_pipeline.py` | Full pipeline script — run this |
| `titanic.csv` | Classification dataset (fetched from seaborn-data) |
| `housing.csv` | Regression dataset (fetched from ageron/handson-ml2) |
| `titanic_model_comparison.csv` | Classification metrics per model |
| `housing_model_comparison.csv` | Regression metrics per model |
| `titanic_feature_importance.png` | Feature importance chart, classification |
| `housing_feature_importance.png` | Feature importance chart, regression |

## How to run

```
pip install pandas numpy scikit-learn matplotlib seaborn
python task5_pipeline.py
```

## Part A — Classification: Titanic survival

**Feature engineering:** `FamilySize` (`sibsp + parch + 1`), `IsAlone`
flag; missing `age` filled with median, missing `embarked` with mode.

**Preprocessing:** `ColumnTransformer` — `StandardScaler` on numeric
features, `OneHotEncoder(drop='first')` on categorical features
(`sex`, `embarked`), wrapped in a single `Pipeline` per model so
preprocessing is refit correctly on each train fold.

**Split:** 80/20, stratified on `survived`.

**Models trained:** Logistic Regression (baseline) vs. Random Forest.

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 80.4% | 0.729 | 0.783 | 0.681 | **0.851** |
| Random Forest | 81.6% | 0.752 | 0.781 | 0.725 | 0.829 |

Random Forest edges out on accuracy and F1; Logistic Regression has the
higher ROC-AUC and was selected as best model, then tuned further with
`GridSearchCV` (`C=1`, CV ROC-AUC 0.857).

**Feature importance (Random Forest):** `fare` (0.264), `sex_male`
(0.261), `age` (0.241) are the three strongest predictors — pclass,
family size, and embarkation port matter much less.

**Business insight:**
- Survival rate by sex: female 74.2% vs. male 18.9%
- Survival rate by class: 1st 63.0%, 2nd 47.3%, 3rd 24.2%
- Fare (a proxy for both class and cabin location) is as strong a
  predictor as gender — wealthier passengers had meaningfully better
  odds of survival.

## Part B — Regression: California housing prices

**Preprocessing:** dropped 207 rows with missing `total_bedrooms`;
`ColumnTransformer` — `StandardScaler` on numeric columns,
`OneHotEncoder` on `ocean_proximity`.

**Split:** 80/20, no stratification (continuous target).

**Models trained:** Linear Regression (baseline) vs. Random Forest
Regressor.

| Model | RMSE | R² | MAE |
|---|---|---|---|
| Linear Regression | $69,298 | 0.649 | $50,413 |
| Random Forest Regressor | **$48,758** | **0.826** | **$31,730** |

Random Forest clearly outperforms — it captures non-linear
relationships (e.g. location clusters) that a linear model can't.

**Feature importance (Random Forest):** `median_income` dominates
(0.485 — nearly half the model's decision weight), followed by
`ocean_proximity_INLAND` (0.143), `longitude` (0.109), and `latitude`
(0.105). Structural features (rooms, bedrooms, households) matter far
less than income and location.

## Common pitfalls avoided

- **Data leakage:** train/test split performed before fitting any
  preprocessing step; `Pipeline` ensures the scaler/encoder are fit
  only on training data, never on test data.
- **Overfitting:** compared a simple baseline (Logistic/Linear
  Regression) against a more complex model (Random Forest) rather than
  reporting a single model's numbers in isolation.
- **Correlation vs. causation:** insights are phrased as observed
  predictive relationships ("fare correlates strongly with survival"),
  not causal claims — fare is likely a proxy for class/cabin location,
  not survival's actual cause.

## Notes

- Datasets were fetched live from public GitHub sources
  (`seaborn-data` for Titanic, `ageron/handson-ml2` for housing) since
  none was supplied with the task brief — matches the brief's own
  "Dataset Recommendation" section exactly.
- `GridSearchCV` was demonstrated on the winning classification model
  family (Logistic Regression) as the brief's "Pro Tip" suggests;
  swap in Random Forest's grid if you want that tuned instead.
