"""
Task 5: Machine Learning Modeling
------------------------------------------------------------------
Full pipeline: raw data -> preprocessing -> model training ->
evaluation -> interpretation, for both a classification problem
(Titanic survival) and a regression problem (California housing
prices), per the task brief's recommended datasets.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, mean_squared_error, r2_score
)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ==================================================================
# PART A — CLASSIFICATION: Titanic survival prediction
# ==================================================================
print("="*70)
print("PART A: CLASSIFICATION — Titanic survival")
print("="*70)

df = pd.read_csv('titanic.csv')
print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# --- 1. Data preparation & feature engineering ---
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

df['FamilySize'] = df['sibsp'] + df['parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked',
            'FamilySize', 'IsAlone']
X = df[features]
y = df['survived']

categorical_features = ['sex', 'embarked']
numerical_features = ['pclass', 'age', 'sibsp', 'parch', 'fare',
                       'FamilySize', 'IsAlone']

# --- 2. Preprocessing pipeline ---
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# --- 3. Train/test split (stratified, before any fitting) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Model training: 2+ algorithms ---
clf_models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

clf_results = {}
for name, model in clf_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    clf_results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
    }
    print(f"\n{name} performance:")
    for k, v in clf_results[name].items():
        print(f"  {k}: {v:.4f}")

clf_results_df = pd.DataFrame(clf_results).T
print("\nModel comparison:\n", clf_results_df)

best_clf_name = clf_results_df['ROC-AUC'].idxmax()
print(f"\nBest classifier by ROC-AUC: {best_clf_name}")

# --- 5. Hyperparameter tuning (bonus) on the best model family ---
if best_clf_name == 'Random Forest':
    param_grid = {'classifier__n_estimators': [50, 100, 200],
                  'classifier__max_depth': [5, 10, None]}
else:
    param_grid = {'classifier__C': [0.01, 0.1, 1, 10]}

grid_search = GridSearchCV(clf_models[best_clf_name], param_grid,
                            scoring='roc_auc', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"\nTuned {best_clf_name} best params: {grid_search.best_params_}")
print(f"Tuned {best_clf_name} CV ROC-AUC: {grid_search.best_score_:.4f}")

# --- 6. Feature importance (Random Forest) ---
rf_pipeline = clf_models['Random Forest']
feature_names = (numerical_features +
                  list(rf_pipeline.named_steps['preprocessor']
                       .named_transformers_['cat']
                       .get_feature_names_out(categorical_features)))
importances = rf_pipeline.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}) \
    .sort_values('importance', ascending=False)
print("\nFeature importance (Random Forest):\n", importance_df)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x='importance', y='feature', color='#2a78d6')
plt.title('Random Forest feature importance — Titanic survival')
plt.xlabel('Importance score')
plt.tight_layout()
plt.savefig('titanic_feature_importance.png', dpi=150)
plt.close()
print("Saved: titanic_feature_importance.png")

# --- 7. Business insight ---
survival_by_sex = df.groupby('sex')['survived'].mean() * 100
survival_by_class = df.groupby('pclass')['survived'].mean() * 100
print("\nSurvival rate by sex (%):\n", survival_by_sex.round(1))
print("\nSurvival rate by class (%):\n", survival_by_class.round(1))

clf_results_df.to_csv('titanic_model_comparison.csv')

# ==================================================================
# PART B — REGRESSION: California housing price prediction
# ==================================================================
print("\n\n" + "="*70)
print("PART B: REGRESSION — California housing prices")
print("="*70)

housing = pd.read_csv('housing.csv')
print("Dataset shape:", housing.shape)
print("\nMissing values:\n", housing.isnull().sum())

# --- Preprocessing ---
housing = housing.dropna()  # total_bedrooms has some missing values
X_housing = housing.drop('median_house_value', axis=1)
y_housing = housing['median_house_value']

categorical_features_h = ['ocean_proximity']
numerical_features_h = [c for c in X_housing.columns if c not in categorical_features_h]

housing_preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features_h),
    ('cat', OneHotEncoder(), categorical_features_h)
])

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

regression_models = {
    'Linear Regression': Pipeline([
        ('preprocessor', housing_preprocessor),
        ('regressor', LinearRegression())
    ]),
    'Random Forest Regressor': Pipeline([
        ('preprocessor', housing_preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
}

reg_results = {}
for name, model in regression_models.items():
    model.fit(X_train_h, y_train_h)
    y_pred_h = model.predict(X_test_h)

    rmse = np.sqrt(mean_squared_error(y_test_h, y_pred_h))
    r2 = r2_score(y_test_h, y_pred_h)
    mae = np.mean(np.abs(y_test_h - y_pred_h))

    reg_results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
    print(f"\n{name}: RMSE = ${rmse:,.0f}, R2 = {r2:.4f}, MAE = ${mae:,.0f}")

reg_results_df = pd.DataFrame(reg_results).T
print("\nRegression model comparison:\n", reg_results_df)

best_reg_name = reg_results_df['R2'].idxmax()
print(f"\nBest regressor by R2: {best_reg_name}")

# --- Feature importance for housing RF ---
rf_h_pipeline = regression_models['Random Forest Regressor']
feature_names_h = (numerical_features_h +
                    list(rf_h_pipeline.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .get_feature_names_out(categorical_features_h)))
importances_h = rf_h_pipeline.named_steps['regressor'].feature_importances_
importance_h_df = pd.DataFrame({'feature': feature_names_h, 'importance': importances_h}) \
    .sort_values('importance', ascending=False)
print("\nFeature importance (Random Forest Regressor):\n", importance_h_df)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_h_df, x='importance', y='feature', color='#1baf7a')
plt.title('Random Forest feature importance — California housing price')
plt.xlabel('Importance score')
plt.tight_layout()
plt.savefig('housing_feature_importance.png', dpi=150)
plt.close()
print("Saved: housing_feature_importance.png")

reg_results_df.to_csv('housing_model_comparison.csv')

print("\n\nAll outputs saved: titanic_model_comparison.csv, housing_model_comparison.csv,")
print("titanic_feature_importance.png, housing_feature_importance.png")
