#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bayesian Model Training with Hyperparameter Optimization
This script implements Bayesian classification models on the startup founder retention dataset.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import optuna
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("BAYESIAN MODEL TRAINING WITH HYPERPARAMETER OPTIMIZATION")
print("="*60)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n[1/8] Loading and preprocessing data...")

# Load datasets
train_df = pd.read_csv('../dataset/train_dropped_column.csv')
test_df = pd.read_csv('../dataset/test_dropped_column.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Separate features and target
X_train = train_df.drop(['retention_status'], axis=1)
y_train = train_df['retention_status']
X_test = test_df.copy()

# Encode target variable
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

print(f"Target classes: {le.classes_}")

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

# Encode categorical variables
def encode_features(df, categorical_columns, label_encoders=None):
    """Encode categorical features using label encoding"""
    df_encoded = df.copy()
    
    if label_encoders is None:
        label_encoders = {}
        for col in categorical_columns:
            le_col = LabelEncoder()
            df_encoded[col] = le_col.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le_col
    else:
        for col in categorical_columns:
            le_col = label_encoders[col]
            df_encoded[col] = df_encoded[col].astype(str).apply(
                lambda x: le_col.transform([x])[0] if x in le_col.classes_ else -1
            )
    
    return df_encoded, label_encoders

X_train_encoded, feature_encoders = encode_features(X_train, categorical_cols)
X_test_encoded, _ = encode_features(X_test, categorical_cols, feature_encoders)

# Handle missing values
for col in numerical_cols:
    if X_train_encoded[col].isnull().any():
        median_val = X_train_encoded[col].median()
        X_train_encoded[col].fillna(median_val, inplace=True)
        X_test_encoded[col].fillna(median_val, inplace=True)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

print("Data preprocessing completed!")

# ============================================================================
# 2. BASELINE MODELS
# ============================================================================
print("\n[2/8] Training baseline models...")

baseline_models = {
    'Gaussian NB': GaussianNB(),
    'Multinomial NB': MultinomialNB(),
    'Bernoulli NB': BernoulliNB()
}

baseline_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in baseline_models.items():
    # Use appropriate data
    if name == 'Gaussian NB':
        X_data = X_train_scaled
    else:
        X_data = np.abs(X_train_encoded.values)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_data, y_train_encoded, cv=cv, scoring='accuracy')
    
    # Train on full data
    model.fit(X_data, y_train_encoded)
    
    baseline_results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"{name:20s} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 3. HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ============================================================================
print("\n[3/8] Optimizing hyperparameters with Optuna...")

# Gaussian NB optimization
def objective_gaussian_nb(trial):
    var_smoothing = trial.suggest_float('var_smoothing', 1e-12, 1e-5, log=True)
    model = GaussianNB(var_smoothing=var_smoothing)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train_scaled, y_train_encoded, cv=cv, scoring='accuracy')
    return scores.mean()

print("\nOptimizing Gaussian Naive Bayes...")
study_gnb = optuna.create_study(direction='maximize', study_name='Gaussian_NB')
study_gnb.optimize(objective_gaussian_nb, n_trials=50, show_progress_bar=True)
print(f"Best GNB Accuracy: {study_gnb.best_trial.value:.4f}")
print(f"Best GNB Params: {study_gnb.best_trial.params}")

# Multinomial NB optimization
def objective_multinomial_nb(trial):
    alpha = trial.suggest_float('alpha', 0.01, 10.0, log=True)
    fit_prior = trial.suggest_categorical('fit_prior', [True, False])
    model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
    X_data = np.abs(X_train_encoded.values)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_data, y_train_encoded, cv=cv, scoring='accuracy')
    return scores.mean()

print("\nOptimizing Multinomial Naive Bayes...")
study_mnb = optuna.create_study(direction='maximize', study_name='Multinomial_NB')
study_mnb.optimize(objective_multinomial_nb, n_trials=50, show_progress_bar=True)
print(f"Best MNB Accuracy: {study_mnb.best_trial.value:.4f}")
print(f"Best MNB Params: {study_mnb.best_trial.params}")

# Bernoulli NB optimization
def objective_bernoulli_nb(trial):
    alpha = trial.suggest_float('alpha', 0.01, 10.0, log=True)
    binarize = trial.suggest_float('binarize', 0.0, 1.0)
    fit_prior = trial.suggest_categorical('fit_prior', [True, False])
    model = BernoulliNB(alpha=alpha, binarize=binarize, fit_prior=fit_prior)
    X_data = np.abs(X_train_encoded.values)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_data, y_train_encoded, cv=cv, scoring='accuracy')
    return scores.mean()

print("\nOptimizing Bernoulli Naive Bayes...")
study_bnb = optuna.create_study(direction='maximize', study_name='Bernoulli_NB')
study_bnb.optimize(objective_bernoulli_nb, n_trials=50, show_progress_bar=True)
print(f"Best BNB Accuracy: {study_bnb.best_trial.value:.4f}")
print(f"Best BNB Params: {study_bnb.best_trial.params}")

# ============================================================================
# 4. TRAIN BEST MODELS
# ============================================================================
print("\n[4/8] Training best models with optimized hyperparameters...")

best_models = {}

# Gaussian NB
best_gnb = GaussianNB(**study_gnb.best_params)
best_gnb.fit(X_train_scaled, y_train_encoded)
best_models['Gaussian NB (Optimized)'] = best_gnb

# Multinomial NB
best_mnb = MultinomialNB(**study_mnb.best_params)
X_train_nonneg = np.abs(X_train_encoded.values)
best_mnb.fit(X_train_nonneg, y_train_encoded)
best_models['Multinomial NB (Optimized)'] = best_mnb

# Bernoulli NB
best_bnb = BernoulliNB(**study_bnb.best_params)
best_bnb.fit(X_train_nonneg, y_train_encoded)
best_models['Bernoulli NB (Optimized)'] = best_bnb

print("Best models trained successfully!")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================
print("\n[5/8] Evaluating models...")

evaluation_results = {}

for name, model in best_models.items():
    # Select appropriate data
    if 'Gaussian' in name:
        X_eval = X_train_scaled
    else:
        X_eval = X_train_nonneg
    
    # Predictions
    y_pred = model.predict(X_eval)
    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_train_encoded, y_pred)
    precision = precision_score(y_train_encoded, y_pred)
    recall = recall_score(y_train_encoded, y_pred)
    f1 = f1_score(y_train_encoded, y_pred)
    roc_auc = roc_auc_score(y_train_encoded, y_pred_proba)
    
    evaluation_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")

# Create evaluation metrics comparison
metrics_df = pd.DataFrame(evaluation_results).T
metrics_df = metrics_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]

print("\n" + "="*60)
print("MODEL COMPARISON:")
print("="*60)
print(metrics_df.round(4))

# Save to CSV
metrics_df.to_csv('model_evaluation_metrics.csv')
print("\nMetrics saved to 'model_evaluation_metrics.csv'")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n[6/8] Creating visualizations...")

# Metrics comparison
fig, ax = plt.subplots(figsize=(14, 6))
metrics_df.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim([0, 1])
ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('model_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, results) in enumerate(evaluation_results.items()):
    cm = confusion_matrix(y_train_encoded, results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=le.classes_, yticklabels=le.classes_)
    axes[idx].set_title(f'{name}\nConfusion Matrix', fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
for name, results in evaluation_results.items():
    fpr, tpr, _ = roc_curve(y_train_encoded, results['y_pred_proba'])
    roc_auc = results['roc_auc']
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations saved!")

# ============================================================================
# 7. SELECT BEST MODEL AND GENERATE PREDICTIONS
# ============================================================================
print("\n[7/8] Generating predictions...")

# Select best model based on F1 score
best_model_name = max(evaluation_results.items(), key=lambda x: x[1]['f1'])[0]
best_model = best_models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"F1 Score: {evaluation_results[best_model_name]['f1']:.4f}")

# Generate predictions on test set
if 'Gaussian' in best_model_name:
    X_test_final = X_test_scaled
else:
    X_test_final = np.abs(X_test_encoded.values)

test_predictions = best_model.predict(X_test_final)
test_predictions_proba = best_model.predict_proba(X_test_final)[:, 1]

# Convert predictions back to original labels
test_predictions_labels = le.inverse_transform(test_predictions)

print(f"\nPrediction distribution:")
print(pd.Series(test_predictions_labels).value_counts())

# Create submission file
submission_df = pd.DataFrame({
    'founder_id': test_df['founder_id'],
    'retention_status': test_predictions_labels,
    'prediction_probability': test_predictions_proba
})

# Save submission
submission_df[['founder_id', 'retention_status']].to_csv('bayesian_submission.csv', index=False)
submission_df.to_csv('bayesian_submission_with_proba.csv', index=False)

print("\nSubmission files created:")
print("  - bayesian_submission.csv")
print("  - bayesian_submission_with_proba.csv")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n[8/8] Creating summary report...")

summary = f"""
{'='*60}
BAYESIAN MODEL TRAINING SUMMARY
{'='*60}

Dataset Information:
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Number of features: {X_train.shape[1]}
- Target classes: {le.classes_}

Models Evaluated:
1. Gaussian Naive Bayes (Optimized)
2. Multinomial Naive Bayes (Optimized)
3. Bernoulli Naive Bayes (Optimized)

Best Model: {best_model_name}

Best Model Performance:
- Accuracy:  {evaluation_results[best_model_name]['accuracy']:.4f}
- Precision: {evaluation_results[best_model_name]['precision']:.4f}
- Recall:    {evaluation_results[best_model_name]['recall']:.4f}
- F1 Score:  {evaluation_results[best_model_name]['f1']:.4f}
- ROC AUC:   {evaluation_results[best_model_name]['roc_auc']:.4f}

Optimized Hyperparameters:
- Gaussian NB:     {study_gnb.best_params}
- Multinomial NB:  {study_mnb.best_params}
- Bernoulli NB:    {study_bnb.best_params}

Output Files:
- bayesian_submission.csv
- bayesian_submission_with_proba.csv
- model_evaluation_metrics.csv
- model_metrics_comparison.png
- confusion_matrices.png
- roc_curves.png

{'='*60}
TRAINING COMPLETED SUCCESSFULLY!
{'='*60}
"""

print(summary)

# Save summary to file
with open('bayesian_model_summary.txt', 'w') as f:
    f.write(summary)

print("\nSummary saved to 'bayesian_model_summary.txt'")
print("\n" + "="*60)
print("ALL TASKS COMPLETED!")
print("="*60)
