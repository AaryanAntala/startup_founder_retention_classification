"""
SVM Model Training Script

This script trains Support Vector Machine (SVM) models on the cleaned dataset.
It supports both Linear and RBF kernels and includes comprehensive preprocessing.
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def load_data():
    """Load the cleaned training and test datasets."""
    print_section("STEP 1: LOADING DATA")
    
    train_path = 'dataset/train_dropped_column.csv'
    test_path = 'dataset/test_dropped_column.csv'
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✓ Training data loaded: {train_df.shape}")
    print(f"✓ Test data loaded: {test_df.shape}")
    
    return train_df, test_df

def prepare_features_and_target(train_df, test_df):
    """Prepare features and target variable from the datasets."""
    print_section("STEP 2: PREPARING FEATURES AND TARGET")
    
    # Separate features and target for training
    X = train_df.drop(['retention_status', 'founder_id'], axis=1, errors='ignore')
    y = train_df['retention_status']
    
    # Store test IDs and features
    test_ids = test_df['founder_id'].copy()
    X_test = test_df.drop(['founder_id'], axis=1, errors='ignore')
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"✓ Target classes: {label_encoder.classes_}")
    unique, counts = np.unique(y_encoded, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"✓ Class distribution: {class_dist}")
    
    # Identify feature types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\n✓ Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"✓ Numerical features ({len(numerical_cols)}): {numerical_cols}")
    
    return X, y_encoded, X_test, test_ids, categorical_cols, numerical_cols, label_encoder

def encode_categorical_features(X, X_test, categorical_cols):
    """Encode categorical features using ordinal encoding where appropriate, one-hot elsewhere."""
    print_section("STEP 3: ENCODING CATEGORICAL FEATURES")
    
    # Define ordinal mappings for features with natural order
    ordinal_mappings = {
        'work_life_balance_rating': ['Poor', 'Fair', 'Good', 'Excellent'],
        'venture_satisfaction': ['Low', 'Medium', 'High', 'Very High'],
        'startup_performance_rating': ['Below Average', 'Average', 'Low', 'High'],
        'startup_reputation': ['Poor', 'Fair', 'Good', 'Excellent'],
        'founder_visibility': ['Low', 'Medium', 'High', 'Very High'],
        'startup_stage': ['Entry', 'Mid', 'Senior'],
        'team_size_category': ['Small', 'Medium', 'Large'],
        'education_background': ['High School', 'Associate Degree', 'Bachelor\'s Degree', 
                                 'Master\'s Degree', 'PhD']
    }
    
    # Create ordinal encoder
    ordinal_encoder = OrdinalEncoder(
        categories=[ordinal_mappings[col] if col in ordinal_mappings 
                   else sorted(X[col].dropna().unique()) for col in categorical_cols],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    # Encode categorical features
    X_categorical_encoded = ordinal_encoder.fit_transform(X[categorical_cols])
    X_test_categorical_encoded = ordinal_encoder.transform(X_test[categorical_cols])
    
    # Convert to DataFrame
    X_categorical_df = pd.DataFrame(
        X_categorical_encoded, 
        columns=[f"{col}_encoded" for col in categorical_cols],
        index=X.index
    )
    X_test_categorical_df = pd.DataFrame(
        X_test_categorical_encoded,
        columns=[f"{col}_encoded" for col in categorical_cols],
        index=X_test.index
    )
    
    # Combine with numerical features
    X_encoded = pd.concat([X[numerical_cols], X_categorical_df], axis=1)
    X_test_encoded = pd.concat([X_test[numerical_cols], X_test_categorical_df], axis=1)
    
    print(f"✓ Encoded {len(categorical_cols)} categorical features using ordinal encoding")
    print(f"✓ Final feature count: {X_encoded.shape[1]} features")
    
    return X_encoded, X_test_encoded, ordinal_encoder

def scale_features(X_train, X_val, X_test):
    """Scale features using StandardScaler."""
    print_section("STEP 4: SCALING FEATURES")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for readability
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"✓ Features scaled using StandardScaler")
    print(f"  Training set mean: {X_train_scaled.mean().mean():.6f}, std: {X_train_scaled.std().mean():.6f}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def train_svm_model(X_train, y_train, X_val, y_val, kernel='rbf', use_grid_search=True, verbose=1):
    """Train an SVM model with optional hyperparameter tuning."""
    print_section(f"STEP 5: TRAINING {kernel.upper()} SVM MODEL")
    
    if kernel == 'linear':
        if use_grid_search:
            print("Performing grid search for Linear SVM...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'class_weight': [None, 'balanced']
            }
            svm = SVC(kernel='linear', random_state=RANDOM_STATE, probability=True)
            grid_search = GridSearchCV(
                svm, param_grid, cv=5, scoring='accuracy', 
                n_jobs=-1, verbose=verbose
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"✓ Best parameters: {grid_search.best_params_}")
        else:
            model = SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE, probability=True)
            model.fit(X_train, y_train)
            print("✓ Model trained with default parameters")
    
    elif kernel == 'rbf':
        if use_grid_search:
            print("Performing grid search for RBF SVM...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'class_weight': [None, 'balanced']
            }
            svm = SVC(kernel='rbf', random_state=RANDOM_STATE, probability=True)
            grid_search = GridSearchCV(
                svm, param_grid, cv=5, scoring='accuracy',
                n_jobs=-1, verbose=verbose
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"✓ Best parameters: {grid_search.best_params_}")
        else:
            model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE, probability=True)
            model.fit(X_train, y_train)
            print("✓ Model trained with default parameters")
    
    # Evaluate on validation set
    start_time = time.time()
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, average='weighted')
    recall = recall_score(y_val, y_val_pred, average='weighted')
    f1 = f1_score(y_val, y_val_pred, average='weighted')
    roc_auc = roc_auc_score(y_val, y_val_proba)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"\n✓ Validation Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"  Prediction time: {prediction_time:.2f}s")
    
    return model, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_mean,
        'cv_std': cv_std
    }

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()

def plot_roc_curve(y_true, y_proba, title="ROC Curve", save_path=None):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✓ ROC curve saved to: {save_path}")
    plt.close()

def save_model_and_preprocessors(model, scaler, ordinal_encoder, label_encoder, kernel, metrics):
    """Save the trained model and preprocessing objects."""
    print_section("STEP 6: SAVING MODEL AND PREPROCESSORS")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = f'models/svm_{kernel}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = f'models/scaler_{kernel}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Save ordinal encoder
    encoder_path = f'models/ordinal_encoder_{kernel}.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(ordinal_encoder, f)
    print(f"✓ Ordinal encoder saved to: {encoder_path}")
    
    # Save label encoder
    label_encoder_path = f'models/label_encoder_{kernel}.pkl'
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Label encoder saved to: {label_encoder_path}")
    
    # Save metrics
    metrics_path = f'models/svm_{kernel}_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"SVM {kernel.upper()} Model Metrics\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"CV Score (mean): {metrics['cv_mean']:.4f}\n")
        f.write(f"CV Score (std): {metrics['cv_std']:.4f}\n")
    print(f"✓ Metrics saved to: {metrics_path}")

def generate_predictions(model, X_test_scaled, test_ids, label_encoder, kernel):
    """Generate predictions on test set and save to CSV."""
    print_section("STEP 7: GENERATING PREDICTIONS")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate predictions
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Convert back to original labels
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'founder_id': test_ids,
        'retention_status': y_test_pred_labels
    })
    
    # Save predictions
    output_path = f'results/svm_{kernel}_predictions.csv'
    submission_df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")
    print(f"  Total predictions: {len(submission_df)}")
    print(f"  Predictions distribution:")
    print(submission_df['retention_status'].value_counts())
    
    return submission_df

def main():
    """Main function to run the complete pipeline."""
    # Parse command line arguments first
    parser = argparse.ArgumentParser(description='Train SVM model on cleaned dataset')
    parser.add_argument('--kernel', type=str, choices=['linear', 'rbf', 'both'], 
                      default=None, help='SVM kernel to use (linear/rbf/both)')
    parser.add_argument('--no-grid-search', action='store_true', 
                      help='Skip grid search for faster training')
    parser.add_argument('--interactive', action='store_true',
                      help='Use interactive kernel selection')
    
    args = parser.parse_args()
    
    print_section("SVM MODEL TRAINING PIPELINE")
    
    start_time = time.time()
    
    try:
        # Step 1: Load data
        train_df, test_df = load_data()
        
        # Step 2: Prepare features and target
        X, y, X_test, test_ids, categorical_cols, numerical_cols, label_encoder = \
            prepare_features_and_target(train_df, test_df)
        
        # Step 3: Encode categorical features
        X_encoded, X_test_encoded, ordinal_encoder = \
            encode_categorical_features(X, X_test, categorical_cols)
        
        # Step 4: Split data
        print_section("STEP 4: SPLITTING DATA")
        X_train, X_val, y_train, y_val = train_test_split(
            X_encoded, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        print(f"✓ Training set: {X_train.shape[0]} samples")
        print(f"✓ Validation set: {X_val.shape[0]} samples")
        
        # Step 5: Scale features
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = \
            scale_features(X_train, X_val, X_test_encoded)
        
        # Step 6: Determine kernel choice
        if args.interactive or args.kernel is None:
            print_section("KERNEL SELECTION")
            print("Choose SVM kernel:")
            print("1. Linear SVM (faster, good for linearly separable data)")
            print("2. RBF SVM (slower, good for non-linear data)")
            print("3. Both (train and compare both)")
            
            user_choice = input("\nEnter choice (1/2/3) [default: 3]: ").strip() or "3"
            choice = user_choice
        else:
            choice_map = {'linear': '1', 'rbf': '2', 'both': '3'}
            choice = choice_map.get(args.kernel, '3')
            print_section("KERNEL SELECTION")
            print(f"Using kernel: {args.kernel.upper()} (from command line argument)")
        
        use_grid_search = not args.no_grid_search
        if not use_grid_search:
            print("⚠ Grid search disabled - using default hyperparameters")
        
        models = {}
        all_metrics = {}
        
        if choice in ['1', '3']:
            model_linear, metrics_linear = train_svm_model(
                X_train_scaled, y_train, X_val_scaled, y_val, 
                kernel='linear', use_grid_search=use_grid_search, verbose=1 if use_grid_search else 0
            )
            models['linear'] = model_linear
            all_metrics['linear'] = metrics_linear
            
            # Plot confusion matrix and ROC curve
            y_val_pred_linear = model_linear.predict(X_val_scaled)
            y_val_proba_linear = model_linear.predict_proba(X_val_scaled)[:, 1]
            
            plot_confusion_matrix(
                y_val, y_val_pred_linear, 
                title="Linear SVM - Confusion Matrix",
                save_path=f"results/confusion_matrix_linear.png"
            )
            plot_roc_curve(
                y_val, y_val_proba_linear,
                title="Linear SVM - ROC Curve",
                save_path=f"results/roc_curve_linear.png"
            )
            
            # Save model
            save_model_and_preprocessors(
                model_linear, scaler, ordinal_encoder, label_encoder,
                'linear', metrics_linear
            )
            
            # Generate predictions
            generate_predictions(
                model_linear, X_test_scaled, test_ids, label_encoder, 'linear'
            )
        
        if choice in ['2', '3']:
            model_rbf, metrics_rbf = train_svm_model(
                X_train_scaled, y_train, X_val_scaled, y_val,
                kernel='rbf', use_grid_search=use_grid_search, verbose=1 if use_grid_search else 0
            )
            models['rbf'] = model_rbf
            all_metrics['rbf'] = metrics_rbf
            
            # Plot confusion matrix and ROC curve
            y_val_pred_rbf = model_rbf.predict(X_val_scaled)
            y_val_proba_rbf = model_rbf.predict_proba(X_val_scaled)[:, 1]
            
            plot_confusion_matrix(
                y_val, y_val_pred_rbf,
                title="RBF SVM - Confusion Matrix",
                save_path=f"results/confusion_matrix_rbf.png"
            )
            plot_roc_curve(
                y_val, y_val_proba_rbf,
                title="RBF SVM - ROC Curve",
                save_path=f"results/roc_curve_rbf.png"
            )
            
            # Save model
            save_model_and_preprocessors(
                model_rbf, scaler, ordinal_encoder, label_encoder,
                'rbf', metrics_rbf
            )
            
            # Generate predictions
            generate_predictions(
                model_rbf, X_test_scaled, test_ids, label_encoder, 'rbf'
            )
        
        # Compare models if both were trained
        if choice == '3' and len(models) == 2:
            print_section("MODEL COMPARISON")
            comparison_df = pd.DataFrame(all_metrics).T
            print("\nModel Performance Comparison:")
            print(comparison_df.round(4))
            
            # Save comparison
            comparison_df.to_csv('results/model_comparison.csv')
            print("\n✓ Comparison saved to: results/model_comparison.csv")
            
            # Determine best model
            best_model_kernel = comparison_df['roc_auc'].idxmax()
            print(f"\n✓ Best model based on ROC-AUC: {best_model_kernel.upper()} SVM")
        
        total_time = time.time() - start_time
        print_section("TRAINING COMPLETE")
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

