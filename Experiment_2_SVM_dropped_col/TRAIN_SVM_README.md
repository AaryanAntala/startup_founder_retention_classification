# SVM Model Training Script - README

## Overview
This script (`train_svm_model.py`) trains Support Vector Machine (SVM) models on the cleaned dataset to predict founder retention status. It supports both Linear and RBF kernels with comprehensive preprocessing and evaluation.

## Features

- **Data Loading**: Loads cleaned datasets (`train_dropped_column.csv` and `test_dropped_column.csv`)
- **Preprocessing**: 
  - Categorical feature encoding (ordinal encoding for ordered categories)
  - Feature scaling (StandardScaler)
  - Train-validation split (80-20 stratified split)
- **Model Training**: 
  - Linear SVM (fast, good for linearly separable data)
  - RBF SVM (handles non-linear patterns)
  - Grid search for hyperparameter tuning
- **Evaluation**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC score
  - 5-fold cross-validation
  - Confusion matrix visualization
  - ROC curve visualization
- **Model Persistence**: Saves trained models and preprocessors for future use
- **Prediction Generation**: Creates CSV files with test set predictions

## Requirements

Make sure you have the following installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

### Basic Usage (Interactive Mode)

Run the script from the project root directory - it will prompt you to choose a kernel:

```bash
python train_svm_model.py
```

The script will prompt you to choose:
- **Option 1**: Train Linear SVM only
- **Option 2**: Train RBF SVM only  
- **Option 3**: Train and compare both models (default)

### Command Line Arguments

You can also specify options via command line:

```bash
# Train Linear SVM only
python train_svm_model.py --kernel linear

# Train RBF SVM only
python train_svm_model.py --kernel rbf

# Train both (default)
python train_svm_model.py --kernel both

# Skip grid search for faster training
python train_svm_model.py --kernel linear --no-grid-search

# Force interactive mode (ignore command line arguments)
python train_svm_model.py --interactive
```

### Available Options

- `--kernel {linear,rbf,both}`: Choose which SVM kernel(s) to train
- `--no-grid-search`: Skip hyperparameter tuning (faster, uses default parameters)
- `--interactive`: Force interactive mode to choose kernel at runtime

### Command Line Arguments (Optional)

You can modify the script to accept command-line arguments, or edit the script directly to change:
- `use_grid_search`: Set to `False` to skip hyperparameter tuning (faster training)
- `RANDOM_STATE`: Random seed for reproducibility

## Input Files

The script expects the following files in the `dataset/` directory:
- `train_dropped_column.csv` - Training dataset (with target variable)
- `test_dropped_column.csv` - Test dataset (without target variable)

## Output Files

### Models Directory (`models/`)
- `svm_linear_model.pkl` - Trained Linear SVM model
- `svm_rbf_model.pkl` - Trained RBF SVM model
- `scaler_linear.pkl` / `scaler_rbf.pkl` - Feature scalers
- `ordinal_encoder_linear.pkl` / `ordinal_encoder_rbf.pkl` - Categorical encoders
- `label_encoder_linear.pkl` / `label_encoder_rbf.pkl` - Target label encoder
- `svm_linear_metrics.txt` / `svm_rbf_metrics.txt` - Model performance metrics

### Results Directory (`results/`)
- `svm_linear_predictions.csv` - Test set predictions (Linear SVM)
- `svm_rbf_predictions.csv` - Test set predictions (RBF SVM)
- `confusion_matrix_linear.png` - Confusion matrix visualization
- `confusion_matrix_rbf.png` - Confusion matrix visualization
- `roc_curve_linear.png` - ROC curve visualization
- `roc_curve_rbf.png` - ROC curve visualization
- `model_comparison.csv` - Performance comparison (if both models trained)

## Preprocessing Details

### Categorical Encoding

The script uses **ordinal encoding** for categorical features with natural order:

**Ordinal Features**:
- `work_life_balance_rating`: Poor → Fair → Good → Excellent
- `venture_satisfaction`: Low → Medium → High → Very High
- `startup_performance_rating`: Below Average → Average → Low → High
- `startup_reputation`: Poor → Fair → Good → Excellent
- `founder_visibility`: Low → Medium → High → Very High
- `startup_stage`: Entry → Mid → Senior
- `team_size_category`: Small → Medium → Large
- `education_background`: High School → Associate → Bachelor's → Master's → PhD

**Binary Features** (automatically encoded):
- `founder_gender`, `working_overtime`, `remote_operations`, etc.

### Feature Scaling

All features are standardized using `StandardScaler`:
- Mean = 0, Standard Deviation = 1
- Critical for SVM performance (SVMs are sensitive to feature scales)

## Hyperparameter Tuning

The script performs grid search with the following parameter grids:

### Linear SVM
- `C`: [0.1, 1, 10, 100]
- `class_weight`: [None, 'balanced']

### RBF SVM
- `C`: [0.1, 1, 10, 100]
- `gamma`: ['scale', 'auto', 0.001, 0.01, 0.1]
- `class_weight`: [None, 'balanced']

**Note**: Grid search uses 5-fold cross-validation and may take several minutes. Set `use_grid_search=False` for faster training with default parameters.

## Model Evaluation Metrics

For each model, the script reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision score
- **Recall**: Weighted recall score
- **F1-Score**: Weighted F1-score
- **ROC-AUC**: Area under the ROC curve
- **CV Score**: 5-fold cross-validation accuracy (mean ± std)

## Usage Example

```python
# Example: Using the trained model for inference
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load saved model and preprocessors
with open('models/svm_rbf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler_rbf.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/ordinal_encoder_rbf.pkl', 'rb') as f:
    ordinal_encoder = pickle.load(f)

# Load and preprocess new data (same steps as training)
# ... preprocessing steps ...

# Make predictions
predictions = model.predict(new_data_scaled)
```

## Performance Tips

1. **Faster Training**: Set `use_grid_search=False` to skip hyperparameter tuning
2. **Memory Usage**: For large datasets, consider reducing the parameter grid size
3. **Parallel Processing**: Grid search uses `n_jobs=-1` to utilize all CPU cores
4. **Model Selection**: RBF SVM typically performs better for non-linear data but is slower to train

## Troubleshooting

### File Not Found Error
- Ensure `train_dropped_column.csv` and `test_dropped_column.csv` exist in the `dataset/` directory
- Run `handle_data_inconsistencies.ipynb` first to generate cleaned datasets

### Memory Issues
- Reduce the parameter grid size in `train_svm_model()` function
- Use Linear SVM only (faster and uses less memory)
- Process data in batches if needed

### Slow Training
- Skip grid search (`use_grid_search=False`)
- Use Linear SVM instead of RBF
- Reduce training data size for initial testing

## Next Steps

1. **Evaluate Model Performance**: Review confusion matrices and ROC curves
2. **Compare Models**: Check `model_comparison.csv` to select the best model
3. **Feature Engineering**: Experiment with feature combinations or transformations
4. **Ensemble Methods**: Consider combining multiple models for better performance

## Notes

- The script uses stratified train-validation split to maintain class distribution
- All preprocessing is saved to ensure consistent transformation for new data
- The target variable is encoded: 0 = "Left", 1 = "Stayed"
- Random seed is set to 42 for reproducibility

