# Experiment 8: Bayesian Model Training with Hyperparameter Optimization

## Overview
This experiment implements Bayesian classification models on the startup founder retention dataset using advanced hyperparameter optimization techniques.

## Models Implemented

### 1. Gaussian Naive Bayes
- Assumes features follow a Gaussian (normal) distribution
- Best for continuous numerical features
- Optimized hyperparameter: `var_smoothing` (variance smoothing parameter)

### 2. Multinomial Naive Bayes
- Suitable for discrete count data
- Works well with frequency-based features
- Optimized hyperparameters:
  - `alpha`: Additive (Laplace/Lidstone) smoothing parameter
  - `fit_prior`: Whether to learn class prior probabilities

### 3. Bernoulli Naive Bayes
- Designed for binary/boolean features
- Useful for text classification and binary feature sets
- Optimized hyperparameters:
  - `alpha`: Additive smoothing parameter
  - `binarize`: Threshold for binarizing features
  - `fit_prior`: Whether to learn class prior probabilities

## Hyperparameter Optimization

### Optuna Framework
- **Trials per model**: 50
- **Optimization direction**: Maximize accuracy
- **Cross-validation**: 5-fold stratified
- **Search strategy**: Tree-structured Parzen Estimator (TPE)

### Search Spaces
- **Gaussian NB**: `var_smoothing` ∈ [1e-12, 1e-5] (log scale)
- **Multinomial NB**: 
  - `alpha` ∈ [0.01, 10.0] (log scale)
  - `fit_prior` ∈ {True, False}
- **Bernoulli NB**:
  - `alpha` ∈ [0.01, 10.0] (log scale)
  - `binarize` ∈ [0.0, 1.0]
  - `fit_prior` ∈ {True, False}

## Evaluation Metrics

The models are evaluated using comprehensive metrics:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve

## Visualizations

The notebook generates the following visualizations:
1. **Baseline Models Comparison**: Bar chart comparing initial model performance
2. **Optimization History**: Track of optimization progress over trials
3. **Parameter Importances**: Relative importance of each hyperparameter
4. **Parallel Coordinate Plot**: Multi-dimensional parameter visualization
5. **Confusion Matrices**: Detailed prediction breakdown for each model
6. **ROC Curves**: True positive vs false positive rate comparison
7. **Precision-Recall Curves**: Precision vs recall trade-off visualization
8. **Metrics Comparison**: Comprehensive bar chart of all evaluation metrics

## Output Files

### Predictions
- `bayesian_submission.csv`: Final predictions with founder_id and retention_status
- `bayesian_submission_with_proba.csv`: Predictions with probability scores

### Analysis
- `model_evaluation_metrics.csv`: Detailed metrics for all models
- `bayesian_model_summary.txt`: Text summary of the experiment
- `optuna_studies.pkl`: Saved Optuna study objects for future analysis

### Visualizations
- `baseline_models_comparison.png`
- `gnb_optimization_history.png`
- `mnb_param_importances.png`
- `bnb_parallel_coordinate.png`
- `confusion_matrices.png`
- `roc_curves.png`
- `precision_recall_curves.png`
- `model_metrics_comparison.png`

## Key Features

### 1. Robust Preprocessing
- Label encoding for categorical variables
- Standard scaling for numerical features
- Missing value imputation using median
- Non-negative transformation for Multinomial/Bernoulli NB

### 2. Cross-Validation
- Stratified K-Fold (5 splits) ensures balanced class distribution
- Prevents overfitting and provides reliable performance estimates

### 3. Model Selection
- Automatic selection of best model based on F1 score
- Comprehensive comparison across all metrics
- Detailed performance analysis

### 4. Reproducibility
- Fixed random seed (42) for consistent results
- Saved study objects for reproducible optimization
- Detailed logging of all hyperparameters

## Usage

### Running the Notebook
```bash
# Navigate to the experiment directory
cd Experiment_8_Bayesian

# Run the notebook
jupyter notebook bayesian_model_training.ipynb
```

### Required Libraries
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
optuna
plotly
kaleido  # For saving plotly figures
```

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn optuna plotly kaleido
```

## Results Interpretation

### Best Model Selection
The best model is automatically selected based on the highest F1 score, which balances precision and recall. This is particularly important for imbalanced datasets.

### Hyperparameter Impact
- **var_smoothing** (Gaussian NB): Controls the portion of the largest variance added to variances for stability
- **alpha** (Multinomial/Bernoulli NB): Prevents zero probabilities in the model
- **binarize** (Bernoulli NB): Threshold for converting features to binary
- **fit_prior** (Multinomial/Bernoulli NB): Whether to use class frequencies from training data

### Performance Considerations
- Gaussian NB works best when features are approximately normally distributed
- Multinomial NB is effective for count-based features
- Bernoulli NB excels with binary features

## Advantages of Bayesian Models

1. **Fast Training**: Very quick to train, even on large datasets
2. **Low Memory**: Minimal memory requirements
3. **Probabilistic Predictions**: Provides probability estimates for predictions
4. **Handles Missing Data**: Can work with incomplete feature sets
5. **No Hyperparameter Tuning Required**: Works well with default parameters (though optimization improves performance)
6. **Interpretable**: Clear probabilistic interpretation of predictions

## Limitations

1. **Independence Assumption**: Assumes features are independent (rarely true in practice)
2. **Continuous Features**: Gaussian NB assumes normal distribution
3. **Zero Frequency Problem**: Requires smoothing for unseen feature combinations
4. **Linear Decision Boundary**: May not capture complex non-linear relationships

## Future Improvements

1. **Feature Engineering**: Create interaction features to capture dependencies
2. **Ensemble Methods**: Combine Bayesian models with other classifiers
3. **Semi-Supervised Learning**: Leverage unlabeled data
4. **Bayesian Networks**: Model feature dependencies explicitly
5. **Online Learning**: Implement incremental learning for streaming data

## References

- Scikit-learn Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
- Optuna Documentation: https://optuna.readthedocs.io/
- Bayesian Classification Theory: https://en.wikipedia.org/wiki/Naive_Bayes_classifier

## Contact

For questions or issues related to this experiment, please refer to the main project documentation.
