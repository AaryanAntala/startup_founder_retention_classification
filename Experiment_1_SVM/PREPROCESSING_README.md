# Data Preprocessing for Founder Retention Classification

## Project Overview
This document summarizes the data preprocessing steps performed on the founder retention dataset to prepare it for training a Support Vector Machine (SVM) classifier. The goal is to predict whether a founder will stay with or leave their startup.

---

## Dataset Information

### Original Dataset
- **Training samples**: 59,611
- **Test samples**: 14,900
- **Original features**: 22 (excluding ID and target)
- **Target variable**: `retention_status` (Binary: "Stayed" or "Left")
- **Class distribution**: 
  - Stayed: 31,265 (52.4%)
  - Left: 28,346 (47.6%)
  - **Note**: Dataset is relatively balanced, which is favorable for classification

### Features After Preprocessing
- **Total features**: 26 (reduced through proper ordinal encoding)
- **Feature types**: 
  - 7 numerical features
  - 15 categorical features (8 ordinal, 7 nominal)

---

## Preprocessing Steps

### 1. Missing Value Imputation

**Problem**: The dataset contained 31,064 missing values across multiple features, with some features having up to 17% missing data.

**Features with Missing Values**:
- `work_life_balance_rating`: 17.0% missing
- `venture_satisfaction`: 12.0% missing
- `num_dependents`: 8.0% missing
- `years_since_founding`: 7.0% missing
- `team_size_category`: 5.0% missing
- `monthly_revenue_generated`: 3.0% missing

**Strategy**:
- **Numerical features**: **Median imputation**
  - Used median instead of mean to be robust against outliers
  - Applied to: `monthly_revenue_generated`, `num_dependents`, `years_since_founding`
  
- **Categorical features**: **Mode imputation**
  - Replaced missing values with the most frequent category
  - Applied to: `work_life_balance_rating`, `venture_satisfaction`, `team_size_category`

**Rationale**: 
- Median imputation for numerical features prevents outliers from skewing the imputed values
- Mode imputation for categorical features maintains the most common distribution
- Alternative approaches like deletion would have resulted in losing ~17% of the data

---

### 2. Categorical Variable Encoding

**Challenge**: SVM algorithms require numerical input, but the dataset contains 15 categorical features.

#### 2.1 Ordinal Encoding (8 features)
Features with natural ordering were encoded using label encoding to preserve their ordinal relationships:

1. **`work_life_balance_rating`**: Poor (1) → Fair (2) → Good (3) → Excellent (4)
2. **`venture_satisfaction`**: Low (1) → Medium (2) → High (3) → Very High (4)
3. **`startup_performance_rating`**: Low (1) → Below Average (2) → Average (3) → High (4)
4. **`startup_stage`**: Seed (1) → Early (2) → Growth (3) → Mature (4) → Expansion (5)
5. **`education_background`**: High School (1) → Associate Degree (2) → Bachelor's Degree (3) → Master's Degree (4) → PhD (5)
6. **`team_size_category`**: Small (1) → Medium (2) → Large (3)
7. **`startup_reputation`**: Poor (1) → Fair (2) → Good (3) → Excellent (4)
8. **`founder_visibility`**: Low (1) → Medium (2) → High (3) → Very High (4)

**Rationale**: These features have a clear progression, and preserving this order helps the SVM understand the magnitude of differences. Proper ordinal encoding reduces dimensionality compared to one-hot encoding while maintaining the meaningful ordering.

#### 2.2 One-Hot Encoding (7 features)
Nominal features without inherent ordering were one-hot encoded:
- `founder_gender` (Male, Female)
- `founder_role` (Technology, Healthcare, Education, Media, Finance)
- `working_overtime` (Yes, No)
- `personal_status` (Single, Married, Divorced)
- `remote_operations` (Yes, No)
- `leadership_scope` (Yes, No)
- `innovation_support` (Yes, No)

**Rationale**: 
- One-hot encoding prevents the model from assuming ordinal relationships that don't exist
- Uses `drop_first=True` to avoid multicollinearity (dummy variable trap)
- By correctly identifying ordinal features, we reduced the feature space from 34 to 26 features

#### 2.3 Target Variable Encoding
- `retention_status`: "Left" → 0, "Stayed" → 1
- Used LabelEncoder for binary target encoding

---

### 3. Feature Scaling (Standardization)

**Method**: StandardScaler (Z-score normalization)

**Formula**: 
```
z = (x - μ) / σ
```
where μ is the mean and σ is the standard deviation

**Result**:
- Mean of scaled features: ≈ 0
- Standard deviation of scaled features: ≈ 1

**Rationale**:
This is the **most critical preprocessing step for SVM**:

1. **SVMs are distance-based algorithms**: They rely on computing distances between data points in feature space. Features with larger scales will dominate the distance calculations.

2. **Example**: Without scaling, `monthly_revenue_generated` (range: 1,316 to 56,050) would have vastly more influence than `founder_age` (range: 6 to 59) simply due to its larger scale, even if age is more predictive.

3. **Impact on decision boundary**: Proper scaling ensures all features contribute equally to the SVM's decision boundary, improving both accuracy and convergence speed.

4. **Kernel functions**: Most SVM kernel functions (RBF, polynomial) are sensitive to feature magnitudes. Scaling ensures consistent kernel computations.

5. **Standardization vs. Normalization**: StandardScaler was chosen over Min-Max scaling because:
   - It's more robust to outliers
   - It doesn't bound features to a fixed range, which can be limiting
   - It works better with the RBF kernel commonly used in SVMs

---

### 4. Correlation Analysis

**Purpose**: Identify and address multicollinearity (highly correlated features).

**Process**:
- Computed correlation matrix for all 34 features
- Visualized using a heatmap
- Searched for feature pairs with |correlation| > 0.8

**Results**: 
- **No highly correlated features found** (threshold > 0.8)
- This indicates no severe multicollinearity issues
- All features can be retained without redundancy concerns

**Rationale**:
- Multicollinearity can make the SVM model unstable and less interpretable
- Removing redundant features reduces dimensionality and improves training efficiency
- In this dataset, features are sufficiently independent

---

### 5. Train-Validation Split

**Split Ratio**: 80% training, 20% validation

**Configuration**:
- Training set: 47,688 samples
- Validation set: 11,923 samples
- **Stratified split**: Maintains class distribution in both sets
  - Training: Stayed (52.4%), Left (47.6%)
  - Validation: Stayed (52.4%), Left (47.6%)

**Rationale**:
- Stratification ensures both sets are representative of the overall class distribution
- Validation set allows for unbiased model evaluation during training
- 80-20 split provides sufficient training data while keeping adequate validation data

---

## Output Files

The following preprocessed files have been generated:

### For Model Training
1. **`X_train_preprocessed.csv`** (59,611 × 34)
   - Full training feature set, scaled and encoded
   
2. **`y_train.csv`** (59,611 × 1)
   - Training labels (0 = Left, 1 = Stayed)

3. **`X_train_split.csv`** (47,688 × 34)
   - Training portion of the train-validation split
   
4. **`X_val_split.csv`** (11,923 × 34)
   - Validation portion for model tuning
   
5. **`y_train_split.csv`** & **`y_val_split.csv`**
   - Corresponding labels for split data

### For Final Predictions
6. **`X_test_preprocessed.csv`** (14,900 × 34)
   - Test set with identical preprocessing applied

### Transformation Objects
7. **`scaler.pkl`**
   - Fitted StandardScaler for transforming new data
   
8. **`label_encoder.pkl`**
   - LabelEncoder for inverse transforming predictions

---

## Why These Steps Are Critical for SVM

### 1. **Handling Missing Data**
- SVMs cannot process missing values
- Imputation preserves sample size and maintains statistical properties

### 2. **Encoding Categorical Variables**
- SVMs require numerical input
- Proper encoding (ordinal vs. one-hot) preserves information structure

### 3. **Feature Scaling** ⭐ **MOST IMPORTANT FOR SVM**
- **Without scaling**: Features with larger ranges dominate
- **With scaling**: All features contribute equally to decision boundary
- **Impact**: Can improve accuracy by 10-30% for SVMs
- **Convergence**: Dramatically speeds up training time

### 4. **Correlation Analysis**
- Reduces redundancy and computational cost
- Improves model stability and interpretability

### 5. **Stratified Split**
- Ensures unbiased evaluation
- Prevents overfitting detection issues

---

## Key Characteristics for SVM Training

✅ **Ready for SVM Training**:
- All features are numerical
- All features are scaled (mean ≈ 0, std ≈ 1)
- No missing values
- No severe multicollinearity
- Balanced classes (52% vs 48%)
- Sufficient training samples (47,688)

🎯 **Expected Model Performance Factors**:
- 26 features (reduced from 34 through proper ordinal encoding)
- Lower dimensionality improves SVM training efficiency and reduces overfitting risk
- Balanced dataset reduces need for class weighting
- Clean data quality should allow good generalization

---

## Next Steps

1. **Train SVM Classifier**
   - Start with linear kernel as baseline
   - Try RBF kernel for non-linear decision boundary
   - Use cross-validation for hyperparameter tuning (C, gamma)

2. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix analysis
   - ROC-AUC curve

3. **Hyperparameter Tuning**
   - Grid Search or Random Search for optimal C and gamma
   - Consider class weights if needed

4. **Generate Predictions**
   - Apply trained model to `X_test_preprocessed.csv`
   - Use `label_encoder.pkl` to convert predictions back to "Stayed"/"Left"

---

## Technical Details

### Libraries Used
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Preprocessing, scaling, encoding, and splitting
- `matplotlib` & `seaborn`: Visualization

### Preprocessing Pipeline
```python
1. Load data
2. Impute missing values (median/mode)
3. Encode categorical variables (ordinal/one-hot)
4. Scale features (StandardScaler)
5. Check for correlations
6. Split train-validation
7. Save preprocessed datasets
```

---

## Contact & Notes

- **Date Processed**: November 25, 2025
- **Preprocessing Notebook**: `analysis_and_preprocessing.ipynb`
- **Dataset Location**: `dataset/` directory

**Important**: When making predictions on new data, ensure the same preprocessing pipeline is applied in the exact same order using the saved `scaler.pkl` and `label_encoder.pkl` objects.
