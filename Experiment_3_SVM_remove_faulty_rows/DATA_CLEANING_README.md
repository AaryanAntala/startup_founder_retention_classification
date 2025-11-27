# Data Cleaning Documentation

## Overview
This document describes the data cleaning process performed on the startup founder retention dataset to address data quality issues identified in `data_quality_check.ipynb`.

## Date: November 26, 2025

---

## Issues Identified

### 1. Age Inconsistencies
Based on the quality check analysis, we found:
- **3 rows** where `years_with_startup > founder_age` (impossible timeline)
- **39,355 rows** where `years_since_founding > founder_age` (company founded before founder was born)
- **Total age-inconsistent rows**: 39,358 (66.02% of dataset)

### 2. Missing Values
- Multiple columns had missing values
- Some rows had multiple missing values across different features

---

## Data Cleaning Process

### Step 1: Handle Age Inconsistencies

**Decision**: Remove rows with impossible age relationships

**Criteria for removal**:
1. `years_with_startup > founder_age` - Founders cannot have worked longer than they've been alive
2. `years_since_founding > founder_age` - Company cannot be older than the founder

**Rationale**:
- These are **logically impossible** data points that would introduce noise
- While this removes ~66% of data, keeping it would compromise model integrity
- The remaining data has valid, interpretable age relationships

**Impact**:
- **Original training data**: 59,611 rows
- **Rows removed**: ~39,358
- **Cleaned training data**: ~20,253 rows (33.98% retained)

### Step 2: Handle Missing Values

**Strategy**: Imputation based on data type

#### Numerical Features
- **Method**: Median imputation
- **Rationale**: 
  - Median is robust to outliers
  - Preserves the central tendency of the distribution
  - Doesn't introduce extreme values
- **Applied to**: `num_dependents`, `years_since_founding`, `monthly_revenue_generated`, etc.

#### Categorical Features  
- **Method**: Mode imputation (most frequent value)
- **Rationale**:
  - Maintains the most common pattern in the data
  - Appropriate for categorical variables
  - Less likely to create unusual category combinations
- **Applied to**: `work_life_balance_rating`, `venture_satisfaction`, `team_size_category`, etc.

#### Rows with Excessive Missing Values
- **Threshold**: Rows with >50% missing values
- **Action**: Removed (if any found)
- **Rationale**: Rows with too many missing values provide little information

### Step 3: Consistency Across Train/Test

**Important**: 
- Missing value imputation uses **training data statistics only**
- Same imputation values applied to test data to prevent data leakage
- No test data was used to compute medians/modes

---

## Output Files

### Created Files:
1. **`dataset/train_cleaned.csv`** - Cleaned training data
   - Age inconsistencies removed
   - Missing values imputed
   - Ready for model training

2. **`dataset/test_cleaned.csv`** - Cleaned test data
   - Missing values imputed using training data statistics
   - No rows removed (age issues only affected training data)
   - Ready for predictions

3. **`data_cleaning.ipynb`** - Cleaning process notebook
   - Complete code for reproducibility
   - Visualizations of before/after distributions
   - Detailed logging of all changes

---

## Quality Checks Performed

### Post-Cleaning Verification:
- ✓ No missing values remain in training data
- ✓ No missing values remain in test data  
- ✓ No age inconsistencies in cleaned training data
- ✓ Target variable distribution preserved (checked for class imbalance changes)
- ✓ All columns retained (only rows removed/imputed)

---

## Impact on Model Training

### Positive Impacts:
- **Data quality**: Only logically valid data points retained
- **Model reliability**: Predictions based on consistent, valid relationships
- **Feature integrity**: Age-related features now have valid ranges
- **No missing values**: Models won't fail on missing data

### Considerations:
- **Reduced sample size**: 66% of training data removed due to age issues
- **Potential bias**: Remaining data may not represent the full population
- **Recommendation**: Monitor model performance carefully; consider:
  - Cross-validation for robust performance estimates
  - Checking if remaining data is representative
  - Potentially collecting more high-quality data if possible

---

## Alternative Approaches Considered

### Option 1: Drop `years_since_founding` column entirely
- **Pros**: Keep all 59,611 rows
- **Cons**: Lose predictive information (column was statistically significant)
- **Decision**: Not chosen due to column's predictive value

### Option 2: Cap/transform extreme values
- **Pros**: Retain more data
- **Cons**: Introduces artificial values that may mislead the model
- **Decision**: Not chosen to maintain data integrity

### Option 3: Keep only "reasonable" founding ages (18-65)
- **Pros**: Most conservative approach
- **Cons**: Would remove 91.9% of data (too aggressive)
- **Decision**: Not chosen due to excessive data loss

---

## Recommendations for Future Data Collection

1. **Validate at source**: Implement data validation rules during data collection
2. **Age checks**: Ensure `years_with_startup ≤ founder_age`
3. **Founding age checks**: Ensure `years_since_founding ≤ founder_age`
4. **Clear definitions**: Document what "years_since_founding" means (company age vs. founder tenure)
5. **Missing value reduction**: Make critical fields mandatory

---

## Usage Instructions

### For Model Training:
```python
import pandas as pd

# Load cleaned data
train_df = pd.read_csv('dataset/train_cleaned.csv')
test_df = pd.read_csv('dataset/test_cleaned.csv')

# Proceed with feature engineering and modeling
# No additional cleaning needed for age issues or missing values
```

### For Reproducing the Cleaning:
```bash
# Open and run the cleaning notebook
jupyter notebook data_cleaning.ipynb

# Or run all cells programmatically
jupyter nbconvert --to notebook --execute data_cleaning.ipynb
```

---

## Contact & Questions

If you have questions about the cleaning process or need to modify the approach:
1. Review `data_quality_check.ipynb` for detailed quality analysis
2. Review `data_cleaning.ipynb` for step-by-step cleaning process
3. Modify the cleaning logic in `data_cleaning.ipynb` as needed

---

**Last Updated**: November 26, 2025  
**Cleaned By**: Automated data cleaning pipeline  
**Original Data**: `dataset/train.csv`, `dataset/test.csv`  
**Cleaned Data**: `dataset/train_cleaned.csv`, `dataset/test_cleaned.csv`
