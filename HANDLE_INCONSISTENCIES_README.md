# Data Inconsistencies Handling - README

## Overview
This document describes the data cleaning steps performed to handle data inconsistencies identified in the data quality check notebook. Both training and test datasets are processed.

## Notebook
**File**: `handle_data_inconsistencies.ipynb`

## Steps Performed

### Step 1: Handle Age Inconsistencies (Training Data Only)
**Issue**: Some rows had inconsistent data where `years_with_startup > founder_age`, which is logically impossible (a founder cannot have worked with a startup longer than their age).

**Action Taken**:
- Identified rows where `years_with_startup > founder_age`
- Found 3 such inconsistent rows in training data
- **Removed these 3 rows from the training dataset only**
- **No rows removed from test dataset** (as per requirement)

**Training Rows Removed**: 3
- Founder ID 50946: Age 13, Years with startup 22
- Founder ID 4762: Age 6, Years with startup 9
- Founder ID 34398: Age 23, Years with startup 26

**Test Data**: All rows kept intact, even if they have age inconsistencies

### Step 2: Drop years_since_founding Column (Both Datasets)
**Issue**: The `years_since_founding` column had significant data quality issues:
- 66.02% of rows had impossible values (company founded before founder was born)
- 77.80% had suspicious values (founded before age 10)
- Only 8.10% had reasonable founding ages (18-65)

**Action Taken**:
- Dropped the `years_since_founding` column from both training and test datasets
- This column was causing many logical inconsistencies and was deemed unreliable

**Column Removed**: `years_since_founding` (from both datasets)

### Step 3: Analyze Missing Values (Both Datasets)
**Analysis Performed**:
1. Identified all columns with missing values in both datasets
2. Calculated missing value counts and percentages for each dataset
3. Analyzed patterns in missing data (rows with multiple missing values)
4. Visualized missing values distribution for both datasets

**Training Data Missing Values**:
- `work_life_balance_rating`: ~10,144 missing (~17.0%)
- `venture_satisfaction`: ~7,164 missing (~12.0%)
- `num_dependents`: ~4,780 missing (~8.0%)
- `team_size_category`: ~2,992 missing (~5.0%)
- `monthly_revenue_generated`: ~1,800 missing (~3.0%)

**Test Data Missing Values**: Similar patterns as training data (exact counts may vary)

Note: The `years_since_founding` column (which had 4,184 missing values in training) was already dropped in Step 2.

### Step 4: Handle Missing Values Appropriately (Both Datasets)
**Important Strategy**:
- **Calculate imputation statistics from TRAINING data only**
- **Apply the same values to both training and test datasets**
- This ensures consistency and prevents data leakage

**Imputation Approach**:
- **Numerical columns**: Fill with median calculated from training data only
- **Categorical columns**: Fill with mode (most frequent value) calculated from training data only
- Both datasets use identical imputation values from training statistics

**Why This Matters**:
- **Consistency**: Same preprocessing for train and test ensures model performance matches deployment
- **No Data Leakage**: Test data statistics don't influence training preprocessing
- **Realistic**: Matches production scenario where only training statistics are available

**Columns Imputed** (for both datasets using training statistics):

**Numerical Columns** (filled with training median):
- `num_dependents`: Filled with training median value
- `monthly_revenue_generated`: Filled with training median value

**Categorical Columns** (filled with training mode):
- `work_life_balance_rating`: Filled with training mode (most frequent rating)
- `venture_satisfaction`: Filled with training mode (most frequent satisfaction level)
- `team_size_category`: Filled with training mode (most frequent team size category)

**Verification**:
- After imputation, verified that no missing values remain in either dataset
- Confirmed that test dataset uses training-derived imputation values

## Results

### Training Dataset Changes:
- **Original dataset**: 59,611 rows, 24 columns
- **Rows removed**: 3 (age inconsistencies)
- **Columns removed**: 1 (`years_since_founding`)
- **Final dataset**: 59,608 rows, 23 columns
- **Missing values**: 0 (all handled through imputation)

### Test Dataset Changes:
- **Original dataset**: 14,900 rows, 23 columns
- **Rows removed**: 0 (all rows kept as requested)
- **Columns removed**: 1 (`years_since_founding`)
- **Final dataset**: 14,900 rows, 22 columns
- **Missing values**: 0 (all handled through imputation)

### Output Files:
- **Training dataset**: `dataset/train_dropped_column.csv`
- **Test dataset**: `dataset/test_dropped_column.csv`

## Data Quality Improvements

1. **Logical Consistency**: Removed impossible age combinations
2. **Data Reliability**: Removed unreliable column with high inconsistency rate
3. **Completeness**: All missing values have been appropriately handled
4. **Data Integrity**: Dataset is now ready for further analysis and modeling

## Next Steps

The cleaned datasets are now ready for:
- Feature engineering
- Exploratory data analysis
- Model training
- Further preprocessing steps

## Notes

- **Training data**: Rows with age inconsistencies were removed (3 rows)
- **Test data**: All rows were kept intact (no rows removed) as per requirement
- **Missing value imputation**: 
  - Imputation statistics (median/mode) are calculated from **training data only**
  - The same values are applied to both training and test datasets
  - This ensures consistency and prevents data leakage from test into training preprocessing
- **Imputation methods**:
  - Median was chosen for numerical imputation as it's more robust to outliers than mean
  - Mode was chosen for categorical imputation as it preserves the distribution of categorical data
- The `years_since_founding` column was dropped from both datasets due to high inconsistency rates
- All imputation was performed after removing inconsistent rows (training only) and dropping the problematic column

