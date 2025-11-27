"""
Additional EDA Plots for Founder Retention Classification
This script creates comprehensive visualizations for the EDA report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load data
train_df = pd.read_csv('dataset/train.csv')

# Create output directory for plots
output_dir = Path('eda_plots')
output_dir.mkdir(exist_ok=True)

print("Generating comprehensive EDA plots...")

# ============================================================================
# 1. CORRELATION HEATMAP FOR NUMERICAL FEATURES
# ============================================================================
print("\n1. Creating correlation heatmap...")
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'founder_id' in numerical_cols:
    numerical_cols.remove('founder_id')

plt.figure(figsize=(12, 10))
correlation_matrix = train_df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '01_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. TARGET VARIABLE DISTRIBUTION WITH PERCENTAGES
# ============================================================================
print("2. Creating target variable distribution...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
retention_counts = train_df['retention_status'].value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(retention_counts.index, retention_counts.values, color=colors, edgecolor='black', linewidth=1.5)

# Add percentage labels
total = len(train_df)
for bar in bars:
    height = bar.get_height()
    percentage = (height / total) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xlabel('Retention Status', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Retention Status (Target Variable)', fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '02_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. NUMERICAL FEATURES VS TARGET (BOX PLOTS)
# ============================================================================
print("3. Creating box plots for numerical features vs target...")
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Numerical Features Distribution by Retention Status', fontsize=16, fontweight='bold')

for idx, col in enumerate(numerical_cols[:7]):  # Top 7 numerical features
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    train_df.boxplot(column=col, by='retention_status', ax=ax, patch_artist=True)
    ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_xlabel('Retention Status', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    plt.sca(ax)
    plt.xticks(rotation=0)

# Remove empty subplots
for idx in range(len(numerical_cols[:7]), 9):
    row = idx // 3
    col_idx = idx % 3
    fig.delaxes(axes[row, col_idx])

plt.tight_layout()
plt.savefig(output_dir / '03_numerical_vs_target_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. CATEGORICAL FEATURES DISTRIBUTION (TOP FEATURES)
# ============================================================================
print("4. Creating categorical features distribution...")
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
if 'retention_status' in categorical_cols:
    categorical_cols.remove('retention_status')

top_categorical = ['founder_gender', 'founder_role', 'work_life_balance_rating', 
                   'startup_performance_rating', 'education_background', 'startup_stage']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution of Key Categorical Features', fontsize=16, fontweight='bold')

for idx, col in enumerate(top_categorical):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    value_counts = train_df[col].value_counts()
    bars = ax.bar(range(len(value_counts)), value_counts.values, 
                  color=sns.color_palette("husl", len(value_counts)), 
                  edgecolor='black', linewidth=1)
    
    ax.set_xticks(range(len(value_counts)))
    ax.set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=9)
    ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / '04_categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. STACKED BAR CHARTS - CATEGORICAL VS TARGET
# ============================================================================
print("5. Creating stacked bar charts for categorical vs target...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Categorical Features vs Retention Status', fontsize=16, fontweight='bold')

for idx, col in enumerate(top_categorical):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Create crosstab
    ct = pd.crosstab(train_df[col], train_df['retention_status'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, ax=ax, color=['#e74c3c', '#2ecc71'], 
            edgecolor='black', linewidth=0.5)
    
    ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Percentage (%)', fontsize=10)
    ax.legend(title='Retention', loc='upper right', fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '05_categorical_vs_target_stacked.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. MISSING VALUES VISUALIZATION
# ============================================================================
print("6. Creating missing values visualization...")
missing_data = train_df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

if len(missing_data) > 0:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    missing_pct = (missing_data / len(train_df)) * 100
    bars = ax.barh(range(len(missing_pct)), missing_pct.values, 
                   color='#e74c3c', edgecolor='black', linewidth=1)
    
    ax.set_yticks(range(len(missing_pct)))
    ax.set_yticklabels(missing_pct.index)
    ax.set_xlabel('Missing Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, missing_pct.values)):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}% ({int(missing_data.values[i]):,})',
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_missing_values.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# 7. AGE DISTRIBUTION BY RETENTION STATUS
# ============================================================================
print("7. Creating age distribution comparison...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram
for status in train_df['retention_status'].unique():
    data = train_df[train_df['retention_status'] == status]['founder_age']
    axes[0].hist(data, bins=30, alpha=0.6, label=status, edgecolor='black')

axes[0].set_xlabel('Founder Age', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Age Distribution by Retention Status', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Violin plot
sns.violinplot(data=train_df, x='retention_status', y='founder_age', 
               palette=['#e74c3c', '#2ecc71'], ax=axes[1])
axes[1].set_xlabel('Retention Status', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Founder Age', fontsize=12, fontweight='bold')
axes[1].set_title('Age Distribution (Violin Plot)', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '07_age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. REVENUE VS YEARS WITH STARTUP (SCATTER)
# ============================================================================
print("8. Creating scatter plot for revenue vs years...")
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

for status, color in zip(['Left', 'Stayed'], ['#e74c3c', '#2ecc71']):
    mask = train_df['retention_status'] == status
    ax.scatter(train_df[mask]['years_with_startup'], 
               train_df[mask]['monthly_revenue_generated'],
               alpha=0.4, s=20, label=status, color=color)

ax.set_xlabel('Years with Startup', fontsize=12, fontweight='bold')
ax.set_ylabel('Monthly Revenue Generated', fontsize=12, fontweight='bold')
ax.set_title('Revenue vs Years with Startup by Retention Status', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '08_revenue_vs_years_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 9. WORK-LIFE BALANCE VS VENTURE SATISFACTION HEATMAP
# ============================================================================
print("9. Creating work-life balance vs satisfaction heatmap...")
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create crosstab
ct = pd.crosstab(train_df['work_life_balance_rating'], 
                 train_df['venture_satisfaction'])

sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
            linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_xlabel('Venture Satisfaction', fontsize=12, fontweight='bold')
ax.set_ylabel('Work-Life Balance Rating', fontsize=12, fontweight='bold')
ax.set_title('Work-Life Balance vs Venture Satisfaction', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '09_worklife_vs_satisfaction.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 10. KEY STATISTICS SUMMARY
# ============================================================================
print("10. Creating summary statistics visualization...")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.axis('off')

summary_text = f"""
DATASET SUMMARY STATISTICS
{'='*60}

📊 DATASET SIZE:
   • Training Samples: {len(train_df):,}
   • Features: {len(train_df.columns) - 2} (excluding ID and target)
   • Numerical Features: {len(numerical_cols)}
   • Categorical Features: {len(categorical_cols)}

🎯 TARGET VARIABLE (Retention Status):
   • Stayed: {(train_df['retention_status'] == 'Stayed').sum():,} ({(train_df['retention_status'] == 'Stayed').sum()/len(train_df)*100:.1f}%)
   • Left: {(train_df['retention_status'] == 'Left').sum():,} ({(train_df['retention_status'] == 'Left').sum()/len(train_df)*100:.1f}%)
   • Class Balance: Relatively balanced dataset

📈 NUMERICAL FEATURES SUMMARY:
   • Founder Age: {train_df['founder_age'].min():.0f} - {train_df['founder_age'].max():.0f} (mean: {train_df['founder_age'].mean():.1f})
   • Years with Startup: {train_df['years_with_startup'].min():.0f} - {train_df['years_with_startup'].max():.0f} (mean: {train_df['years_with_startup'].mean():.1f})
   • Monthly Revenue: ${train_df['monthly_revenue_generated'].min():.0f} - ${train_df['monthly_revenue_generated'].max():.0f}
   • Funding Rounds Led: {train_df['funding_rounds_led'].min():.0f} - {train_df['funding_rounds_led'].max():.0f}

❌ MISSING VALUES:
   • Total Missing: {train_df.isnull().sum().sum():,} ({train_df.isnull().sum().sum()/(len(train_df)*len(train_df.columns))*100:.2f}% of all values)
   • Features with Missing: {(train_df.isnull().sum() > 0).sum()}
   • Most Missing: {missing_data.index[0] if len(missing_data) > 0 else 'None'} ({missing_data.values[0] if len(missing_data) > 0 else 0:,} values)

👥 KEY CATEGORICAL DISTRIBUTIONS:
   • Gender: Male ({(train_df['founder_gender'] == 'Male').sum():,}), Female ({(train_df['founder_gender'] == 'Female').sum():,})
   • Top Role: {train_df['founder_role'].value_counts().index[0]} ({train_df['founder_role'].value_counts().values[0]:,})
   • Most Common Stage: {train_df['startup_stage'].value_counts().index[0]} ({train_df['startup_stage'].value_counts().values[0]:,})
"""

ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, 
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / '10_summary_statistics.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ All plots saved to '{output_dir}/' directory!")
print("\nGenerated plots:")
print("  1. Correlation heatmap")
print("  2. Target distribution with percentages")
print("  3. Numerical features vs target (box plots)")
print("  4. Categorical features distribution")
print("  5. Categorical vs target (stacked bars)")
print("  6. Missing values visualization")
print("  7. Age distribution comparison")
print("  8. Revenue vs years scatter plot")
print("  9. Work-life balance vs satisfaction heatmap")
print(" 10. Summary statistics")
