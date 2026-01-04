import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("bank_marketing_data_eda.xlsx")

### Stat_Summary
print(df.describe())
print(df.describe(include='object'))
print(df['y'].value_counts(normalize=True) * 100)
summary = df.describe(include='all').transpose()
print(summary)
summary.to_excel("summary_stat.xlsx")

### Co-relation Matrix

num_df = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = num_df.corr()
print(corr_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()

