### Load Data

import pandas as pd
df = pd.read_excel("bank_marketing_data_eda.xlsx")

### Working with object data only

object_columns = df.select_dtypes(include=['object']).columns

### Labeling Data

from sklearn.preprocessing import LabelEncoder

# Create a copy to keep your original data safe
df_encoded = df.copy()
label_encoders = {}

for col in object_columns:
    le = LabelEncoder()
    # We fill NaNs with 'Unknown' because LabelEncoder cannot handle NaNs
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

import numpy as np
from scipy.stats import chi2_contingency

def get_cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Assuming your target column name is 'y'
results = []
for col in object_columns:
    score = get_cramers_v(df_encoded[col], df_encoded['y'])
    results.append({'Feature': col, 'Cramers_V': score})

# Create a summary table
cramers_df = pd.DataFrame(results).sort_values(by='Cramers_V', ascending=False)
print(cramers_df)

num_df = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = num_df.corr()

import matplotlib.pyplot as plt
import seaborn as sns

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

# 1. Define columns to drop based on Low Correlation (Cramer's V < 0.05)
# From your results: loan (0.0), housing (0.009), day_of_week (0.023)
low_corr_cols = ['loan', 'housing', 'day_of_week']

# 2. Define columns to drop for Multicollinearity (Pearson > 0.90)
# Keeping nr_employed as the strongest representative of the economic cluster
# Dropping redundant float and categorical versions of the others
multicollinear_cols = [
    'euribor3m', 'euribor_cat',
    'emp.var.rate', 'emp_var_rate_cat'
]

# 3. Drop Data Leakage and Redundant numerical columns
# 'duration' is unknown before a call; 'age' is replaced by 'age_group'
leakage_and_redundant = ['duration', 'age','pdays','nr.employed','campaign','cons.price.idx','cons.conf.idx']

# Combine all lists for dropping
cols_to_remove = low_corr_cols + multicollinear_cols + leakage_and_redundant

# Execute the drop on df_encoded
# Using errors='ignore' ensures the code runs even if some columns were already dropped
df_final = df_encoded.drop(columns=cols_to_remove, errors='ignore')

#print(df_final.info())

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 1. Separate Features and Target
X = df_final.drop(columns=['y'])
y = df_final['y']

# 2. Split into Training and Testing sets (Best practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Apply Scaler
scaler = StandardScaler()

# Fit only on training data to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for readability
X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)


# Initialize and train the model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_final, y_train)

X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)

# Make predictions
y_pred = log_reg.predict(X_test_final)
y_prob = log_reg.predict_proba(X_test_final)[:, 1]# Probabilities for ROC-AUC

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(log_reg, X_test_final, y_test,
                                      display_labels=['No Churn', 'Churn'],
                                      cmap='Blues', ax=ax)
plt.title('Confusion Matrix: Logic-Based Features')
plt.show()

# Calculate ROC and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# Extract and sort the coefficients
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': log_reg.coef_[0]
}).sort_values(by='Importance', ascending=False)

print("Top Drivers of Success:")
print(importance)