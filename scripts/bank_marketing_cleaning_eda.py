import pandas as pd
import numpy as np

df = pd.read_csv("bankmarketing.csv")

df['y'] = df['y'].map({'no': 0, 'yes': 1})

# 1) AGE CATEGORIZATION (for EDA only)
# -------------------------
age_bins = [-np.inf, 17, 25, 35, 45, 60, np.inf]   # include <18 just in case
age_labels = ['under_18', '18_25', '26_35', '36_45', '46_60', '60_plus']

df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)
df['age_group'] = df['age_group'].astype('category')  # nicer dtype for plotting

# 2) CAMPAIGN CATEGORIZATION (for EDA only)
# campaign = number of contacts during this campaign; values can be large (e.g., 56)
campaign_bins = [ -np.inf, 3, 6, 10, np.inf ]
campaign_labels = ['1-3', '4-6', '7-10', '10+']

df['campaign_frequency'] = pd.cut(df['campaign'], bins=campaign_bins, labels=campaign_labels, right=True)
df['campaign_frequency'] = df['campaign_frequency'].astype('category')


# 3) PDAYS: EDA bins
# EDA bins:
def pdays_eda_bin(x):
    if x == 999:
        return 'no_prev_contact'
    if 0 <= x <= 5:
        return '0-5_very_recent'
    if 6 <= x <= 10:
        return '6-10_recent'
    if x >= 11:
        return '11+_long_ago'
    return 'unknown'

df['pdays_group'] = df['pdays'].apply(pdays_eda_bin).astype('category')

# 4) Employment Variation Rate
df['emp_var_rate_cat'] = pd.cut(
    df['emp.var.rate'],
    bins=[-4, -3, -1, 0.5, 2],
    labels=['very_low', 'low', 'medium', 'high']
)

# 5) Consumer Price Index
df['cons_price_cat'] = pd.cut(
    df['cons.price.idx'],
    bins=[92, 93, 94, 95],
    labels=['low_CPI', 'medium_CPI', 'high_CPI']
)

# 6) Consumer Confidence Index
df['cons_conf_cat'] = pd.cut(
    df['cons.conf.idx'],
    bins=[-60, -45, -38, -33, -25],
    labels=['very_low_conf', 'low_conf', 'medium_conf', 'high_conf']
)

# 7) Euribor 3-month Rate
df['euribor_cat'] = pd.cut(
    df['euribor3m'],
    bins=[0, 1.5, 2.5, 3.5, 4.5, 6],
    labels=['very_low_rate', 'low_rate', 'medium_rate', 'high_rate', 'very_high_rate']
)

# 8) Number of Employees
df['nr_employed_cat'] = pd.cut(
    df['nr.employed'],
    bins=[4900, 5000, 5100, 5200, 5300],
    labels=['very_low_emp', 'low_emp', 'medium_emp', 'high_emp']
)


df.to_excel("bank_marketing_data_eda.xlsx", index=False)

