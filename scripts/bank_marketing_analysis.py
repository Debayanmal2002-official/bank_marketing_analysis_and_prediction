import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("bank_marketing_data_eda.xlsx")

#0) Last Campain overall Success Rate

success_counts = df['y'].value_counts()
labels = ['No Subscription', 'Subscribed']
sizes = [success_counts[0], success_counts[1]]
colors = ['#FF6F61', '#6BCB77']

plt.figure(figsize=(7,7))
plt.pie(sizes, labels=None, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title("Overall Last Campaign Subscription Success Rate", fontsize=15)
plt.legend(labels, loc="upper right")
plt.tight_layout()
plt.show()


# 1) Age Range DISTRIBUTION

age_counts = df['age_group'].value_counts()
age_percent = (age_counts / age_counts.sum() * 100).round(2).astype(str) + '%'
legend_labels = [f"{grp} — {pct}" for grp, pct in zip(age_counts.index, age_percent)]

plt.figure(figsize=(8,8))
wedges, _ = plt.pie(
    age_counts,
    labels=None,
    autopct=None,
    startangle=90,
    wedgeprops={'width':0.35, 'edgecolor':'white'}
)
plt.legend(
    wedges,
    legend_labels,
    title="Age Groups",
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)
plt.title('Age Group Distribution')
plt.tight_layout()
plt.show()

# 2) Success Rate by Different Column(Vertical Graph)

def plot_success_rate_h(df, col, save=False):
    success = df.groupby(col)['y'].mean().reset_index()
    success['y'] = success['y'] * 100
    success = success.sort_values('y', ascending=False).reset_index(drop=True)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=success, x='y', y=col, hue=col, palette='Set3', legend=False)
    for index, value in enumerate(success['y']):
        plt.text(value + 0.5, index, f"{value:.1f}%", va='center')
    display_col = col.replace('_', ' ').capitalize()
    plt.title(f'Subscription Success Rate by {col}')
    plt.xlabel('Subscription Rate (%)')
    plt.ylabel(f'{col} Category')
    plt.xlim(0, success['y'].max() + 10)
    plt.tight_layout()
    if save:
        file_name = f"plots/{col}_success_rate.png"
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.show()

    return success

cols_h = ['age_group', 'job','education','month','poutcome']
for c in cols_h:
    plot_success_rate_h(df, c, save=False)

# 3) Success Rate by Different Column(Horizontal Graph)

def plot_success_rate_v(df, col, save=False):
    success = df.groupby(col)['y'].mean().reset_index()
    success['y'] = success['y'] * 100
    success = success.sort_values('y', ascending=False).reset_index(drop=True)
    order = success[col].tolist()
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=success, x=col, y='y', hue=col, order=order, palette='Set1', legend=False)
    for bar, val in zip(ax.patches, success['y']):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.text(x, y + 0.6, f"{val:.1f}%", ha='center', va='bottom', fontsize=10)
    display_col = col.replace('_', ' ').capitalize()
    plt.title(f'Subscription Success Rate by Status of {display_col}')
    plt.xlabel(f'{display_col} Status')
    plt.ylabel('Subscription Rate (%)')
    plt.ylim(0, success['y'].max() + 5)
    if save:
        file_name = f"plots/{col}_success_rate.png"
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.show()


    return success

cols_v = ['marital', 'default', 'housing', 'loan', 'contact',
        'day_of_week','campaign_frequency','pdays_group','previous',
          'emp_var_rate_cat','cons_price_cat','cons_conf_cat',
          'euribor_cat','nr_employed_cat']
for c in cols_v:
    plot_success_rate_v(df, c, save=False)

