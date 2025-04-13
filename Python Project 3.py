import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# folder for saving graphs
output_folder = 'eda_outputs'
try:
    os.makedirs(output_folder, exist_ok=True)
except OSError as e:
    print(f"Couldn't create folder {output_folder}: {e}")
    exit()


sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12

# Data Loading 
try:
    data = pd.read_csv("COVID-19_Outcomes_by_Vaccination_Status_-_Historical.csv")
except FileNotFoundError:
    print("Oops, can't find the dataset! Check the file path.")
    exit()

# Data before cleaning 
print("Here's the data before cleaning:")
print(data.info())

# Turn 'Week End' into dates 
data['Week End'] = pd.to_datetime(data['Week End'], format='%m/%d/%Y')

# Fill missing numbers with 0 to keep plots clean
columns_to_fill = ['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate',
                   'Crude Vaccinated Ratio', 'Crude Boosted Ratio',
                   'Age-Adjusted Unvaccinated Rate', 'Age-Adjusted Vaccinated Rate',
                   'Age-Adjusted Boosted Rate', 'Age-Adjusted Vaccinated Ratio',
                   'Age-Adjusted Boosted Ratio', 'Population Unvaccinated',
                   'Population Vaccinated', 'Population Boosted',
                   'Outcome Unvaccinated', 'Outcome Vaccinated', 'Outcome Boosted']
for col in columns_to_fill:
    data[col] = data[col].fillna(0)

# Fix age group names (80-200 is weird)
data['Age Group'] = data['Age Group'].replace('80-200', '80+')

#Checking for duplicate rows
print("\nDuplicate Rows:", data.duplicated().sum())
data = data.drop_duplicates()

# See the cleaned data
print("\nCleaned Data:")
print(data.info())

# Save the cleaned data
cleaned_path = os.path.abspath(os.path.join(output_folder, 'cleaned_covid_data.csv'))
data.to_csv(cleaned_path, index=False)
print("\nSaved cleaned data to:", cleaned_path)

# Objective 1: Show how vaccines affect cases, hospitalizations, deaths
all_age_data = data[data['Age Group'] == 'All']

# Make plots for each outcome
fig, axes = plt.subplots(3, 1, figsize=(12, 12))
outcomes = ['Cases', 'Hospitalizations', 'Deaths']
colors = ['salmon', 'skyblue', 'lightgreen']
labels = ['Unvaccinated', 'Vaccinated', 'Boosted']

for i in range(3):
    outcome_data = all_age_data[all_age_data['Outcome'] == outcomes[i]]
    dates = outcome_data['Week End']
    unvac_rates = outcome_data['Unvaccinated Rate']
    vac_rates = outcome_data['Vaccinated Rate']
    boost_rates = outcome_data['Boosted Rate']
    axes[i].stackplot(dates, unvac_rates, vac_rates, boost_rates,
                      labels=labels, colors=colors, alpha=0.8)
    axes[i].set_title(f'{outcomes[i]} Rates Over Time')
    axes[i].set_ylabel('Rate per 100,000')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle='--', alpha=0.7)

axes[2].set_xlabel('Date')
plt.xticks(rotation=45)
plt.suptitle('Vaccination Trends')
plt.tight_layout()
save_path = os.path.abspath(os.path.join(output_folder, 'vaccination_effectiveness.png'))
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()

# Objective 2: Check death rates by age in 2023
data_2023 = data[data['Week End'].dt.year == 2023]
death_data = data_2023[data_2023['Outcome'] == 'Deaths']

age_groups = sorted(death_data['Age Group'].unique())
n_groups = len(age_groups)

unvac_rates = []
vac_rates = []
boost_rates = []
for age in age_groups:
    age_data = death_data[death_data['Age Group'] == age]
    unvac_rates.append(age_data['Unvaccinated Rate'].mean())
    vac_rates.append(age_data['Vaccinated Rate'].mean())
    boost_rates.append(age_data['Boosted Rate'].mean())

# Bar plot for ages
plt.figure(figsize=(14, 8))
bar_width = 0.25
index = np.arange(n_groups)
plt.bar(index, unvac_rates, bar_width, label='Unvaccinated', color='salmon')
plt.bar(index + bar_width, vac_rates, bar_width, label='Vaccinated', color='skyblue')
plt.bar(index + 2 * bar_width, boost_rates, bar_width, label='Boosted', color='lightgreen')
plt.title('Death Rates by Age in 2023')
plt.xlabel('Age Group')
plt.ylabel('Death Rate per 100,000')
plt.xticks(index + bar_width, age_groups, rotation=45)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
save_path = os.path.abspath(os.path.join(output_folder, 'age_group_risk.png'))
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()

# Horizontal bar plot
plt.figure(figsize=(12, 8))
y_index = np.arange(n_groups)
plt.barh(y_index - bar_width, unvac_rates, bar_width, label='Unvaccinated', color='salmon')
plt.barh(y_index, vac_rates, bar_width, label='Vaccinated', color='skyblue')
plt.barh(y_index + bar_width, boost_rates, bar_width, label='Boosted', color='lightgreen')
plt.title('Death Rates by Age in 2023 (Sideways)')
plt.ylabel('Age Group')
plt.xlabel('Death Rate per 100,000')
plt.yticks(y_index, age_groups)
plt.legend()
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
save_path = os.path.abspath(os.path.join(output_folder, 'age_group_risk_horizontal.png'))
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()

# Objective 3: how rates spread
outcomes = ['Cases', 'Hospitalizations', 'Deaths']
status_list = ['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate']
status_names = ['Unvaccinated', 'Vaccinated', 'Boosted']
box_data = []
box_labels = []
box_status = []
for outcome in outcomes:
    for status in status_list:
        rates = data[data['Outcome'] == outcome][status]
        box_data.extend(rates)
        box_labels.extend([outcome] * len(rates))
        box_status.extend([status_names[status_list.index(status)]] * len(rates))
box_df = pd.DataFrame({'Outcome': box_labels, 'Rate': box_data, 'Status': box_status})

# Box plot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Outcome', y='Rate', hue='Status', data=box_df)
plt.title('Rate Spread by Outcome')
plt.xlabel('Outcome')
plt.ylabel('Rate per 100,000 (Log Scale)')
plt.yscale('log')
plt.legend(title='Vaccination Status')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
save_path = os.path.abspath(os.path.join(output_folder, 'outcome_distribution.png'))
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()

# Histogram with curve
fig, axes = plt.subplots(3, 1, figsize=(12, 12))
for i in range(3):
    outcome_data = box_df[box_df['Outcome'] == outcomes[i]]
    sns.histplot(data=outcome_data, x='Rate', hue='Status', kde=True, ax=axes[i],
                 palette=['salmon', 'skyblue', 'lightgreen'], alpha=0.6, bins=30)
    axes[i].set_title(f'{outcomes[i]} Rates')
    axes[i].set_xlabel('Rate per 100,000')
    axes[i].set_ylabel('Count')
    axes[i].set_xscale('log')
    axes[i].legend(title='Vaccination Status')
    axes[i].grid(True, linestyle='--', alpha=0.7)
axes[2].set_xlabel('Rate per 100,000 (Log Scale)')
plt.suptitle('Rate Spread with Curve')
plt.tight_layout()
save_path = os.path.abspath(os.path.join(output_folder, 'outcome_distribution_hist.png'))
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()

# Objective 4: Check missing data
data_missing = pd.read_csv("COVID-19_Outcomes_by_Vaccination_Status_-_Historical.csv")
missing_percent = []
for col in data_missing.columns:
    missing_count = data_missing[col].isnull().sum()
    total_rows = len(data_missing)
    percent = (missing_count / total_rows) * 100
    missing_percent.append(percent)

# Bar plot
plt.figure(figsize=(12, 6))
plt.bar(data_missing.columns, missing_percent, color='teal', edgecolor='black')
plt.title('Missing Data Check')
plt.xlabel('Columns')
plt.ylabel('Missing (%)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(missing_percent):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')
plt.tight_layout()
save_path = os.path.abspath(os.path.join(output_folder, 'missing_data.png'))
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()

# Objective 5: Look at boosters after Sep 2021
booster_data = data[data['Week End'] >= '2021-09-01']
death_booster = booster_data[(booster_data['Outcome'] == 'Deaths') & (booster_data['Age Group'] == 'All')]

# Scatter plot
plt.figure(figsize=(12, 8))
dates_numeric = (death_booster['Week End'] - death_booster['Week End'].min()).dt.days
plt.scatter(death_booster['Vaccinated Rate'], death_booster['Boosted Rate'],
            c=dates_numeric, cmap='plasma', s=100, alpha=0.6)
plt.colorbar(label='Days Since Sep 2021')
plt.title('Vaccinated vs. Boosted Deaths')
plt.xlabel('Vaccinated Death Rate per 100,000')
plt.ylabel('Boosted Death Rate per 100,000')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
save_path = os.path.abspath(os.path.join(output_folder, 'booster_impact.png'))
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()

# Objective 6: to show how rates connect
all_age_data = data[data['Age Group'] == 'All']
corr_data = all_age_data[['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Rate Connections')
plt.tight_layout()
save_path = os.path.abspath(os.path.join(output_folder, 'correlation_heatmap.png'))
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()

# Pair plot for patterns
pair_data = all_age_data[['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate', 'Outcome']]
pair_plot = sns.pairplot(pair_data, hue='Outcome', palette='husl', height=2.5)
pair_plot.figure.suptitle('Rate Patterns', y=1.02)
save_path = os.path.abspath(os.path.join(output_folder, 'correlation_pairplot.png'))
pair_plot.figure.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Pie charts for outcome shares
outcome_counts = {'Cases': [0, 0, 0], 'Hospitalizations': [0, 0, 0], 'Deaths': [0, 0, 0]}
for outcome in outcome_counts:
    outcome_data = data[data['Outcome'] == outcome]
    outcome_counts[outcome][0] = outcome_data['Outcome Unvaccinated'].sum()
    outcome_counts[outcome][1] = outcome_data['Outcome Vaccinated'].sum()
    outcome_counts[outcome][2] = outcome_data['Outcome Boosted'].sum()

plt.figure(figsize=(12, 8))
for i, outcome in enumerate(['Cases', 'Hospitalizations', 'Deaths']):
    plt.subplot(1, 3, i + 1)
    plt.pie(outcome_counts[outcome], labels=['Unvaccinated', 'Vaccinated', 'Boosted'],
            autopct='%1.1f%%', colors=['salmon', 'skyblue', 'lightgreen'])
    plt.title(f'{outcome} Share')
plt.suptitle('Outcome Shares')
plt.tight_layout()
save_path = os.path.abspath(os.path.join(output_folder, 'outcome_pie.png'))
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()

# Objective 7: Check rates by month
data['Month'] = data['Week End'].dt.month

months = range(1, 13)
outcomes = ['Cases', 'Hospitalizations', 'Deaths']
seasonal_rates = {'Cases': [], 'Hospitalizations': [], 'Deaths': []}
for month in months:
    month_data = data[(data['Month'] == month) & (data['Age Group'] == 'All')]
    for outcome in outcomes:
        outcome_data = month_data[month_data['Outcome'] == outcome]
        avg_rate = outcome_data['Unvaccinated Rate'].mean()
        if np.isnan(avg_rate):
            avg_rate = 0
        seasonal_rates[outcome].append(avg_rate)

# Stacked bar plot
fig = plt.figure(figsize=(12, 8))
plt.bar(months, seasonal_rates['Cases'], label='Cases', color='salmon')
plt.bar(months, seasonal_rates['Hospitalizations'], bottom=seasonal_rates['Cases'],
        label='Hospitalizations', color='skyblue')
bottom = np.array(seasonal_rates['Cases']) + np.array(seasonal_rates['Hospitalizations'])
plt.bar(months, seasonal_rates['Deaths'], bottom=bottom, label='Deaths', color='lightgreen')
plt.title('Unvaccinated Rates by Month')
plt.xlabel('Month')
plt.ylabel('Average Rate per 100,000')
plt.xticks(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
save_path = os.path.abspath(os.path.join(output_folder, 'seasonal_patterns.png'))
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()

# Objective 8: Look at coverage vs. deaths
total_pop = data['Population Unvaccinated'] + data['Population Vaccinated'] + data['Population Boosted']
data['Vaccinated Coverage'] = data['Population Vaccinated'] / total_pop
data['Boosted Coverage'] = data['Population Boosted'] / total_pop

all_age_deaths = data[(data['Age Group'] == 'All') & (data['Outcome'] == 'Deaths')]

# Histogram
plt.figure(figsize=(12, 8))
plt.hist(all_age_deaths['Vaccinated Coverage'], bins=20, alpha=0.5, label='Vaccinated Coverage', color='skyblue')
plt.hist(all_age_deaths['Boosted Coverage'], bins=20, alpha=0.5, label='Boosted Coverage', color='lightgreen')
plt.title('Vaccination Coverage Spread')
plt.xlabel('Coverage Proportion')
plt.ylabel('Count')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
save_path = os.path.abspath(os.path.join(output_folder, 'coverage_distribution.png'))
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()

# Scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(all_age_deaths['Vaccinated Coverage'], all_age_deaths['Unvaccinated Rate'],
            label='Vaccinated Coverage', color='skyblue', alpha=0.6)
plt.scatter(all_age_deaths['Boosted Coverage'], all_age_deaths['Unvaccinated Rate'],
            label='Boosted Coverage', color='lightgreen', alpha=0.6)
plt.title('Coverage vs. Unvaccinated Deaths')
plt.xlabel('Coverage Proportion')
plt.ylabel('Unvaccinated Death Rate per 100,000')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
save_path = os.path.abspath(os.path.join(output_folder, 'coverage_vs_deaths.png'))
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()
