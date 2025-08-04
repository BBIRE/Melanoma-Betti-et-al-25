import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import chi2_contingency

# --------------------------
# STEP 1: Compute Signature Score
# --------------------------

# Load TPM-normalized expression data
cbio = pd.read_csv('data_mrna_seq_tpm.txt', sep='\t')
cbio = cbio[cbio['Hugo_Symbol'].isin(final_genes)].T
cbio.columns = cbio.loc['Hugo_Symbol']
cbio = cbio.drop(['Hugo_Symbol', 'Entrez_Gene_Id'])

# Reorder to match final gene list
cbio = cbio[final_genes]

# Compute normalized dormancy score based on up/down gene sets
v1 = pd.DataFrame(cbio[gene_list_1].mean(axis=1), columns=['value'])  # Dormancy-like genes
v2 = pd.DataFrame(cbio[gene_list_2].mean(axis=1), columns=['value'])  # Proliferation-like genes
final = (v1 - v2) / (abs(v1) + abs(v2))  # Normalized signature score

# --------------------------
# STEP 2: Load and Filter Clinical Data
# --------------------------

clinical = pd.read_csv('mel_dfci_2019_clinical_data.tsv', sep='\t')
clinical = clinical[clinical['Biopsy Site'] == 'skin']  # Only skin biopsies
clinical['Progress Free Survival (Months)'] = pd.to_numeric(clinical['Progress Free Survival (Months)'], errors='coerce')

# Remove short follow-up LIVING cases to reduce bias
median_OS_indeceased = clinical[clinical['Overall Survival Status'] == '1:DECEASED']['Overall Survival (Months)'].median()
OS_index = clinical[(clinical['Overall Survival Status'] == '0:LIVING') & 
                    (clinical['Overall Survival (Months)'] < median_OS_indeceased)].index
clinical = clinical.drop(OS_index)

# --------------------------
# STEP 3: Merge Expression Signature and Clinical Data
# --------------------------

merged_df = pd.merge(final, clinical, left_index=True, right_on='Sample ID')
merged_df['Group'] = merged_df['value'].apply(lambda x: 'Dormants' if x > 0 else 'Proliferants')
merged_df['Progression Free Status'] = merged_df['Progression Free Status'].replace({
    '1:PROGRESSION': 1,
    '0:CENSORED': 0
})
merged_df = merged_df.dropna(subset=['Progress Free Survival (Months)', 'Progression Free Status'])

# --------------------------
# STEP 4: Kaplan-Meier PFS Curve + Summary Table
# --------------------------

kmf = KaplanMeierFitter()
color_map = {'Proliferants': '#e41a1c', 'Dormants': '#377eb8'}

fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=300, gridspec_kw={'height_ratios': [3, 1]})
ax, ax_table = axes
groups = merged_df['Group'].unique()
summary_data = []

for group in groups:
    ix = merged_df['Group'] == group
    T = merged_df.loc[ix, 'Progress Free Survival (Months)']
    E = merged_df.loc[ix, 'Progression Free Status']
    
    kmf.fit(T, event_observed=E, label=group)
    kmf.plot_survival_function(ax=ax, color=color_map[group])
    summary_data.append([group, len(T), E.sum(), len(T) - E.sum()])

# Log-rank p-value
if len(groups) == 2:
    res = logrank_test(
        merged_df[merged_df['Group'] == groups[0]]['Progress Free Survival (Months)'],
        merged_df[merged_df['Group'] == groups[1]]['Progress Free Survival (Months)'],
        event_observed_A=merged_df[merged_df['Group'] == groups[0]]['Progression Free Status'],
        event_observed_B=merged_df[merged_df['Group'] == groups[1]]['Progression Free Status']
    )
    ax.text(0.5, 0.1, f'p-value = {res.p_value:.3f}', transform=ax.transAxes,
            ha='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

# Style KM curve
ax.set_title('Kaplan-Meier Survival Curve (PFS)')
ax.set_xlabel('Months')
ax.set_ylabel('PFS Probability')
ax.grid(False)

# Table
summary_df = pd.DataFrame(summary_data, columns=['Group', 'Total', 'Events', 'Censored'])
table = ax_table.table(cellText=summary_df.values, colLabels=summary_df.columns,
                       cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)
for key, cell in table.get_celld().items():
    cell.set_linewidth(1)
    cell.set_edgecolor('black')
ax_table.axis('off')

plt.tight_layout()
plt.savefig('Kaplan_Meier_Survival_Curve_PFS_NEW.png', dpi=300, bbox_inches='tight')
plt.show()

# --------------------------
# STEP 5: Kaplan-Meier OS Curve (Same logic, different endpoint)
# --------------------------

merged_df['Overall Survival Status'] = merged_df['Overall Survival Status'].replace({
    '1:DECEASED': 1, '0:LIVING': 0
})
merged_df = merged_df.dropna(subset=['Overall Survival Status', 'Overall Survival (Months)'])

fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=300, gridspec_kw={'height_ratios': [3, 1]})
ax, ax_table = axes
summary_data = []

for group in groups:
    ix = merged_df['Group'] == group
    T = merged_df.loc[ix, 'Overall Survival (Months)']
    E = merged_df.loc[ix, 'Overall Survival Status']
    
    kmf.fit(T, event_observed=E, label=group)
    kmf.plot_survival_function(ax=ax, color=color_map[group])
    summary_data.append([group, len(T), E.sum(), len(T) - E.sum()])

if len(groups) == 2:
    res = logrank_test(
        merged_df[merged_df['Group'] == groups[0]]['Overall Survival (Months)'],
        merged_df[merged_df['Group'] == groups[1]]['Overall Survival (Months)'],
        event_observed_A=merged_df[merged_df['Group'] == groups[0]]['Overall Survival Status'],
        event_observed_B=merged_df[merged_df['Group'] == groups[1]]['Overall Survival Status']
    )
    ax.text(0.5, 0.1, f'p-value = {res.p_value:.3f}', transform=ax.transAxes,
            ha='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

ax.set_title('Kaplan-Meier Survival Curve (OS)')
ax.set_xlabel('Months')
ax.set_ylabel('OS Probability')
ax.grid(False)

summary_df = pd.DataFrame(summary_data, columns=['Group', 'Total', 'Events', 'Censored'])
table = ax_table.table(cellText=summary_df.values, colLabels=summary_df.columns,
                       cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)
for key, cell in table.get_celld().items():
    cell.set_linewidth(1)
    cell.set_edgecolor('black')
ax_table.axis('off')

plt.tight_layout()
plt.savefig('Kaplan_Meier_Survival_Curve_OS_NEW.png', dpi=300, bbox_inches='tight')
plt.show()

# --------------------------
# STEP 6: Association with Radiographic Response
# --------------------------

small = merged_df[['value', 'Group', 'Best Radiographic Response (RECIST 1.1)', 'Cycles On Therapy']].copy()

# Simplify response categories
response_mapping = {
    'Partial Response': 'Response',
    'Complete Response': 'Response',
    'Progressive Disease': 'Progression',
    'Mixed Response': 'Response',
    'Stable Disease': 'Stable'
}
small['Best Radiographic Response (RECIST 1.1)'] = small['Best Radiographic Response (RECIST 1.1)'].replace(response_mapping)
small = small.dropna(subset=['Cycles On Therapy', 'Group'])

# Bar plot: Response by group
crosstab_data = pd.crosstab(small['Best Radiographic Response (RECIST 1.1)'], small['Group'])
crosstab_data.plot(kind='bar', stacked=True, color=color_map)
plt.title('Best Radiographic Response by Group')
plt.xlabel('')
plt.legend(title='Group', bbox_to_anchor=(0.7, 1), loc='upper left')

chi2_stat, p_chi, dof, expected = chi2_contingency(crosstab_data)
plt.text(2, 20, f'Chi2 p = {p_chi:.3f}', ha='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

plt.tight_layout()
plt.show()
