import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

# --------------------------
# INPUT ASSUMPTIONS:
# --------------------------
# - `df`: gene expression dataframe (rows: samples, cols: genes + 'group')
# - `de`: DESeq2-like table with 'Unnamed: 0' (gene), 'log2FoldChange'
# - 'group' column in `df` includes: 'Dormants', 'Proliferants', and 'Unknown'
# - `custom_palette`: color dictionary mapping groups to colors
# --------------------------

# --- Select DE genes highly different between groups ---
up_dorm = list(de[de['log2FoldChange'] < -2]['Unnamed: 0'])   # Overexpressed in Dormants
down_dorm = list(de[de['log2FoldChange'] > 2]['Unnamed: 0'])  # Overexpressed in Proliferants
sig_biomarkers = up_dorm + down_dorm

# --- Step 1: PCA on DE genes to define structure among Dormants and Proliferants ---
de_genes = list(de['Unnamed: 0'])
df_pca_input = df[df['group'].isin(['Dormants', 'Proliferants'])][de_genes]
group_labels = df.loc[df_pca_input.index, 'group']

# Standardize gene expression before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pca_input)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Get loadings to identify most discriminative genes for PC1/PC2
loadings = pd.DataFrame(pca.components_.T, index=de_genes, columns=['PC1', 'PC2'])
loadings['abs_PC1'] = loadings['PC1'].abs()
loadings['abs_PC2'] = loadings['PC2'].abs()

# --- Step 2: Find optimal gene subset for classification using AUC ---
best_auc = 0
best_n = 0
best_genes = []
auc_results = []

for n in range(2, 30):
    top_pc1 = loadings.sort_values('abs_PC1', ascending=False).head(n)
    top_pc2 = loadings.sort_values('abs_PC2', ascending=False).head(n)
    selected_genes = pd.Index(top_pc1.index.tolist() + top_pc2.index.tolist()).unique().tolist()

    df_grouped_iter = df[df['group'].isin(['Dormants', 'Proliferants', 'Unknown'])].groupby('group')[selected_genes].aggregate(list)

    genes_up = [g for g in selected_genes if g in de[de['log2FoldChange'] < 0]['Unnamed: 0'].values]
    genes_down = [g for g in selected_genes if g in de[de['log2FoldChange'] > 0]['Unnamed: 0'].values]

    sig_iter = compute_signature_scores(df_grouped_iter, genes_up, genes_down, df)
    sig_iter['Group'] = df.loc[sig_iter.index, 'group']

    binary_df = sig_iter[sig_iter['Group'].isin(['Dormants', 'Proliferants'])]
    y_true = (binary_df['Group'] == 'Dormants').astype(int)
    y_score = binary_df['SignatureScore']
    auc = roc_auc_score(y_true, y_score)
    auc_results.append((n, auc))

    if auc > best_auc:
        best_auc = auc
        best_n = n
        best_genes = selected_genes

# --- Step 3: Final Signature Computation using best PCA-derived genes ---

def group_expression_lists(df, genes, group_labels):
    return df[df['group'].isin(group_labels)].groupby('group')[genes].aggregate(list)

def compute_signature_scores(df_grouped, genes_up, genes_down, original_df):
    def compute_median_expression(row, genes):
        return [np.mean([row[col][i] for col in genes]) for i in range(len(row.iloc[0]))]

    df_grouped = df_grouped.copy()
    df_grouped['Up'] = df_grouped.apply(lambda row: compute_median_expression(row, genes_up), axis=1)
    df_grouped['Down'] = df_grouped.apply(lambda row: compute_median_expression(row, genes_down), axis=1)

    def normalized_score(up_list, down_list):
        return [round((u - d) / (abs(u) + abs(d) + 1e-5), 2) for u, d in zip(up_list, down_list)]

    scores = {}
    for group in df_grouped.index:
        score_list = normalized_score(df_grouped.loc[group]['Up'], df_grouped.loc[group]['Down'])
        sample_indices = original_df[original_df['group'] == group].index
        scores.update(dict(zip(sample_indices, score_list)))

    return pd.DataFrame.from_dict(scores, orient='index', columns=['SignatureScore'])

df_grouped = group_expression_lists(df, best_genes, ['Dormants', 'Proliferants', 'Unknown'])
genes_up = [g for g in best_genes if g in de[de['log2FoldChange'] < 0]['Unnamed: 0'].values]
genes_down = [g for g in best_genes if g in de[de['log2FoldChange'] > 0]['Unnamed: 0'].values]
sig_df = compute_signature_scores(df_grouped, genes_up, genes_down, df)
sig_df['Group'] = df.loc[sig_df.index, 'group']

# Save results
sig_df.to_csv('signature_values.csv')
print("Signature genes from PCA:", best_genes)

# --- Step 4: Final Visualization (2x2 Panel) ---

# Prepare PCA plot data again (optional)
pca_df = df[df['group'].isin(['Dormants', 'Proliferants'])][de_genes]
pca_labels = df[df['group'].isin(['Dormants', 'Proliferants'])]['group']
pca_model = PCA(n_components=2)
pcs = pca_model.fit_transform(pca_df)
pca_plot_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
pca_plot_df['Group'] = pca_labels.values

# Extract AUC curve
x_vals, y_vals = zip(*auc_results)

fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=300)

# Panel 1: PCA Plot
sns.scatterplot(
    data=pca_plot_df,
    x='PC1',
    y='PC2',
    hue='Group',
    palette=custom_palette,
    ax=axes[0, 0],
    alpha=0.8,
    edgecolor='black'
)
axes[0, 0].set_title('PCA of DE Genes')
axes[0, 0].legend(title='Group')
axes[0, 0].grid(False)

# Panel 2: AUC vs. N genes
axes[0, 1].plot(x_vals, y_vals, marker='o', color='black')
axes[0, 1].set_title('AUC vs. Top PCA Genes')
axes[0, 1].set_xlabel('Top genes from PC1 and PC2')
axes[0, 1].set_ylabel('AUC')
axes[0, 1].axvline(best_n, linestyle='--', color='red', label=f'Best: {best_n} genes')
axes[0, 1].legend()
axes[0, 1].grid(False)

# Panel 3: Violin Plot
sns.violinplot(
    data=sig_df,
    x='Group',
    y='SignatureScore',
    order=['Proliferants', 'Dormants', 'Unknown'],
    inner='box',
    palette=custom_palette,
    ax=axes[1, 0]
)
for violin in axes[1, 0].collections:
    violin.set_alpha(0.8)

axes[1, 0].axhline(0, linestyle='--', color='gray', linewidth=1)
axes[1, 0].set_title('Dormancy/Proliferation Signature')
axes[1, 0].set_xlabel('Group')
axes[1, 0].set_ylabel('Signature Score')
axes[1, 0].grid(False)

# Panel 4: ROC Curve
df_bin = sig_df[sig_df['Group'].isin(['Dormants', 'Proliferants'])]
y_true = (df_bin['Group'] == 'Dormants').astype(int)
y_score = df_bin['SignatureScore']
auc = roc_auc_score(y_true, y_score)
fpr, tpr, _ = roc_curve(y_true, y_score)

axes[1, 1].plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='black')
axes[1, 1].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[1, 1].set_title('ROC Curve')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].legend()
axes[1, 1].grid(False)

# Final save
plt.tight_layout()
plt.savefig("panel_with_pca.png", dpi=300, bbox_inches='tight')
plt.show()
