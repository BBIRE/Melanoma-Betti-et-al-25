# WGCNA Pipeline for Gene Expression Data
# Description: This script performs WGCNA network analysis on RNA-Seq count data
# Reference: Langfelder, P. & Horvath, S. (2008). WGCNA: an R package for weighted correlation network analysis. BMC Bioinformatics.

# ====================
# Load Required Libraries
# ====================
suppressPackageStartupMessages({
  library(WGCNA)
  library(DESeq2)
  library(dplyr)
  library(tidyverse)
  library(gridExtra)
  library(ggplot2)
  library(factoextra)
  library(cluster)
  library(pheatmap)
  library(RColorBrewer)
})

# Allow multi-threading
enableWGCNAThreads()

# ====================
# Parameters
# ====================
soft_power_values <- c(1:10, seq(12, 20, 2))
soft_power_chosen <- 12
output_dir <- "results/WGCNA"
input_data_file <- "data/DataTraits.RData" # <-- Replace with relative path if needed

# ====================
# Load Data
# ====================
message("Loading data...")
load(input_data_file)  # Expected to load 'data' and 'traits'

# ====================
# Step 1: Soft-Thresholding Power Selection
# ====================
message("Calculating soft-thresholding powers...")
sft <- pickSoftThreshold(data,
                         dataIsExpr = TRUE,
                         powerVector = soft_power_values,
                         networkType = 'signed',
                         verbose = 5)

sft_data <- sft$fitIndices

# Plot Scale-Free Topology Fit
scale_free_plot <- ggplot(sft_data, aes(Power, SFT.R.sq, label = Power)) +
  geom_point() +
  geom_text(nudge_y = 0.1) +
  geom_hline(yintercept = 0.8, color = 'red') +
  labs(x = 'Power', y = 'Scale-free topology model fit, signed R^2') +
  theme_classic()

# Plot Mean Connectivity
connectivity_plot <- ggplot(sft_data, aes(Power, mean.k., label = Power)) +
  geom_point() +
  geom_text(nudge_y = 0.1) +
  labs(x = 'Power', y = 'Mean connectivity') +
  theme_classic()

# Display plots
grid.arrange(scale_free_plot, connectivity_plot, nrow = 2)

# ====================
# Step 2: Network Construction and Module Detection
# ====================
message("Constructing network and detecting modules...")
# Temporarily save original cor function
temp_cor <- cor
cor <- WGCNA::cor

net <- blockwiseModules(data,
                        power = soft_power_chosen,
                        TOMType = 'signed',
                        maxBlockSize = 15000,
                        minModuleSize = 30,
                        mergeCutHeight = 0.25,
                        numericLabels = FALSE,
                        pamRespectsDendro = FALSE,
                        saveTOMs = FALSE,
                        randomSeed = 1234,
                        verbose = 3)

cor <- temp_cor  # Restore default cor

# ====================
# Step 3: Module Visualization
# ====================
moduleColors <- net$colors
MEs <- orderMEs(moduleEigengenes(data, moduleColors)$eigengenes)

# Barplot: Module Sizes
module_summary <- data.frame(table(moduleColors))
module_summary <- module_summary[order(module_summary$Freq, decreasing = TRUE), ]

par(mar = c(10, 5, 3, 1))
bar_labels <- paste0(module_summary$moduleColors, " (", module_summary$Freq, ")")
barplot(module_summary$Freq,
        col = as.character(module_summary$moduleColors),
        ylab = "# Genes",
        names.arg = bar_labels,
        las = 2, cex.names = 1.1)

# Dendrogram + Module Colors
geneTree <- net$dendrograms[[1]]
plotDendroAndColors(geneTree,
                    moduleColors[net$blockGenes[[1]]],
                    "Module Colors",
                    dendroLabels = FALSE,
                    hang = 0.03,
                    addGuide = TRUE,
                    guideHang = 0.05)

# ====================
# Step 4: Module-Trait Associations
# ====================
message("Computing module-trait correlations...")
status_map <- c("concordante" = 0, "discordante" = 1)
traits$GruppoNum <- status_map[traits$FENOTIPO]

nSamples <- nrow(data)

module_trait_corr <- cor(MEs, traits$GruppoNum, use = 'p')
module_trait_corr_pval <- corPvalueStudent(module_trait_corr, nSamples)

# Heatmap of correlations
module_trait_df <- data.frame("Correlation" = module_trait_corr,
                              "P.value" = module_trait_corr_pval)

pheatmap(module_trait_df["Correlation"],
         color = colorRampPalette(c("blue", "white", "red"))(50),
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         display_numbers = matrix(round(module_trait_df$P.value, 2)),
         fontsize_number = 10,
         number_color = 'black',
         legend = TRUE,
         main = "Module-Trait Relationship")

# ====================
# Step 5: Module Gene Extraction
# ====================
module_gene_mapping <- as.data.frame(moduleColors)

# Example: Extract genes from selected modules
darkmagenta_genes <- rownames(module_gene_mapping)[module_gene_mapping$moduleColors == 'darkmagenta']
darkgrey_genes <- rownames(module_gene_mapping)[module_gene_mapping$moduleColors == 'darkgrey']
midnightblue_genes <- rownames(module_gene_mapping)[module_gene_mapping$moduleColors == 'midnightblue']

# ====================
# Step 6: Module Membership & Gene Significance
# ====================
ModuleMembership <- cor(data, MEs, use = 'p')
ModuleMembership_pvalue <- corPvalueStudent(ModuleMembership, nSamples)

GeneSignificance <- cor(data, traits$GruppoNum, use = 'p')
GeneSignificance_pvalue <- corPvalueStudent(GeneSignificance, nSamples)

# ====================
# Step 7: Save Results
# ====================
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
save_file <- file.path(output_dir, "WGCNA_results.RData")

save(net, MEs, module_summary,
     module_trait_df, module_gene_mapping,
     darkmagenta_genes, darkgrey_genes, midnightblue_genes,
     ModuleMembership, ModuleMembership_pvalue,
     GeneSignificance, GeneSignificance_pvalue,
     file = save_file)

message("Analysis complete. Results saved to: ", save_file)
