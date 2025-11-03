# Requirements
scanpy = 1.10.1\
pytorch = 2.2.2

# How to use HIDF
We provide four Jupyter tutorials for researchers, please see in tutorial.

# training_tutorial
This Jupyter tutorial includes: training HIDF model, saving cell type deconvolution results, storing the cell-spot mapping matrix, and keeping the trained model parameters for interpretability analysis.

# analysis_tutorial
This Jupyter tutorials demonstrates how to load deconvolution results and compute multiple evaluation metrics, including RMSE, Spearman's correlation, and Moran's I.

# visualize_tutorial
This Jupyter tutorials features multi-level visualization of deconvolution results, enabling the display of cell-type distributions as well as more refined cell-subtype distributions.

# interpretability_tutorial
This Jupyter tutorials covers interpretability analysis for the trained HIDF model, utilizing parameter sensitivity analysis based on cell-type information as its core method.

# Hierarchical_analysis_tutorial
This Jupyter tutorials includes a hierarchical visualization analysis, which integrates hierarchical information from the single-cell reference data with the cell-spot mapping matrix.