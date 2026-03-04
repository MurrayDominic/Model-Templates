# Model Templates

A collection of machine learning template notebooks with synthetic test data, ready to run out-of-the-box. Each template follows a consistent structure: data loading, preprocessing, hyperparameter tuning via GridSearchCV, model evaluation, and diagnostic visualizations.

## Getting Started

### Prerequisites
- Python 3.11+
- Conda environment (included in `.conda/`)

### Required Packages
```bash
# Core (included in environment)
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels

# Boosting models
pip install xgboost lightgbm catboost

# Neural networks (optional, for PyTorch section)
pip install torch
```

### Usage
1. Open any notebook in Jupyter / VS Code
2. Each notebook includes synthetic test data — just **Run All**
3. To use your own data, uncomment the `pd.read_csv("data.csv")` line and update feature names

## Notebooks

| Notebook | Type | Description |
|----------|------|-------------|
| `EDA.ipynb` | Exploration | Data profiling, distributions, correlations, missing values |
| `GLM Template.ipynb` | Supervised | Linear, Poisson, Gamma, Tweedie, Logistic regression |
| `Random Forest Template.ipynb` | Supervised | Bagged decision trees with OOB error and feature importance |
| `Logistic Regression Template.ipynb` | Classification | Detailed logistic regression with ROC/PR curves, threshold optimization |
| `SGD Classifier & Regressor Template.ipynb` | Supervised | Online learning with stochastic gradient descent, convergence curves |
| `XGBoost Template.ipynb` | Supervised | Gradient boosted trees with early stopping and learning curves |
| `LightGBM Template.ipynb` | Supervised | Leaf-wise gradient boosting, fast training, split/gain importance |
| `CatBoost Template.ipynb` | Supervised | Ordered boosting with native categorical feature support |
| `KMeans Template.ipynb` | Unsupervised | Partition clustering with elbow method and silhouette analysis |
| `HDBSCAN Template.ipynb` | Unsupervised | Density-based clustering with automatic cluster detection and noise handling |
| `Neural Network Template.ipynb` | Deep Learning | MLP via sklearn + PyTorch alternative section |

## Template Structure

### Supervised Templates
All supervised templates follow a consistent pipeline:

1. **Documentation** — method overview, when to use, assumptions, references
2. **Imports**
3. **Synthetic test data** — 8000 samples, mixed numeric + categorical features
4. **Train/test split** — with stratification fallback for continuous targets
5. **Preprocessing** — `ColumnTransformer` with imputation, scaling, and one-hot encoding
6. **Parameter grids** — model-specific hyperparameters for `GridSearchCV`
7. **Training** — `train_models()` function with optional feature selection
8. **Results summary** — comparison DataFrame across model variants
9. **Best model selection** — by R² (regression) or ROC-AUC (classification)
10. **Feature importance / coefficients**
11. **Diagnostics** — residual plots (regression), confusion matrix + ROC/PR curves (classification)
12. **Profile plots** — actual vs predicted by feature

### Unsupervised Templates
Clustering templates follow a different structure:

1. **Documentation** — method overview, when to use, assumptions, references
2. **Imports**
3. **Synthetic data** — with natural cluster structure
4. **Preprocessing** — same `ColumnTransformer` pipeline
5. **Parameter exploration** — elbow/silhouette (KMeans) or min_cluster_size sweep (HDBSCAN)
6. **Cluster evaluation** — silhouette, Calinski-Harabasz, Davies-Bouldin scores
7. **Cluster visualization** — PCA 2D projection
8. **Cluster profiling** — feature means per cluster, heatmap

## Notes

- All supervised templates support both **regression and classification** (switch target in the synthetic data cell)
- Logistic Regression is classification-only with detailed threshold analysis
- Clustering templates have no target variable or train/test split
- Boosting templates (XGBoost, LightGBM, CatBoost) include **early stopping demos** separate from GridSearchCV
- The Neural Network template includes both sklearn MLP and a **PyTorch alternative** section
