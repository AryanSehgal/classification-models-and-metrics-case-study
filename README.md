# Classification Models & Metrics in Machine Learning

A hands-on collection of Jupyter notebooks exploring **classification algorithms** (Logistic Regression and K-Nearest Neighbors) and the **evaluation metrics** used to judge them. The notebooks combine intuition, math, and scikit-learn code, and use real-style datasets (telecom churn, wholesale customers, diabetes) to show *why* the choice of model and metric matters, especially on imbalanced data.

---

## Notebooks

| Notebook | What it covers |
| --- | --- |
| [`Classification_Metrics.ipynb`](Classification_Metrics.ipynb) | Why accuracy fails on imbalanced data; precision, recall, F1, ROC and AUC; confusion matrices; threshold tuning; a metric cheat sheet |
| [`Logistic_Regression.ipynb`](Logistic_Regression.ipynb) | Intuition behind logistic regression, the sigmoid function, log loss, why MSE is not used, regularization, and hyperparameter tuning on a churn dataset |
| [`KNN.ipynb`](KNN.ipynb) | K-Nearest Neighbors end to end: distance metrics, weighted KNN, bias-variance, outliers, KNN imputation, and handling imbalanced data (class weights, under/oversampling, SMOTE) |

### Suggested reading order

1. **Classification_Metrics**: learn how to evaluate a classifier.
2. **Logistic_Regression**: build and tune a first classifier.
3. **KNN**: a second, non-parametric classifier, plus imbalanced-data techniques that use both models and the metrics from step 1.

---

## Topics in detail

### 1. Classification Metrics
- The problem with accuracy on imbalanced datasets
- **Precision** and **Recall**, with the fishing-net analogy
- **F1 score** and why it uses the harmonic mean (not arithmetic or geometric)
- Bank-fraud case study comparing accuracy, precision, recall, F1 and FPR
- `classification_report` (macro vs. weighted averages, support)
- **ROC curve** and **AUC**: how thresholds shape the curve, and AUC properties and limitations
- Precision-Recall curves for severe imbalance
- Metric cheat sheet, and performance metrics vs. loss functions
- Evaluating a model on the Pima diabetes dataset: confusion matrix, sensitivity, specificity, threshold adjustment, cross-validated AUC

### 2. Logistic Regression
- From linear regression to a separating hyperplane
- The **sigmoid** function
- **Log loss**, and why MSE + sigmoid gives a non-convex loss
- Train / validation / test split and feature scaling
- `predict` vs. `predict_proba` and the default 0.5 threshold
- **Regularization** (`C = 1/λ`) and hyperparameter tuning with `make_pipeline`

### 3. K-Nearest Neighbors
- Geometric intuition and the KNN algorithm
- Distance metrics: **Euclidean** and **Manhattan**
- KNN with scikit-learn (on PCA-reduced and on original data)
- **Weighted KNN** (inverse-distance weights)
- Assumptions of KNN, and train/test time complexity
- **Bias-variance trade-off** and choosing `K`
- Multi-class classification and the effect of outliers
- **KNN-based imputation** for missing values (numeric and categorical)
- **Imbalanced data**: how it affects KNN and Logistic Regression, and how to handle it
  - Weighted loss (`class_weight`)
  - Random undersampling and oversampling
  - **SMOTE** (Synthetic Minority Oversampling Technique)
  - Churn prediction case study comparing these approaches

---

## Quick metric cheat sheet

| Situation | Metric to prefer |
| --- | --- |
| You need class probabilities | Log loss |
| Classes are balanced | Accuracy |
| Imbalanced, and false positives are costly | Precision |
| Imbalanced, and false negatives are costly | Recall |
| Imbalanced, and you need a balance of both | F1 score |
| You care about both classes across all thresholds | ROC-AUC |
| Severe imbalance | PR-AUC |

---

## Datasets

The notebooks download their data from Google Drive using `gdown`. The first cell(s) of each notebook fetch what is needed.

| Notebook | Dataset(s) | Used for |
| --- | --- | --- |
| Classification_Metrics | `pred_data.csv`, `pima-indians-diabetes (1).csv` | Metric walkthroughs and the diabetes classifier |
| Logistic_Regression | `churn_logistic.csv` | Telecom customer churn prediction |
| KNN | `binary_class.csv`, `knn_imputation.csv`, `Churn.csv` | Wholesale-customer classification, KNN imputation, imbalanced churn prediction |

> If `gdown` fails, the notebooks include commented-out `wget` alternatives, or you can pin an older version with `pip install gdown==4.6.0`.

---

## Getting started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab (or run the notebooks in Google Colab)

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn gdown jupyter
```

### Run

```bash
jupyter notebook
```

Then open any of the notebooks and run the cells top to bottom.

---

## Libraries used

- **NumPy**, **pandas**: data handling
- **Matplotlib**, **seaborn**: visualization
- **scikit-learn**: models, preprocessing, metrics, imputation, PCA
- **imbalanced-learn**: `RandomUnderSampler`, `RandomOverSampler`, `SMOTE`
- **gdown**: fetching datasets

---

## Repository structure

```
.
├── Classification_Metrics.ipynb
├── Logistic_Regression.ipynb
├── KNN.ipynb
└── README.md
```

---

## Key takeaways

- **Accuracy can be misleading.** A model that always predicts the majority class can score highly while being useless. Check precision, recall, and F1.
- **Pick the metric that matches the cost of errors.** Spam filters favor precision; cancer screening and fraud detection favor recall.
- **AUC is threshold-independent**, but it becomes less reliable under severe imbalance, where PR curves are the better choice.
- **Imbalance affects both models.** Logistic regression's loss is dominated by the majority class, and KNN gets biased when `K` is large relative to the dataset.
- **In the churn case study, class-weighted logistic regression outperformed SMOTE**, so SMOTE is not always the best fix. Always validate with the right metric.

---

## Contributing

Suggestions and improvements are welcome. Feel free to open an issue or submit a pull request.

## License

Add your preferred license here (for example, MIT).
