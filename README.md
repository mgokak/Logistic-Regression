# Logistic Regression – Binary, Multiclass & Model Evaluation

## Overview

This repository contains a Jupyter Notebook that demonstrates **Logistic Regression** applied to different real-world classification scenarios. The focus of this notebook is on **practical implementation**, including multiclass classification, handling imbalanced datasets, hyperparameter tuning, and model evaluation.

---

## Table of Contents

1. Installation  
2. Project Structure  
3. Logistic Regression for Multiclass Classification  
4. Logistic Regression for Imbalanced Data  
5. Hyperparameter Tuning  
6. Model Evaluation Metrics  
7. ROC Curve & ROC–AUC 

---

## Installation

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Project Structure

- `LogisticRegression_BinaryMultiClassification.ipynb` – Notebook covering logistic regression with tuning and evaluation techniques.

---

## Logistic Regression for Multiclass Classification

This section applies logistic regression to **multiclass classification problems**, where the target variable has more than two classes.

Key points:
- Logistic regression can handle multiclass problems using `scikit-learn`
- Predictions are made directly using class labels
- Performance is evaluated using classification metrics

Basic commands used:
```python
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
```

---

## Logistic Regression for Imbalanced Dataset

This section focuses on applying logistic regression when the dataset is **imbalanced**.

Key points:
- Imbalanced data can lead to biased predictions
- Accuracy alone may not be sufficient
- Additional evaluation metrics are required

Common commands:
```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## Hyperparameter Tuning

To improve model performance, **hyperparameter tuning** is performed using cross-validation.

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
```

### Randomized Search
```python
from sklearn.model_selection import RandomizedSearchCV

random = RandomizedSearchCV(LogisticRegression(), param_distributions, cv=5)
random.fit(X_train, y_train)
```

---

## Model Evaluation Metrics

Model performance is evaluated using:

### Accuracy
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

### Confusion Matrix
```python
confusion_matrix(y_test, y_pred)
```

### Classification Report
```python
classification_report(y_test, y_pred)
```

---

## ROC Curve & ROC–AUC

The notebook also evaluates model performance using the **ROC curve** and **ROC–AUC score**.

Key points:
- ROC curve shows trade-off between true positive rate and false positive rate
- ROC–AUC summarizes model performance across thresholds

Common commands:
```python
from sklearn.metrics import roc_curve, roc_auc_score
```

---


## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  
DePaul University
