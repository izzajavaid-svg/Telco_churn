# Telco Customer Churn Prediction — Logistic Regression

## Overview

This project builds an end-to-end **customer churn prediction model** using Logistic Regression on the IBM Telco Customer Churn dataset.

The goal is to understand the complete supervised learning workflow, starting from data inspection and preprocessing, through baseline modeling and feature engineering, and finally evaluating the model using business-oriented classification metrics and decision-threshold analysis.

The project focuses on understanding **why each step is performed**, rather than only optimizing model performance.

---

## Dataset

The project uses the **IBM Telco Customer Churn** dataset.

* **Rows:** 7,043
* **Features:** 21 columns
* **Target:** `Churn`
* `Churn = No` → `0`
* `Churn = Yes` → `1`

The target distribution is approximately:

* **73.46%** — No Churn
* **26.54%** — Churn

Because the target is imbalanced, accuracy alone is not sufficient for evaluating the model.

---

## Project Objectives

This project covers:

* Supervised learning workflow
* Train/test splitting
* Stratified sampling
* One-hot encoding
* Missing-value handling
* Feature scaling
* Logistic Regression
* Cross-validation
* Feature engineering
* Error analysis
* Learning curves
* Bias vs. variance analysis
* Precision vs. recall trade-offs
* Decision-threshold selection
* PR-AUC evaluation
* Business-oriented model evaluation

---

## Data Preprocessing

### TotalCharges

`TotalCharges` was initially stored as an object/string column even though it represents numerical values.

It was converted using:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

Invalid or blank values were converted to `NaN`.

The resulting missing values were handled using median imputation inside the preprocessing pipeline.

### Target Encoding

The original `Churn` values were converted into binary labels:

```text
No  → 0
Yes → 1
```

### Customer ID

`customerID` was excluded from the model because it is an identifier rather than a meaningful predictive feature.

---

## Train/Test Split

The dataset was divided into:

* **80% training data**
* **20% test data**

A stratified split was used to preserve the original churn ratio:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

Final sizes:

* Training: **5,634**
* Test: **1,409**

The churn proportion remained approximately **26.5%** in both sets.

---

## Baseline Model

A simple Logistic Regression model was first trained using only three numerical features:

* `tenure`
* `MonthlyCharges`
* `TotalCharges`

The baseline preprocessing consisted of:

1. Median imputation
2. Standard scaling
3. Logistic Regression

### Baseline Results

At the default classification threshold of 0.50:

| Metric    | Churn Class |
| --------- | ----------: |
| Precision |        0.60 |
| Recall    |        0.42 |
| F1 Score  |        0.49 |

Overall accuracy was approximately **77%**.

The baseline model correctly identified 157 of the 374 actual churners but missed 217 churners.

This showed that accuracy alone could hide poor performance on the churn class.

---

## Cross-Validation

A **5-fold Stratified Cross-Validation** was used to evaluate the stability of the baseline model.

Configuration:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

F1 scores across the folds were:

```text
0.5483
0.5237
0.5506
0.4913
0.4846
```

### Cross-Validation Result

**Mean F1:** `0.5197`
**Standard deviation:** `0.0277`

This indicates that model performance was reasonably consistent across different validation folds.

---

## Feature Engineering

Additional features were introduced to improve the baseline model.

### 1. Tenure Bucket

Customer tenure was divided into groups:

```text
0–12 months
13–24 months
25–48 months
49–72 months
```

This provides the model with a representation of different stages of the customer lifecycle.

### 2. Charge Ratio

A charge ratio was created:

```python
df["charge_ratio"] = (
    df["MonthlyCharges"] / (df["tenure"] + 1)
)
```

The `+1` prevents division by zero.

### 3. Contract

The categorical `Contract` feature was added:

```text
Month-to-month
One year
Two year
```

It was converted into numerical features using one-hot encoding.

---

## Preprocessing Pipeline

The engineered model uses a `ColumnTransformer` and `Pipeline`.

### Numerical Features

* Median imputation
* Standard scaling

### Categorical Features

* Most-frequent imputation
* One-hot encoding

The preprocessing and Logistic Regression model were combined into a single pipeline.

This helps prevent preprocessing inconsistencies and ensures that transformations are learned from the training data.

`handle_unknown="ignore"` was used for one-hot encoding so that unseen categorical values do not cause the model to fail during inference.

---

## Engineered Model Results

At the default threshold of 0.50:

| Metric          | Baseline | Engineered |
| --------------- | -------: | ---------: |
| Accuracy        |     0.77 |       0.79 |
| Churn Precision |     0.60 |     0.6464 |
| Churn Recall    |     0.42 |     0.4545 |
| Churn F1        |     0.49 |     0.5338 |

The engineered model improved both precision and recall for the churn class.

The number of false negatives decreased from:

```text
217 → 204
```

meaning the engineered model identified 13 additional churners at the default threshold.

---

## Error Analysis

Ten misclassified test examples were manually inspected.

The sample contained:

* **7 false negatives**
* **3 false positives**

One notable failure case was an actual churner with:

* 61 months of tenure
* Two-year contract
* Churn probability of approximately 4.4%

The model predicted that this customer would not churn, but the customer actually churned.

This demonstrates that the current feature set does not capture every factor influencing customer churn.

The model also produced false positives for some short-tenure, month-to-month customers whose characteristics appeared risky according to the model but who ultimately did not churn.

---

