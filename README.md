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

## Learning Curve and Bias/Variance Analysis

Learning curves were generated using F1 score.

At the largest training size:

* Training F1 ≈ **0.5338**
* Validation F1 ≈ **0.5331**

The training and validation curves were very close, providing no strong evidence of overfitting.

Both scores improved as more training data was added and then began to plateau around an F1 score of 0.53.

This suggests that the model generalizes reasonably well, but the current feature set may be limiting further performance improvement.

---

## Decision Threshold Analysis

Logistic Regression produces churn probabilities rather than only final class labels.

The default threshold is:

```text
0.50
```

However, the best threshold depends on the business cost of false positives and false negatives.

For this project, the business assumption was:

> Missing a true churner is more expensive than contacting a customer who ultimately does not churn.

Therefore, higher recall was prioritized.

Several thresholds were evaluated.

| Threshold |  Precision |     Recall |         F1 |
| --------: | ---------: | ---------: | ---------: |
|      0.20 |     0.4556 |     0.8636 |     0.5965 |
|      0.25 |     0.4772 |     0.8102 |     0.6006 |
|  **0.30** | **0.5091** | **0.7513** | **0.6069** |
|      0.35 |     0.5277 |     0.6872 |     0.5970 |
|      0.40 |     0.5502 |     0.6150 |     0.5808 |
|      0.45 |     0.5879 |     0.5455 |     0.5659 |
|      0.50 |     0.6464 |     0.4545 |     0.5338 |
|      0.55 |     0.6959 |     0.3610 |     0.4754 |
|      0.60 |     0.7448 |     0.2888 |     0.4162 |

A threshold of **0.30** was selected because it substantially increased recall while also producing the highest F1 score among the tested thresholds.

---

## Final Test Results

Using a decision threshold of **0.30**, the final model achieved:

| Metric    |      Score |
| --------- | ---------: |
| Precision | **0.5091** |
| Recall    | **0.7513** |
| F1 Score  | **0.6069** |
| PR-AUC    | **0.6379** |

### Confusion Matrix

```text
[[764 271]
 [ 93 281]]
```

This corresponds to:

* **TN:** 764
* **FP:** 271
* **FN:** 93
* **TP:** 281

The model correctly identified **281 of 374 actual churners**, resulting in a churn recall of approximately **75.1%**.

The trade-off is that 271 non-churners were also incorrectly flagged as potential churners.

---

## Key Findings

1. A simple numerical Logistic Regression baseline achieved approximately **0.49 churn F1** at the default threshold.
2. Feature engineering improved churn F1 to approximately **0.53** at the default threshold.
3. Cross-validation produced a mean F1 of approximately **0.52 ± 0.03**.
4. Learning curves showed no strong evidence of overfitting.
5. Lowering the classification threshold increased churn recall substantially.
6. A threshold of **0.30** produced:

   * **75.13% recall**
   * **50.91% precision**
   * **60.69% F1**
   * **63.79% PR-AUC**
7. The main model weakness is false negatives among customers whose observed features make them appear low-risk.

---

## Production Monitoring

If this model were deployed in production, monitoring would be important.

Potential monitoring areas include:

### Input Data Drift

Monitor changes in:

* `tenure`
* `MonthlyCharges`
* `TotalCharges`
* `Contract`
* Other important customer characteristics

### Prediction Drift

Monitor whether the distribution of predicted churn probabilities changes significantly over time.

### Model Performance

Once actual churn outcomes become available, monitor:

* Precision
* Recall
* F1
* False-negative rate
* PR-AUC

A significant deterioration could indicate that customer behavior has changed and the model needs to be retrained.

---

## Limitations

The final model uses a relatively small set of features and Logistic Regression, which provides a simple and interpretable baseline but may not capture complex non-linear relationships.

The threshold was selected based on the stated business assumption that false negatives are more costly than false positives.

Further improvements could include:

* Adding additional relevant customer features
* Testing different regularization strengths
* Treating tenure buckets as categorical features
* Testing non-linear models
* Using a dedicated validation set for threshold selection
* Monitoring model performance after deployment

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter / Google Colab

---

## Project Structure

```text
Telco_churn/
│
├── Telco_Customer_Churn.csv
├── telco_churn.ipynb
└── README.md
```

---

## Conclusion

This project demonstrates a complete supervised learning workflow for customer churn prediction.

The final Logistic Regression model prioritizes recall because the business assumption makes missed churners more costly than unnecessary retention outreach. With a threshold of 0.30, the model identifies approximately 75% of actual churners while achieving an F1 score of 0.6069 and PR-AUC of 0.6379.

The results show that basic feature engineering and threshold selection can meaningfully improve a simple Logistic Regression model, while the error analysis and learning curves highlight areas for future improvement.
