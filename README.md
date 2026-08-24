# Looking Beyond Accuracy: Subgroup Recall in Adult Income Classification

**Logistic regression vs. a small neural network on the UCI Adult dataset**

Break Through Tech AI capstone, cleaned up and published here mostly so it is not only sitting in a course notebook.

This project started as a straightforward binary-classification comparison: predict whether an individual in the UCI Adult dataset earns more than \$50K using logistic regression and a small feed-forward neural network. The more interesting result ended up being elsewhere. The two models perform almost identically on overall F1, but they do **not** behave identically across sex subgroups.

The goal here is therefore less "which model wins?" and more: **what disappears when model evaluation stops at one aggregate score?**

---

## Table of Contents

- [Project Context](#project-context)
- [Data](#data)
- [Questions](#questions)
- [Method](#method)
- [Model Results](#model-results)
- [Subgroup Results](#subgroup-results)
- [Weighted Sensitivity Check](#weighted-sensitivity-check)
- [Paired Model Comparison](#paired-model-comparison)
- [Key Takeaways](#key-takeaways)
- [Limitations](#limitations)
- [Reproducibility](#reproducibility)

---

## Project Context

The dataset is moderately imbalanced: **24.08%** of observations have income above \$50K, while **75.92%** do not. A classifier can therefore achieve fairly high accuracy while still performing poorly on the positive class.

There is also a substantial difference in the observed positive rate by sex:

| Group | Income > \$50K |
|---|---:|
| Female | 10.95% |
| Non-female | 30.57% |

That made aggregate accuracy a particularly incomplete evaluation target. I use F1 for model selection, then explicitly examine recall and precision by subgroup.

> **Schema note:** the public UCI source uses `Male` and `Female`. The course version of the dataset relabeled `Male` as `Non-Female`, and the notebook preserves that schema for compatibility. `Non-Female` should not be interpreted as a modern gender-identity category.

---

## Data

The analysis uses **32,561 observations** from the UCI Adult dataset.

The final modeling split is stratified and fixed across both model families:

| Partition | Rows |
|---|---:|
| Training | 22,792 |
| Validation | 4,884 |
| Test | 4,885 |

The final test set is held out until all preprocessing, model selection, early stopping, and probability-threshold selection are complete.

### Preprocessing

A few choices were made specifically to avoid letting preprocessing decisions leak information from validation or test data:

- Numeric variables are median-imputed and standardized using the training set.
- Categorical variables use an explicit `Missing` category and one-hot encoding.
- `education` is dropped because it is redundant with `education-num`.
- `fnlwgt` is excluded as a predictor and retained only for a weighted sensitivity analysis.
- Missing native-country values are kept distinct rather than silently treated as U.S.-native.
- `capital-gain` and `capital-loss` are treated as zero-inflated, heavily right-skewed variables:
  - separate zero/nonzero indicators are retained;
  - `capital-gain` is clipped at the **training-set 99th percentile (\$15,024)**;
  - amounts are transformed with `log1p` before standardization.

---

## Questions

The project asks three related questions:

1. How does a regularized logistic regression compare with a small neural network on this tabular classification problem?
2. Do similar aggregate scores hide different behavior across sex subgroups?
3. Does the subgroup result persist when test observations are reweighted using `fnlwgt`?

---

## Method

### Logistic regression

The logistic-regression pipeline is tuned using **5-fold cross-validation on the training set only**. The selected model used:

- `C = 10`
- balanced class weights
- L2 regularization
- best cross-validation F1: **0.6790**

After fitting, the classification threshold is selected on the validation set to maximize F1.

- Validation-selected threshold: **0.62**
- Validation F1 at that threshold: **0.7178**

### Neural network

The neural network uses the same raw features and exactly the same train/validation/test observations.

Architecture:

```text
47 transformed inputs
      ↓
Dense(64, ReLU)
      ↓
Dense(32, ReLU)
      ↓
Dense(1, sigmoid)
```

Training uses:

- SGD, learning rate `0.01`
- binary cross-entropy
- balanced class weights
- weighted validation loss
- early stopping with patience of 20 epochs
- best weights restored

Training stopped after **78 epochs**. As with logistic regression, the final classification threshold was chosen using validation F1 only.

- Validation-selected threshold: **0.63**
- Validation F1 at that threshold: **0.7248**

---

## Model Results

On the untouched test set:

| Metric | Logistic Regression | Neural Network |
|---|---:|---:|
| Accuracy | **0.8416** | 0.8299 |
| F1 | 0.6955 | **0.6964** |
| Validation-selected threshold | 0.62 | 0.63 |

The headline result is almost a tie.

The neural network improves F1 by less than one tenth of a percentage point, while logistic regression retains about **1.17 percentage points more accuracy**. On aggregate performance alone, there is little reason to claim that the extra neural-network complexity meaningfully "wins."

The more interesting difference appears after breaking performance out by subgroup.

---

## Subgroup Results

### Recall

| Group | Logistic Regression | Neural Network | Change |
|---|---:|---:|---:|
| Female | 0.6522 | **0.7391** | **+8.69 pp** |
| Non-female | 0.7702 | **0.8236** | **+5.34 pp** |
| Recall gap | **11.80 pp** | **8.45 pp** | **−3.35 pp** |

The neural network finds more of the actual positive cases in **both** groups, but the improvement is larger among female observations. As a result, the female/non-female recall gap narrows from about **11.8 points to 8.5 points**.

Counts make that difference more concrete:

| Model / Group | Actual positives | True positives found | Predicted positive |
|---|---:|---:|---:|
| Logistic — Female | 184 | 120 | 192 |
| Logistic — Non-female | 992 | 764 | 1,174 |
| Neural network — Female | 184 | 136 | 227 |
| Neural network — Non-female | 992 | 817 | 1,334 |

Relative to logistic regression, the network correctly identifies **16 additional female positives** and **53 additional non-female positives** on the same test set.

### Precision

That extra recall is not free:

| Group | Logistic Regression | Neural Network |
|---|---:|---:|
| Female precision | **0.6250** | 0.5991 |
| Non-female precision | **0.6508** | 0.6124 |

The network predicts the positive class more often, increasing true positives but also false positives. This is why I do not interpret the narrower recall gap as evidence that the neural network is simply "fairer." It represents a different error trade-off.

---

## Weighted Sensitivity Check

`fnlwgt` is a census sampling weight, not a predictive feature. I therefore keep it out of both models but use it in a sensitivity analysis of the reported subgroup metrics.

### Weighted recall

| Group | Logistic Regression | Neural Network |
|---|---:|---:|
| Female | 0.6472 | **0.7388** |
| Non-female | 0.7744 | **0.8242** |
| Recall gap | **12.72 pp** | **8.54 pp** |

The weighted result is extremely close to the unweighted result. Reweighting changes the exact rates slightly, but not the substantive pattern: the neural network has higher recall in both groups and a smaller female/non-female recall gap.

Weighted precision shows the same trade-off seen in the raw test sample:

| Group | Logistic Regression | Neural Network |
|---|---:|---:|
| Female | **0.6461** | 0.6039 |
| Non-female | **0.6768** | 0.6298 |

---

## Paired Model Comparison

Because both models make predictions for the **same people**, their errors are paired rather than independent.

I use McNemar's exact test on actual positive cases to ask whether one model is systematically correct on cases the other misses.

| Positive cases | LR correct / NN wrong | NN correct / LR wrong | Exact p-value |
|---|---:|---:|---:|
| All | 32 | **101** | 1.59 × 10⁻⁹ |
| Female | 7 | **23** | 0.0052 |
| Non-female | 25 | **78** | 1.61 × 10⁻⁷ |

Among positive cases, the neural network is substantially more often the model that gets a discordant case right. That supports the recall difference seen above; it does **not** imply that the network is uniformly better, since logistic regression still has higher overall accuracy and precision.

---

## Key Takeaways

The project ended up being a useful example of why model evaluation is not one number.

If I reported only F1, I would conclude that these models are effectively tied. If I reported only accuracy, I would prefer logistic regression. If I cared specifically about finding positive cases, the neural network is meaningfully better on this split. If I then break recall down by sex, I see that the improvement is not distributed evenly.

So the main result is not that a neural network beats logistic regression. It is that **two models with nearly identical aggregate F1 can have meaningfully different error profiles**, and those differences become visible only after deciding which errors matter and for whom.

That is also why the subgroup analysis here is descriptive rather than a blanket fairness verdict. Recall, precision, calibration, false-positive rates, thresholds, and the underlying label distribution all answer different questions. A model can look better under one criterion and worse under another.

---

## Limitations

- **One train/validation/test split.** The test set is properly held out, but subgroup metrics are still point estimates from one seeded split.
- **Thresholds optimize global F1.** They are not chosen to equalize subgroup performance or optimize calibration.
- **Only sex is evaluated in detail.** Race and native-country information are included in the model but not given equivalent subgroup analyses.
- **Historical data.** The dataset reflects 1994 census-era data and should not be treated as a description of the present-day labor market.
- **Native country is compressed.** Known non-U.S. countries are grouped together for modeling.
- **Capital-gain preprocessing is a modeling choice.** The training-fitted 99th-percentile cap and `log1p` transformation reduce sensitivity to the extreme upper tail, but other defensible transformations are possible.
- **`fnlwgt` weighting is a sensitivity check, not a full survey-design analysis.**
- **No calibration analysis.** Similar F1 or recall does not imply equally calibrated predicted probabilities.
- **No third tabular model family.** A gradient-boosted tree would be a useful comparison.

---

## Reproducibility

The notebook uses a fixed random seed (316) and fits all preprocessing using training data only.

The public UCI Adult dataset is downloaded automatically if the original course CSV is not available, with schema mappings applied so the analysis can run against either version.

To reproduce the environment:
`pip install -r requirements.txt`

Then run `main.ipynb`, which contains the complete analysis from exploratory analysis and preprocessing through model fitting, subgroup evaluation, weighted sensitivity checks, and paired tests.
