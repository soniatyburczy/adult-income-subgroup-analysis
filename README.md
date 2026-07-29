# Income > $50K — logistic regression vs. a small neural network

Break Through Tech AI capstone, cleaned up and put here mostly so it isn't only sitting in a
course notebook. Binary classification on the 1994 US Census (Adult) data: does an individual
earn more than $50,000.

The point isn't the model comparison. It's that **both model families miss high-earning women
at a much higher rate than high-earning men and aggregate metrics can't see it. 

## Results

| Metric | Logistic Regression | Neural Network |
|---|---|---|
| Accuracy | 0.8186 | 0.8299 |
| F1 | 0.6941 | 0.7029 |
| Recall, female | 0.7507 | 0.7619 |
| Recall, non-female | 0.8732 | 0.8491 |
| Recall gap | 12.25 pts | 8.72 pts |
| Precision, female | 0.5969 | 0.6071 |
| Precision, non-female | 0.5824 | 0.6063 |

The network's gap is 3.5 points smaller. In counts, on 2,352 test positives (357 female,
1,995 non-female):

| | Female positives found | Non-female positives found |
|---|---|---|
| Logistic Regression | 268 / 357 | 1,742 / 1,995 |
| Neural Network | 272 / 357 | 1,694 / 1,995 |
| Change | **+4** | **−48** |

McNemar's test on the paired predictions, restricted to positives:

| Subset | LR caught, NN missed | NN caught, LR missed | p |
|---|---|---|---|
| All positives | 107 | 63 | 0.0009 |
| Female positives | 14 | 18 | 0.60 |
| Non-female positives | 93 | 45 | 0.0001 |

So the network's apparent gain for women is not statistically detectable — the net +4 is 14
losses against 18 gains, churn rather than progress — while its loss for men is. The smaller
recall gap is a real harm to the majority group plus noise in the minority group. A reader
seeing only "12.25 → 8.72" would conclude the opposite.

Both models used `class_weight='balanced'`, which reweights the global 76/24 label imbalance
but does nothing about the ~89/11 imbalance *inside* the female subgroup. Those are different
quantities, and the second is what produces the gap.

**Precision tells a different story than recall.** It's within 1.5 points across groups in
both models while recall differs by 9–12. Same predictions, same test set, two standard
fairness criteria, opposite verdicts.

**The result is not fragile.** The notebook was rerun across a dataset swap (course CSV →
public UCI), an encoding fix (`drop_first=True`), a TensorFlow major version, and a
regularizer change the grid search made on its own (L2 → L1). No metric moved more than half
a point. Side-by-side table in section 7.6.

## Data

Outputs come from the public UCI Adult file, so they reproduce from a clean clone.

An earlier run used the course-provided `censusData.csv` — a modified copy with injected
missingness in `age` and `hours-per-week` and `capital-gain` capped at 14,084 instead of the
99,999 top-code. Not redistributed here. `load_census()` prefers it at `data/censusData.csv`
if you have it, and otherwise downloads the public file and renames it into the same schema.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
```

NOTE: the grid search still passes `penalty`, deprecated in
sklearn 1.8 in favor of `l1_ratio`, so expect warnings.

## Known issues

Section 8 of the notebook has the full list. The short version:

- `capital-gain`'s 99,999 top-code isn't clipped, and it dominates the coefficient table.
- L1 zeroes 13 of 44 coefficients; among correlated predictors that choice is close to
  arbitrary, so absence from the model isn't evidence of no association.
- Reference levels are pandas' alphabetical default, not deliberate choices.
- Only sex is broken out as a subgroup to investigate. Race and national origin are in the model, but remain unmeasured.
- "Non-Female" is a course relabeling of the 1994 census `Male` category; `sex_selfID` implies
  a self-identification the data does not contain.
- One split, one seed, fixed 0.5 threshold, no calibration check.

## Further Work

- Per-subgroup thresholds — near-equal precision alongside a large recall gap is the
  signature of a threshold effect as much as a representational one, so this comes first.
- Clip or bin `capital-gain` and refit.
- SMOTE and resampling against `class_weight`, scored on subgroup recall rather than F1.
- Gradient boosting as a third model family, same subgroup breakdown, same paired test.
- FPR gaps, equalized odds, calibration by group.
- Repeated splits with intervals on the subgroup rates.
