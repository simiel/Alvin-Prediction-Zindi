# Alvin Prediction

This repository is a [Zindi](https://zindi.africa) competition entry for the AlvinApp challenge (the starter notebook also titles it AvinApp). It predicts which of 13 spending categories a customer assigned to a merchant purchase, from the purchase amount, two timestamps, an M-Pesa send-money flag, and a few customer attributes. The labeled set is small and uneven, and the leaderboard expects a full probability vector per transaction, scored with log loss. The notebooks, competition tables, and submission files are the whole project: a record of the analysis and the models that were actually fit and submitted.

## Problem

Each row is one purchase. `Train.csv` has the category the customer assigned (`MERCHANT_CATEGORIZED_AS`). `Test.csv` withholds it. A valid submission follows `SampleSubmission.csv`: `Transaction_ID` plus one probability column for each category.

`VariableDefinitions.csv` describes the fields. The tables in this repository contain:

| Column | Role |
| --- | --- |
| `Transaction_ID` | Row id used in the submission |
| `USER_ID` | Customer id |
| `MERCHANT_NAME` | Merchant string |
| `MERCHANT_CATEGORIZED_AS` | Target category (train only) |
| `MERCHANT_CATEGORIZED_AT` | When the customer categorized the merchant |
| `PURCHASED_AT` | When the purchase was made |
| `PURCHASE_VALUE` | Purchase amount |
| `IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY` | True when the merchant is not a registered business name |
| `USER_AGE`, `USER_GENDER`, `USER_HOUSEHOLD`, `USER_INCOME` | Customer attributes |

The definitions file also lists `USER_EMAIL`. That column is not in the CSV tables here.

Scale of the labeled problem:

- **373** labeled transactions, **25** customers, **218** merchant names. Purchase values run from 10 to 150,000.
- **558** test transactions, **30** customers, **314** merchant names. Every training customer also appears in the test set. **92** merchant names appear in both.
- Age is missing for **312 / 373** training rows (21 of 25 customers) and **473 / 558** test rows. Gender is missing only for customer `ID_R4T8Q46J` (6 training rows, 5 test rows).
- The 13 classes are heavily skewed. Bills & Fees has 78 training rows; Education and Rent / Mortgage have 2 each.

| Category | Training rows |
| --- | ---: |
| Bills & Fees | 78 |
| Groceries | 50 |
| Data & WiFi | 43 |
| Miscellaneous | 43 |
| Family & Friends | 41 |
| Going out | 41 |
| Transport & Fuel | 29 |
| Shopping | 21 |
| Emergency fund | 12 |
| Health | 6 |
| Loan Repayment | 5 |
| Rent / Mortgage | 2 |
| Education | 2 |

`extra_data.csv` has **10,000** further transactions in the same schema, from 39 customers (24 shared with train, 28 with test) and 2,781 merchant names. `MERCHANT_CATEGORIZED_AS` is empty on every row, and age is missing on 8,842 rows. It is unlabeled side data. Neither notebook trains on it.

## Approach

Two notebooks try two representations of the same table.

**`Starter_notebook.ipynb`** is the competition starter, carried through a first submission. It plots spend by category, encodes the target to inspect correlation (the numeric fields have little linear association with the label), drops fully empty rows, fills missing gender with the majority value `Male`, and fills missing age with the column median. Merchant name is one-hot encoded after train and test are concatenated, so both sides share the same columns (446 features on the modeling matrix). Timestamps, user id, and the target are then dropped. A decision tree and a random forest are trained on a 90/10 split (`test_size=0.1`, `random_state=101`).

**`Solution.ipynb`** uses a short numeric feature set and keeps time of day. The features are the hour of `MERCHANT_CATEGORIZED_AT`, `PURCHASE_VALUE`, the hour of `PURCHASED_AT`, the M-Pesa flag, `USER_INCOME`, `USER_GENDER`, and `USER_HOUSEHOLD`. Gender and the M-Pesa flag are mapped to 0/1; any other gender value, including missing, becomes 1 (male). Merchant name, user id, and age are left out. The executed path fits `XGBClassifier` and writes `predict_proba` output to `xgb.csv`.

The same notebook still defines the estimators used on the way to that file:

- `LinearSVC` wrapped in `CalibratedClassifierCV`. The `jetting_off` commits fit this and write `jetting_off.csv`. The `LinearSVCBagging` commit bags 10 calibrated SVMs (`n_estimators=10`, `random_state=0`).
- `DecisionTreeClassifier`, and `RandomForestClassifier(criterion='log_loss', n_estimators=100, random_state=42)`.
- `LGBMClassifier(learning_rate=0.05)`. The `lightlr02` commit fits this and writes `lightlr02.csv`.
- `BaggingClassifier` around the forest, with 10 members.
- `StackingClassifier` whose base models are a 10-tree random forest and a bagged calibrated SVM. The commit that adds `Stack.csv` uses a random forest as the final estimator. The version in the notebook now, which also writes `Stack2withbagprob.csv`, uses logistic regression and `stack_method='predict_proba'`.

A dictionary of tree criteria and estimator counts is declared in the notebook and left unused.

Predicted probabilities can be clipped with `pimp` to the closed range 0.025–0.975. `lightlr02pimp.csv` shows that floor. The checked-in `xgb.csv` matches the unclipped XGBoost probabilities. Clipping is not followed by a renormalization, so clipped files such as `lightlr02pimp.csv`, `grid.csv`, and `forestbagikam.csv` no longer sum to 1 across the 13 classes. `grid.csv` and `ld.csv` are identical.

## Repository structure

```
Train.csv, Test.csv, extra_data.csv
SampleSubmission.csv, VariableDefinitions.csv
Starter_notebook.ipynb     # EDA, preprocessing, first model, first submission
Solution.ipynb             # numeric features, defined ensembles, XGBoost submission
*.csv                      # probability submissions from the experiment trail
```

Submission files, in the order they were added:

| File | Commit | What the notebook trail shows |
| --- | --- | --- |
| `first_submission.csv` | `Init` | Starter random forest. Hard labels expanded to 13 columns |
| `jetting_off.csv` | `jetting_off`, updated in `Jetting off, a little better` | Calibrated `LinearSVC` probabilities. The second message calls the update a small improvement |
| `linearSVCBagging.csv` | `LinearSVCBagging` | 10-member bag of a calibrated `LinearSVC`. The cell in that commit writes `DBagging.csv`; the file committed under this name is the artifact |
| `DTreeBagging.csv` | `RForestBagging` | 10-member bag of `DecisionTreeClassifier`, written by that commit's notebook |
| `RForestBagging.csv` | `RForestBagging` | Added with the tree bag. `RandomForestClassifier` is imported in that commit; the saved cell writes `DTreeBagging.csv` |
| `Stack.csv` | `Stack2withbag` | Stack of a 10-tree forest and a bagged calibrated SVM, random forest as the final estimator, then `predict_proba` |
| `Stack2withbag.csv` | `Stack2withbag` | Added in the same commit. The saved cell writes `Stack.csv` |
| `Stack2withbagprob.csv` | `Stack2withbagprob` | Same stack with logistic regression and `stack_method='predict_proba'` |
| `lightlr02.csv` | `lightlr02` | `LGBMClassifier(learning_rate=0.05)` probabilities |
| `light.csv` | `lightlr02` | Added beside `lightlr02.csv` |
| `xgb.csv` | `All Good 90th Position` | Unclipped `XGBClassifier` probabilities from the current solution notebook |
| `lightlr02pimp.csv` | `All Good 90th Position` | LightGBM probabilities after the 0.025 floor |
| `forestbagikam.csv`, `forestbagikammpp.csv` | `All Good 90th Position` | Forest-named submissions from the final commit. The `mpp` file sums to 1 per row; `forestbagikam.csv` does not. The saved notebook writes `xgb.csv` |
| `grid.csv`, `ld.csv` | `All Good 90th Position` | Byte-identical copies of one clipped probability file |

For the first submission, the starter predicts 11 of the 13 classes. Two rows are then overwritten with `Education` and `Rent / Mortgage` so the one-hot file contains every column log loss requires. The notebook text marks that overwrite as a demonstration and says the model itself should cover every class, or a missing column should be added and filled with zeros.

## Results

Figures below are the ones printed in the notebooks or recorded in git. The only public standing stored in the repository is the 90th-place commit message.

On the starter's 38-row holdout:

- Decision tree (`max_depth=5`, `min_samples_leaf=8`, `random_state=101`): **accuracy 0.342**
- Random forest (scikit-learn defaults): **accuracy 0.632**

Several rare classes are absent from that 38-row slice, so these accuracies are a rough local check.

`Solution.ipynb` prints an `XGBClassifier` accuracy of **0.946** from `model.score` on the same 373 rows the model was fit on.

The final commit, `a48a577` (`All Good 90th Position`), records a **90th-place** finish.

## Tech stack

Python 3 notebooks (the kernel metadata is IPython's `python3` kernel), with pandas, NumPy, Matplotlib, Seaborn, scikit-learn, LightGBM, and XGBoost. The starter imports scikit-learn's decision tree, random forest, `train_test_split`, accuracy, classification report, and log loss. The solution notebook adds linear SVM, logistic regression, probability calibration, bagging, stacking, LightGBM, and XGBoost.

## Getting started

Place the CSVs next to the notebooks and run them from the repository root. `Starter_notebook.ipynb` sets `path = ""`. `Solution.ipynb` reads `Train.csv`, `Test.csv`, and `SampleSubmission.csv` from the working directory. A Google Colab Drive path in the starter is commented out.

Libraries the notebooks import:

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `lightgbm` and `xgboost` (`Solution.ipynb`)

No dependency file, environment variables, build script, or test command is part of the repository. Re-running `Solution.ipynb` writes `xgb.csv`. Re-running the starter's submission cell writes `first_submission.csv`.

## Project status

Finished competition entry. The modeling lives in two notebooks, and each public iteration is kept as a CSV plus a commit. The last of those commits records 90th place.

## What this demonstrates

- A complete competition pass on a 13-class probability task: data audit, aligned train and test features, a baseline, and submissions in the exact sample-file layout log loss expects.
- Two deliberate representations of a tiny table. One widens merchant name into aligned one-hot columns across train and test. The other keeps amount, income, household, gender, the M-Pesa flag, and the hour of each timestamp, which is a reasonable choice when age is missing for most customers and merchant strings are sparse.
- A model sweep across calibrated linear SVM, decision tree, random forest, LightGBM, XGBoost, bagging, and stacking, with the probability files left in the tree so the sequence can be checked.
- Care with the metric itself. Hard labels are expanded to class columns, SVM scores are calibrated before they are treated as probabilities, and extreme probabilities are clipped when that variant is submitted.
- An honest read of the data. The labeled set is 373 rows, two classes have two examples, age is missing for 21 of 25 customers, and the 10,000-row side file has the category column and no labels. The local holdout numbers and the in-sample XGBoost score are reported separately, and the public result on record is 90th place.
