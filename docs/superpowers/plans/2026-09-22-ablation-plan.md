# Feature Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the five frozen Phase 1 representative models on all 17 predictors versus the same models after removing Negative Symptom Score, GAF Score, and Positive Symptom Score, using leakage-safe paired five-fold CV and an explicitly exploratory fixed-test evaluation.

**Architecture:** Extend the shared preprocessing factory so it accepts a selected predictor list while preserving the current full-feature callers. Keep estimator construction, configured pipelines, holdout metrics, and data loading in `preprocessing_and_cv.py`; both `full/full.py` and the focused `ablation/` runner import from that shared module. The ablation runner constructs one shared set of stratified folds, writes per-fold and paired summaries, evaluates both frozen feature sets on the existing test file, and records the test-set exposure limitation in a protocol and interpretation file.

**Tech Stack:** Python 3.14 virtual environment, pandas, NumPy, scikit-learn, standard-library `unittest`, existing project modules.

**Spec:** User's feature-ablation requirements in the conversation; no separate design document was requested.

## Global Constraints

- Keep the current headerless 18-column schema and the existing `-1/1` to `0/1` label conversion.
- Use the same 8,000 training rows, the same `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` split indices, and the same six primary metrics for full and ablated conditions.
- Fit every enabled data-dependent operation inside each training fold; fit final test preprocessing only on all training rows. The frozen configurations disable numeric scaling, and `knn_002` uses Euclidean distance, so neither scaling nor Mahalanobis covariance is fitted in this experiment.
- Delete exactly `Negative Symptom Score`, `GAF Score`, and `Positive Symptom Score` for the ablated condition.
- Freeze the current Phase 1 representative configurations: `logistic_002`, `knn_002`, `perceptron_002`, `mlp_016`, and `random_forest_001`.
- Treat test metrics as exploratory because Phase 1 evaluated the same test set for all 146 candidate configurations.
- Preserve the existing uncommitted Phase 2 changes.

## Review Focus

- A selected-column preprocessor must never silently reintroduce one of the three ablated variables; test that transformed feature names contain only the requested columns.
- Full and ablated conditions must consume identical row indices in every fold; test that the shared split list is reused.
- If a Mahalanobis k-NN configuration is used later, its covariance must be fitted on each fold's training rows through the shared estimator pipeline. The frozen ablation configuration uses Euclidean distance.
- The six metrics must use the positive class `1` and specificity must use class `0`; test a known confusion matrix.
- The test evaluation must fit on all training rows and must be reported separately from CV summaries; inspect the protocol and output schema.

---

### Task 1: Make the existing preprocessing selectable by feature list

**Files:**
- Modify: `preprocessing_and_cv.py`
- Modify: `full/full.py`
- Test: `tests/test_ablation.py`

**Interfaces:**
- `build_preprocessor(feature_columns: Sequence[str] | None = None) -> ColumnTransformer` keeps the current full-feature default and derives numerical/categorical transformer columns from the supplied predictor list.
- `build_pipeline(model: object, feature_columns: Sequence[str] | None = None) -> Pipeline` passes the selected columns to the preprocessor.
- `preprocessing_and_cv.build_configured_pipeline(..., feature_columns: Sequence[str] | None = None)` keeps current callers unchanged and forwards the list to `build_pipeline`.

- [ ] **Step 1: Write the failing tests**

  Add tests that call `build_preprocessor(feature_columns=ABLATION_FEATURES)` and assert that fitting and transforming a small frame succeeds, that no dropped feature appears in `get_feature_names_out()`, and that `build_configured_pipeline` accepts the same selected list.

- [ ] **Step 2: Run the focused tests and verify the expected failure**

  Run: `python.exe -m unittest tests.test_ablation -v`

  Expected: FAIL because the current preprocessor has no `feature_columns` parameter.

- [ ] **Step 3: Implement the minimal configurable preprocessing**

  Add validation against `NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS`, preserve caller order only for the DataFrame, derive the two transformer lists by membership, and retain `handle_unknown="ignore"`, dense one-hot output, and numeric `StandardScaler`. Add the optional argument to `build_pipeline` and `build_configured_pipeline` without changing default full-feature behavior.

- [ ] **Step 4: Run the focused tests and verify they pass**

  Run: `python.exe -m unittest tests.test_ablation -v`

  Expected: the preprocessing-selection tests pass.

### Task 2: Add the paired ablation runner and output artifacts

**Files:**
- Create: `ablation/__init__.py`
- Create: `ablation/run_ablation.py`
- Create: `ablation/README.md`
- Test: `tests/test_ablation.py`

**Interfaces:**
- `make_shared_splits(y, cv_splits=5, random_state=42) -> list[tuple[np.ndarray, np.ndarray]]` returns the one shared fold list.
- `run_ablation(...) -> dict[str, pd.DataFrame]` returns `fold_results`, `cv_summary`, `paired_differences`, and `test_results` and writes them plus the protocol and interpretation files.
- The runner imports `build_configured_pipeline`, `evaluate_test_set`, and `load_datasets` directly from `preprocessing_and_cv`; it does not import `full.full`.

- [ ] **Step 1: Extend the failing tests**

  Add tests for shared split identity, fixed configuration IDs, six-metric confusion-matrix scoring, and the required output column names.

- [ ] **Step 2: Run the focused tests and verify the expected failure**

  Run: `python.exe -m unittest tests.test_ablation -v`

  Expected: FAIL because `ablation.run_ablation` does not exist.

- [ ] **Step 3: Implement the minimal runner**

  Define the exact dropped set and five frozen configurations. Build one split list, fit fresh selected-column pipelines for each model/feature-set/fold, calculate Accuracy, Precision, Recall, Specificity, F1, and ROC-AUC, and retain fold identifiers and configuration metadata. Aggregate mean and sample standard deviation, calculate paired `ablated - full` deltas from matching folds, then fit each condition on all training rows for a separate test table with confusion counts. Write CSVs under `results/ablation`, a JSON protocol containing parameters, seed, folds, feature sets, preprocessing, and test-exposure status, and a numerical Markdown interpretation naming the largest changes.

- [ ] **Step 4: Run the focused tests and verify they pass**

  Run: `python.exe -m unittest tests.test_ablation -v`

  Expected: all focused tests pass.

### Task 3: Run the experiment and verify the complete delivery

**Files:**
- Create: `results/ablation/ablation_fold_results.csv`
- Create: `results/ablation/ablation_cv_summary.csv`
- Create: `results/ablation/ablation_cv_paired_differences.csv`
- Create: `results/ablation/ablation_test_results.csv`
- Create: `results/ablation/ablation_protocol.json`
- Create: `results/ablation/ablation_interpretation.md`

**Interfaces:**
- The command `python.exe -m ablation.run_ablation` runs from the project root and uses the default `training_data.csv`, `testing_data.csv`, and `results/ablation` paths.

- [ ] **Step 1: Run the full test suite**

  Run: `python.exe -m unittest discover -v`

  Expected: all repository tests pass.

- [ ] **Step 2: Run the ablation experiment**

  Run: `python.exe -m ablation.run_ablation`

  Expected: five models × two feature sets × five folds are written, followed by ten fixed-test rows and all protocol/interpretation artifacts.

- [ ] **Step 3: Verify output integrity and numerical interpretation**

  Check that there are 50 fold rows, 10 CV summary rows, 5 paired-difference rows, and 10 test rows; every ablated row records exactly 14 predictors; all six metrics are finite; full and ablated folds have matching fold IDs; and the interpretation names changes from the generated values without causal or clinical claims.

- [ ] **Step 4: Run the full test suite again after the experiment**

  Run: `python.exe -m unittest discover -v`

  Expected: all repository tests pass after result generation.
