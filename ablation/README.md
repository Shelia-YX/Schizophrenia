# Fixed-parameter feature ablation

Run this module from the project root with the configured project interpreter:

```text
python.exe -m ablation.run_ablation
```

The defaults read `training_data.csv` and `testing_data.csv`, then write to
`results/ablation/`. Optional paths and model subsets are available through
`--training`, `--testing`, `--output-dir`, and `--models`.

The experiment compares the five Phase 1 representative configurations on
all 17 predictors and after removing `Negative Symptom Score`, `GAF Score`,
and `Positive Symptom Score`. It creates one shared five-fold stratified split
list (`random_state=42`) and fits preprocessing independently inside every
training fold. The final test rows use a fresh pipeline fitted on all training
rows. The frozen configurations all set `scale_numeric=false`, so the
numeric scaler is passthrough in this run; if scaling is enabled in a future
run, it must be fitted inside each training fold. The k-NN configuration uses
Euclidean distance and does not fit a Mahalanobis covariance matrix.

Outputs:

- `ablation_fold_results.csv`: one row per model, feature set, and fold;
- `ablation_cv_summary.csv`: five-fold means and sample standard deviations;
- `ablation_cv_paired_differences.csv`: paired ablated-minus-full fold deltas;
- `ablation_test_results.csv`: fixed-test metrics, confusion counts, and an
  `Evaluation Status=exploratory` field;
- `ablation_protocol.json`: feature sets, frozen parameters, folds, seed, and
  preprocessing/test-use metadata;
- `ablation_interpretation.md`: numerical interpretation generated from the
  saved outputs.

The fixed test file has already been evaluated repeatedly during Phase 1 for
all 146 candidate configurations. Its ablation rows are therefore exploratory
and are not a strict independent final validation.
