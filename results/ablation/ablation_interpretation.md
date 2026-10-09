# Feature ablation interpretation

The fixed comparison removes Negative Symptom Score, GAF Score, and Positive Symptom Score together. Deltas are ablated minus full; negative values mean the metric decreased after removal.

## Largest paired CV changes

- knn - Recall: full mean 1.000000; paired delta -0.285281 (ablated - full).
- knn - F1: full mean 1.000000; paired delta -0.166532 (ablated - full).
- knn - Accuracy: full mean 1.000000; paired delta -0.082375 (ablated - full).
- perceptron - Recall: full mean 1.000000; paired delta -0.079221 (ablated - full).
- logistic - Recall: full mean 1.000000; paired delta -0.076190 (ablated - full).

## Fixed-test comparison

The test rows below were generated after fitting each pipeline on all training rows. They are exploratory because Phase 1 evaluated this same test set for all 146 candidate configurations; they were not used to select the ablation feature set or parameters.

- knn - Recall: full 1.000000; test delta -0.273830 (ablated - full).
- knn - F1: full 1.000000; test delta -0.158635 (ablated - full).
- knn - Accuracy: full 1.000000; test delta -0.079000 (ablated - full).
- perceptron - Recall: full 1.000000; test delta -0.076256 (ablated - full).
- logistic - Recall: full 1.000000; test delta -0.050260 (ablated - full).

These changes describe predictive performance on this dataset. They do not establish causal effects, clinical utility, clinical diagnostic validity, or early-prediction capability.
