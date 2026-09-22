# Feature ablation interpretation

The fixed comparison removes Negative Symptom Score, GAF Score, and Positive Symptom Score together. Deltas are ablated minus full; negative values mean the metric decreased after removal.

## Largest paired CV changes

- knn - Recall: full mean 1.000000; paired delta -0.277922 (ablated - full).
- knn - F1: full mean 1.000000; paired delta -0.161577 (ablated - full).
- logistic - Recall: full mean 1.000000; paired delta -0.147186 (ablated - full).
- knn - Accuracy: full mean 1.000000; paired delta -0.080250 (ablated - full).
- logistic - F1: full mean 1.000000; paired delta -0.079564 (ablated - full).

## Fixed-test comparison

The test rows below were generated after fitting each pipeline on all training rows. They are exploratory because Phase 1 evaluated this same test set for all 146 candidate configurations; they were not used to select the ablation feature set or parameters.

- knn - Recall: full 1.000000; test delta -0.270364 (ablated - full).
- knn - F1: full 1.000000; test delta -0.156313 (ablated - full).
- logistic - Recall: full 1.000000; test delta -0.114385 (ablated - full).
- knn - Accuracy: full 1.000000; test delta -0.078000 (ablated - full).
- perceptron - Recall: full 1.000000; test delta -0.076256 (ablated - full).

These changes describe predictive performance on this dataset. They do not establish causal effects, clinical utility, clinical diagnostic validity, or early-prediction capability.
