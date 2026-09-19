from sklearn.linear_model import LogisticRegression
import pandas as pd

from preprocessing_and_cv import (
    load_datasets,
    build_pipeline,
    evaluate_pipelines,
)

X_train, y_train, X_test, y_test = load_datasets(
    "training_data.csv",
    "testing_data.csv",
)

LR_pipeline = build_pipeline(
    LogisticRegression(
        C=0.1,
        max_iter=2000,
    ),
)

results = evaluate_pipelines(
    pipelines={"Logistic Regression": LR_pipeline},
    X_train=X_train,
    y_train=y_train,
    cv_splits=5,
    random_state=42,
    n_jobs=-1,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

print(results.to_string(index=False))