from sklearn.neighbors import KNeighborsClassifier

from preprocessing_and_cv import (
    load_datasets,
    build_pipeline,
    evaluate_pipelines,
)

X_train, y_train, X_test, y_test = load_datasets(
    "training_data.csv",
    "testing_data.csv",
)

knn_pipeline = build_pipeline(
    KNeighborsClassifier(
        n_neighbors=21,
        metric="manhattan",
    )
)

results = evaluate_pipelines(
    pipelines={"k-NN": knn_pipeline},
    X_train=X_train,
    y_train=y_train,
    cv_splits=5,
    random_state=42,
    n_jobs=-1,
)

print(results)