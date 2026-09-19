import pandas as pd

train = pd.read_csv("training_data.csv", header=None)
test = pd.read_csv("testing_data.csv", header=None)

all_columns = list(train.columns)
feature_columns = all_columns[:-1]
target_column = all_columns[-1]

# 完全相同的记录：特征和Diagnosis都相同
exact_duplicates = train.merge(
    test,
    on=all_columns,
    how="inner"
).drop_duplicates()

# 特征完全相同的记录，不考虑Diagnosis
feature_duplicates = train.merge(
    test,
    on=feature_columns,
    how="inner",
    suffixes=("_train", "_test")
)

# 特征相同但Diagnosis不同
conflicting_duplicates = feature_duplicates[
    feature_duplicates[f"{target_column}_train"]
    != feature_duplicates[f"{target_column}_test"]
]

print("Training rows:", len(train))
print("Testing rows:", len(test))
print(
    "Exact duplicate rows across training and testing:",
    len(exact_duplicates)
)
print(
    "Feature-identical cross-set matches:",
    len(feature_duplicates)
)
print(
    "Feature-identical but label-conflicting matches:",
    len(conflicting_duplicates)
)

positive_col = 12
negative_col = 13
gaf_col = 14
diagnosis_col = train.columns[-1]

for column in [positive_col, negative_col, gaf_col]:
    summary = train.groupby(diagnosis_col)[column].agg(
        ["count", "min", "max", "mean", "std", "median"]
    )

    print(f"\nColumn {column}")
    print(summary)