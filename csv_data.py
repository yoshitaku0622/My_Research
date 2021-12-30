import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# csv読み込み
df = pd.read_csv("./governer_data.csv")
train_inds, test_inds = next(GroupShuffleSplit(test_size=0.1, n_splits=2, random_state=7).split(df, groups=df["patient_id"]))
train, valid = df.iloc[train_inds], df.iloc[test_inds]

print(train)
print(valid)