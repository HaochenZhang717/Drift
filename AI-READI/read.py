import pandas as pd

df = pd.read_parquet("glucose_train.parquet")

print(df.head())
print(df.shape)