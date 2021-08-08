import pandas as pd

df = pd.read_csv('./datasets/esol.txt', delimiter=",")

print(df.head())