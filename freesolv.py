import pandas as pd

df = pd.read_csv('./datasets/freesolv.txt', delimiter="; ", engine="python")
to_drop = [
    "experimental reference (original or paper this value was taken from)", 
    "calculated reference",
    "text notes."
]
df.drop(to_drop, inplace=True, axis=1)

new_names = {
    "compound id (and file prefix)": "compound id",
    "iupac name (or alternative if IUPAC is unavailable or not parseable by OEChem)": "iupac name"
}
df.rename(columns=new_names, inplace=True)

regex = r'(\d{7})'
extr = df["compound id"].str.extract(regex, expand=False)
df['compound id'] = pd.to_numeric(extr, downcast='unsigned') # not working

df.set_index("compound id", inplace=True)

print(df.head())