import numpy as np
import sys,os
import pickle, gzip
import pandas as pd
from tqdm import tqdm

def create_index(df):
    # create index col and remove source columns
    df["id"] = df.apply(lambda row: str(int(row[0])) + "_" + str(int(row[1])) + "_" + str(int(row[2])), axis=1)
#     del df["sample"]
#     del df["segment"]
#     del df["frame"]
    df.set_index("id", inplace=True)

data = pd.read_csv(sys.argv[1])
create_index(data)

nsamples = 100
print("Taking subset to train PCA nsamples=", nsamples)
subset = np.random.choice(range(len(data)), nsamples, replace=False)
data_subset = data.iloc[subset]

print("Computing PCA")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(data_subset.values[:,4:])


results = []

for i, row in tqdm(data.iterrows()):
    
    toemb = row.values[4:]
    iemb = pca.transform([toemb])
    
    
    results.append([int(row["sample"]),
                    int(row["segment"]),
                    int(row["frame"]),*iemb[0].astype("float32")])
    
print("Creating dataframe")
results_df = pd.DataFrame(results, columns=["sample", "segment", "frame", *[""]*(len(results[0])-3)])
results_df.index.name = "id"

print("Writing CSV")
results_df.to_csv("test_emb_pca.csv")

