import numpy as np
import sys,os
import pickle, gzip
import pandas as pd
from tqdm import tqdm
import gc, gzip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('embeddings_file', help='File with embeddings')
parser.add_argument('target_file', nargs='?', type=str, default='test_emb_pca.csv.gz', help='')
parser.add_argument('pca_nsamples', nargs='?', type=int, default=1000, help='')
parser.add_argument('pca_dim', nargs='?', type=int, default=2, help='')
args = parser.parse_args()
print(args)

def create_index(df):
    # create index col and remove source columns
    df["id"] = df.apply(lambda row: str(int(row["sample"])) + "_" + str(int(row["segment"])) + "_" + str(int(row["frame"])), axis=1)
#     del df["sample"]
#     del df["segment"]
#     del df["frame"]
    df.set_index("id", inplace=True)


np.random.seed(0)
def getSubset(num_samples):
    tosample = np.random.choice(lines_emb, num_samples, replace=False)
    subset = []
    with gzip.open(args.embeddings_file, 'rb') as f:
        header = f.readline().decode('ascii')
        
        for i, line in enumerate(f):
            if (i in tosample):
                subset.append(line.decode('ascii').split(","))
    data = pd.DataFrame(subset, columns=header.replace(" ","").split(","))
    create_index(data)
    
    return data

## get counts
lines_emb = 0
with gzip.open(args.embeddings_file, 'rb') as f:
    for line in f:
        lines_emb += 1

print("Taking subset to train PCA nsamples=", args.pca_nsamples)
data = getSubset(args.pca_nsamples)

print("Computing PCA")
from sklearn.decomposition import PCA
pca = PCA(n_components=args.pca_dim)
pca.fit(data.values[:,4:])


results = []

ft= gzip.open(args.target_file, 'wt')
#ft= open('test_emb_pca.csv', 'wt')
ft.write("sample, segment, frame," + ",".join(map(str, range(args.pca_dim))) + " \n")
with gzip.open(args.embeddings_file, 'rb') as f:
        header = f.readline().decode('ascii').replace("\n","").split(",")

        for i, line in tqdm(enumerate(f)):
            row = line.decode('ascii').replace("\n","").split(",")
            
            toemb = np.asarray(row[4:],dtype="float32")
            iemb = pca.transform([toemb])
            
            ft.write(str(row[0]) + "," + 
                str(row[1]) + "," + 
                str(row[2]) + "," +
                ','.join(iemb[0].astype("float32").astype("str")) + 
                "\n")
            gc.collect()

