import numpy as np
import sys,os
import pickle, gzip
import pandas as pd
from tqdm import tqdm
import gc, gzip
import argparse
import scipy.fftpack

parser = argparse.ArgumentParser()
parser.add_argument('embeddings_file', help='File with embeddings')
parser.add_argument('target_file', nargs='?', type=str, default='test_emb_rand.csv.gz', help='')
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

ft= gzip.open(args.target_file, 'wt')
#ft= open('test_emb_pca.csv', 'wt')
ft.write("sample, segment, frame," + ",".join(map(str, range(1024))) + " \n")
with gzip.open(args.embeddings_file, 'rb') as f:
        header = f.readline().decode('ascii').replace("\n","").split(",")

        for i, line in tqdm(enumerate(f)):
            row = line.decode('ascii').replace("\n","").split(",")
            
            toemb = np.asarray(row[3:],dtype="float32")
            
            # just make it random to test pipeline
            iemb = np.random.rand(2)
            
            ft.write(str(row[0]) + "," + 
                str(row[1]) + "," + 
                str(row[2]) + "," +
                ','.join(iemb.astype("float32").astype("str")) + 
                "\n")
            gc.collect()

