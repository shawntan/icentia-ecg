import numpy as np
import gzip
import pandas as pd
import sys,os

def create_index(df, remove_meta=False):
    # create index col and remove source columns
    df["id"] = df.apply(lambda row: str(int(row["sample"])) + "_" + str(int(row["segment"])) + "_" + str(int(row["frame"])), axis=1)
    
    if remove_meta:
        del df["sample"]
        del df["segment"]
        del df["frame"]
    df.set_index("id", inplace=True)

# get a subset of the data without reading every line into memory.
def getSubset(num_samples, lines_emb=420493, seed=0, 
              embeddings_file="test_emb.csv.gz", 
              labels_file="test_labels.csv.gz"):
    
    np.random.seed(seed)
    tosample = np.random.choice(lines_emb, num_samples, replace=False)
    subset = []
    with gzip.open(embeddings_file, 'rb') as f:
        header = f.readline().decode('ascii').replace("\n","")
        
        for i, line in enumerate(f):
            if (i in tosample):
                a = line.decode('ascii').replace("\n","").split(",")
                a = [float(i) for i in a] #convert here for memory
                subset.append(a)
                
    aheader = header.replace(" ","").split(",")
    aheader = aheader[:len(subset[0])] # patch to deal with incorrect length of header
    data = pd.DataFrame(subset, columns=aheader)
    create_index(data, remove_meta=True)
    
    data = data.astype("float32")
    
    subset = []
    with gzip.open(labels_file, 'rb') as f:
        header = f.readline().decode('ascii').replace("\n","")
        
        for i, line in enumerate(f):
            if (i in tosample):
                subset.append(line.decode('ascii').replace("\n","").split(","))
    labels = pd.DataFrame(subset, columns=header.replace(" ","").split(","))
    create_index(labels)
    
    # order by labels
    labels = labels.loc[data.index]
    
    return data, labels
