import numpy as np
import gzip
import pandas as pd
import sys,os

def create_index(df):
    # create index col and remove source columns
    df["id"] = df.apply(lambda row: str(int(row["sample"])) + "_" + str(int(row["segment"])) + "_" + str(int(row["frame"])), axis=1)
    df.set_index("id", inplace=True)

def getSubset(num_samples, lines_emb=420493, seed=0):
    np.random.seed(seed)
    tosample = np.random.choice(lines_emb, num_samples, replace=False)
    subset = []
    with gzip.open("test_emb.csv.gz", 'rb') as f:
        header = f.readline().decode('ascii').replace("\n","")
        
        for i, line in enumerate(f):
            if (i in tosample):
                subset.append(line.decode('ascii').replace("\n","").split(","))
    aheader = header.replace(" ","").split(",")
    aheader = aheader[:len(subset[0])] # patch to deal with incorrect length of header
    data = pd.DataFrame(subset, columns=aheader)
    create_index(data)
    
    metadata = data.iloc[:,:3]
    data = data.iloc[:,3:].astype("float32")
    
    subset = []
    with gzip.open("test_labels.csv.gz", 'rb') as f:
        header = f.readline().decode('ascii').replace("\n","")
        
        for i, line in enumerate(f):
            if (i in tosample):
                subset.append(line.decode('ascii').replace("\n","").split(","))
    labels = pd.DataFrame(subset, columns=header.replace(" ","").split(","))
    create_index(labels)
    
    # order by labels
    labels = labels.loc[data.index]
    
    return metadata, data, labels
