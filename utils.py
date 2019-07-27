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

def lines_in_file(embeddings_file):
    with gzip.open(embeddings_file, 'rb') as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# does some caching
def getSubset(num_samples, cache=True, seed=0, 
              embeddings_file="test_emb.csv.gz", 
              labels_file="test_labels.csv.gz"):
    args = locals()
    
    if not args.pop("cache", None): # pop and remove from dict
        return getSubsetCore(**args)
    
    subset = None
    args_hash = hash(str(args))
    args_file = hash(str(os.path.getsize(embeddings_file)) + str(os.path.getmtime(embeddings_file)))
    filename = ".cache/{}_{}.pkl.gz".format(args_hash, args_file)
    
    import pickle
    
    if os.path.exists(filename):
        print("Loading cache {}".format(filename))
        subset = pickle.load(gzip.GzipFile(filename, 'rb'))

    if subset is None:
        print("Building cache {}".format(filename))
        if not os.path.exists(".cache"):
            os.makedirs(".cache")
    
        subset = getSubsetCore(**args)
        pickle.dump(subset, gzip.GzipFile(filename, 'wb') , protocol=2)
        
    return subset
    
# get a subset of the data without reading every line into memory.
def getSubsetCore(num_samples,seed,embeddings_file,labels_file):
    
    lines_emb = lines_in_file(labels_file) # get number to pick from
    
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
    
    # convert to small number
    labels = labels.astype("int32")
    labels["btype"] = labels["btype"].values.astype("int8")
    labels["rtype"] = labels["rtype"].values.astype("int8")
    
    return data, labels


btype_names = {
    0:"0 Undefined",
    1:"1 Normal",
    2:"2 ESSV (PAC)",
    3:"3 Aberrated",
    4:"4 ESV (PVC)"
}
rtype_names = {
    0:"0 Null/Undefined",
    1:"1 End (essentially noise)",
    2:"2 Noise",
    3:"3 NSR (normal sinusal rhythm)",
    4:"4 AFib",
    5:"5 AFlutter"
}
