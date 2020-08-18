import numpy as np
import gzip
import pandas as pd
import sys,os
import hashlib
import pickle
import os 

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

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
              embeddings_file=os.path.join(dir_path,"test_emb.csv.gz"), 
              labels_file=os.path.join(dir_path,"test_labels.csv.gz"),
              balanced=None):
    args = locals()
    
    if not args.pop("cache", None): # pop and remove from dict
        return getSubsetCore(**args)
    
    subset = None
    
    args_hash = hashlib.md5(str.encode(str(args))).hexdigest()
    args_file = hashlib.md5(str.encode(str(os.path.getsize(embeddings_file)) + str(os.path.getmtime(embeddings_file)))).hexdigest()
    filename = dir_path + "/.cache/{}_{}_{}.pkl.gz".format(num_samples, args_hash, args_file)
    
    if os.path.exists(filename):
        print("Loading cache {}".format(filename))
        subset = pickle.load(gzip.GzipFile(filename, 'rb'))

    if subset is None:
        print("Building cache {}".format(filename))
        if not os.path.exists(dir_path + "/.cache"):
            os.makedirs(dir_path + "/.cache")
    
        subset = getSubsetCore(**args)
        pickle.dump(subset, gzip.GzipFile(filename, 'wb') , protocol=2)
        
    return subset
    
def getSubsetFile(filename):
    return pickle.load(gzip.GzipFile(filename, 'rb'))
    
# get a subset of the data without reading every line into memory.
def getSubsetCore(num_samples,seed,embeddings_file,labels_file,balanced):
    
    
    labels_raw = pd.read_csv(labels_file)
    #lines_emb = lines_in_file(labels_file) # get number to pick from
    
    # convert to small number
    labels_raw = labels_raw.astype("int32")
    labels_raw["btype"] = labels_raw["btype"].values.astype("int8")
    labels_raw["rtype"] = labels_raw["rtype"].values.astype("int8")
    
    # filter for only the targets we care about
    btype_targets = [1,2,4]
    rtype_targets = [3,4,5]
    
    # mark these to not use
    labels_raw.loc[~labels_raw["btype"].isin(btype_targets),"btype"] = -1
    labels_raw.loc[~labels_raw["rtype"].isin(rtype_targets),"rtype"] = -1
    
    # relabel
    for new_class, old_class in enumerate(btype_targets):
        labels_raw.loc[labels_raw["btype"] == old_class, "btype"] = new_class
        
    for new_class, old_class in enumerate(rtype_targets):
        labels_raw.loc[labels_raw["rtype"] == old_class, "rtype"] = new_class
    
    np.random.seed(seed)
    
    if balanced != None:
        assert balanced in ["btype","rtype"], balanced
        n_samples_class = num_samples//len(labels_raw[balanced].unique())
        tosample = []
        np.random.seed(seed)
        for clazz in labels_raw[balanced].unique():
            if clazz != -1:
                urn = np.where(labels_raw[balanced] == clazz)[0]
                tosample.append(np.random.choice(urn,n_samples_class,replace=False))
        tosample = np.concatenate(tosample)
    else:
        tosample = np.random.choice(len(labels_raw), num_samples, replace=False)
        
    #print(labels_raw.iloc[tosample]["btype"].value_counts())
    
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

    create_index(labels_raw)
    
    #print(labels_raw.iloc[tosample]["btype"].value_counts())
    
    # order labels by data (this also ensures the code above was correct)
    labels = labels_raw.loc[data.index]
    
    return data, labels#, labels_raw, tosample


btype_names = {
    0:"Normal",
    1:"ESSV (PAC)",
    2:"ESV (PVC)"
}
rtype_names = {
    0:"NSR",
    1:"AFib",
    2:"AFlutter"
}

# btype_names_raw = {
#     0:"0 Undefined",
#     1:"1 Normal",
#     2:"2 ESSV (PAC)",
#     3:"3 Aberrated",
#     4:"4 ESV (PVC)"
# }
# rtype_names_raw = {
#     0:"0 Null/Undefined",
#     1:"1 End (essentially noise)",
#     2:"2 Noise",
#     3:"3 NSR (normal sinusal rhythm)",
#     4:"4 AFib",
#     5:"5 AFlutter"
# }
