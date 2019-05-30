import numpy as np
import sys,os
import pickle, gzip
import pandas as pd


test_labels = pd.read_csv("test_labels.csv")
test_labels.set_index("id", inplace=True)

frame_length = 2**11+1
filename = ""
labels = None
results = []
for index, row in test_labels.iterrows():
    new_filename = "/home/cohenjos/projects/rpp-bengioy/jpcohen/icentia12k/" + str(row["sample"]) + "_batched.pkl.gz"
    if filename != new_filename: # do fancy caching
        print(new_filename)
        filename = new_filename
        data = pickle.load(gzip.open(new_filename))
    
    # get frame which should be 2049 with the center at the frame index
    input_from = row["frame"]-(int(frame_length/2))
    input_to = row["frame"]+(int(frame_length/2))+1
    input_seq = data[row["segment"]][input_from:input_to]

    # compute the embedding
    
    emb = input_seq # baseline
    
    results.append([row["sample"],row["segment"],row["frame"],*emb])
    
    
results_df = pd.DataFrame(results, columns=["sample", "segment", "frame", *[""]*(len(results[0])-3)])
results_df.index.name = "id"

results_df.to_csv("test_emb.csv")

