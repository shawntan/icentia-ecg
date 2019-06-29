#!/usr/bin/env python3
import numpy as np
import sys,os
import pickle, gzip
import pandas as pd
import gc, gzip
import model
import torch

enc = model.Autoencoder()
enc.eval()
def encode(x):
    x = torch.from_numpy(x)
    emb = enc.autoencode_1.encode(x[None, None, :])[0, :, 0].detach().numpy()
    return emb

test_labels = pd.read_csv("test_labels.csv.zip")
test_labels.set_index("id", inplace=True)

frame_length = 2**11+1
filename = ""
labels = None
results = []

#f= open("test_emb.csv","w+")
f= gzip.open('test_emb_enc.csv.gz', 'wt')
f.write("sample, segment, frame," + ",".join(map(str, range(frame_length))) + " \n")

for index, row in test_labels.iterrows():
    new_filename = sys.argv[1] + "/" + str(row["sample"]) + "_batched.pkl.gz"
    if filename != new_filename: # do fancy caching
        print(new_filename)
        filename = new_filename
        data = pickle.load(gzip.open(new_filename))

    # get frame which should be 2049 with the center at the frame index
    input_from = row["frame"]-(int(frame_length/2))
    input_to = row["frame"]+(int(frame_length/2))+1
    input_seq = encode(data[row["segment"]][input_from:input_to])

    # compute the embedding

    emb = np.copy(input_seq) # baseline

    f.write(str(row["sample"]) + "," + 
            str(row["segment"]) + "," + 
            str(row["frame"]) + "," +
            ','.join(emb.astype("float32").astype("str")) + 
            "\n")

    gc.collect()

f.close()
