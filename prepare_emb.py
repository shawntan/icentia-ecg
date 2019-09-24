#!/usr/bin/env python3
import numpy as np
import sys,os
import pickle, gzip
import pandas as pd
import gc, gzip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data')
parser.add_argument('-labels_file', nargs='?', default="test_labels.csv.gz")
parser.add_argument('-embeddings_file', nargs='?', default="test_emb.csv.gz")
parser.add_argument('-segment_length', nargs='?', type=int, default=1000000)
parser.add_argument('-frame_length', nargs='?', type=int, default=2**11+1)
args = parser.parse_args()

test_labels = pd.read_csv(args.labels_file)
test_labels.set_index("id", inplace=True)

filename = ""
labels = None
results = []

#f= open("test_emb.csv","w+")
f= gzip.open(args.embeddings_file, 'wt')
f.write("sample, segment, frame," + ",".join(map(str, range(args.frame_length))) + " \n")

for index, row in test_labels.iterrows():
    new_filename = args.data + "/" + ("%05d" % row["sample"]) + "_batched.pkl.gz"
    if filename != new_filename: # do fancy caching
        print(new_filename)
        filename = new_filename
        data = pickle.load(gzip.open(new_filename))
    
    # get frame which should be 2049 with the center at the frame index
    input_from = row["frame"]-(int(args.frame_length/2))
    input_to = row["frame"]+(int(args.frame_length/2))+1
    input_seq = data[row["segment"]][input_from:input_to]

    # compute the embedding
    
    emb = np.copy(input_seq) # baseline
    
    f.write(str(row["sample"]) + "," + 
            str(row["segment"]) + "," + 
            str(row["frame"]) + "," +
            ','.join(emb.astype("float32").astype("str")) + 
            "\n")
    
    if index % 1000 == 0:
        gc.collect()
    
f.close()
