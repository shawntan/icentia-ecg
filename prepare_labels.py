import numpy as np
import sys,os
import pickle, gzip
import pandas as pd


segment_length = 1000000
frame_length = 2**11+1

num_segments = 50
num_labels = 5
max_idx_per_label = 20

test_labels = []
for i in range(10001, 12971): # range of text examples
    filename = "/home/cohenjos/projects/rpp-bengioy/jpcohen/icentia12k/" + str(i) + "_batched_lbls.pkl.gz"
    print(filename)
    labels = pickle.load(gzip.open(filename))
    for segment in range(num_segments):
        for label in range(num_labels):
            idx_toselect = labels[segment][label]
            
            # filter below 1/2 frame
            idx_toselect = list(filter(lambda x: x >(frame_length/2), idx_toselect))
            
            # filter above length-1/2 frame
            idx_toselect = list(filter(lambda x: x <(segment_length-(frame_length/2)), idx_toselect))
            
            # keep number of samples per label to less than 20
            if len(idx_toselect) > max_idx_per_label:
                idx_toselect = np.random.choice(labels[0][0], max_idx_per_label, replace=False)
            
            for idx in idx_toselect:
                test_label = [i, segment, idx, label]
                test_labels.append(test_label)
                
                
                
data = pd.DataFrame(test_labels, columns=["sample","segment","frame", "label"])
data.index.name = "id"

import collections
print("label stats:",collections.Counter(data.label))

data.to_csv("test_labels.csv")