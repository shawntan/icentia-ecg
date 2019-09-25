import numpy as np
import sys,os,random
import pickle,gzip
import pandas as pd
from tqdm import tqdm
import argparse,distutils.util
import collections

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path')
parser.add_argument('-start_idx', nargs='?', type=int, default=0)
parser.add_argument('-end_idx', nargs='?', type=int, default=11000)
args = parser.parse_args()

print(args)

def count_labels(sample_id, segment_id, segment_labels):
    return sum([c.shape[0] for c in segment_labels['btype']])

total_beats = 0 
total_segments = 0
total_frames = 0
for sample_id in range(args.start_idx, args.end_idx): # range of text examples
    filename = os.path.join(args.dataset_path, ("%05d" % sample_id) + "_batched_lbls.pkl.gz")
    print("{}/{} ".format(sample_id, args.end_idx) + filename)
    
    if (not os.path.isfile(filename)):
        print("##### File missing")
        continue

    segments = pickle.load(gzip.open(filename))
    total_segments += len(segments)
    for segment_id, segment_labels in enumerate(segments):
        total_frames += len(segment_labels)
        total_beats += count_labels(sample_id, segment_id, segment_labels)
        
    print("total_beats:", total_beats, "total_frames:", total_frames, "total_segments:", total_segments)


print("Done")
