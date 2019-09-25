import numpy as np
import sys,os,random
import pickle,gzip
import pandas as pd
from tqdm import tqdm
import argparse,distutils.util
import collections

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path')
parser.add_argument('-labels_file', nargs='?', default="test_labels.csv.gz")
parser.add_argument('-segment_length', nargs='?', type=int, default=1000000)
parser.add_argument('-frame_length', nargs='?', type=int, default=2**11+1)
parser.add_argument('-num_segments', nargs='?', type=int, default=50)
parser.add_argument('-max_idx_per_label', nargs='?', type=int, default=1)
parser.add_argument('-start_idx', nargs='?', type=int, default=9000)
parser.add_argument('-end_idx', nargs='?', type=int, default=11000)
parser.add_argument('-relabel', nargs='?', type=lambda x:bool(distutils.util.strtobool(x)), default=True)
args = parser.parse_args()

print(args)

def build_label_dict(segment, label_type):
    toreturn = dict()
    for label, idxs in enumerate(segment[label_type]):
        for idx in idxs:
            toreturn[idx] = label
    return toreturn

def extract_labels(sample_id, segment_id, segment_labels, test_labels):
    
        np.random.seed(0)
        random.seed(0)
        
        btype = build_label_dict(segment_labels, "btype")
        rtype = build_label_dict(segment_labels, "rtype")
        
        all_idx_toselect = []
        
        for label, idx_toselect in enumerate(segment_labels["btype"]):

            # remove idxs that don't have rtype labeled
            idx_toselect = list(filter(lambda x: x in rtype, idx_toselect))
            
            # filter below 1/2 frame
            idx_toselect = list(filter(lambda x: x >(args.frame_length/2), idx_toselect))

            # filter above length-1/2 frame
            idx_toselect = list(filter(lambda x: x <(args.segment_length-(args.frame_length/2)), idx_toselect))

            # keep number of samples per label to less than max_idx_per_label
            if len(idx_toselect) > args.max_idx_per_label:
                idx_toselect = np.random.choice(idx_toselect, args.max_idx_per_label, replace=False)
                
            all_idx_toselect += list(idx_toselect)
                
        for label, idx_toselect in enumerate(segment_labels["rtype"]):

            # remove idxs that don't have btype labeled
            idx_toselect = list(filter(lambda x: x in btype, idx_toselect))
            
            # filter below 1/2 frame
            idx_toselect = list(filter(lambda x: x >(args.frame_length/2), idx_toselect))

            # filter above length-1/2 frame
            idx_toselect = list(filter(lambda x: x <(args.segment_length-(args.frame_length/2)), idx_toselect))

            # keep number of samples per label to less than max_idx_per_label
            if len(idx_toselect) > args.max_idx_per_label:
                idx_toselect = np.random.choice(idx_toselect, args.max_idx_per_label, replace=False)
            
            all_idx_toselect += list(idx_toselect)

        for idx in np.unique(all_idx_toselect):
            this_btype = btype[idx]
            if args.relabel:
                if 1 == btype[idx]:
                    for i in range(idx-args.frame_length//2, idx+args.frame_length//2):
                        if i in btype and btype[i] in [2,4]:
                            this_btype = btype[i]
                if btype[idx] != this_btype:
                    print("A normal was relabeled as {}".format(this_btype))
            
            test_label = [sample_id, segment_id, idx, this_btype, rtype[idx]]
            test_labels.append(test_label)

test_labels = []
for sample_id in range(args.start_idx, args.end_idx): # range of text examples
    filename = os.path.join(args.dataset_path, ("%05d" % sample_id) + "_batched_lbls.pkl.gz")
    print("{}/{} ".format(sample_id, args.end_idx) + filename)
    if (not os.path.isfile(filename)):
        print("##### File missing")
        continue

    segments = pickle.load(gzip.open(filename))
    for segment_id, segment_labels in enumerate(segments):
        extract_labels(sample_id, segment_id, segment_labels, test_labels=test_labels)
    
data = pd.DataFrame(test_labels, columns=["sample","segment","frame", "btype", "rtype"])
data.index.name = "id"
print("btype stats:",collections.Counter(data.btype))
print("rtype stats:",collections.Counter(data.rtype))
data.to_csv(args.labels_file)

# data_rtype = pd.DataFrame(test_labels_rtype, columns=["sample","segment","frame", "label"])
# data_rtype.index.name = "id"
# print("data_rtype stats:",collections.Counter(data_rtype.label))
# data_rtype.to_csv("test_labels_rtype.csv.gz")

print("Done")
