import matplotlib.pyplot as plt
import pandas as pd
import wfdb
import itertools
from tqdm import tqdm
import json
import os
import collections

data_path = "/network/scratch/o/ortizgas/transit_datasets/icentia11k_wfdb/wfdb/"


cache_dir = "./index/"
os.makedirs(cache_dir, exist_ok=True)
def put(key, value):
    path = f'{cache_dir}/{key.replace("/","")}'
    json.dump(value, open(path, "w"))
    
def get(key):
    path = f'{cache_dir}/{key.replace("/","")}'
    if os.path.exists(path):
        return json.load(open(path,"r"))
    return False


print("Counting from files")
iterator = itertools.product(range(0,11000), range(0,50))
for patient_id, segment_id in tqdm(iterator, total=11000*50):
    key = f'p{patient_id:05d}_s{segment_id:02d}'
    path = f'p{patient_id:05d}/p{patient_id:05d}_s{segment_id:02d}'
    
    if not get(f'{key}'):
        filename = f'{data_path}/{path}'
        if os.path.exists(f'{filename}.atr'):
            ann = wfdb.rdann(filename, "atr")
            
            put(f'{key}', {
                'symbols': collections.Counter(ann.symbol),
                'aux_notes': collections.Counter(ann.aux_note)
            })


print("Gathering stats")
all_symbols = collections.Counter()
all_aux_notes = collections.Counter()
files = 0
iterator = itertools.product(range(0,11000), range(0,50))
for patient_id, segment_id in tqdm(iterator, total=11000*50):
    key = f'p{patient_id:05d}_s{segment_id:02d}'
    
    index = get(key)
    if index:
        files += 1

        all_symbols += collections.Counter(index["symbols"])
        all_aux_notes += collections.Counter(index["aux_notes"])

        
print("Total Files: ", files)
print("Symbols: ", all_symbols)
print("Aux Notes: ", all_aux_notes)
