import numpy as np
import sys,os
import pickle, gzip
import pandas as pd
from tqdm import tqdm
import gc, gzip
import argparse
import utils
import encoders
from inspect import getmembers, isfunction

parser = argparse.ArgumentParser()
parser.add_argument('encode_method', type=str, choices=[o for o in dir(encoders) if not o.startswith("_")])
parser.add_argument('embeddings_file', nargs='?', default='test_emb.csv.gz', type=str, help='File with embeddings')
args = parser.parse_args()
print(args)

enc = getattr(encoders, args.encode_method)()

target_file = "test_emb_" + args.encode_method + ".csv.gz"

ft= gzip.open(target_file, 'wt')
#ft= open('test_emb_pca.csv', 'wt')
needs_header = True
emb_length = None

with gzip.open(args.embeddings_file, 'rb') as f:
        header = f.readline().decode('ascii').replace("\n","").split(",")

        for i, line in tqdm(enumerate(f)):
            row = line.decode('ascii').replace("\n","").split(",")
            
            toemb = np.asarray(row[3:],dtype="float32")
            
            try:
                iemb = enc.encode(toemb)
            except ValueError as e:
                print(e)
                iemb = np.zeros(emb_length)
            
            if needs_header:
                emb_length = len(iemb)
                print("emb length:",len(iemb)) 
                ft.write("sample, segment, frame," + ",".join(map(str, range(len(iemb)))) + " \n")
                needs_header = False
            
            ft.write(str(row[0]) + "," + 
                str(row[1]) + "," + 
                str(row[2]) + "," +
                ','.join(iemb.astype("float32").astype("str")) + 
                "\n")
            gc.collect()

