import numpy as np
import sys,os
import pickle, gzip
import pandas as pd
import matplotlib,matplotlib.pyplot as plt
import utils
import encoders
import glob
from sklearn.decomposition import PCA
from tqdm import tqdm
import gc
import data_io
import random

if __name__ == "__main__":
    directory = sys.argv[1]

    frame_length = 2**11 + 1
    patient_samples = 10
    context = int(frame_length / 2)

    filenames = (glob.glob(directory + "/?_batched.pkl.gz") +\
                 glob.glob(directory + "/??_batched.pkl.gz") +\
                 glob.glob(directory + "/???_batched.pkl.gz") +\
                 glob.glob(directory + "/????_batched.pkl.gz"))
    random.shuffle(filenames)

    data_raw = np.empty((len(filenames) * patient_samples, frame_length))
    i = 0

    def _loaded_files():
        for fname in tqdm(filenames):
            # print("Loading", fname)
            yield pickle.load(gzip.open(fname, 'rb'))

    data_stream = data_io.threaded(_loaded_files(), queue_size=20)


    for data in data_stream:
        for _ in range(patient_samples):
            segment = np.random.randint(data.shape[0])
            frame = np.random.randint(data.shape[1] - 2 * context) + context
            input_from = frame - context
            input_to = frame + context + 1
            input_seq = data[segment, input_from:input_to]
            data_raw[i] = input_seq
            i += 1
        if i % 1000 == 0:
            gc.collect()
    data_raw = data_raw[:i]
    pickle.dump(data_raw, open('data.pkl', 'wb'))
    pca = PCA(n_components=100)
    pca.fit(data_raw)

    pickle.dump((pca.mean_, pca.components_), open("pca.pkl.gz","bw"), protocol=2)
