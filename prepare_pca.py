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
    filenames = [ directory + "/%05d_batched.pkl.gz" % i
                  for i in range(9000) ]
    random.shuffle(filenames)

    data_raw = np.empty((len(filenames) * patient_samples, frame_length))
    i = 0

    def _loaded_files():
        for fname in tqdm(filenames):
            # print("Loading", fname)
            yield pickle.load(gzip.open(fname, 'rb'))

    def load_file(fname):
        data = pickle.load(gzip.open(fname, 'rb'))
        segment = np.arange(data.shape[0])
        np.random.shuffle(segment)
        segment = segment[:patient_samples]
        segment = np.random.randint(data.shape[0])
        frame = np.random.randint(data.shape[1] - 2 * context) + context
        input_from = frame - context
        input_to = frame + context + 1
        input_seq = data[segment, input_from:input_to]
        return input_seq

    data_stream = data_io.multiprocess(filenames, load_file, worker_count=10)
    # data_stream = data_io.threaded(_loaded_files(), queue_size=20)


    for input_seq in tqdm(data_stream, total=len(filenames)):
        data_raw[i:i+patient_samples] = input_seq
        i += patient_samples
        if i % 1000 == 0:
            gc.collect()
    print("Done.")
    data_raw = data_raw[:i]
    pickle.dump(data_raw, open('data.pkl', 'wb'))

    pca = PCA(n_components=100)
    pca.fit(data_raw)
    pickle.dump((pca.mean_, pca.components_), open("pca_100.pkl.gz","bw"), protocol=2)

    pca = PCA(n_components=50)
    pca.fit(data_raw)
    pickle.dump((pca.mean_, pca.components_), open("pca_50.pkl.gz","bw"), protocol=2)

    pca = PCA(n_components=10)
    pca.fit(data_raw)
    pickle.dump((pca.mean_, pca.components_), open("pca_10.pkl.gz","bw"), protocol=2)
