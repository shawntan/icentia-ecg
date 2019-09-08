import numpy as np
import sys,os
import pickle, gzip
import pandas as pd
import matplotlib,matplotlib.pyplot as plt
import utils
import encoders

data_raw, labels = utils.getSubset(30000, embeddings_file="train_emb.csv.gz",labels_file="train_labels.csv.gz")

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
pca.fit(data_raw.values)

pickle.dump((pca.mean_, pca.components_), open("pca.pkl.gz","bw"), protocol=2)
