import argparse
import numpy as np
import matplotlib.pylab as plt
import sys
import pandas as pd
import sklearn, sklearn.model_selection, sklearn.neighbors
import sklearn.linear_model
import gzip
import utils
import encoders
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('embeddings_file', help='File with embeddings')
parser.add_argument('num_examples', nargs='?', type=int, default=5000, help='')
parser.add_argument('num_trials', nargs='?', type=int, default=10, help='')
parser.add_argument('labels_file', nargs='?', default="test_labels.csv.gz", help='')
parser.add_argument('-model', type=str, default="knn", choices=["knn", "lr"],help='Model to evaluate embeddings with.')
parser.add_argument('-encode_method', type=str, default=None, choices=[o for o in dir(encoders) if not o.startswith("_")], help='to encode the signals on the fly')
args = parser.parse_args()

print(args)

enc = None
if args.encode_method != None:
    enc = getattr(encoders, args.encode_method)()
    print("Encoder:",enc)

## get counts
lines_emb = 0
with gzip.open(args.embeddings_file, 'rb') as f:
    for line in f:
        lines_emb += 1

lines_labels = 0
with gzip.open(args.labels_file, 'rb') as f:
    for line in f:
        lines_labels += 1

print("lines_emb:", lines_emb)

if lines_labels != lines_emb:
    print(" !! Issue with coverage of labels. The data must align to the labels.")
    sys.exit()

def evaluate(num_examples, num_trials):
    
    all_acc = []
    for i in range(num_trials):
        
        print("Generating subset", i)
        
        data, labels = utils.getSubset(num_examples, embeddings_file=args.embeddings_file)
        
        # remove class 0
        data = data[labels["label"] != 0]
        labels = labels[labels["label"] != 0]
        
        if enc:
            newdata = []
            for emb in tqdm(data.values):
                newdata.append(enc.encode(emb))
            data = np.asarray(newdata)
        
        import collections
        print(collections.Counter(labels["label"]))

        X, X_test, y, y_test = \
            sklearn.model_selection.train_test_split(data, labels["label"], 
                                                     train_size=len(labels)//2, 
                                                     test_size=len(labels)//2, 
                                                     stratify=labels["label"],
                                                     random_state=i)
        print("X", X.shape, "X_test", X_test.shape)
        if args.model == "knn":
            model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        elif args.model == "lr":
            model = sklearn.linear_model.LogisticRegression(multi_class="auto")
        else:
            print("Unknown model")
            sys.exit();
            
        print(model)
        model = model.fit(X, y.values.flatten())

        acc = (model.predict(X_test) == y_test.values.flatten()).mean()
        all_acc.append(acc)
        
        print("   Run " + str(i) + ", Accuracy:",acc) 

    return np.asarray(all_acc).mean(), np.asarray(all_acc).std()
    
    
    
mean,stdev = evaluate(args.num_examples, args.num_trials)

print("Accuracy:",round(mean,3), "+-", round(stdev,3), "num_trials:",args.num_trials, args) 

    
    
    
    
    
    
