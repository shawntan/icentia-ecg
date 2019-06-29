import argparse
import numpy as np
import matplotlib.pylab as plt
import sys
import pandas as pd
import sklearn, sklearn.model_selection, sklearn.neighbors
import gzip
import utils


parser = argparse.ArgumentParser()
parser.add_argument('embeddings_file', help='File with embeddings')
parser.add_argument('num_train_examples', nargs='?', type=int, default=5000, help='')
parser.add_argument('num_test_examples', nargs='?', type=int, default=5000, help='')
parser.add_argument('num_trials', nargs='?', type=int, default=10, help='')
parser.add_argument('labels_file', nargs='?', default="test_labels.csv.gz", help='')
args = parser.parse_args()

print(args)

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

def evaluate(num_train_examples, num_test_examples, num_trials):
    
    all_acc = []
    for i in range(num_trials):
        
        print("Generating subset", i)
        metadata, data, labels = utils.getSubset(num_train_examples+num_test_examples, embeddings_file=args.embeddings_file)
        
        import collections
        print(collections.Counter(labels))
        X, X_test, y, y_test = \
            sklearn.model_selection.train_test_split(data, labels, 
                                                     train_size=num_train_examples, 
                                                     test_size=num_test_examples, 
                                                     stratify=labels,
                                                     random_state=i)
        print("X", X.shape, "X_test", X_test.shape)
        model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        model = model.fit(X, y.values.flatten())

        acc = (model.predict(X_test) == y_test.values.flatten()).mean()
        all_acc.append(acc)
        
        print("   Run " + str(i) + ", Accuracy:",acc) 

    return np.asarray(all_acc).mean(), np.asarray(all_acc).std()
    
    
    
mean,stdev = evaluate(args.num_train_examples, args.num_test_examples, args.num_trials)

print("Accuracy:",round(mean,3), "+-", round(stdev,3), "num_trials:",args.num_trials) 

    
    
    
    
    
    
