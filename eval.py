import argparse
import numpy as np
import matplotlib.pylab as plt
import sys
import pandas as pd
import sklearn, sklearn.model_selection, sklearn.neighbors
import gzip


parser = argparse.ArgumentParser()
parser.add_argument('embeddings_file', help='File with embeddings')
parser.add_argument('num_train_examples', nargs='?', type=int, default=1000, help='')
parser.add_argument('num_trials', nargs='?', type=int, default=10, help='')
parser.add_argument('labels_file', nargs='?', default="test_labels.csv.gz", help='')
args = parser.parse_args()

print(args)

def create_index(df):
    # create index col and remove source columns
    df["id"] = df.apply(lambda row: str(int(row[0])) + "_" + str(int(row[1])) + "_" + str(int(row[2])), axis=1)
    del df["sample"]
    del df["segment"]
    del df["frame"]
    df.set_index("id", inplace=True)


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
    
        
np.random.seed(0)
def getSubset(num_samples):
    tosample = np.random.choice(lines_emb, num_samples, replace=False)
    subset = []
    with gzip.open(args.embeddings_file, 'rb') as f:
        header = f.readline().decode('ascii')
        
        for i, line in enumerate(f):
            if (i in tosample):
                subset.append(line.decode('ascii').split(","))
    data = pd.DataFrame(subset, columns=header.replace(" ","").split(","))
    create_index(data)
    
    subset = []
    with gzip.open(args.labels_file, 'rb') as f:
        header = f.readline().decode('ascii')
        
        for i, line in enumerate(f):
            if (i in tosample):
                subset.append(line.decode('ascii').split(","))
    labels = pd.DataFrame(subset, columns=header.replace(" ","").split(","))
    create_index(labels)
    
    # order by labels
    data = data.loc[labels.index]
    
    return data, labels

def evaluate(num_train_examples, num_test_examples, num_trials):
    
    all_acc = []
    for i in range(num_trials):
        
        print("Generating subset", i)
        data, labels = getSubset(num_train_examples+num_test_examples)
        
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
    
    
    
mean,stdev = evaluate(args.num_train_examples, 3000, args.num_trials)

print("Accuracy:",round(mean,3), "+-", round(stdev,3), "num_trials:",args.num_trials) 

    
    
    
    
    
    
