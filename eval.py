import argparse
import numpy as np
import matplotlib.pylab as plt
import sys
import pandas as pd
import sklearn, sklearn.model_selection, sklearn.neighbors


parser = argparse.ArgumentParser()
parser.add_argument('embeddings_file', help='File with embeddings')
parser.add_argument('num_train_examples', nargs='?', type=int, default=1000, help='')
parser.add_argument('num_trials', nargs='?', type=int, default=10, help='')
parser.add_argument('labels_file', nargs='?', default="test_labels.csv", help='')
args = parser.parse_args()

print(args)

def create_index(df):
    # create index col and remove source columns
    df["id"] = df.apply(lambda row: str(int(row[0])) + "_" + str(int(row[1])) + "_" + str(int(row[2])), axis=1)
    del df["sample"]
    del df["segment"]
    del df["frame"]
    df.set_index("id", inplace=True)
    

data = pd.read_csv(args.embeddings_file)
labels = pd.read_csv(args.labels_file)

create_index(data)
create_index(labels)

# order by labels
if len(labels.index) != len(data.index):
    print(" !! Issue with coverage of labels. The data must align to the labels.")
    sys.exit()

data = data.loc[labels.index]

print("Loaded data", data.shape)

def evaluate(embeddings, labels, num_train_examples, num_test_examples, num_trials):
    
    all_acc = []
    for i in range(num_trials):
        X, X_test, y, y_test = \
            sklearn.model_selection.train_test_split(embeddings, labels, 
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
    
mean,stdev = evaluate(data, labels, args.num_train_examples, min(3000,len(data)-args.num_train_examples), args.num_trials)

print("Accuracy:",round(mean,3), "+-", round(stdev,3), "num_trials:",args.num_trials) 

    
    
    
    
    
    