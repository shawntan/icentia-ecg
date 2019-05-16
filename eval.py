import argparse
import numpy as np
import matplotlib.pylab as plt
import sys
import pandas as pd
import sklearn, sklearn.model_selection, sklearn.neighbors


parser = argparse.ArgumentParser()
parser.add_argument('embeddings_file', help='File with embeddings')
parser.add_argument('num_train_examples', nargs='?', type=int, default=2, help='')
parser.add_argument('num_trials', nargs='?', type=int, default=10, help='')
parser.add_argument('labels_file', nargs='?', default="test_labels.csv", help='')
args = parser.parse_args()


def create_index(df):
    # create index col and remove source columns
    df["id"] = df.apply(lambda row: row[0] + "_" + str(row[1]), axis=1)
    del df[0]
    del df[1]
    df.set_index("id", inplace=True)
    

data = pd.read_csv(args.embeddings_file, header=None)
labels = pd.read_csv(args.labels_file, header=None)

create_index(data)
create_index(labels)

# order by labels
data = data.loc[labels.index]


def evaluate(embeddings, labels, num_train_examples=10, num_trials=1):
    
    all_acc = []
    for i in range(num_trials):
        X, X_test, y, y_test = \
            sklearn.model_selection.train_test_split(data, labels, 
                                                     train_size=num_train_examples, 
                                                     test_size=len(data)-num_train_examples, 
                                                     stratify=labels,
                                                     random_state=i)

        model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        model = model.fit(X, y.values.flatten())

        acc = (model.predict(X_test) == y_test.values.flatten()).mean()
        all_acc.append(acc)

    return np.asarray(all_acc).mean(), np.asarray(all_acc).std()
    
    

mean,stdev = evaluate(data, labels, args.num_train_examples, args.num_trials)

print("Accuracy:",mean, "+-", stdev, "num_trials:",args.num_trials) 
    
    
    
    
    
    