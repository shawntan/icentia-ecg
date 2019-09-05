# icentia-ecg
Working on Icentia ECG data.

# Evaluation

To prepare raw embeddings file:

```
./prepare_labels.py -labels_file test_labels_v2.csv.gz /path/to/icentia12k/
./prepare_emb.py -labels_file test_labels_v2.csv.gz -embeddings_file test_emb_v2.csv.gz /path/to/icentia12k/
```

To run evalations:

```bash
./eval.py test_emb.csv.gz

Namespace(embeddings_file='test_emb.csv.gz', labels_file='test_labels.csv.gz', num_test_examples=5000, num_train_examples=5000, num_trials=10)
lines_emb: 420493
Generating subset 0
X (5000, 2049) X_test (5000, 2049)
   Run 0, Accuracy: 0.5208
Generating subset 1
X (5000, 2049) X_test (5000, 2049)
   Run 1, Accuracy: 0.5
Generating subset 2
X (5000, 2049) X_test (5000, 2049)
   Run 2, Accuracy: 0.5204
Generating subset 3
X (5000, 2049) X_test (5000, 2049)
   Run 3, Accuracy: 0.5138
Generating subset 4
X (5000, 2049) X_test (5000, 2049)
   Run 4, Accuracy: 0.531
Generating subset 5
X (5000, 2049) X_test (5000, 2049)
   Run 5, Accuracy: 0.5108
Generating subset 6
X (5000, 2049) X_test (5000, 2049)
   Run 6, Accuracy: 0.52
Generating subset 7
X (5000, 2049) X_test (5000, 2049)
   Run 7, Accuracy: 0.5188
Generating subset 8
X (5000, 2049) X_test (5000, 2049)
   Run 8, Accuracy: 0.5272
Generating subset 9
X (5000, 2049) X_test (5000, 2049)
   Run 9, Accuracy: 0.4902
Accuracy: 0.515 +- 0.012 num_trials: 10

```
