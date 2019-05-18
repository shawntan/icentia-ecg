import numpy as np
import torch
import torch.optim as optim
import math
import glob
# import torch.nn as nn
from model import Autoencoder
import data_io



#if __name__ == "__main__":
#    signal_data_frames = []
#    for filename in files:
#        print("Loading", filename, "...")
#        data = np.load(filename)
#        signal_data = data['signal'].astype(np.float32)
#        signal_data_ = (signal_data[:-(signal_data.shape[0] % length)] /
#                        data['norm_factor']).reshape((-1, length))
#        signal_data_frames.append(signal_data_)
#
#    signal_data_train = np.concatenate(signal_data_frames[:-2], axis=0)
#    np.random.shuffle(signal_data_train)
#    train_mean = signal_data_train.mean()
#    train_std = signal_data_train.std()
#
#    signal_data_valid = np.concatenate(signal_data_frames[-2:], axis=0)
#
#    print(signal_data_train.shape)
#    signal_data_batched = signal_data_train

def data_stream():
    directory = "/home/shawntan/projects/rpp-bengioy/jpcohen/icentia-ecg-dataset"
    filenames = glob.glob(directory + "/*_batched.npz")
    return data_io.stream_file_list(filenames, buffer_count=25, batch_size=5)


if __name__ == "__main__":
    model = Autoencoder(0, 1).cuda()
    # valid_data = torch.from_numpy(signal_data_valid).cuda()[:, None, :]

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 25
    # batch_count = signal_data_batched.shape[0] // batch_size
    # print("Batch count:", batch_count)
    best_loss = np.inf
    for epoch in range(epochs):
        running_loss = 0.0
        i = 0
        model.train()
        for data in data_stream():
            # get the inputs
            input = torch.from_numpy(data.astype(np.float32)).cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = model(input)
            # print(loss)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()

            i += 1
            if i % 200 == 0:    # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i, running_loss / 500))
                running_loss = 0.0

        model.eval()
#        with torch.no_grad():
#            val_batch_counts = valid_data.size(0) // batch_size
#            valid_loss = sum(
#                model(valid_data[i * batch_size:(i+1) * batch_size])
#                for i in range(val_batch_counts)
#            ) / val_batch_counts
#            if valid_loss < best_loss:
#                print("Best valid loss:", valid_loss)
#                with open('model.pt', 'wb') as f:
#                    torch.save(model, f)
#                best_loss = valid_loss
#            else:
#                print("Valid loss:", valid_loss)
#
#        np.random.shuffle(signal_data_batched)
