import numpy as np
import torch
import torch.optim as optim
import math
import glob
# import torch.nn as nn
from model import Autoencoder
import data_io
import sys

report_every = 20
frame_length = 2**11 + 1
def data_stream(filenames, shuffle=True):
    return data_io.stream_file_list(filenames,
                                    buffer_count=25,
                                    batch_size=5,
                                    shuffle=shuffle)


if __name__ == "__main__":
    directory = sys.argv[1]
    filenames = (glob.glob(directory + "/?_batched.pkl.gz") +\
                 glob.glob(directory + "/??_batched.pkl.gz") +\
                 glob.glob(directory + "/???_batched.pkl.gz") +\
                 glob.glob(directory + "/????_batched.pkl.gz"))
    train_count = int(len(filenames) * 0.9)
    train_filenames = filenames[:train_count]
    valid_filenames = filenames[train_count:]


    model = Autoencoder(0, 1).cuda()
    # valid_data = torch.from_numpy(signal_data_valid).cuda()[:, None, :]

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    epochs = 25
    # batch_count = signal_data_batched.shape[0] // batch_size
    # print("Batch count:", batch_count)
    best_loss = np.inf
    i = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for data in data_stream(train_filenames):
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
            running_loss += loss.data.item()

            i += 1
            if i % report_every == 0:    # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i, running_loss / report_every))
                running_loss = 0.0
            if i % report_every * 10 == 0:
                model.eval()
                with torch.no_grad():
                    total_loss = 0.
                    count = 0
                    for data in data_stream(valid_filenames):
                        # get the inputs
                        input = torch.from_numpy(data.astype(np.float32)).cuda()
                        loss = model(input)
                        total_loss += loss.data.item()
                        count += 1
                    valid_loss = total_loss / count
                    if valid_loss < best_loss:
                        print("Best valid loss:", valid_loss)
                        with open('model.pt', 'wb') as f:
                            torch.save(model, f)
                        best_loss = valid_loss
                    else:
                        print("Valid loss:", valid_loss)
                model.train()

