import numpy as np
import torch
import torch.optim as optim
import math
import torch.nn.functional as F
import glob
# import torch.nn as nn


length = 513
bottleneck_size = 32
data_location = "/network/tmp1/tanjings/icentia-ecg/icentia-mila-research/"
files = glob.glob(data_location + "*.npz")[:10]


if __name__ == "__main__":
    signal_data_frames = []
    for filename in files:
        print("Loading", filename, "...")
        data = np.load(filename)
        signal_data = data['signal'].astype(np.float32)
        signal_data_ = (signal_data[:-(signal_data.shape[0] % length)] /
                        data['norm_factor']).reshape((-1, length))
        signal_data_frames.append(signal_data_)

    signal_data_train = np.concatenate(signal_data_frames[:-2], axis=0)
    np.random.shuffle(signal_data_train)
    train_mean = signal_data_train.mean()
    train_std = signal_data_train.std()

    signal_data_valid = np.concatenate(signal_data_frames[-2:], axis=0)

    print(signal_data_train.shape)
    signal_data_batched = signal_data_train


class Autoencoder(torch.nn.Module):
    def __init__(self, mean, std):
        super(Autoencoder, self).__init__()
        self.mean = mean
        self.std = std
        stack_spec = [
            (1, 64, 129, 32),
            (64, 128, 7, 1),
            (128, 256, 5, 1),
            (256, 512, 3, 1),
            (512, 128, 1, 1),
            (128, bottleneck_size, 1, 1)
        ]

        activation = torch.nn.ELU()
        dropout = torch.nn.Dropout(0.5)

        encode_ops = []
        for in_c, out_c, kernel, stride in stack_spec:
            conv_op = torch.nn.Conv1d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=kernel,
                stride=stride,
                dilation=1, groups=1, bias=True
            )
            encode_ops.append(conv_op)
            encode_ops.append(activation)
            encode_ops.append(dropout)
        self.encode = torch.nn.Sequential(*encode_ops)


        decode_ops = []
        for out_c, in_c, kernel, stride in stack_spec[::-1]:
            conv_op = torch.nn.ConvTranspose1d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=kernel,
                stride=stride,
                dilation=1, groups=1, bias=True
            )
            decode_ops.append(conv_op)
            decode_ops.append(activation)
            decode_ops.append(dropout)
        self.decode = torch.nn.Sequential(*decode_ops)



    def forward(self, input):
        # print(input.size())
        input = (input - self.mean) / self.std
        encoding = self.encode(input)
        # print(encoding.size())
        output = self.decode(F.elu(encoding))
        # print(output.size())
        loss = torch.mean((output - input)**2)
        return loss

if __name__ == "__main__":
    model = Autoencoder(train_mean, train_std).cuda()
    valid_data = torch.from_numpy(signal_data_valid).cuda()[:, None, :]

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 25
    batch_size = 256
    batch_count = signal_data_batched.shape[0] // batch_size
    print("Batch count:", batch_count)
    best_loss = np.inf
    for epoch in range(epochs):
        running_loss = 0.0

        model.train()
        for i in range(batch_count):
            # get the inputs
            data = signal_data_batched[i * batch_size:
                                       (i + 1) * batch_size, None, :]
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
            if (i + 1) % 500 == 0:    # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

        model.eval()
        with torch.no_grad():
            val_batch_counts = valid_data.size(0) // batch_size
            valid_loss = sum(
                model(valid_data[i * batch_size:(i+1) * batch_size])
                for i in range(val_batch_counts)
            ) / val_batch_counts
            if valid_loss < best_loss:
                print("Best valid loss:", valid_loss)
                with open('model.pt', 'wb') as f:
                    torch.save(model, f)
                best_loss = valid_loss
            else:
                print("Valid loss:", valid_loss)

        np.random.shuffle(signal_data_batched)
