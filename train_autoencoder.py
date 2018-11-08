import numpy as np
import torch
import torch.optim as optim
import math
import torch.nn.functional as F
# import torch.nn as nn

data = np.load('./AF4025.npz')
signal_data = data['signal']
length = 513
signal_data = (signal_data[:-(signal_data.shape[0] % length)] /
               data['norm_factor']) \
              .astype(np.float32) \
              .reshape((-1, length))
np.random.shuffle(signal_data)

validation_count = int(math.ceil(signal_data.shape[0] * 0.2))

signal_data_train, signal_data_valid = (signal_data[:-validation_count],
                                        signal_data[-validation_count:])
print(validation_count)
print(signal_data_train.shape)
signal_data_batched = signal_data_train
bottleneck_size = 64


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encode = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=129,
                stride=32,
                dilation=1, groups=1, bias=True
            ),
            torch.nn.ELU(),
            torch.nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=7,
                stride=1,
                dilation=1, groups=1, bias=True
            ),
            torch.nn.ELU(),
            torch.nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                stride=1,
                dilation=1, groups=1, bias=True
            ),
            torch.nn.ELU(),
            torch.nn.Conv1d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                dilation=1, groups=1, bias=True
            ),
            torch.nn.ELU(),
            torch.nn.Conv1d(
                in_channels=512,
                out_channels=128,
                kernel_size=1, stride=1, dilation=1, groups=1, bias=True
            ),
            torch.nn.ELU(),
            torch.nn.Conv1d(
                in_channels=128,
                out_channels=bottleneck_size,
                kernel_size=1, stride=1, dilation=1, groups=1, bias=True
            )
        )

        self.decode = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=bottleneck_size,
                out_channels=128,
                kernel_size=1, stride=1, dilation=1, groups=1, bias=True
            ),
            torch.nn.ELU(),
            torch.nn.Conv1d(
                in_channels=128,
                out_channels=512,
                kernel_size=1, stride=1, dilation=1, groups=1, bias=True
            ),
            torch.nn.ELU(),
            torch.nn.ConvTranspose1d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                dilation=1, groups=1, bias=True
            ),
            torch.nn.ELU(),
            torch.nn.ConvTranspose1d(
                in_channels=256,
                out_channels=128,
                kernel_size=5,
                stride=1,
                dilation=1, groups=1, bias=True
            ),
            torch.nn.ELU(),
            torch.nn.ConvTranspose1d(
                in_channels=128,
                out_channels=64,
                kernel_size=7,
                stride=1,
                dilation=1, groups=1, bias=True
            ),
            torch.nn.ELU(),
            torch.nn.ConvTranspose1d(
                in_channels=64,
                out_channels=1,
                kernel_size=129,
                stride=32,
                dilation=1, groups=1, bias=True
            )
        )

    def forward(self, input):
        # print(input.size())
        encoding = self.encode(input)
        # print(encoding.size())
        output = self.decode(F.elu(encoding))
        # print(output.size())
        loss = torch.mean((output - input)**2)
        return loss


valid_data = torch.from_numpy(signal_data_valid).cuda()[:, None, :]

model = Autoencoder().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

batch_size = 64
batch_count = signal_data_batched.shape[0] // batch_size
print("Batch count:", batch_count)
best_loss = np.inf
for epoch in range(50):
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
        valid_loss = model(valid_data)
        if valid_loss < best_loss:
            print("Best valid loss:", valid_loss)
            with open('model.pt', 'wb') as f:
                torch.save(model, f)
            best_loss = valid_loss
        else:
            print("Valid loss:", valid_loss)

    np.random.shuffle(signal_data_batched)
