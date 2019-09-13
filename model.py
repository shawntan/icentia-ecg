import torch
from torch import nn
import torch.nn.functional as F


def receptive_field(op_params):
    _, _, erfield, estride = op_params[0]
    for i in range(1, len(op_params)):
        _, _, kernel, stride = op_params[i]
        one_side = erfield // 2
        erfield = (kernel - 1) * estride + 1 + 2 * one_side
        estride = estride * stride
        if erfield % 2 == 0:
            print("EVEN", erfield)
        print(erfield, estride)
    return erfield, estride


class ResidualEncoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activation=torch.nn.ELU(), dropout=0.1, last=False):
        super(ResidualEncoder, self).__init__()
        self.last = last

        self.conv_op = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=1, groups=1, bias=True
        )

        self.nin_op = torch.nn.Conv1d(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1, groups=1, bias=True
        )
        self.res_op = torch.nn.Conv1d(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1, groups=1, bias=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = activation
        self.bn = nn.BatchNorm1d(2 * out_channels)

    def forward(self, x):
        z_ = self.bn(self.conv_op(x))
        z = self.dropout(self.activation(z_))
        y_ = self.nin_op(z)
        if not self.last:
            y = self.dropout(self.activation(y_))
            return y + self.res_op(z_)
        else:
            return y_

class ResidualDecoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activation=torch.nn.ELU(), dropout=0.5, last=False):
        super(ResidualDecoder, self).__init__()
        self.last = last
        self.conv_op = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            dilation=1, groups=1, bias=True
        )
        self.nonlin = torch.nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1, groups=1, bias=True
        )
        self.res_op = torch.nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1, groups=1, bias=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = activation
        self.bn = nn.BatchNorm1d(2 * out_channels)

    def forward(self, x):
        z_ = self.bn(self.conv_op(x))
        z = self.dropout(self.activation(z_))
        y_ = self.nonlin(z)
        # print(y_.size(), z.size())
        if not self.last:
            y = self.dropout(self.activation(y_))
            return y + self.res_op(z_)
        else:
            return y_

class ConvAutoencoder(torch.nn.Module):
    def __init__(self, stack_spec, debug=True):
        super(ConvAutoencoder, self).__init__()
        activation = torch.nn.ELU()
        encode_ops = []
        dropout = torch.nn.Dropout(0.5)

        for i, (in_c, out_c, kernel, stride) in enumerate(stack_spec):
            last = i == (len(stack_spec)-1)
            encode_ops.append(ResidualEncoder(in_c, out_c, kernel, stride,
                                              dropout=0.1,
                                              last=last))
            if not last:
                pass

        #    encode_ops.append(dropout)
        # encode_ops = encode_ops[:-1]

        self.encode = torch.nn.Sequential(*encode_ops)
        erfield, estride = receptive_field(stack_spec)
        if debug:
            print("Effective receptive field:", erfield, estride)
            self.test_conv = torch.nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=erfield,
                stride=estride,
                padding=0,
                dilation=1, groups=1, bias=True
            )

        decode_ops = []
        for i, (out_c, in_c, kernel, stride) in enumerate(stack_spec[::-1]):
            last = (i == len(stack_spec)-1)
            decode_ops.append(ResidualDecoder(in_c, out_c, kernel, stride,
                                              dropout=0.1,
                                              last=last))
            if not last:
                pass


            #    decode_ops.append(dropout)
            # decode_ops = decode_ops[:-1]
            self.decode = torch.nn.Sequential(*decode_ops)
        self.activation = activation
        self.dropout = dropout
        self.debug = debug

    def forward(self, x):
        encoding = self.encode(x)
        output = self.decode(encoding)
        return output



class Autoencoder(torch.nn.Module):
    def __init__(self, mean=0, std=1, bottleneck_size=32):
        super(Autoencoder, self).__init__()
        self.mean = mean
        self.std = std
        activation = torch.nn.ELU()
        self.dropout = dropout = torch.nn.Dropout(0.5)

        frame_dim = 100
        segment_dim = 256
        patient_dim = 256
        # Output should be [batch, *, 4089]
        # (  1,  16, 2049, 256),
#        self.autoencode_1 = ConvAutoencoder([
#                # in, out, kernel, stride
#                (  1, 512, 1025, 512),
#                (512, frame_dim,   3,   4),
#            ],
#        )
        self.autoencode_1 = ConvAutoencoder([
                # in, out, kernel, stride
                (  1, frame_dim, 2049,  2048),
            ],
        )
 
        # print(self.autoencode_1)

        self.encode_2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                frame_dim, 128, 5, 2,
                padding=0, dilation=1, groups=1, bias=True
            ),
            activation,
            torch.nn.MaxPool1d(3),
            dropout,
            torch.nn.Conv1d(
                128, 128, 5, 2,
                padding=0, dilation=1, groups=1, bias=True
            ),
            activation,
            torch.nn.MaxPool1d(3),
            dropout,
            torch.nn.Conv1d(
                128, segment_dim, 3, 1,
                padding=0, dilation=1, groups=1, bias=True
            ),
            activation,
        )

        self.encode_3_1 = torch.nn.Sequential(
            nn.Linear(segment_dim, patient_dim, 1),
            activation,
            dropout,
            nn.Linear(patient_dim, patient_dim, 1),
        )

        self.encode_3_2 = torch.nn.Sequential(
            nn.Linear(patient_dim * 2, patient_dim),
            activation,
            dropout
        )


        self.frame_transform = nn.Linear(frame_dim, frame_dim * 2)
        self.segment_transform = nn.Linear(segment_dim, frame_dim * 2)
        self.patient_transform = nn.Linear(patient_dim, frame_dim * 2)

        self.decode_transform = nn.Sequential(
            activation,
            nn.Linear(frame_dim * 2, frame_dim),
        )
        self.frame_bn = nn.BatchNorm1d(frame_dim)

    def encode_3(self, x, input_shape):
        x = x.view(input_shape[0], input_shape[1], x.size(-1))
        h_1 = self.encode_3_1(x)
        h_2 = torch.cat([torch.max(h_1, dim=1)[0],
                         torch.min(h_1, dim=1)[0]], dim=-1)
        emb = self.encode_3_2(h_2)
        return emb

    def encode(self, input_flat):
        encoding_1 = self.frame_bn(self.autoencode_1.encode(input_flat))
        return encoding_1

    def decode(self, encoding):
        output = self.autoencode_1.decode(encoding)
        return output

    def forward(self, input):
        input = (input - self.mean) / self.std
        input_flat = input.view(-1, 1, input.size(-1))

        # encoding_2 = torch.max(self.encode_2(encoding_1), dim=-1)[0]

        # encoding_3 = self.encode_3(encoding_2, input.size())

        # frame_rep = self.frame_transform(encoding_1.permute(0, 2, 1))

        # decode_rep = self.decode_transform(
        #     frame_rep
            # segment_rep +
            # patient_rep
        # ).permute(0, 2, 1)
        output = self.decode(self.encode(input_flat))
        output = output.view(input.size())
        loss = torch.sqrt(torch.mean((output - input)**2))
        # loss = torch.mean(abs(output - input))
        return loss


