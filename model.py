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
    return erfield, estride

class ConvAutoencoder(torch.nn.Module):
    def __init__(self, stack_spec, encoder_only=False, debug=False):
        super(ConvAutoencoder, self).__init__()
        activation = torch.nn.ELU()
        dropout = torch.nn.Dropout(0.5)
        encode_ops = []
        for in_c, out_c, kernel, stride in stack_spec:
            conv_op = torch.nn.Conv1d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=kernel,
                stride=stride,
                padding=0,
                dilation=1, groups=1, bias=True
            )
            encode_ops.append(conv_op)
            encode_ops.append(activation)
            encode_ops.append(dropout)
        self.encode = torch.nn.Sequential(*encode_ops)
        erfield, estride = receptive_field(stack_spec)
        print("Effective receptive field:", erfield, estride)
        self.test_conv = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=erfield,
            stride=estride,
            padding=0,
            dilation=1, groups=1, bias=True
        )

        if not encoder_only:
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
        self.activation = activation
        self.dropout = dropout
        self.debug = debug

    def forward(self, x):
        encoding = self.encode(x)
        output = self.decode(F.elu(encoding))
        return output



class Autoencoder(torch.nn.Module):
    def __init__(self, mean, std, bottleneck_size=32):
        super(Autoencoder, self).__init__()
        self.mean = mean
        self.std = std
        activation = torch.nn.ELU()
        dropout = torch.nn.Dropout(0.5)

        frame_dim = 128
        segment_dim = 256
        patient_dim = 256
        # Output should be [batch, *, 4089]
        # (  1,  16, 2049, 256),
        self.autoencode_1 = ConvAutoencoder([
                # in, out, kernel, stride
                ( 1, 16, 129,  64),
                (16, 32,   7,   4),
                (32, 64,   3,   2),
                (64, frame_dim,   3,   2),
            ],
            debug=True
        )

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

    def encode_3(self, x, input_shape):
        x = x.view(input_shape[0], input_shape[1], x.size(-1))
        h_1 = self.encode_3_1(x)
        h_2 = torch.cat([torch.max(h_1, dim=1)[0],
                         torch.min(h_1, dim=1)[0]], dim=-1)
        emb = self.encode_3_2(h_2)
        return emb


    def forward(self, input):
        input = (input - self.mean) / self.std
        input_flat = input.view(-1, 1, input.size(-1))

        encoding_1 = self.autoencode_1.encode(input_flat)
        if True:
            test_out = self.autoencode_1.test_conv(input_flat)
            assert(encoding_1.size(-1) == test_out.size(-1))
            print(input_flat.size(), encoding_1.size(), test_out.size())

        encoding_2 = torch.max(self.encode_2(encoding_1), dim=-1)[0]
        # print(encoding.size())

        encoding_3 = self.encode_3(encoding_2, input.size())

        frame_rep = self.frame_transform(
                encoding_1.permute(0, 2, 1))
        segment_rep = self.segment_transform(encoding_2)[:, None, :]
        patient_rep = encoding_3[:, None, :]\
                        .repeat(1, input.size(1), 1)\
                        .view(-1, encoding_3.size(-1))[:, None, :]


        decode_rep = self.decode_transform(
            frame_rep +
            segment_rep +
            patient_rep
        ).permute(0, 2, 1)

        output = self.autoencode_1.decode(decode_rep)
        output = output.view(input.size())
        loss = torch.mean((output - input)**2)
        return loss


