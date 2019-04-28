import torch

from torch import nn
from torch.autograd import Variable
import numpy as np
from config import Config


def empty_variable():
    return Variable(torch.FloatTensor(np.empty(0))).to(Config.device)


class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        self.cnn_1 = nn.Conv2d(in_channels=Config.input_dim[0],
                               out_channels=Config.cnn1_dim,
                               kernel_size=Config.kernel_1,
                               padding=Config.padding_1)
        self.cnn_2 = nn.Conv2d(in_channels=Config.cnn1_dim,
                               out_channels=Config.cnn2_dim,
                               kernel_size=Config.kernel_2,
                               padding=Config.padding_2)
        self.cnn_3 = nn.Conv2d(in_channels=Config.cnn2_dim,
                               out_channels=Config.cnn3_dim,
                               kernel_size=Config.kernel_3,
                               padding=Config.padding_3)
        self.cnn_4 = nn.Conv2d(in_channels=Config.cnn3_dim,
                               out_channels=Config.cnn4_dim,
                               kernel_size=Config.kernel_4,
                               padding=Config.padding_4)
        self.cnn_5 = nn.Conv2d(in_channels=Config.cnn4_dim,
                               out_channels=Config.cnn5_dim,
                               kernel_size=Config.kernel_5,
                               padding=Config.padding_5)
        self.cnn_6 = nn.Conv2d(in_channels=Config.cnn5_dim,
                               out_channels=Config.cnn6_dim,
                               kernel_size=Config.kernel_6,
                               padding=Config.padding_6)
        self.cnn_7 = nn.Conv2d(in_channels=Config.cnn6_dim,
                               out_channels=Config.cnn7_dim,
                               kernel_size=Config.kernel_7,
                               padding=Config.padding_7)
        self.pooling_1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pooling_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pooling_3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pooling_4 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pooling_5 = nn.MaxPool2d(kernel_size=(1, 2))
        self.batch_norm_1 = nn.BatchNorm2d(512).to(Config.device)
        self.batch_norm_2 = nn.BatchNorm2d(512).to(Config.device)
        self.activation = nn.ReLU().to(Config.device)

    def forward(self, x):
        x = self.activation(self.cnn_1(x))
        x = self.pooling_1(x)
        x = self.activation(self.cnn_2(x))
        x = self.pooling_2(x)
        x = self.activation(self.cnn_3(x))
        x = self.activation(self.cnn_4(x))
        x = self.pooling_3(x)
        x = self.activation(self.cnn_5(x))
        x = self.batch_norm_1(x)
        x = self.activation(self.cnn_6(x))
        x = self.batch_norm_2(x)
        x = self.pooling_4(x)
        x = self.activation(self.cnn_7(x))
        x = self.pooling_5(x)
        b, c, w, h = x.shape
        # --> x : batch - chanel - x - y
        return x.permute(0, 3, 1, 2).view(b, -1).view(b, w*h*c)


class LSTMModule(nn.Module):
    def __init__(self):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_size=Config.lstm_dim[0],
                            hidden_size=Config.lstm_dim[1],
                            num_layers=2,
                            bidirectional=True,
                            dropout=Config.dropout)

    def forward(self, x):
        x = self.lstm(x)
        return x


class FCModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCModule, self).__init__()
        self.fc_1 = nn.Linear(input_dim, 1024)
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.fc_out(x)
        return x


class ImageExtract(nn.Module):
    def __init__(self):
        super(ImageExtract, self).__init__()

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.permute(2, 0, 1, 3).contiguous().view(w, -1)
        x_out = empty_variable()
        for i in range(0, Config.length_seq*2, 2):
            tmp = empty_variable()
            for j in range(Config.size_image[0]):
                tmp = torch.cat((tmp, x[i+j]))
            tmp = tmp.contiguous()
            x_out = torch.cat((x_out, tmp))
        x_out = x_out.view((Config.length_seq, Config.size_image[0], b, c, Config.size_image[1]))
        x_out = x_out.permute((0, 2, 3, 1, 4))
        return x_out


class ImageGenerate(nn.Module):
    def __init__(self):
        super(ImageGenerate, self).__init__()

    def forward(self, x):

        pass


class CRNNCTC(nn.Module):
    def __init__(self):
        super(CRNNCTC, self).__init__()
        self.batch_size = 0
        self.cnn_part = CNNModule().to(Config.device)
        self.lstm_part = LSTMModule().to(Config.device)
        self.fc_part = FCModule(Config.lstm_dim[1] * 2, Config.output_dim).to(Config.device)
        self.image_extract = ImageExtract().to(Config.device)
        self.image_pre_gen = ImageGenerate().to(Config.device)

    def forward(self, x):
        x = self.image_extract(x)
        x = torch.stack([self.cnn_part(x[t]) for t in range(Config.length_seq)])
        x, _ = self.lstm_part(x)
        x = torch.stack([self.fc_part(x[t]) for t in range(Config.length_seq)])
        # print(x.shape)
        return x
