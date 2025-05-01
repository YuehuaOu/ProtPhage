from torch.nn import Module, Conv1d, BatchNorm1d, MaxPool1d, functional as F, Dropout, Linear, LazyLinear
import torch


class Cnn(Module):
    """
    CNN model
    """

    def __init__(self, output_dim=1, input_dim=1024, drop_out=0, stride=2):
        super(Cnn, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.drop_out = drop_out

        self.kernel_1 = 3
        self.channel_1 = 32

        self.conv_1 = Conv1d(kernel_size=self.kernel_1, out_channels=self.channel_1, in_channels=1, stride=1)
        self.normalizer_1 = BatchNorm1d(self.channel_1)
        self.pooling_1 = MaxPool1d(kernel_size=self.kernel_1, stride=stride)

        self.dropout = Dropout(p=drop_out)
        self.fc1 = LazyLinear(64)
        self.normalizer_2 = BatchNorm1d(64)
        self.fc2 = Linear(64, 2)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # (batch, embedding_dim) -> (batch, 1, embedding_dim)
        c_1 = self.pooling_1(F.relu(self.normalizer_1(self.conv_1(x))))

        c_2 = torch.flatten(c_1, start_dim=1)
        c_2 = self.dropout(c_2)
        out = F.relu(self.normalizer_2(self.fc1(c_2)))
        out = self.fc2(out)
        out = torch.softmax(out, dim=-1)
        return out


class Cnn_muti(Module):
    """
    CNN model
    """

    def __init__(self, output_dim=1, input_dim=1024, drop_out=0, stride=2):
        super(Cnn_muti, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.drop_out = drop_out

        self.kernel_1 = 3
        self.channel_1 = 32

        self.conv_1 = Conv1d(kernel_size=self.kernel_1, out_channels=self.channel_1, in_channels=1, stride=1)
        self.normalizer_1 = BatchNorm1d(self.channel_1)
        self.pooling_1 = MaxPool1d(kernel_size=self.kernel_1, stride=stride)

        self.dropout = Dropout(p=drop_out)
        self.fc1 = LazyLinear(64)
        self.normalizer_2 = BatchNorm1d(64)
        self.fc2 = Linear(64, 7)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # (batch, embedding_dim) -> (batch, 1, embedding_dim)
        c_1 = self.pooling_1(F.relu(self.normalizer_1(self.conv_1(x))))

        c_2 = torch.flatten(c_1, start_dim=1)
        c_2 = self.dropout(c_2)
        out = F.relu(self.normalizer_2(self.fc1(c_2)))
        out = self.fc2(out)
        out = torch.softmax(out, dim=-1)
        return out