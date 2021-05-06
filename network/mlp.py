import torch as th
import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MLP_base(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, device):
        super(MLP_base, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)
        self.device = device

    def forward(self, x):
        hidden1 = self.relu(self.fc1(x))
        hidden2 = self.relu(self.fc2(hidden1))
        output = self.fc3(hidden2)
        output_clipped = th.max(th.min(output,  th.ones(size=output.shape).to(self.device)), th.zeros(size=output.shape).to(self.device))
        return output_clipped

class MLP(nn.Module):
    def __init__(self, input_size,  device, D=4, W=128, output_size=1, skips=[2]):
        """
        """
        super(MLP, self).__init__()
        self.D = D
        self.W = W
        self.input_ch_views = input_size
        self.skips = skips


        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_size, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_size, W) for i in
                                        range(D - 1)])

        self.device = device
        self.output_linear = nn.Linear(W, 1)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = th.nn.functional.relu(h)
            if i in self.skips:
                h = th.cat([x, h], -1)

        output = self.output_linear(h)
        output_clipped = th.max(th.min(output,  th.ones(size=output.shape).to(self.device)), th.zeros(size=output.shape).to(self.device))
        return output_clipped
