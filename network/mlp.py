import torch as th
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size1  = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)


    def forward(self, x):
        hidden1 = self.relu(self.fc1(x))
        hidden2 = self.relu(self.fc2(hidden1))
        output = self.fc3(hidden2)
        output_clipped = th.max(th.min(output,  th.ones(size=output.shape)), th.zeros(size=output.shape))
        return output_clipped
