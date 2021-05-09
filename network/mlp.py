import torch as th
import torch.nn as nn
import torch


def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

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


if __name__ == '__main__':
    a = torch.arange(100) / 100
    a = a.unsqueeze(1)
    b = a[::4]
    posa = positional_encoding(a, num_encoding_functions=6)
    posb = positional_encoding(b, num_encoding_functions=6)
    import matplotlib.pyplot as plt
    plt.plot(posa[0])
    plt.plot(posb[0])
    plt.show()
    plt.plot(posa[1])
    plt.plot(posb[1])
    plt.show()
    (posa[4] == posb[1]).all()
