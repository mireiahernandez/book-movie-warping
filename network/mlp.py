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
        output_clipped = th.max(th.min(output,  th.ones(size=output.shape).to(self.device)),
                                th.zeros(size=output.shape).to(self.device))
        return output_clipped


class MLP_2dir(nn.Module):
    def __init__(self, input_size,  device, D=4, W=128, PE=6, output_size=1, skips=[2]):
        """
        """
        super(MLP_2dir, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.device = device
        self.PE = PE
        self.pts_linears_m2b = nn.ModuleList(
            [nn.Linear(input_size, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_size, W) for i in
                                        range(D - 1)])
        self.output_linear_m2b = nn.Linear(W, 1)
        self.pts_linears_b2m = nn.ModuleList(
            [nn.Linear(input_size, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_size, W) for i in
                                        range(D - 1)])
        self.output_linear_b2m = nn.Linear(W, 1)

    def forward_m2b(self, x):
        h1 = x
        for i, l in enumerate(self.pts_linears_m2b):
            h1 = self.pts_linears_m2b[i](h1)
            h1 = th.nn.functional.relu(h1)
            if i in self.skips:
                h1 = th.cat([x, h1], -1)
        output_m2b = self.output_linear_m2b(h1)
        output_clipped_m2b = th.max(th.min(output_m2b,  th.ones(size=output_m2b.shape).to(self.device)),
                                th.zeros(size=output_m2b.shape).to(self.device))
        return output_clipped_m2b

    def forward_b2m(self, x):
        h2 = x
        for i, l in enumerate(self.pts_linears_b2m):
            h2 = self.pts_linears_b2m[i](h2)
            h2 = th.nn.functional.relu(h2)
            if i in self.skips:
                h2 = th.cat([x, h2], -1)
        output_b2m = self.output_linear_b2m(h2)
        output_clipped_b2m = th.max(th.min(output_b2m,  th.ones(size=output_b2m.shape).to(self.device)),
                                th.zeros(size=output_b2m.shape).to(self.device))
        return output_clipped_b2m

    def forward(self, m, b):
        m1 = self.forward_m2b(positional_encoding(b.unsqueeze(1), self.PE))
        b1 = self.forward_b2m(positional_encoding(m.unsqueeze(1), self.PE))

        # this needs positional encoding
        b2m_m2b_b = self.forward_b2m(positional_encoding(m1, self.PE))
        m2b_b2m_m = self.forward_m2b(positional_encoding(b1, self.PE))

        return m1, b1, b2m_m2b_b, m2b_b2m_m


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
