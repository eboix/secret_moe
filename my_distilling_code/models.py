import torch
import torch.nn as nn

class ParallelMLPs(nn.Module):
    def __init__(self, input_dim, output_dim, intermediate_dim=128, multi_index_dim=2, m=127, top_k=None, bias=False,
                 init_scale=0.01, dtype=torch.float32):
        super(ParallelMLPs, self).__init__()
        self.m = m
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.multi_index_dim = multi_index_dim
        self.top_k = top_k
        self.bias = bias
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

        # Parameters E, F, G, H, where
        # E is of dimension (m, output_dim, k)
        self.E = nn.Parameter(torch.randn(m, output_dim, multi_index_dim, dtype=dtype) * init_scale)
        # F is of dimension (m, k, intermediate_dim)
        self.F = nn.Parameter(torch.randn(m, multi_index_dim, intermediate_dim, dtype=dtype) * init_scale)
        # G is of dimension (m, intermediate_dim, k)
        self.G = nn.Parameter(torch.randn(m, intermediate_dim, multi_index_dim, dtype=dtype) * init_scale)
        # H is of dimension (m, k, input_dim)
        self.H = nn.Parameter(torch.randn(m, multi_index_dim, input_dim, dtype=dtype) * init_scale)
        if bias:
            self.bias_G = nn.Parameter(torch.zeros(m, intermediate_dim,1, dtype=dtype))
            self.bias_E = nn.Parameter(torch.zeros(output_dim,1, dtype=dtype))

    def forward(self, x):
        # x is of shape (batch_size, input_dim))
        # left-multiply by H
        xlin = self.linear(x)
        x = torch.einsum('mki,bi->mkb', self.H, x)
        # x is now of shape (m, k, batch_size)
        # left-multiply by G
        x = torch.einsum('mtk,mkb->mtb', self.G, x)
        if self.bias:
            # add bias for G
            x = x + self.bias_G
        # x is now of shape (m, intermediate_dim, batch_size)
        # apply GeLU activation elementwise
        x = torch.nn.functional.gelu(x)
        # left-multiply by F
        x = torch.einsum('mkt,mtb->mkb', self.F, x)
        # x is now of shape (m, k, batch_size)
        # left-multiply by E
        x = torch.einsum('mok,mkb->mob', self.E, x)
        # x is now of shape (m, output_dim, batch_size)
        if self.top_k is not None:
            # compute the squared norm of x for each index in the zero dimension (m)
            # and select the top-k indices
            norms_sq = torch.sum(x ** 2, dim=1)  # shape (m, batch_size)
            top_k_indices = torch.topk(norms_sq, self.top_k, dim=0).indices # shape (top_k, batch_size)
            # gather the top-k elements from x
            x = torch.gather(x, 0, top_k_indices.unsqueeze(1).expand(-1, x.shape[1], -1))
            # x is now of shape (top_k, output_dim, batch_size)
            # check if the shape is correct

        # sum over the first dimension (m)
        x = torch.sum(x, dim=0)
        if self.bias:
            # add bias for E
            x = x + self.bias_E
        # x is now of shape (output_dim, batch_size)
        # transpose to get (batch_size, output_dim)
        x = x.transpose(0, 1)
        x += xlin
        return x