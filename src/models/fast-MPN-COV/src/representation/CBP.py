import torch
import torch.nn as nn

class CBP(nn.Module):
    def __init__(self, input_dim=2048, output_dim=2048, weight_std=0.01, *args, **kwargs):
        super(CBP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer('weights_', torch.randn(2, input_dim, output_dim) * weight_std)

    def forward(self, x):
        x_flat = x.view(-1, self.input_dim)
        sketch1 = x_flat.mm(self.weights_[0])
        sketch2 = x_flat.mm(self.weights_[1])
        s1_fft = torch.fft.fft(torch.complex(sketch1, torch.zeros_like(sketch1)))
        s2_fft = torch.fft.fft(torch.complex(sketch2, torch.zeros_like(sketch2)))
        res = torch.fft.ifft(s1_fft * s2_fft).real
        return res
