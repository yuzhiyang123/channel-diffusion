# Complex NN layers
import torch.nn as nn
import torch


def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None):
        super(ComplexLinear, self).__init__()
        self.lr = nn.Linear(in_features, out_features, bias, device)
        self.li = nn.Linear(in_features, out_features, bias, device)

    def forward(self, xx):
        return apply_complex(self.lr, self.li, xx)


class ComplexAvgPool1d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super(ComplexAvgPool1d, self).__init__()
        self.lr = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)
        self.li = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    def forward(self, xx):
        return apply_complex(self.lr, self.li, xx)


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', device=None):
        super(ComplexConv1d, self).__init__()
        self.lr = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding,
                            dilation, groups, bias, padding_mode, device)
        self.li = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding,
                            dilation, groups, bias, padding_mode, device)

    def forward(self, xx):
        return apply_complex(self.lr, self.li, xx)


class ComplexConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                 groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        super(ComplexConvTranspose1d, self).__init__()
        self.lr = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding,
                                     output_padding, groups, bias, dilation, padding_mode, device)
        self.li = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding,
                                     output_padding, groups, bias, dilation, padding_mode, device)

    def forward(self, xx):
        return apply_complex(self.lr, self.li, xx)


class ComplexGELU(nn.Module):
    def __init__(self):
        super(ComplexGELU, self).__init__()
        self.lr = nn.GELU()
        self.li = nn.GELU()

    def forward(self, xx):
        return self.lr(xx.real) + 1j * self.li(xx.real)
