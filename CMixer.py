# A simple complex mixer implement of the paper:
# Z. Chen, Z. Zhang, Z. Yang, et al., “Channel mapping based on interleaved learning with complex-domain MLP-Mixer,” IEEE Wireless Communications Letters, vol. 13, no. 5, pp. 1369–1373, 2024.
# Only used for benchmark tests.
from torch import nn
from myComplexLayers import *

class CMixerUnit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CMixerUnit, self).__init__()
        self.layers = nn.Sequential(ComplexLinear(in_dim, out_dim * 2),
                                    ComplexGELU(),
                                    ComplexLinear(out_dim * 2, out_dim))

    def forward(self, x):
        return self.layers(x)


class CMixerLayer(nn.Module):
    def __init__(self, in_car, out_car, in_ant, out_ant):
        super(CMixerLayer, self).__init__()
        self.layer_car = CMixerUnit(in_car, out_car)
        self.layer_ant = CMixerUnit(in_ant, out_ant)

    def forward(self, x):
        x_new = self.layer_car.forward(x).transpose(1, 2)
        x_new = self.layer_ant.forward(x_new).transpose(1, 2)
        return x + x_new


class CMixer(nn.Module):
    def __init__(self, in_car, out_car, in_ant, out_ant):
        super(CMixer, self).__init__()
        assert in_ant == out_ant
        self.input_linear = ComplexLinear(in_car, out_car)
        self.layers = nn.Sequential(*[CMixerLayer(out_car, out_car, out_ant, out_ant) for _ in range(5)])

    def forward(self, x):
        mean = x.mean(dim=[1, 2], keepdim=True)
        var = x.var(dim=[1, 2], keepdim=True, unbiased=False)
        x = (x - mean) / var
        x = self.layers.forward(self.input_linear.forward(x))
        return x * var + mean
