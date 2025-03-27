# Definition of NN with time embedding
import torch
import torch.nn as nn
import math

from myComplexLayers import ComplexLinear, ComplexAvgPool1d, ComplexGELU, ComplexConv1d, ComplexConvTranspose1d


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_step, embed_dim=256, flat=2):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_step, embed_dim), persistent=False)
        mid_dim = int(embed_dim * flat)
        self.projection = nn.Sequential(
            ComplexLinear(embed_dim, mid_dim, bias=True),
            ComplexGELU(),
            ComplexLinear(mid_dim, embed_dim, bias=True),
        )

    def forward(self, t):
        if t.dtype in [torch.int32, torch.int64]:
            x = self.embedding[t]
        else:
            x = self._lerp_embedding(t)
        return self.projection(x)

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_step, embed_dim):
        steps = torch.arange(max_step).unsqueeze(1)  # [T, 1]
        dims = torch.arange(embed_dim).unsqueeze(0)  # [1, E]
        table = steps * torch.exp(-math.log(max_step)
                                  * dims / embed_dim)  # [T, E]
        table = torch.exp(1j * table)
        return table


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, mid=None, flat=2, embed=None):
        super(LinearBlock, self).__init__()
        if mid is None:
            if in_dim > out_dim:
                middle_dim = int(in_dim * flat)
            else:
                middle_dim = int(out_dim * flat)
        else:
            middle_dim = mid

        self.layers = nn.Sequential(
            ComplexLinear(in_dim, middle_dim),
            ComplexGELU(),
            ComplexLinear(middle_dim, out_dim))
        if embed is not None:
            self.time_embedding_scale = nn.Sequential(
                ComplexGELU(),
                ComplexLinear(embed, in_dim)
            )
            self.time_embedding_shift = nn.Sequential(
                ComplexGELU(),
                ComplexLinear(embed, in_dim)
            )
            self.time_embedding_gate = nn.Sequential(
                ComplexGELU(),
                ComplexLinear(embed, out_dim)
            )
        self.embed = (embed is not None)

    def forward(self, x, t=None):
        if self.embed:
            shift = self.time_embedding_shift(t)
            scale = self.time_embedding_scale(t)
            x = x * (1 + scale) * shift
            return self.time_embedding_gate(t) * self.layers(x)
        else:
            return self.layers(x)



class Model(nn.Module):
    def __init__(self, configs, configs_car, configs_ant):
        super(Model, self).__init__()
        self.configs = configs
        self.configs_car = configs_car
        self.configs_ant = configs_ant
        if configs.channel_independence is None:
            self.channel_independence = True
        else:
            self.channel_independence = (configs.channel_independence == 1)

        self.num_layers = configs.num_layers
        # 0: car -> ant, 1: ant -> car,
        if configs.car_first == 1:
            self.framework_indicator = 0
        else:
            self.framework_indicator = 1
        if configs_car.model == 'Linear':
            self.car_layers = nn.ModuleList([LinearBlock(configs_car.input_len, configs_car.seq_len,
                                                         flat=configs_car.flat, embed=configs_car.embed_dim)])
            for i in range(configs.num_layers - 1):
                self.car_layers.append(LinearBlock(configs_car.seq_len, configs_car.seq_len,
                                                   flat=configs_car.flat, embed=configs_car.embed_dim))

        if configs_ant.model == 'Linear':
            self.ant_layers = nn.ModuleList([LinearBlock(configs_ant.input_len, configs_ant.seq_len,
                                                         flat=configs_ant.flat, embed=configs_ant.embed_dim)])
            for i in range(configs.num_layers - 1):
                self.ant_layers.append(LinearBlock(configs_ant.seq_len, configs_ant.seq_len,
                                                   flat=configs_ant.flat, embed=configs_ant.embed_dim))
        self.time_embedding = DiffusionEmbedding(configs.num_train_timesteps, configs_car.embed_dim, configs_ant.flat)


    # xx: B * ant * car, t: B
    def forward(self, xx, t):
        t = self.time_embedding(t).unsqueeze(1)

        if self.framework_indicator == 0:
            # total_params = sum(p.numel() for p in self.parameters())
            # print(f"Total number of trainable parameters: {total_params}")
            for i in range(self.num_layers):
                xx_ = self.car_layers[i].forward(xx, t)
                xx = xx + xx_
                # xx_ = rearrange(xx, 'b ant car -> b car ant')
                xx = xx.transpose(-1,-2)
                xx_ = self.ant_layers[i].forward(xx, t)
                xx = xx + xx_
                xx = xx.transpose(-1,-2)
                # xx_ = rearrange(xx_, 'b car ant -> b ant car')
        elif self.framework_indicator == 1:
            for i in range(self.num_layers):
                # xx = rearrange(xx_, 'b ant car -> b car ant')
                xx = xx.transpose(-1,-2)
                xx_ = self.ant_layers[i].forward(xx, t)
                xx = xx + xx_
                xx = xx.transpose(-1,-2)
                # xx_ = rearrange(xx, 'b car ant -> b ant car')
                xx_ = self.car_layers[i].forward(xx, t)
                xx = xx + xx_
        else:
            raise ValueError

        return xx
