# Util functions and classes. Including signal generation and csv logging
import csv
from dataclasses import dataclass
import torch
import math

from sympy.core.parameters import distribute
from torch.utils.data import TensorDataset
import os
import numpy as np


def MSE(X, keepdim=False):
    if hasattr(X, 'real'):
        X = X.real * X.real + X.imag * X.imag
    else:
        X = X * X
    return torch.mean(X, dim=tuple(range(1, X.dim())), keepdim=keepdim)


class SignalGeneration:
    def __init__(self, config, noise_config):
        self.num_ant = config.num_ant
        self.num_car = config.num_car
        self.sp_params = SignalProcessingParams(noise_power=noise_config.noise_power,
                                                is_NMSE=noise_config.is_NMSE)

    def get_signal_processing_unit(self, sp_unit_name):
        if hasattr(self, "sp_module_"+sp_unit_name):
            return getattr(self, "sp_module_"+sp_unit_name)
        elif hasattr(self, sp_unit_name):
            return getattr(self, sp_unit_name)
        else:
            raise NotImplementedError

    # H: bs * num_ant * num_car
    def generate_signal(self, H):
        raise NotImplementedError


@dataclass
class SignalProcessingParams:
    noise_power: float
    is_NMSE: bool
    X: [torch.Tensor, None] = None
    Y: [torch.Tensor, None] = None
    sigma_H: [torch.Tensor, None] = None
    sigma_H_sqrt: [torch.Tensor, None] = None
    noise_power_real: [torch.Tensor, None] = None
    noise_power_real_sqrt: [torch.Tensor, None] = None

    def set_params(self, X, Y, **kwargs):
        self.X = X
        self.Y = Y
        self.sigma_H = MSE(self.Y)
        self.sigma_H_sqrt = torch.sqrt(self.sigma_H)
        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def add_noise(self):
        if self.is_NMSE:
            self.Y = self.Y * (self.noise_power * torch.randn_like(self.Y) + 1)
            self.noise_power_real = self.noise_power * self.sigma_H
        else:
            self.Y = self.Y + self.noise_power * torch.randn_like(self.Y)
            self.noise_power_real = self.noise_power * torch.ones(1,)
        self.noise_power_real_sqrt = torch.sqrt(self.noise_power_real)

    def select(self, sel):
        self.Y = torch.index_select(self.Y, 0, sel)
        self.sigma_H_sqrt = torch.index_select(self.sigma_H_sqrt, 0, sel)
        self.sigma_H = torch.index_select(self.sigma_H, 0, sel)
        self.noise_power_real = torch.index_select(self.noise_power_real, 0, sel)
        self.noise_power_real_sqrt = torch.index_select(self.noise_power_real_sqrt, 0, sel)
        self.X = torch.index_select(self.X, 0, sel)

    def calc_err(self, X_pred, sel=None):
        X_err = X_pred - self.X
        if sel is not None:
            X_err = torch.index_select(X_err, 0, sel)
        # print(X_err)
        err = X_err.real.abs().lt(0.01) * X_err.imag.abs().lt(0.01)
        return 1 - torch.mean(err.to(torch.float32), dim=tuple(range(1, X_err.dim())))

    def restart(self, noise_power, is_NMSE):
        self.X = None
        self.Y = None
        self.sigma_H = None
        self.sigma_H_sqrt = None
        self.noise_power_real = None
        self.noise_power_real_sqrt = None
        self.noise_power = noise_power
        self.is_NMSE = is_NMSE


def load_dataset_deepmimo(args):
# def get_dataset(ratio: list, seed=1234, dataset_path='/mnt/HD2/yyz/MIMOlocdata32/', name='.'):
    dest_train = os.path.join(args.dataset_dir, args.dataset_file_name, "channel_train.pt")
    dest_test = os.path.join(args.dataset_dir, args.dataset_file_name, "channel_test.pt")
    dest_val = os.path.join(args.dataset_dir, args.dataset_file_name, "channel_val.pt")  # not always exist

    if os.path.exists(dest_train) and os.path.exists(dest_test):
        channel_train = torch.load(dest_train)
        channel_test = torch.load(dest_test)
        if os.path.exists(dest_val):
            channel_val = torch.load(dest_val)
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test),
                    TensorDataset(channel_val),]
        else:
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test)]
    else:
        channel_file = os.path.join(args.dataset_dir, args.dataset_file_name, "data.npy")
        print("Loading channels", channel_file)

        channel = np.load(channel_file)  # N*1*ant*car complex
        channel_torch = torch.tensor(channel).squeeze(1)  # N*ant*car cfloat tensor
        num_data = channel_torch.shape[0]

        # Normalization
        channel_torch = channel_torch * 1e5

        perm = torch.randperm(num_data)

        num = int(args.ratio[0] * num_data)
        ids = perm[0:num]
        channel_train = channel_torch[ids]
        torch.save(channel_train, dest_train)

        num2 = int(args.ratio[1] * num_data)
        ids = perm[num:num+num2]
        channel_test = channel_torch[ids]
        torch.save(channel_test,  dest_test)

        if len(args.ratio) == 3:
            num3 = int(args.ratio[2] * num_data)
            ids = perm[num+num2:num+num2+num3]
            channel_val = channel_torch[ids]
            torch.save(channel_val,  dest_val)
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test),
                    TensorDataset(channel_val),]
        else:
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test)]

def load_dataset(args):
    if args.dataset == "deepmimo":
        return load_dataset_deepmimo(args)
    else:
        raise NotImplementedError


class Logger:
    def __init__(self, *args):
        self.register(*args)

    def register(self, *args):
        for i in range(len(args)):
            n = args[i]
            setattr(self, n, LogUnit())

    def add(self, **kwargs):
        for k, v in kwargs.items():
            getattr(self, k).update(v)

    def get_result(self):
        name_list = []
        mean_list = []
        dist_list = []
        for attr in dir(self):
            if not attr.startswith('_'):
                attr_ = getattr(self, attr)
                if isinstance(attr_, LogUnit):
                    mean, dist = attr_.get_result()
                    name_list.append(attr)
                    mean_list.append(mean)
                    dist_list.append(dist)
        return name_list, mean_list, dist_list

    def write(self, csvwriter_mean, csvwriter_dist, write_name=False):
        name_list, mean_list, dist_list = self.get_result()
        if write_name:
            csvwriter_mean.writerow(name_list)
        csvwriter_mean.writerow(mean_list)
        for line in dist_list:
            csvwriter_dist.writerow(line)

    def clear(self, name):
        getattr(self, name).clear()

    def clear_all(self):
        for attr in dir(self):
            if not attr.startswith('_'):
                attr = getattr(self, attr)
                if isinstance(attr, LogUnit):
                    attr.clear()


class LogUnit:
    def __init__(self):
        self.sum = 0
        self.num = 0
        self.dist = [0] * 100

    def update(self, val):
        for v in val:
            v = float(v)
            self.sum += v
            self.num += 1
            v_floor = math.floor(v * 100)
            if v_floor < 100:
                self.dist[v_floor] += 1
            else:
                self.dist[99] += 1

    def get_result(self):
        mean = 0 if self.num==0 else self.sum/self.num
        return mean, self.dist

    def clear(self):
        self.sum = 0
        self.num = 0
        self.dist = [0] * 100
