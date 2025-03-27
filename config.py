# Configs
# NOTE: some of the configs are abundant and related to other works which have not been removed.
# Please do not make changes here and directly work on the .sh files unless carefully checked.
import argparse
import torch
import os
import random
import numpy as np
import json
import time

C = 299792458

fix_seed = 1234
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


class EmptyClass:
    pass


def str_or_int(value):
    try:
        return int(value)
    except ValueError:
        return value


def get_config(name=None):
    if name is None:
        return get_config_from_input()
    else:
        with (open(os.path.join(name, 'config.json'), 'r') as f):
            configs = argparse.Namespace(**json.load(f))
        with (open(os.path.join(name, 'config_car.json'), 'r') as f):
            configs_car = argparse.Namespace(**json.load(f))
        with (open(os.path.join(name, 'config_ant.json'), 'r') as f):
            configs_ant = argparse.Namespace(**json.load(f))
        with (open(os.path.join(name, 'config_diff.json'), 'r') as f):
            configs_diff = argparse.Namespace(**json.load(f))
        with (open(os.path.join(name, 'config_noise.json'), 'r') as f):
            configs_noise = argparse.Namespace(**json.load(f))

        configs.BS_ant = torch.tensor(configs.BS_ant)
        if torch.cuda.is_available():
            configs.BS_ant = configs.BS_ant.cuda()

        return configs, configs_car, configs_ant, configs_diff, configs_noise


def get_config_from_input():
    parser = argparse.ArgumentParser(description='LargeMLPMixer')

    # basic config for data preprocessing
    parser.add_argument('--raw_data_dir', type=str, default='../../Dataset',
                        help='raw dataset directory')
    parser.add_argument('--basic_file_name', type=str, default='all',
                        help='raw dataset basic file name, all means all the file under the directory')
    # 'MIMO_MovingPointDemo.paths.t001_48.r%3d'
    parser.add_argument('--file_ids', nargs='+', type=int, default=[55], help='dataset basic file ids')
    parser.add_argument('--max_num_path', type=int, default=25, help='max # of paths')
    parser.add_argument('--gain_offset', type=float, default=70, help='gain offset in dB')
    parser.add_argument('--gain_gate', type=float, default=-70,
                        help='gain gate in dB. If no path gains exceed this gate, this user is aborted')

    # basic config
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='dataset directory')
    parser.add_argument('--dataset', type=str, default='wirelessinsight',
                        help='dataset, wirelessinsight or deepmimo')
    parser.add_argument('--divided_dataset_dir', type=str, default='divided',
                        help='directory for divided datasets under dataset directory')
    parser.add_argument('--dataset_file_name', type=str, default='dataset.pth', help='dataset file name')
    parser.add_argument('--ratio', nargs='+', type=float, default=[0.8, 0.2],
                        help='ratios for [train, test(, val)] datasets')

    # communication scenario config
    parser.add_argument('--scenario', type=str, default='stable',
                        help='scenario, options: [stable, moving], in case of moving, moving params should be given')
    parser.add_argument('--num_car', type=int, default=1024, help='# of carriers')
    parser.add_argument('--sampling_freq', type=float, default=20000000, help='sampling frequency')
    parser.add_argument('--freq', type=float, default=3.5e9,
                        help='carrier frequency, do not change unless changing datasets')
    parser.add_argument('--ant_shape', nargs='+', type=int, default=[1, 1, 1], help='Shape of BS antennas')

    # NN config
    # Overall mixer NN config
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--num_layers', type=int, default=5, help='# of mixer layers')
    parser.add_argument('--car_first', type=int, default=1, help='0: antenna first 1: carrier first')

    # required options
    parser.add_argument('--model_car', type=str, required=True, default='Linear',
                        help='Model name for carriers, options:[Linear, Conv, TimeMixer]')
    parser.add_argument('--model_ant', type=str, required=True, default='Linear',
                        help='Model name for BS antennas, options:[Linear, Conv, TimeMixer]')

    # Configs for carrier/BS ant units
    # For Linear/Conv
    parser.add_argument('--flat_car', type=float, default=2,
                        help='inflated param of middle layers for carriers for linear/conv')
    parser.add_argument('--flat_ant', type=float, default=2,
                        help='inflated param of middle layers for BS antennas for linear/conv')
    parser.add_argument('--kernel_size_car', type=int, default=15, help='kernel size for carriers for conv')
    parser.add_argument('--kernel_size_ant', type=int, default=15, help='kernel size for BS antennas for conv')

    # For TimeMixer
    parser.add_argument('--down_sampling_method_car', type=str, default='conv', help='down sampling method for carriers')
    parser.add_argument('--down_sampling_layers_car', type=int, default=5, help='num of down sampling layers for carriers')
    parser.add_argument('--down_sampling_window_car', type=int, default=2, help='down sampling window size for carriers')
    parser.add_argument('--linear_gate_car', type=int, default=512, help='gate between Linear and Conv for carriers')
    parser.add_argument('--enc_in_car', type=int, default=10, help='conv kernel size of conv down sampling for carriers')
    parser.add_argument('--down_sampling_method_ant', type=str, default='conv', help='down sampling method for BS ant')
    parser.add_argument('--down_sampling_layers_ant', type=int, default=5, help='num of down sampling layers for BS ant')
    parser.add_argument('--down_sampling_window_ant', type=int, default=2, help='down sampling window size for BS ant')
    parser.add_argument('--linear_gate_ant', type=int, default=64, help='gate between Linear and Conv for BS ant')
    parser.add_argument('--enc_in_ant', type=int, default=10, help='conv kernel size of conv down sampling for BS ant')

    parser.add_argument('--moving_avg_car', type=int, default=25, help='window size of moving average for carriers')
    parser.add_argument('--top_k_car', type=int, default=5, help='for TimesBlock for carriers')
    parser.add_argument('--decomp_method_car', type=str, default='moving_avg',
                        help='method of series decompsition for carriers, only support moving_avg or dft_decomp')
    parser.add_argument('--moving_avg_ant', type=int, default=25, help='window size of moving average for BS ant')
    parser.add_argument('--top_k_ant', type=int, default=5, help='for TimesBlock for BS ant')
    parser.add_argument('--decomp_method_ant', type=str, default='moving_avg',
                        help='method of series decompsition for BS ant, only support moving_avg or dft_decomp')

    # optimization
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--drop_last', type=bool, default=False, help='data loader drop last')

    # required options
    parser.add_argument('--exp_name', type=str, default='', help='name of experiment, can be omitted')
    parser.add_argument('--task', type=str, default='train', required=True, help='train or generate')
    parser.add_argument('--lr', type=float, default=1e-3, help='original learning rate')
    parser.add_argument('--step_size', type=int, default=1, help='learning rate decay step size')
    parser.add_argument('--gamma_lr', type=float, default=0.95, help='learning rate decay each step')
    parser.add_argument('--num_epochs', type=int, default=400, help='# of training epochs')
    parser.add_argument('--save_model_epochs', type=int, default=5, help='saving every # epochs')
    parser.add_argument('--save_image_epochs', type=int, default=5, help='overall testing every # epochs')
    parser.add_argument('--mixed_precision', type=str, default='no', help='mixed precision training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')

    # diffusion
    parser.add_argument('--num_remaining_channels', type=int, default=16, help='remaining channels in diff')
    parser.add_argument('--substitute_channel', type=bool, default=False, help='substitute channel in diff')
    parser.add_argument('--recovery_err_gate', type=float, default=2., help='recovery error in diff')
    parser.add_argument('--num_samples', type=int, default=8, help='# of samples in diff')
    parser.add_argument('--num_train_timesteps', type=int, default=1000, help='# of training timesteps in diff')
    parser.add_argument('--all_time_steps', type=int, default=100, help='# of generation timesteps in diff')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='beta starting value')
    parser.add_argument('--beta_end', type=float, default=0.02, help='beta ending value')
    parser.add_argument('--beta_schedule', type=str, default='linear', help='beta schedule')
    parser.add_argument('--clip_sample', type=bool, default=False, help='clip sample')
    parser.add_argument('--set_alpha_to_one', type=bool, default=True, help='set alpha to one')
    parser.add_argument('--steps_offset', type=int, default=0, help='steps offset')
    parser.add_argument('--prediction_type', type=str, default='epsilon', help='prediction type')
    parser.add_argument('--thresholding', type=bool, default=False, help='thresholding')
    parser.add_argument('--dynamic_thresholding_ratio', type=float, default=0.995, help='dynamic thresholding ratio')
    parser.add_argument('--clip_sample_range', type=float, default=1.0, help='clip sample range')
    parser.add_argument('--sample_max_value', type=float, default=1.0, help='sample max value')
    parser.add_argument('--timestep_spacing', type=str, default='leading', help='timestep spacing')
    parser.add_argument('--rescale_betas_zero_snr', type=bool, default=False, help='rescale betas zero snr')
    parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
    parser.add_argument('--gamma', type=float, default=0.3, help='gamma in diffusion')
    parser.add_argument('--eta1', type=float, default=0.3, help='eta1 in diffusion')
    parser.add_argument('--eta2', type=float, default=0.1, help='eta2 in diffusion')
    parser.add_argument('--eta3', type=float, default=0.6, help='eta3 in diffusion')
    parser.add_argument('--use_grad', type=bool, default=True, help='use_grad in diffusion')

    # sp module
    parser.add_argument('--pilot_spacing', type=int, default=16, help='pilot spacing (in # of subcarriers)')
    parser.add_argument('--modulation', type=str, default='QPSK', help='modulation type, QPSK or QAM16')
    parser.add_argument('--decoding', type=str, default='hard', help='decoding method, soft or hard')
    parser.add_argument('--sp_unit_name', type=str, default='naive', help='signal processing unit name')
    parser.add_argument('--use_noise_as_init', type=bool, default=True, help='use noise as init signal')

    # noise
    parser.add_argument('--noise_pow_dB', type=float, default=0.0, help='noise power')
    parser.add_argument('--noise_NMSE', type=bool, default=True, help='noise is NMSE (true) or MSE (false)')

    configs = parser.parse_args()

    # GPU config
    configs.use_gpu = True if torch.cuda.is_available() else False

    s = configs.ant_shape

    configs.num_ant = s[0] * s[1] * s[2]
    BS_ant = torch.zeros(s[0], s[1], s[2], 3)
    BS_ant = BS_ant + torch.arange(-(s[0] - 1) / 2, s[0] / 2)\
        .view(s[0], 1, 1, 1) * (torch.tensor([1, 0, 0]).view(1, 1, 1, 3))
    BS_ant = BS_ant + torch.arange(-(s[1] - 1) / 2, s[1] / 2)\
        .view(1, s[1], 1, 1) * (torch.tensor([0, 1, 0]).view(1, 1, 1, 3))
    BS_ant = BS_ant + torch.arange(-(s[2] - 1) / 2, s[2] / 2)\
        .view(1, 1, s[2], 1) * (torch.tensor([0, 0, 1]).view(1, 1, 1, 3))
    configs.BS_ant = BS_ant.view(-1, 3).tolist()

    configs.in_car = configs.num_car
    configs.in_ant = configs.num_ant
    configs.step_car = configs.num_car//configs.in_car
    configs.step_ant = configs.num_ant//configs.in_ant
    configs.pilot_cars = [i for i in range(0,configs.num_car,configs.pilot_spacing)]

    configs_car = EmptyClass()
    configs_ant = EmptyClass()

    configs_car.model = configs.model_car
    configs_ant.model = configs.model_ant
    # Linear/Conv
    configs_car.input_len = configs.in_car
    configs_ant.input_len = configs.in_ant
    configs_car.seq_len = configs.num_car
    configs_ant.seq_len = configs.num_ant
    configs_car.flat = configs.flat_car
    configs_ant.flat = configs.flat_ant
    configs_car.kernel_size = configs.kernel_size_car
    configs_ant.kernel_size = configs.kernel_size_ant
    # TimeMixer
    configs_car.down_sampling_method = configs.down_sampling_method_car
    configs_car.down_sampling_layers = configs.down_sampling_layers_car
    configs_car.down_sampling_window = configs.down_sampling_window_car
    configs_car.linear_gate = configs.linear_gate_car
    configs_car.enc_in = configs.enc_in_car
    configs_ant.down_sampling_method = configs.down_sampling_method_ant
    configs_ant.down_sampling_layers = configs.down_sampling_layers_ant
    configs_ant.down_sampling_window = configs.down_sampling_window_ant
    configs_ant.linear_gate = configs.linear_gate_ant
    configs_ant.enc_in = configs.enc_in_ant
    configs_car.embed_dim = configs.embed_dim
    configs_ant.embed_dim = configs.embed_dim

    configs_car.length_list = [configs_car.seq_len // (configs_car.down_sampling_window ** i)
                               for i in range(configs_car.down_sampling_layers+1)]
    configs_ant.length_list = [configs_ant.seq_len // (configs_ant.down_sampling_window ** i)
                               for i in range(configs_ant.down_sampling_layers+1)]
    configs_car.moving_avg = configs.moving_avg_car
    configs_car.top_k = configs.top_k_car
    configs_car.decomp_method = configs.decomp_method_car
    configs_ant.moving_avg = configs.moving_avg_ant
    configs_ant.top_k = configs.top_k_ant
    configs_ant.decomp_method = configs.decomp_method_ant

    configs_noise = EmptyClass()
    configs_noise.noise_power = 10 ** (configs.noise_pow_dB / 10)
    configs_noise.is_NMSE = configs.noise_NMSE

    configs_diff = EmptyClass()
    configs_diff.num_samples = configs.num_samples
    configs_diff.num_remaining_channels = configs.num_remaining_channels
    configs_diff.substitute_channel = configs.substitute_channel
    configs_diff.recovery_err_gate = configs.recovery_err_gate
    configs_diff.num_train_timesteps = configs.num_train_timesteps
    configs_diff.all_time_steps = configs.all_time_steps
    configs_diff.beta_start = configs.beta_start
    configs_diff.beta_end = configs.beta_end
    configs_diff.beta_schedule = configs.beta_schedule
    configs_diff.clip_sample = configs.clip_sample
    configs_diff.set_alpha_to_one = configs.set_alpha_to_one
    configs_diff.steps_offset = configs.steps_offset
    configs_diff.prediction_type = configs.prediction_type
    configs_diff.thresholding = configs.thresholding
    configs_diff.dynamic_thresholding_ratio = configs.dynamic_thresholding_ratio
    configs_diff.clip_sample_range = configs.clip_sample_range
    configs_diff.sample_max_value = configs.sample_max_value
    configs_diff.timestep_spacing = configs.timestep_spacing
    configs_diff.rescale_betas_zero_snr = configs.rescale_betas_zero_snr
    configs_diff.gamma = configs.gamma

    print('Args in experiment:')
    if configs.exp_name == '':
        configs.exp_name = 'Exp_{}_{}_{}_{}_{}_{}_{}'.format(configs.in_ant, configs.in_car,
                                                             configs.num_ant, configs.num_car,
                                                             configs.model_ant, configs.model_car,
                                                             configs_diff.prediction_type)
    os.makedirs(configs.exp_name, exist_ok=True)
    print(configs)
    with open(os.path.join(configs.exp_name, 'config.json'), 'w') as f:
        json.dump(vars(configs), f, indent=4)
    with open(os.path.join(configs.exp_name, 'config_car.json'), 'w') as f:
        json.dump(vars(configs_car), f, indent=4)
    with open(os.path.join(configs.exp_name, 'config_ant.json'), 'w') as f:
        json.dump(vars(configs_ant), f, indent=4)
    with open(os.path.join(configs.exp_name, 'config_diff.json'), 'w') as f:
        json.dump(vars(configs_diff), f, indent=4)
    with open(os.path.join(configs.exp_name, 'config_noise.json'), 'w') as f:
        json.dump(vars(configs_noise), f, indent=4)

    configs.BS_ant = torch.tensor(configs.BS_ant)
    if torch.cuda.is_available():
        configs.BS_ant = configs.BS_ant.cuda()

    return configs, configs_car, configs_ant, configs_diff, configs_noise
