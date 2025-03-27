# Data generation file. Please visit https://www.deepmimo.net/ to download the dataset of 'O1_3p5' scenario and check how to use it.
import DeepMIMO
import numpy as np

param = DeepMIMO.default_params()
param['scenario'] = 'O1_3p5'
param['dataset_folder'] = './'

param['num_paths'] = 25

param['active_BS'] = np.array([3])
param['user_row_first'] = 700
param['user_row_last'] = 1300
param['row_subsampling'] = 1
param['user_subsampling'] = 1

param['ue_antenna']['shape'] = np.array([1, 1, 1])
param['bs_antenna']['shape'] = np.array([8, 1, 4])

param['enable_BS2BS'] = 0
param['activate_OFDM'] = 1
param['OFDM']['bandwidth'] = 0.0192
param['OFDM']['subcarriers'] = 64
param['OFDM']['subcarriers_limit'] = 64

print(param)
dataset = DeepMIMO.generate_data(param)
print(dataset[0]['user']['channel'].shape)
np.save('data.npy', dataset[0]['user']['channel'])
np.save('loc.npy', dataset[0]['user']['location'])
