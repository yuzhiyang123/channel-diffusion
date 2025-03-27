#Example for benchmark training. Changing --task to train for diffusion training

python main_OFDM.py --dataset deepmimo --dataset_file_name 32ant_64car_300k --num_car 64 --ant_shape 32 1 1 --model_car Linear --model_ant Linear --task benchmark --bs 256 --exp_name benchmark_diffusion_10 --prediction_type sample --num_train_timesteps 1000 --num_epochs 50 --save_model_epochs 10 --pilot_spacing 10
