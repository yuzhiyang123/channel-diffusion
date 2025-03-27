# Example for generation

python main_OFDM.py --dataset deepmimo --dataset_file_name 32ant_64car_300k --num_car 64 --ant_shape 32 1 1 \
  --model_car Linear --model_ant Linear --task generate --bs 64 --exp_name step1000samplenew --pilot_spacing 10 \
  --prediction_type sample --num_train_timesteps 1000 --num_epochs 50 --gamma 0.9 --all_time_steps 100 \
  --noise_pow_dB 0 --recovery_err_gate 0 --modulation QAM16 --num_samples 16 --num_remaining_channels 8 \
  --sp_unit_name grad --eta1 0 --eta2 0.4 --eta3 0
