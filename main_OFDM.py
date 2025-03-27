# Main entrance of the project. The detailed training/generation methods are defined in diffusion_training.py and diffusion_generation.py, respectively.
import torch
from accelerate import Accelerator
import os
import csv

from diffusion_training import train_loop, test
from diffusion_generation import DiffusionGeneration
from myDiffuser import MyDDIMScheduler
from utils import load_dataset, Logger
from MixerLayer import Model
from sp_modules import OFDMSignalGeneration
from config import get_config
from CMixer import CMixer


def MSEComplex(x):
    err = x.real * x.real + x.imag * x.imag
    return err.mean(dim=(-1, -2))

def main_OFDM_training(config, config_car, config_ant, config_diff):
    model = Model(config, config_car, config_ant)
    datasets = load_dataset(config)
    train_dataSet = datasets[0]
    test_dataSet = datasets[1]
    train_loader = torch.utils.data.DataLoader(
        train_dataSet,
        batch_size=config.bs,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=config.drop_last)
    test_loader = torch.utils.data.DataLoader(
        test_dataSet,
        batch_size=config.bs,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=config.drop_last)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.step_size,
                                                   gamma=config.gamma_lr, last_epoch=-1)
    noise_scheduler = MyDDIMScheduler(
        num_train_timesteps=config_diff.num_train_timesteps,
        beta_start=config_diff.beta_start,
        beta_end=config_diff.beta_end,
        beta_schedule=config_diff.beta_schedule,
        trained_betas=None,
        clip_sample=config_diff.clip_sample,
        set_alpha_to_one=config_diff.set_alpha_to_one,
        steps_offset=config_diff.steps_offset,
        prediction_type=config_diff.prediction_type,
        thresholding=config_diff.thresholding,
        dynamic_thresholding_ratio=config_diff.dynamic_thresholding_ratio,
        clip_sample_range=config_diff.clip_sample_range,
        sample_max_value=config_diff.sample_max_value,
        timestep_spacing=config_diff.timestep_spacing,
        rescale_betas_zero_snr=config_diff.rescale_betas_zero_snr
    )

    if config.dataset == "deepmimo":
        data_preprocess = lambda data: data[0]
    else:
        raise ValueError

    train_loop(config, model, noise_scheduler, optimizer, train_loader,
               test_loader, lr_scheduler, data_preprocess)


def main_OFDM_generation(config, config_car, config_ant, config_diff, config_noise, model_name):
    noise_scheduler = MyDDIMScheduler(
        num_train_timesteps=config_diff.num_train_timesteps,
        beta_start=config_diff.beta_start,
        beta_end=config_diff.beta_end,
        beta_schedule=config_diff.beta_schedule,
        trained_betas=None,
        clip_sample=config_diff.clip_sample,
        set_alpha_to_one=config_diff.set_alpha_to_one,
        steps_offset=config_diff.steps_offset,
        prediction_type=config_diff.prediction_type,
        thresholding=config_diff.thresholding,
        dynamic_thresholding_ratio=config_diff.dynamic_thresholding_ratio,
        clip_sample_range=config_diff.clip_sample_range,
        sample_max_value=config_diff.sample_max_value,
        timestep_spacing=config_diff.timestep_spacing,
        rescale_betas_zero_snr=config_diff.rescale_betas_zero_snr
    )
    logger = Logger("H_MSE_last", "H_BEST_MSE_last", "Y_NMSE_last", "X_BER_last")
    for i in range(15):
        logger.register("H_MSE_{}".format(i + 1), "H_BEST_MSE_{}".format(i + 1),
                        "H_MSE_gen_{}".format(i + 1), "H_BEST_MSE_gen_{}".format(i + 1),
                        "Y_NMSE_{}".format(i + 1), "X_BER_{}".format(i + 1))
    model = Model(config, config_car, config_ant)
    params_file = os.path.join(config.exp_name, "results", model_name+".pth")
    model.load_state_dict(torch.load(params_file, map_location=torch.device('cpu')))
    datasets = load_dataset(config)
    test_dataSet = datasets[1]
    test_loader = torch.utils.data.DataLoader(
        test_dataSet,
        batch_size=config.bs,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=config.drop_last)

    if config.dataset == "deepmimo":
        data_preprocess = lambda data: data[0]
    else:
        raise ValueError

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # test(model, test_loader, noise_scheduler, data_preprocess)

    signal_generator = OFDMSignalGeneration(config, config_noise, device)
    signal_processing_unit = signal_generator.get_signal_processing_unit(config.sp_unit_name)
    channel_generator = DiffusionGeneration(config, model, noise_scheduler, signal_processing_unit,
                                            signal_generator=signal_generator, logger=logger,
                                            err_to_time=None, generator=None)
    acc = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(config.exp_name, "results", "generation_logs"),
    )
    if acc.is_main_process:
        os.makedirs(os.path.join(config.exp_name, "results"), exist_ok=True)
        acc.init_trackers("generation_example")

    model, test_loader = acc.prepare(model, test_loader)

    def gen(config, config_noise, config_diff):
        signal_generator.restart(config, config_noise, device)
        channel_generator.restart(config, err_to_time=None, generator=None)
        torch.cuda.empty_cache()
        for (i, batch) in enumerate(test_loader):
            clean_channels = data_preprocess(batch)
            # print(clean_channels.shape, 1111)
            sp_params = signal_generator.generate_signal(clean_channels)
            ori_channel = signal_generator.get_ori_channel()
            # ori_channel = clean_channels.unsqueeze(1)
            if config.use_noise_as_init:
                channel_generator.diffusion_generation(sp_params, ori_channel, gamma=config_diff.gamma,
                                                       all_time_steps=config_diff.all_time_steps)
            else:
                raise NotImplementedError

    @torch.no_grad()
    def gen_etas(config, config_diff, config_noise, logger):
        dir_name = config.exp_name + "/gen_results_steps_{}_gamma_{}_num_samples_{}_num_remaining_samples_{}_spacing_{}_{}".format(
            config.all_time_steps, config.gamma, config.num_samples, config.num_remaining_channels, config.pilot_spacing, config.modulation
        )
        os.makedirs(dir_name, exist_ok=True)

        eta1 = [0]
        eta2 = [0.4]
        eta3 = [0]
        noise = [2]
        for i in eta1:
            config.eta1 = i
            for j in eta2:
                config.eta2 = j
                for k in eta3:
                    if i + j + k > 1:
                        break
                    config.eta3 = k
                    print(i, j, k, " start!")
                    file_name = dir_name + "/eta1_{}_eta2_{}_eta3_{}_new".format(i, j, k)
                    f1 = open(file_name + "_mean.csv", "w")
                    csvwriter_mean = csv.writer(f1)
                    f2 = open(file_name + "_distribution.csv", "w")
                    csvwriter_distribution = csv.writer(f2)
                    flag = True
                    for n in noise:
                        noise_power = 10 ** (n / 10)
                        config_noise.noise_power = noise_power
                        config.noise_pow_dB = n
                        config.noise_power = noise_power
                        print(config, vars(config_noise), vars(config_diff))
                        if flag:
                            for (i, batch) in enumerate(test_loader):
                                clean_channels = data_preprocess(batch)
                                # print(clean_channels.shape, 1111)
                                sp_params = signal_generator.generate_signal(clean_channels)
                                ori_channel = signal_generator.get_ori_channel()
                                channel_generator.diffusion_generation(sp_params, ori_channel, gamma=config_diff.gamma,
                                                                        all_time_steps=config_diff.all_time_steps)
                                signal_generator.restart(config, config_noise, device)
                                channel_generator.restart(config, err_to_time=None, generator=None)
                                torch.cuda.empty_cache()
                                break
                            flag = False
                        gen(config, config_noise, config_diff)
                        logger.write(csvwriter_mean, csvwriter_distribution, write_name=False)
                        logger.clear_all()
                    f1.close()
                    f2.close()

    for mod in ["QAM16"]:
        print("Modulation ", mod)
        config.modulation = mod
        for spacing in [10]:
            print("Spacing ", spacing)
            config.pilot_cars = [i for i in range(0,config.num_car,spacing)]
            config.pilot_spacing = spacing
            if True:
                # print("Steps ", steps)
                # config.all_time_steps = steps
                # config_diff.all_time_steps = steps

                # config.num_samples = 1
                # config_diff.num_samples = 1
                # config.num_remaining_channels = 1
                # config_diff.num_remaining_channels = 1
                gen_etas(config, config_diff, config_noise, logger)

                # config.num_samples = 16
                # config_diff.num_samples = 16
                # config.num_remaining_channels = 8
                # config_diff.num_remaining_channels = 8
                # gen_etas(config, config_diff, config_noise, logger)
                #
                # config.num_samples = 32
                # config_diff.num_samples = 32
                # config.num_remaining_channels = 16
                # config_diff.num_remaining_channels = 16
                # gen_etas(config, config_diff, config_noise, logger)
                #
                # config.num_samples = 64
                # config_diff.num_samples = 64
                # config.num_remaining_channels = 16
                # config_diff.num_remaining_channels = 16
                # gen_etas(config, config_diff, config_noise, logger)

def benchmark_training(config, config_car, config_ant, config_diff):
    noise_scheduler = MyDDIMScheduler(
        num_train_timesteps=config_diff.num_train_timesteps,
        beta_start=config_diff.beta_start,
        beta_end=config_diff.beta_end,
        beta_schedule=config_diff.beta_schedule,
        trained_betas=None,
        clip_sample=config_diff.clip_sample,
        set_alpha_to_one=config_diff.set_alpha_to_one,
        steps_offset=config_diff.steps_offset,
        prediction_type=config_diff.prediction_type,
        thresholding=config_diff.thresholding,
        dynamic_thresholding_ratio=config_diff.dynamic_thresholding_ratio,
        clip_sample_range=config_diff.clip_sample_range,
        sample_max_value=config_diff.sample_max_value,
        timestep_spacing=config_diff.timestep_spacing,
        rescale_betas_zero_snr=config_diff.rescale_betas_zero_snr
    )
    model = CMixer(len(config.pilot_cars), config.num_car, config.num_ant, config.num_ant)
    datasets = load_dataset(config)
    train_dataSet = datasets[0]
    test_dataSet = datasets[1]
    train_loader = torch.utils.data.DataLoader(
        train_dataSet,
        batch_size=config.bs,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=config.drop_last)
    test_loader = torch.utils.data.DataLoader(
        test_dataSet,
        batch_size=config.bs,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=config.drop_last)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.step_size,
                                                   gamma=config.gamma_lr, last_epoch=-1)
    if config.dataset == "deepmimo":
        data_preprocess = lambda data: data[0]
    else:
        raise ValueError

        # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.exp_name, "results", "logs"),
    )
    if accelerator.is_main_process:
        os.makedirs(os.path.join(config.exp_name, "results"), exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        # progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        # progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_loader):
            clean_channels = data_preprocess(batch)
            with torch.no_grad():
                mean = torch.mean(clean_channels, dim=(1, 2), keepdim=True)
                var1 = torch.mean(clean_channels.real * clean_channels.real, dim=(1, 2),
                                  keepdim=True) - mean.real * mean.real
                var2 = torch.mean(clean_channels.imag * clean_channels.imag, dim=(1, 2),
                                  keepdim=True) - mean.imag * mean.imag
                std = torch.sqrt(var1 + var2)
            clean_channels = (clean_channels - mean) / std

            # Sample noise to add to the images
            noise = torch.randn_like(clean_channels)
            bs = clean_channels.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_channels.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_channels = noise_scheduler.add_noise(clean_channels, noise, timesteps)
            #noisy_channels = clean_channels#  + noise * torch.rand(bs, 1, 1).to(clean_channels.device)
            noisy_channels = noisy_channels[:, :, 0:config.num_car:config.pilot_spacing]

            with accelerator.accumulate(model):
                # Predict the noise residual
                pred = model(noisy_channels)
                if config.prediction_type == "epsilon":
                    loss = MSEComplex(pred - noise).mean()
                elif config.prediction_type == "sample":
                    loss = MSEComplex(pred - clean_channels).mean()
                elif config.prediction_type == "v_prediction":
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            # progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        lr_scheduler.step()

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            torch.save(model.state_dict(),
                       os.path.join(config.exp_name, "results", "model_epoch%d.pth" % (epoch + 1)))


@torch.inference_mode()
def benchmark_testing(config, config_noise, model_name):
    model = CMixer(len(config.pilot_cars), config.num_car, config.num_ant, config.num_ant)
    params_file = os.path.join(config.exp_name, "results", model_name+".pth")
    model.load_state_dict(torch.load(params_file, weights_only=True, map_location=torch.device('cpu')))
    datasets = load_dataset(config)
    test_dataSet = datasets[1]
    test_loader = torch.utils.data.DataLoader(
        test_dataSet,
        batch_size=config.bs,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=config.drop_last)

    if config.dataset == "deepmimo":
        data_preprocess = lambda data: data[0]
    else:
        raise ValueError

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    acc = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(config.exp_name, "results", "generation_logs"),
    )
    if acc.is_main_process:
        os.makedirs(os.path.join(config.exp_name, "results"), exist_ok=True)
        acc.init_trackers("generation_example")

    model, test_loader = acc.prepare(model, test_loader)

    noise = [0, 2, 4]
    NMSE = []
    for n in noise:
        noise_power = 10 ** (n / 10)
        config_noise.noise_power = noise_power
        config.noise_pow_dB = n
        config.noise_power = noise_power

        acc_err = 0.
        num_samples = 0
        for batch in test_loader:
            clean_channels = data_preprocess(batch)
            mean = torch.mean(clean_channels, dim=(1, 2), keepdim=True)
            var1 = torch.mean(clean_channels.real * clean_channels.real, dim=(1, 2),
                              keepdim=True) - mean.real * mean.real
            var2 = torch.mean(clean_channels.imag * clean_channels.imag, dim=(1, 2),
                              keepdim=True) - mean.imag * mean.imag
            std = torch.sqrt(var1 + var2)
            clean_channels = (clean_channels - mean) / std

            channel = clean_channels[:, :, 0:config.num_car:config.pilot_spacing]

            # Sample noise to add to the images
            noise = torch.randn_like(channel)
            bs = clean_channels.shape[0]

            channel = channel + noise * noise_power
            mean_ = torch.mean(channel, dim=(1, 2), keepdim=True)
            var1_ = torch.mean(channel.real * channel.real, dim=(1, 2),
                              keepdim=True) - mean_.real * mean_.real
            var2_ = torch.mean(channel.imag * channel.imag, dim=(1, 2),
                              keepdim=True) - mean_.imag * mean_.imag
            std_ = torch.sqrt(var1_ + var2_)
            channel_input = (channel - mean_) / std_
            channel_output = model(channel_input) * std_ + mean_
            err = MSEComplex(channel_output - clean_channels).mean()

            acc_err += err * bs
            num_samples += bs

        NMSE.append(acc_err / num_samples)
    with open(os.path.join(config.exp_name, "results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([0,2,4,6])
        writer.writerow(NMSE)


exp_name = None
config, config_car, config_ant, config_diff, config_noise = get_config(exp_name)
model_name = "model_epoch%d" % config.num_epochs
if config.task == 'train':
    main_OFDM_training(config, config_car, config_ant, config_diff)
elif config.task == 'benchmark':
    benchmark_training(config, config_car, config_ant, config_diff)
elif config.task == 'generate':
    main_OFDM_generation(config, config_car, config_ant, config_diff, config_noise, model_name)
elif config.task == 'test_benchmark':
    benchmark_testing(config, config_car, model_name)


