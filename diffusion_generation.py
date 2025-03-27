import torch
import math


def MSEComplex(x):
    err = x.real * x.real + x.imag * x.imag
    return err.mean(dim=(-1, -2))


class DiffusionGeneration:
    def __init__(self,
                 config,
                 model,
                 noise_scheduler,
                 signal_processing_unit,  # recover the transmitted signals & the received signals,
                                          # return loss & enhanced channels
                 signal_generator,
                 logger=None,
                 err_to_time=None,
                 generator=None):
        self.num_samples = config.num_samples
        self.num_remaining_channels = config.num_remaining_channels
        self.substitute_channel = config.substitute_channel
        self.recovery_err_gate = config.recovery_err_gate
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.signal_processing_unit = signal_processing_unit
        self.signal_generator = signal_generator
        self.pilot_mask = signal_generator.pilot_mask
        self.generator = generator
        self.err_to_time = err_to_time
        self.num_car = config.num_car
        self.num_pilot = len(config.pilot_cars)
        self.logger = logger
        self.eta1 = config.eta1
        self.eta2 = config.eta2
        self.eta3 = config.eta3
        self.ori_est = None
        self.Y_err_rem = None
        self.H_rem = None
        self.H_grad_rem = None
        self.recovered_X_rem = None
        self.output_step = (config.all_time_steps // 10)
        if self.output_step == 0:
            self.output_step = 1

    # channel: bs * num_ant * num_car
    def diffusion_generation(self,
                             sp_params,  # should be a data class containing Y and other params
                             channel,  # input channel (original estimation)
                             gamma,
                             all_time_steps=100):
        # 1. generate original noise for diffusion
        channel_shape = channel.shape
        bs = channel_shape[0]
        total_ener =  torch.mean(1 + sp_params.noise_power_real / sp_params.sigma_H)
        delta = math.sqrt(1 - gamma * gamma)
        # gamma = torch.sqrt((self.num_car / self.num_pilot) / total_ener) * gamma
        #
        # # delta = torch.mean(torch.sqrt(sp_params.noise_power_real / sp_params.sigma_H)) * gamma
        # gamma = 2.8
        # delta = 0
        # gamma = gamma.view(-1, 1, 1, 1)
        # delta = delta.view(-1, 1, 1, 1)
        # print(gamma.shape, self.pilot_mask.shape, channel_shape)

        shape_sampled = torch.Size([bs, self.num_remaining_channels * self.num_samples]) + channel_shape[2:]
        channel_gen = torch.randn(shape_sampled, device=channel.device, dtype=channel.dtype)
        channel_gen_mask = torch.ones_like(channel_gen) - self.pilot_mask.unsqueeze(1)
        self.ori_est = torch.sqrt((self.num_car / self.num_pilot) / total_ener) * self.pilot_mask.unsqueeze(1) * channel
        channel_estimated = gamma * self.ori_est

        channel = channel_estimated + delta * channel_gen * channel_gen_mask
        # target_alpha = gamma * gamma * self.num_pilot / self.num_car
        target_alpha = (self.num_pilot / self.num_car) / total_ener
        # print(target_alpha)
        # channel = self.pilot_mask * channel * 10
        t0 = (self.noise_scheduler.get_time_by_alpha(target_alpha) * torch.ones(1,)).to(torch.int32)
        if torch.cuda.is_available():
            t0 = t0.cuda()

        # t0 = 400
        # print(t0)

        # 2. start generation
        # channel: bs * (1 or num_remaining_channels) * num_ant [* mum_car] * num_car
        if self.substitute_channel:
            assert self.err_to_time is not None
            raise NotImplementedError
        else:
            # all_X_recovered = []
            # all_seq = []
            t = t0
            count = 0
            while t > 0:
                count += 1
                t_next = t - t0 // all_time_steps
                if (count % self.output_step) == 0:
                    if_print = count // self.output_step
                else:
                    if_print = 0
                end_flag, channel = self.generate_step(channel, t, t_next, sp_params, if_print=if_print)
                # print(end_flag, 22222)
                if all(end_flag.view(-1)):
                    # print(11111)
                    self.Y_err_rem = None
                    self.H_rem = None
                    self.H_grad_rem = None
                    self.recovered_X_rem = None
                    return None
                t = t_next

            # 3. output
            # channel = channel.reshape(bs, -1, *channel_shape[1:])
            self.generate_last_step(channel, sp_params)
            self.Y_err_rem = None
            self.H_rem = None
            self.H_grad_rem = None
            self.recovered_X_rem = None
            return None

    # channel: bs * (num_remaining_channels * num_samples) * num_ant [* mum_car] * num_car
    def generate_step(self,
                      in_channel,
                      time_step_now,
                      t_next,
                      sp_params,  # a list of parameters used for the signal processing unit
                      if_print=False
                      ):
        shape = in_channel.shape
        shape_sampled = torch.Size([shape[0], -1]) + shape[2:]
        # print(shape_sampled, shape)
        # bs * (num_samples * num_in_channels) * num_car [* mum_car] * num_ant
        out_channel = self.model(torch.flatten(in_channel, start_dim=0, end_dim=1),
                                 time_step_now).reshape(shape_sampled)
        # mse, best_mse = self.signal_generator.calc_H_err_normalized(out_channel)
        recovered_Y_err, recovered_X, H_grad, _ = self.signal_processing_unit(out_channel)
        if if_print > 0:
            mse, best_mse = self.signal_generator.calc_H_err_normalized(out_channel)
            exec("self.logger.add(H_MSE_gen_{}=mse, H_BEST_MSE_gen_{}=best_mse)".format(if_print, if_print))

        if self.Y_err_rem is not None:
            # print(recovered_Y_err[3], self.Y_err_rem[3])
            # print(recovered_Y_err.shape, self.Y_err_rem.shape)
            recovered_Y_err = torch.cat((recovered_Y_err, self.Y_err_rem), 1)
            out_channel = torch.cat((out_channel, self.H_rem), 1)
            H_grad = torch.cat((H_grad, self.H_grad_rem), 1)
            recovered_X = torch.cat((recovered_X, self.recovered_X_rem), 1)
            in_channel = torch.cat((in_channel, in_channel), 1)

        # bs * (num_samples * num_in_channels)
        recovered_Y_err, indices = torch.sort(recovered_Y_err, dim=1, descending=False)
        indices = indices.view(indices.size(0), indices.size(1), *[1 for _ in out_channel.shape[2:]])

        # print(recovered_Y_err[3])
        # Indicating the data id in a batch terminated by the err criteria
        end_flag = recovered_Y_err[:, 0].lt(self.recovery_err_gate * sp_params.noise_power_real).squeeze()
        indices_expanded = indices.expand_as(recovered_X)
        recovered_X = torch.gather(recovered_X, 1, indices_expanded[:, :self.num_remaining_channels])

        # print(best_mse, recovered_Y_err[:,0]/sp_params.sigma_H.view(-1), sp_params.calc_err(recovered_X[:,:1]))
        # print(end_flag)
        # time = time_step_now / self.noise_scheduler.config.num_train_timesteps
        # self.logger.add(time=([time] * int(sum(sel))))
        if if_print > 0:
            mse, best_mse = self.signal_generator.calc_H_err_normalized(in_channel)
            ber = sp_params.calc_err(recovered_X[:,:1])
            exec("self.logger.add(H_MSE_{}=mse, H_BEST_MSE_{}=best_mse, X_BER_{}=ber)".format(if_print, if_print, if_print))
            exec("self.logger.add(Y_NMSE_{}=recovered_Y_err[:,0]/sp_params.sigma_H.view(-1)-sp_params.noise_power)".format(if_print))

        if H_grad.shape[1] > self.num_remaining_channels:
            indices_expanded = indices.expand_as(H_grad)
            indices_expanded = indices_expanded[:, :self.num_remaining_channels]
            out_channel = torch.gather(out_channel, 1, indices_expanded)
            in_channel = torch.gather(in_channel, 1, indices_expanded)
            H_grad = torch.gather(H_grad, 1, indices_expanded)
            recovered_Y_err = recovered_Y_err[:, :self.num_remaining_channels]
        sel = torch.nonzero(~end_flag.view(-1), as_tuple=False).view(-1)
        H_grad = torch.index_select(H_grad, 0, sel)
        out_channel = torch.index_select(out_channel, 0, sel)
        in_channel = torch.index_select(in_channel, 0, sel)
        self.signal_generator.select(sel)
        self.Y_err_rem = torch.index_select(recovered_Y_err, 0, sel)
        self.H_rem = out_channel
        self.H_grad_rem = H_grad
        self.recovered_X_rem = torch.index_select(recovered_X, 0, sel)
        # bs * (num_in_channels) * num_car [* mum_car] * num_ant
        # mse, best_mse = self.signal_generator.calc_H_err_normalized(in_channel)
        # print(mse, best_mse, 11111)
        out_channel = self.noise_scheduler.step(out_channel, time_step_now, t_next, in_channel,
                                                self.num_samples, self.eta1, self.eta2, self.eta3,
                                                model_grad=H_grad, ori_est=self.ori_est, generator=self.generator).prev_sample
        out_channel_power = out_channel.real * out_channel.real + out_channel.imag * out_channel.imag
        out_channel = out_channel / torch.sqrt(out_channel_power.mean(dim=(-2, -1), keepdim=True))
        # mse, best_mse = self.signal_generator.calc_H_err_normalized(out_channel)
        # print(mse, best_mse, 769769)
        # print(end_flag)
        return end_flag, out_channel

    def generate_last_step(self,
                      in_channel,
                      sp_params  # a list of parameters used for the signal processing unit
                      ):
        shape = in_channel.shape
        shape_sampled = torch.Size([shape[0], -1]) + shape[2:]
        # bs * (num_samples * num_in_channels) * num_car [* mum_car] * num_ant
        t0 = torch.zeros(1,)
        if torch.cuda.is_available():
            t0 = t0.cuda()
        out_channel = self.model(torch.flatten(in_channel, start_dim=0, end_dim=1),
                                 t0).reshape(shape_sampled)
        # mse, best_mse = self.signal_generator.calc_H_err_normalized(out_channel)
        # print(best_mse, 11111)
        recovered_Y_err, recovered_X, _, _ = self.signal_processing_unit(out_channel)

        if self.Y_err_rem is not None:
            # print(recovered_Y_err[3], self.Y_err_rem[3])
            recovered_Y_err = torch.cat((recovered_Y_err, self.Y_err_rem), 1)
            out_channel = torch.cat((out_channel, self.H_rem), 1)
            recovered_X = torch.cat((recovered_X, self.recovered_X_rem), 1)
        # bs * (num_samples * num_in_channels)
        recovered_Y_err, indices = torch.sort(recovered_Y_err, dim=1, descending=False)
        # print(recovered_Y_err[:,0]/sp_params.sigma_H.view(-1), 111)
        indices = indices.view(indices.size(0), indices.size(1), *[1 for _ in out_channel.shape[2:]])
        indices_expanded = indices.expand_as(recovered_X)[:, :1]
        recovered_X = torch.gather(recovered_X, 1, indices_expanded)

        mse, best_mse = self.signal_generator.calc_H_err_normalized(out_channel)
        ber = sp_params.calc_err(recovered_X)
        # print(ber, 1780460474036)
        self.logger.add(H_MSE_last=mse, H_BEST_MSE_last=best_mse, X_BER_last=ber)
        self.logger.add(Y_NMSE_last=recovered_Y_err[:,0]/sp_params.sigma_H.view(-1)-sp_params.noise_power)
        # self.logger.add(time=([0] * shape[0]))

    def restart(self,
                config,
                err_to_time=None,
                generator=None):
        self.num_samples = config.num_samples
        self.num_remaining_channels = config.num_remaining_channels
        self.substitute_channel = config.substitute_channel
        self.recovery_err_gate = config.recovery_err_gate
        self.pilot_mask = self.signal_generator.pilot_mask
        self.signal_processing_unit = self.signal_generator.get_signal_processing_unit(config.sp_unit_name)
        self.generator = generator
        self.err_to_time = err_to_time
        self.num_car = config.num_car
        self.num_pilot = len(config.pilot_cars)
        self.eta1 = config.eta1
        self.eta2 = config.eta2
        self.eta3 = config.eta3
        self.ori_est = None
        self.Y_err_rem = None
        self.H_rem = None
        self.H_grad_rem = None
        self.recovered_X_rem = None
        self.output_step = (config.all_time_steps // 10)
        if self.output_step == 0:
            self.output_step = 1
