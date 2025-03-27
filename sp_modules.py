#Basic signal processing units for the traditional demodulation parts
import torch

from utils import SignalGeneration, SignalProcessingParams, MSE


def QPSK_generation(shape, mask=None, dtype=torch.cfloat, device=torch.device('cuda')):
    xx_r = (torch.randint(1, shape, device=device) - 0.5) * 1.414213562
    xx_i = (torch.randint(1, shape, device=device) - 0.5) * 1.414213562
    if mask is not None:
        xx = xx_r * mask + 1j * (xx_i * mask)
    else:
        xx = xx_r + 1j * xx_i
    return xx.to(dtype=dtype, device=device)


def QAM16_generation(shape, mask=None, dtype=torch.cfloat, device=torch.device('cuda')):
    xx_r = (torch.randint(4, shape, device=device) - 1.5) * 0.632455532
    xx_i = (torch.randint(4, shape, device=device) - 1.5) * 0.632455532
    if mask is not None:
        xx = xx_r * mask + 1j * (xx_i * mask)
    else:
        xx = xx_r + 1j * xx_i
    return xx.to(dtype=dtype, device=device)


def QPSK_decoder_hard(u, v=None):
    return torch.sign(u.real) * 0.70710678 + 0.70710678j * torch.sign(u.imag)


def QPSK_decoder_hard_uv(u, v):
    v_r = torch.exp(-u.real * torch.sign(u.real) * 2.82842712 / v)
    u_r = torch.sign(u.real) * 0.70710678
    v_i = torch.exp(-u.imag * torch.sign(u.imag) * 2.82842712 / v)
    u_i = torch.sign(u.imag) * 0.70710678
    return u_r + 1j * u_i, v_r + v_i


def QPSK_decoder_u(u, v):
    p_r = 1 / (1 + torch.exp(-2.82842712 * u.real.div(v)))
    u_post_r = (p_r - 0.5) * 1.41421356
    p_i = 1 / (1 + torch.exp(-2.82842712 * u.imag.div(v)))
    u_post_i = (p_i - 0.5) * 1.41421356
    return u_post_r + 1j * u_post_i


def QPSK_decoder_uv(u, v):
    def demod(u):
        p_1 = 1 / (1 + torch.exp(-2 * u))
        u_post = 2 * p_1 - 1
        v_post = 1 - u_post.pow(2)
        return u_post, v_post

    u_post_r, v_post_r = demod(1.41421356 * u.real.div(v))
    u_post_i, v_post_i = demod(1.41421356 * u.imag.div(v))
    return (u_post_r + 1j * u_post_i) * 0.70710678, (v_post_r + v_post_i) / 2


def QAM16_decoder_hard(u, v=None):
    # print(u)
    u_r = torch.sign(u.real) + torch.sign(u.real - 0.632455532) + torch.sign(u.real + 0.632455532)  # 2/sqrt(10)
    u_i = torch.sign(u.imag) + torch.sign(u.imag - 0.632455532) + torch.sign(u.imag + 0.632455532)  # 2/sqrt(10)
    # print(u_r, u_i)
    # exit(0)
    return 0.316227766 * u_r + 0.316227766j * u_i # 1/sqrt(10)


def QAM16_decoder_hard_uv(u, v):
    def demod_real(u, v):
        uu = 1.264911064 * u.div(v)  # 4/sqrt(10)
        p_1 = torch.exp(uu)
        p_minus_1 = 1 / p_1
        tmp = p_1 * p_1 * p_1
        tmp2 = torch.exp(-3.2 / v)
        p_3 = tmp * tmp2
        p_minus_3 = tmp2 / tmp
        u_post = torch.sign(u) + torch.sign(u - 0.632455532) + torch.sign(u + 0.632455532)  # 2/sqrt(10)
        v_post = (p_minus_3 * (u_post * 6 + 9) + p_minus_1 * (u_post * 2 + 1) +
                  p_1 * (1 - u_post * 2) + p_3 * (9 - u_post * 6))
        v_post = u_post * u_post + v_post / (p_1 + p_minus_1 + p_3 + p_minus_3)
        return u_post * 0.316227766, v_post * 0.1   # 1/sqrt(10)

    u_post_r, v_post_r = demod_real(u.real, v)
    u_post_i, v_post_i = demod_real(u.imag, v)
    return u_post_r + 1j * u_post_i, v_post_r + v_post_i

def QAM16_decoder_u(u, v):
    def demod_real(u, v):
        uu = 1.264911064 * u.div(v)  # 4/sqrt(10)
        p_1 = torch.exp(uu)
        p_minus_1 = 1 / p_1
        tmp = p_1 * p_1 * p_1
        tmp2 = torch.exp(-3.2 / v)
        p_3 = tmp * tmp2
        p_minus_3 = tmp2 / tmp
        u_out = (p_1 - p_minus_1 + 3 * p_3 - 3 * p_minus_3) / (p_1 + p_minus_1 + p_3 + p_minus_3)
        return u_out * 0.316227766  # 1/sqrt(10)

    u_post_r = demod_real(u.real, v)
    u_post_i = demod_real(u.imag, v)
    return u_post_r + 1j * u_post_i


def QAM16_decoder_uv(u, v):
    def demod_real(u, v):
        uu = 1.264911064 * u.div(v)  # 4/sqrt(10)
        p_1 = torch.exp(uu)
        p_minus_1 = 1 / p_1
        tmp = p_1 * p_1 * p_1
        tmp2 = torch.exp(-3.2 / v)
        p_3 = tmp * tmp2
        p_minus_3 = tmp2 / tmp
        sum_p = p_1 + p_minus_1 + p_3 + p_minus_3
        uu = p_1 - p_minus_1 + 3 * p_3 - 3 * p_minus_3
        vv = p_1 + p_minus_1 + 9 * p_3 + 9 * p_minus_3
        u_out = (uu / sum_p)
        v_out = (vv / sum_p)- u_out * u_out
        return u_out * 0.316227766, v_out * 0.1  # 1/sqrt(10)

    u_post_r, v_post_r = demod_real(u.real, v)
    u_post_i, v_post_i = demod_real(u.imag, v)
    return u_post_r + 1j * u_post_i, v_post_r + v_post_i


class OFDMSignalGeneration(SignalGeneration):
    def __init__(self, config, noise_config, device):
        super(OFDMSignalGeneration, self).__init__(config, noise_config)
        self.pilot_mask = torch.zeros((1, 1, self.num_car), device=device)
        self.pilot_mask[:, :, config.pilot_cars] = 1.0
        self.pilot = self.pilot_mask + 0j
        self.data_mask = 1 - self.pilot_mask
        self.H_label = None
        self.H_ener = None
        if config.modulation == "QPSK":
            self.gen = QPSK_generation
            if config.use_grad:
                if config.decoding == "soft":
                    self.decode = QPSK_decoder_uv
                elif config.decoding == "hard":
                    self.decode = QPSK_decoder_hard_uv
                else:
                    raise NotImplementedError
            else:
                if config.decoding == "soft":
                    self.decode = QPSK_decoder_u
                elif config.decoding == "hard":
                    self.decode = QPSK_decoder_hard
                else:
                    raise NotImplementedError
        elif config.modulation == "QAM16":
            self.gen = QAM16_generation
            if config.use_grad:
                if config.decoding == "soft":
                    self.decode = QAM16_decoder_uv
                elif config.decoding == "hard":
                    self.decode = QAM16_decoder_hard_uv
                else:
                    raise NotImplementedError
            else:
                if config.decoding == "soft":
                    self.decode = QAM16_decoder_u
                elif config.decoding == "hard":
                    self.decode = QAM16_decoder_hard
                else:
                    raise NotImplementedError

    # H: bs * num_ant * num_car
    def generate_signal(self, H):
        # print(H.shape, 2222)
        self.H_label = H
        H_pow = H.real * H.real + H.imag * H.imag
        self.H_ener = H_pow.mean(dim=(-2, -1))
        X = self.gen((H.shape[0], 1, self.num_car), mask=self.data_mask, dtype=H.dtype, device=H.device)
        X = X + self.pilot
        Y = H * X
        # print(X)
        # print(self.H_ener.shape, self.H_label.shape, 3333)

        self.sp_params.set_params(X=X.unsqueeze(1), Y=Y.unsqueeze(1))
        self.sp_params.add_noise()
        return self.sp_params

    def sp_module_naive(self, H):
        # print(self.calc_H_err_normalized(H))
        # H = (self.H_label / torch.sqrt(self.H_ener).view(-1, 1, 1)).unsqueeze(1)
        # print(self.calc_H_err_normalized(H))
        hy = (self.sp_params.Y * H.conj()).mean(dim=-2, keepdim=True)
        hh = (H * H.conj()).mean(dim=-2, keepdim=True)
        x_hat = hy / (hh * self.sp_params.sigma_H_sqrt.view(-1, *[1 for _ in hy.shape[1:]]))
        # print(self.sp_params.calc_err(x_hat))
        x_hat = self.decode(x_hat)
        # print(self.sp_params.calc_err(x_hat))
        x_hat = x_hat.real * self.data_mask + 1j * (x_hat.imag * self.data_mask) + self.pilot
        #print(self.sp_params.calc_err(x_hat))

        y_hat = H * x_hat * self.sp_params.sigma_H_sqrt.view(-1, *[1 for _ in hy.shape[1:]])
        y_err = y_hat - self.sp_params.Y
        y_err = y_err.real * y_err.real + y_err.imag * y_err.imag
        y_err = y_err.mean(dim=(-2, -1))
        return y_err, x_hat, None, None

    def get_ori_channel(self):
        return self.sp_params.Y * self.pilot_mask / self.sp_params.sigma_H_sqrt.view(-1, *[1 for _ in self.sp_params.Y.shape[1:]])

    def calc_H_err(self, H):
        err = self.H_label.unsqueeze(1) - H
        err = err.real * err.real + err.imag * err.imag
        MSE = err.mean(dim=(-2, -1))
        NMSE = MSE / self.H_ener.unsqueeze(1)
        return MSE.mean(dim=1), NMSE.mean(dim=1), MSE.min(dim=1)[0], NMSE.min(dim=1)[0]

    def calc_H_err_normalized(self, H, sel=None):
        # print(H.shape, self.H_label.shape, self.H_ener.shape, 3333)
        err = (self.H_label / torch.sqrt(self.H_ener).view(-1, 1, 1)).unsqueeze(1) - H
        if sel is not None:
            err = torch.index_select(err, 0, sel)
        err = err.real * err.real + err.imag * err.imag
        MSE = err.mean(dim=(-2, -1))
        return MSE.mean(dim=1), MSE.min(dim=1)[0]

    def select(self, sel):
        self.sp_params.select(sel)
        self.H_label = torch.index_select(self.H_label, 0, sel)
        self.H_ener = torch.index_select(self.H_ener, 0, sel)
        # print(self.H_ener.shape, self.H_label.shape, 55555)

    def sp_module_grad(self, H):
        hy = (self.sp_params.Y * H.conj()).mean(dim=-2, keepdim=True)
        hh = (H.real * H.real + H.imag * H.imag).mean(dim=-2, keepdim=True)
        x_hat = hy / (hh * self.sp_params.sigma_H_sqrt.view(-1, *[1 for _ in hy.shape[1:]]))
        v_x = self.sp_params.noise_power_real.view(-1, 1, 1, 1) / hh
        x_hat, var_x = self.decode(x_hat, v_x)
        x_hat = x_hat.real * self.data_mask + 1j * (x_hat.imag * self.data_mask) + self.pilot
        var_x = var_x * self.data_mask
        # print(self.sp_params.calc_err(x_hat))

        y_hat = H * x_hat * self.sp_params.sigma_H_sqrt.view(-1, *[1 for _ in hy.shape[1:]])
        y_err = y_hat - self.sp_params.Y
        delta_H = y_err * x_hat / (var_x + (self.sp_params.noise_power_real/self.sp_params.sigma_H).view(-1, 1, 1, 1))
        delta_H_ener = delta_H.real * delta_H.real + delta_H.imag * delta_H.imag
        delta_H_ener = delta_H_ener.mean(dim=(-2, -1), keepdim=True).sqrt()
        y_err = y_err.real * y_err.real + y_err.imag * y_err.imag
        y_err = y_err.mean(dim=(-2, -1))
        return y_err, x_hat, delta_H/delta_H_ener, None

    def restart(self, config, noise_config, device):
        self.pilot_mask = torch.zeros((1, 1, self.num_car), device=device)
        self.pilot_mask[:, :, config.pilot_cars] = 1.0
        self.pilot = self.pilot_mask + 0j
        self.data_mask = 1 - self.pilot_mask
        self.H_label = None
        self.H_ener = None
        if config.modulation == "QPSK":
            self.gen = QPSK_generation
            if config.use_grad:
                if config.decoding == "soft":
                    self.decode = QPSK_decoder_uv
                elif config.decoding == "hard":
                    self.decode = QPSK_decoder_hard_uv
                else:
                    raise NotImplementedError
            else:
                if config.decoding == "soft":
                    self.decode = QPSK_decoder_u
                elif config.decoding == "hard":
                    self.decode = QPSK_decoder_hard
                else:
                    raise NotImplementedError
        elif config.modulation == "QAM16":
            self.gen = QAM16_generation
            if config.use_grad:
                if config.decoding == "soft":
                    self.decode = QAM16_decoder_uv
                elif config.decoding == "hard":
                    self.decode = QAM16_decoder_hard_uv
                else:
                    raise NotImplementedError
            else:
                if config.decoding == "soft":
                    self.decode = QAM16_decoder_u
                elif config.decoding == "hard":
                    self.decode = QAM16_decoder_hard
                else:
                    raise NotImplementedError
        self.sp_params.restart(noise_power=noise_config.noise_power, is_NMSE=noise_config.is_NMSE)
