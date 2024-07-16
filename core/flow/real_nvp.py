from typing import Tuple
from torch import nn
import torch as th
from core.flow.volume_aware_module import VolumeAwareModule
from core.common.loadable_module import LoadbleModule

class RealNvp(VolumeAwareModule, th.nn.Module, LoadbleModule):
    """Define a simple real nvp module, does not include the prior."""

    def __init__(self, var_count, transform_count, conditional_param_count, hidden_size):
        super(RealNvp, self).__init__()
        assert transform_count % 2 == 0
        self.var_count = var_count
        self.conditional_param_count = conditional_param_count
        self.dim = self.var_count + self.conditional_param_count
        self.transform_count = transform_count
        self.hidden_size = hidden_size
        self.kwargs = {'var_count': var_count, 'transform_count': transform_count, "conditional_param_count": conditional_param_count, "hidden_size": hidden_size}
        mask = nn.Parameter(self._get_masks(transform_count), requires_grad=False)
        self.register_buffer('mask', mask)
        self.register_module('t', self._get_t())
        self.register_module('s', self._get_s())
    
    def concat_y_if_needed(self, x_or_z, y):
        if y is None:
            return x_or_z
        return th.cat([x_or_z, y], dim=1)

    def f(self, x, y):
        log_det_J, z = th.zeros(x.shape[0], device=x.device).float(), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            inp = self.concat_y_if_needed(z_, y)
            s = self.s[i](inp) * (1-self.mask[i])
            t = self.t[i](inp) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * th.exp(-s) + z_
            log_det_J -= th.sum(s, dim=1)
        return z, log_det_J

    def g(self, z, y):
        log_det_J, x = th.zeros(z.shape[0], device=z.device).float(), z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            inp = self.concat_y_if_needed(x_, y)
            s = self.s[i](inp)*(1 - self.mask[i])
            t = self.t[i](inp)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * th.exp(s) + t)
            log_det_J = th.sum(s, dim=1)
        return x, log_det_J

    def _get_nets(self):
        return nn.Sequential(
            nn.Linear(self.dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.var_count),
            nn.Tanh()
        )

    def _get_nett(self):
        return nn.Sequential(
            nn.Linear(self.dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.var_count),
        )


    def _get_t(self):
        return th.nn.ModuleList([self._get_nett() for _ in range(len(self.mask))])

    def _get_s(self):
        return th.nn.ModuleList([self._get_nets() for _ in range(len(self.mask))])

    def _get_masks(self, transform_count):
        mask_one = th.concatenate([th.ones(self.var_count//2), th.zeros(self.var_count - self.var_count//2)])
        mask_two = 1 - mask_one
        mask = th.stack([mask_one, mask_two]).repeat(transform_count//2, 1)
        return mask

    def disable_grad(self, disable=True):
        req_grad = not disable
        if req_grad:
            self.train()
        else:
            self.eval()
        for i in range(self.transform_count):
            for module in list(self.t[i]) + list(self.s[i]):
                for p in module.parameters():
                    p.requires_grad = req_grad


def test_real_nvp():
    flow = RealNvp(2, 6, 3, 256).to('cpu')
    x = th.tensor([[2, 3], [0, 0]])
    y = th.tensor([[2, 3, 3], [0, 0, 0]])
    z, logDet = flow.f(x, y)
    assert z.shape == (2, 5)
    assert logDet.shape == (2, )

    # save and load from file test
    save_path = "/tmp/test_model.pt"
    flow.save_module(save_path)
    flow2: RealNvp = RealNvp.load_module(save_path)
    z2, logDet2 = flow2.f(x, y)
    assert th.eq(z, z2).all()
    assert th.eq(logDet, logDet2).all()
