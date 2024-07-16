import torch as th
from core.flow.volume_aware_module import VolumeAwareModule
from core.flow.base_distribution import BaseDistribution

def update_flow_batch(flow: VolumeAwareModule, prior: BaseDistribution, x: th.Tensor, y: th.Tensor, optimizer: th.optim.Optimizer, gradient_clip_value: float, take_log_again: bool):
    """
    Gradient update flow for one batch of data, to maximize log likelyhood.
    """
    z, logDet = flow.f(x, y)
    prior_logp = prior.log_prob(z, y)
    assert logDet.shape == prior_logp.shape
    logp = logDet + prior_logp
    if take_log_again:
        loss = -logp
        more_than_one  = loss>1
        loss[more_than_one] = loss[more_than_one].log()
        loss[~more_than_one] = loss[~more_than_one] - 1
        loss = loss.mean()
    else:
        loss = -logp.mean()
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    th.nn.utils.clip_grad_norm_(flow.parameters(), gradient_clip_value)
    optimizer.step()
    return loss.item()


