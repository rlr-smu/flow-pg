from stable_baselines3.td3.policies import MlpPolicy, Actor
from torch import Tensor
import torch as th
from core.flow.real_nvp import RealNvp
from core.rl.ddpg_pro import DDPGPro
import gym
import numpy as np
import logging

log = logging.getLogger(__name__)


class FlowActor(Actor):
    def __init__(self, *args, flow: RealNvp, flow_function: str, state_indexes, **kwargs):
        super().__init__(*args, **kwargs)
        flow = flow.float()
        self.register_module("flow", flow)
        self.flow_function = flow_function
        self.state_indexes = state_indexes
    
    def apply_flow(self, a_hat, obs):
        funct = getattr(self.flow, self.flow_function)
        if self.state_indexes is not None:
            y = obs.float()[:,self.state_indexes]
        else:
            y  = None
        action, det = funct(a_hat, y)
        return action
    
    def forward(self, obs: Tensor) -> Tensor:
        a_hat =  super().forward(obs)
        return self.apply_flow(a_hat, obs)
        

class MlpFlowPolicy(MlpPolicy):
    def __init__(self, *args, flow_file, flow_function: str, state_indexes, **kwargs):
        self.flow_function = flow_function
        self.flow_file = flow_file
        print("Flow file", flow_file)
        self.state_indexes = state_indexes
        super().__init__(*args, **kwargs)
        
    def make_actor(self, features_extractor = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        flow = RealNvp.load_module(self.flow_file)
        flow.disable_grad(True)
        return FlowActor(flow=flow, flow_function=self.flow_function, state_indexes=self.state_indexes, **actor_kwargs).to(self.device)
    
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation)
    

class FlowPG(DDPGPro):

    def __init__(self, *args, **kwargs):
        kwargs["policy"] = MlpFlowPolicy
        assert "policy_kwargs" in kwargs, "Policy kwargs must be provided"
        assert "flow_file" in kwargs["policy_kwargs"], "Flow file must be provided"
        assert "flow_function" in kwargs["policy_kwargs"], "Flow function must be provided"
        kwargs["policy_kwargs"]["state_indexes"] = kwargs["state_indexes"]
        
        super().__init__(*args, **kwargs)

    def _sample_action(self, learning_starts: int, action_noise = None, n_envs: int = 1):
                # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_a_hat = np.array([self.action_space.sample() for _ in range(n_envs)])
            a_hat = th.from_numpy(unscaled_a_hat).to(self.device)
            observation, vectorized_env = self.policy.obs_to_tensor(self._last_obs)
            unscaled_action = self.policy.actor.apply_flow(a_hat, observation).to("cpu").detach().numpy()
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            # Important: Here send project false to avoid projecition
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False, project=False) 
        

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1) # Add noise in the [-1, 1] space

            # We store the scaled action in the buffer
            unscaled_action = self.policy.unscale_action(scaled_action)
            unscaled_action = self.project_if_infeasible(unscaled_action, self._last_obs, "rollout") # Project in the unscaled space
            buffer_action = self.policy.scale_action(unscaled_action) # Scall back to get the buffer action. But usually most cases nothing happens in the scale as 
        else:
            raise Exception("Discrete case not implemented")

        return unscaled_action, buffer_action