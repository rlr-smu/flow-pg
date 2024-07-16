from typing import Tuple, Union
from numpy import ndarray
import numpy as np
from stable_baselines3 import DDPG
from core.rl.constrained_algorithm import ConstrainedAlgorithm
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.logger import Logger
import gym
from core.constraints import BaseConstraint
import logging

log = logging.getLogger(__name__)
    

class DDPGPro(ConstrainedAlgorithm, DDPG):

    def predict(self, observation: ndarray, state: tuple = None, episode_start: ndarray = None, deterministic: bool = False, project=True) -> tuple:
        unscaled_action, _ = super().predict(observation, state, episode_start, deterministic)
        if not project:
            return unscaled_action, _
        unscaled_action = self.project_if_infeasible(unscaled_action, observation, "eval")
        return unscaled_action, _

    def _sample_action(self, learning_starts: int, action_noise: Union[ActionNoise, None] = None, n_envs: int = 1) -> Tuple[ndarray]:
                # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
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
    


# class FlowMlpPolicy(TD3Policy):
#     """Policy class for FlowPG, actor_kwargs should include flow and problem"""
#     def __init__(self, *args, actor_kwargs, **kwargs):
#         self.extra_actor_kwargs = actor_kwargs
#         super().__init__(*args, **kwargs)
        

#     class FlowActor(Actor):
#         flow: RealNvp
#         problem: BaseRLProblem
#         def __init__(self, flow: RealNvp, problem, *args, **kwargs): #
#             super().__init__(*args, **kwargs)
#             self.problem = problem
#             self.flow = flow
#             n_actions = len(self.action_space.low)
#             self.action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        
#         def apply_flow(self, action, observation):
#             x = th.cat([action, observation[:, self.problem.state_indices].to(action.dtype)], dim=1)
#             return self.flow.g(x)[0][:,:action.shape[1]]

#         def forward(self, obs: th.Tensor, deterministic=True) -> th.Tensor:
#             action = super().forward(obs)
#             if not deterministic:
#                 action = action + th.from_numpy(self.action_noise()).to(self.device)
#                 action = th.clip(action, min=-1.0, max=1.)
#             # print("Before flow", action)
#             action = self.apply_flow(action, obs)
#             # print("After flow", action)
#             return action

#     def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
#         actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
#         flow, problem = self.extra_actor_kwargs['flow'], self.extra_actor_kwargs['problem']
#         return self.FlowActor(flow, problem, **actor_kwargs).to(self.device)
    
#     def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
#         return self.actor(observation, deterministic)


# class DDPGFlow(DDPGProj):
#     def __init__(self, flow: RealNvp, problem: BaseRLProblem, *args, **kwargs):
#         policy_kwargs = {"actor_kwargs": {"flow": flow, "problem": problem}}
#         super().__init__(problem,FlowMlpPolicy, *args, policy_kwargs=policy_kwargs, **kwargs)
    