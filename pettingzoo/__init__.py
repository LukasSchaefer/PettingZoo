from pettingzoo.utils.env import AECEnv
import pettingzoo.utils
import inspect
from pathlib import Path
import os
from gym.envs.registration import EnvSpec, register, registry
from gym.core import Env
from gym.spaces import Box, Discrete, Tuple
import importlib
import numpy as np
__version__ = "1.3.5"


class PettingZooWrapper(Env):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, lib_name, env_name, **kwargs):

        PZEnv = importlib.import_module(f"pettingzoo.{lib_name}.{env_name}")
        print(PZEnv)
        self._env = PZEnv.parallel_env(**kwargs)

        n_agents = self._env.num_agents

        self.action_space = Tuple(
            tuple([self._env.action_spaces[k] for k in self._env.agents])
        )
        self.observation_space = Tuple(
            tuple([self._env.observation_spaces[k] for k in self._env.agents])
        )

        self.n_agents = n_agents

    def reset(self):
        obs = self._env.reset()
        obs = tuple([obs[k] for k in self._env.agents])
        return obs

    def render(self, mode="human"):
        return self._env.render(mode)

    def step(self, actions):
        dict_actions = {}
        for agent, action in zip(self._env.agents, actions):
            dict_actions[agent] = action

        observations, rewards, dones, infos = self._env.step(dict_actions)

        obs = tuple([observations[k] for k in self._env.agents])
        rewards = [rewards[k] for k in self._env.agents]
        dones = [dones[k] for k in self._env.agents]
        info = {}
        if all(dones):
            for k in self._env.agents:
                info = info | infos[k]
        return obs, rewards, dones, info

    def close(self):
        return self._env.close()

class HeuristicPreyWrapper(PettingZooWrapper):
    def __init__(self, lib_name, env_name, **kwargs):
        self.num_good = kwargs["num_good"]
        self.escape_threshold = kwargs["escape_threshold"]
        kwargs.pop("escape_threshold")
        super().__init__( lib_name, env_name, **kwargs)

        self.n_agents -= self.num_good
        self.action_space = Tuple(
            tuple([self._env.action_spaces[k] for k in self._env.agents[:-self.num_good]])
        )
        self.observation_space = Tuple(
            tuple([self._env.observation_spaces[k] for k in self._env.agents[:-self.num_good]])
        )
        self.dist_min = 0.075 + 0.05 # Agent sizes
        self._discrete_actions = [
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1]
        ]
        self.agent_cur_locations = []

    def reset(self):
        obs = self._env.reset()
        obs = tuple([obs[k] for k in self._env.agents])
        self.agent_cur_locations = [obs_[:2] for obs_ in obs]
        return obs[:-self.num_good]

    def get_prey_actions(self):
        prey_actions = []
        for prey_loc in self.agent_cur_locations[-self.num_good:]:
            dists_cur = []
            for predator_loc in self.agent_cur_locations[:-self.num_good]:
                delta_cur_pos = predator_loc - prey_loc
                dists_cur.append(np.sqrt(np.sum(np.square(delta_cur_pos))))
            closest_predator_loc = self.agent_cur_locations[np.argmin(dists_cur)]
            proj_pos = np.array([[prey_loc[0] + dis_act[0], prey_loc[1] + dis_act[1]] for dis_act in self._discrete_actions])
            delta_proj_pos = closest_predator_loc - proj_pos
            dists_proj = np.sqrt(np.sum(np.square(delta_proj_pos), axis=1))

            for i in range(len(proj_pos)):
                for predator_loc in self.agent_cur_locations[:-self.num_good]:
                    proj_dist =  np.sqrt(np.sum(np.square(proj_pos[i] - predator_loc)))
                    if proj_dist < self.dist_min:
                        dists_proj[i] = -99999

            if min(dists_cur) > self.escape_threshold:
                prey_actions.append(0)
            else:
                prey_actions.append(np.argmax(dists_proj) + 1)

        return prey_actions

    def step(self, actions):
        actions.extend(self.get_prey_actions())
        dict_actions = {}
        for agent, action in zip(self._env.agents, actions):
            dict_actions[agent] = action
        observations, rewards, dones, infos = self._env.step(dict_actions)

        obs = tuple([observations[k] for k in self._env.agents])
        rewards = [rewards[k] for k in self._env.agents]
        dones = [dones[k] for k in self._env.agents]

        self.agent_cur_locations = [obs_[:2] for obs_ in obs]
        obs = obs[:-self.num_good]
        rewards = rewards[:-self.num_good]
        dones = dones[:-self.num_good]
        info = {}
        if all(dones):
            for k in self._env.agents[:-self.num_good]:
                info = info | infos[k]
        return obs, rewards, dones, info

envs = Path(os.path.dirname(os.path.realpath(__file__))).glob("**/*_v?.py")
for e in envs:
    name = e.stem.replace("_", "-")
    lib = e.parent.stem
    filename = e.stem

    gymkey = f"pz-{lib}-{name}"
    entry_point = "pettingzoo:PettingZooWrapper"
    if "heuristic-prey" in name:
        entry_point = "pettingzoo:HeuristicPreyWrapper"
    register(
        gymkey,
        entry_point=entry_point,
        kwargs={"lib_name": lib, "env_name": filename,},
    )
    registry.spec(gymkey).gymma_wrappers = tuple()