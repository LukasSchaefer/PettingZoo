from pettingzoo.utils.env import AECEnv
import pettingzoo.utils
import inspect
from pathlib import Path
import os
from gym.envs.registration import EnvSpec, register, registry
from gym.core import Env
from gym.spaces import Box, Discrete, Tuple
import importlib

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

envs = Path(os.path.dirname(os.path.realpath(__file__))).glob("**/*_v?.py")
for e in envs:
    name = e.stem.replace("_", "-")
    lib = e.parent.stem
    filename = e.stem

    gymkey = f"pz-{lib}-{name}"
    register(
        gymkey,
        entry_point="pettingzoo:PettingZooWrapper",
        kwargs={"lib_name": lib, "env_name": filename,},
    )
    registry.spec(gymkey).gymma_wrappers = tuple()