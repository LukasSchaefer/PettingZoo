from gym.core import Env
from gym.spaces import Box, Discrete, Tuple

from collections import defaultdict
import importlib
import numpy as np
from enum import Enum
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

class PreyActions(Enum):
    NOOP = [0, 0]
    LEFT = [-1, 0] 
    RIGHT = [1, 0] 
    DOWN = [0, -1] 
    UP = [0, 1]

class HeuristicPreyWrapper(PettingZooWrapper):
    def __init__(self, lib_name, env_name, **kwargs):
        self.num_preys = kwargs["num_preys"]
        self.vicinity_threshold = kwargs["vicinity_threshold"]
        self.escape_bound = kwargs["escape_bound"]
        kwargs.pop("vicinity_threshold")
        kwargs.pop("escape_bound")
        super().__init__( lib_name, env_name, **kwargs)

        self.n_agents -= self.num_preys
        self.action_space = Tuple(
            tuple([self._env.action_spaces[k] for k in self._env.agents[:-self.num_preys]])
        )
        self.observation_space = Tuple(
            tuple([self._env.observation_spaces[k] for k in self._env.agents[:-self.num_preys]])
        )
        self.dist_min = 0.075 + 0.05 # Agent sizes
        self._discrete_actions = [act.value for act in PreyActions] # Ignore NOOP 
        self.agent_cur_locations = []

    def reset(self):
        obs = self._env.reset()
        obs = tuple([obs[k] for k in self._env.agents])
        self.agent_cur_locations = [obs_[:2] for obs_ in obs]

        self.episode_infos = defaultdict(list)
        return obs[:-self.num_preys]

    def get_prey_actions(self):
        self.agent_cur_locations = np.array(self.agent_cur_locations)

        predator_locs = self.agent_cur_locations[:-self.num_preys]
        prey_locs = self.agent_cur_locations[-self.num_preys:]

        # Retrieve predator-prey distance information
        predator_prey_dists = np.sqrt(np.sum(np.square(np.expand_dims(predator_locs, axis=1) - prey_locs), axis=-1)) # Shape: Nr_Prey x Nr_Predators
        
        # Project possible actions and retrieve distance from each predator to a projection.
        proj_pos = np.expand_dims(prey_locs, axis=1) + self._discrete_actions
        delta_proj_predator = np.repeat(np.expand_dims(predator_locs, axis=(1, 2)), self.num_preys, axis=1) - proj_pos
        proj_predator_distances = np.sqrt(np.sum(np.square(delta_proj_predator), axis= -1)).transpose(1, 0 ,2) # Shape: Nr_Prey x Nr_Predators x Nr_Actions 

        # Choose action leading farthest from nearest predator while solving prey being stuck between two predators.
        closest_predator_indx = np.argmin(predator_prey_dists.transpose(), axis=1)
        dists_proj = np.array([distance[id_] for distance, id_ in zip(proj_predator_distances, closest_predator_indx)]) # Shape: Nr_Prey x Nr_Actions 
        too_close_mask = np.any(proj_predator_distances.transpose(0, 2, 1) < self.dist_min, axis=-1)
        out_of_bounds_mask = np.any(abs(proj_pos) > self.escape_bound, axis=-1)
        dists_proj[too_close_mask] = -np.inf
        dists_proj[out_of_bounds_mask] = -np.inf
        prey_actions = np.argmax(dists_proj, axis=-1)
        
        # Filter unecessary "escape" action
        vicinity_thershold_mask = np.min(predator_prey_dists.transpose(), axis=-1) < self.vicinity_threshold
        prey_actions[vicinity_thershold_mask == 0] = 0
        
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
        obs = obs[:-self.num_preys]
        rewards = rewards[:-self.num_preys]
        dones = dones[:-self.num_preys]

        for k in self._env.agents:
            for metric_name, metric_val in infos[k].items():
                if "episode" in metric_name:
                    self.episode_infos[metric_name].append(metric_val)
        
        info = {}
        if all(dones):
            for k in self._env.agents:
                info = info | infos[k]
            info = info | {k:sum(v) for k, v in self.episode_infos.items()}
            prey_collisions = np.transpose(np.array([v for k,v in self.episode_infos.items() if "collisions_prey" in k]))
            all_prey_collisions = np.all(prey_collisions, axis=1)
            info["all_prey_collided_average"] = 0 if not np.any(all_prey_collisions) else np.sum(prey_collisions[all_prey_collisions])/self.num_preys
            info["best_prey_collided_average"] = np.max(np.sum(all_prey_collisions, axis=0))

        return obs, rewards, dones, info