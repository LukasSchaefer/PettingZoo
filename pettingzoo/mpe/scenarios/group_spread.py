from enum import Enum

import numpy as np

from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario


# Predefined colors for agents and landmarks
COLORS = [
    np.array([0.85, 0.25, 0.25]),
    np.array([0.25, 0.85, 0.25]),
    np.array([0.25, 0.25, 0.85]),
    np.array([0.25, 0.25, 0.25]),
    np.array([0.85, 0.85, 0.25]),
    np.array([0.85, 0.25, 0.85]),
    np.array([0.25, 0.85, 0.85]),
    np.array([1.00, 0.00, 0.00]),
    np.array([0.00, 1.00, 0.00]),
    np.array([0.00, 0.00, 1.00]),
]

class ColorConfig(Enum):
    CONTINUOUS_FIXED = 0    # used fixed set of colors (matched by `num_colors`) which agents observe as continuous vectors
    ONEHOT_FIXED = 1        # used fixed set of colors (matched by `num_colors`) which agents observe as one-hot vectors
    CONTINUOUS_RANDOM = 2   # used random colors which agents observe as continuous vectors (not using `num_colors`!)

    @staticmethod
    def from_str(label):
        """
        Returns the ColorConfig enum value corresponding to the given string label.
        :param label: string label
        :return: ColorConfig enum value
        """
        if label.lower().replace(" ", "_") in ("continuousfixed", "continuous_fixed"):
            return ColorConfig.CONTINUOUS_FIXED
        elif label.lower().replace(" ", "_") in ("onehotfixed", "onehot_fixed", "fixed"):
            return ColorConfig.ONEHOT_FIXED
        elif label.lower().replace(" ", "_") in ("continuousrandom", "continuous_random", "random"):
            return ColorConfig.CONTINUOUS_RANDOM
        else:
            raise NotImplementedError(f"Unknown color config label '{label}'")

    def get_color_values(self, num_colors):
        """
        Generate color values for the given color config and number of colors.
        :param color_config: ColorConfig enum value
        :param num_colors: number of colors
        :return: list of color values (each color value is a numpy array of shape (3,))
        """
        assert num_colors is not None, "Number of colors must be specified!"
        assert num_colors > 0 and num_colors <= len(COLORS), f"Number of colors must be in range [1, {len(COLORS)}]!"

        if self == ColorConfig.CONTINUOUS_RANDOM:
            # generate random set of colors
            return [np.random.random(3) for _ in range(num_colors)]
        elif self == ColorConfig.ONEHOT_FIXED or self == ColorConfig.CONTINUOUS_FIXED:
            # generate fixed set of colors using only individual color channels 
            return COLORS[:num_colors]
        else:
            raise NotImplementedError(f"Unknown color config '{self.name}'")
    
    def get_color_obs_value(self, colors, color_idx):
        """
        Generate color observation value for the given color config, colors and color index.
        :param color_config: ColorConfig enum value
        :param colors: list of color values (each color value is a numpy array of shape (3,))
        :param color_idx: color index
        :return: color observation value (numpy array of shape (3,))
        """
        if self == ColorConfig.CONTINUOUS_RANDOM or self == ColorConfig.CONTINUOUS_FIXED:
            return colors[color_idx]
        elif self == ColorConfig.ONEHOT_FIXED:
            return np.eye(len(colors))[color_idx]
        else:
            raise NotImplementedError(f"Unknown color config '{self.name}'")


class Scenario(BaseScenario):
    def make_world(self, groups, reward_per_group=True, color_config="onehot_fixed", num_colors=None, shuffle_obs_per_agent=True):
        world = World()
        # set any world properties first
        world.dim_c = 2
        self.num_agents = sum(groups)
        self.num_landmarks = sum(groups)
        world.collaborative = True

        self.groups = groups
        # group indices per agent as list of ints
        self.group_indices = [a * [i] for i, a in enumerate(self.groups)]
        self.group_indices = [
            item for sublist in self.group_indices for item in sublist
        ]
        self.reward_per_group = reward_per_group
        self.shuffle_obs_per_agent = shuffle_obs_per_agent

        self.num_colors = len(self.groups) if num_colors is None else num_colors
        self.color_config = ColorConfig.from_str(color_config)

        if self.color_config == ColorConfig.CONTINUOUS_RANDOM and num_colors is not None:
            raise ValueError("Color config is set to CONTINUOUS_RANDOM but `num_colors` is explicitly set. `num_colors` value is not used!")

        self.colors = self.color_config.get_color_values(self.num_colors)

        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent_{}".format(i)
            agent.collide = False
            agent.silent = True
            agent.size = 0.15

        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        if self.color_config == ColorConfig.CONTINUOUS_RANDOM:
            # randomly generate new colors for each episode
            self.colors = self.color_config.get_color_values(self.num_colors)

        # assign one of the available colors to a group
        color_idxs = np.random.choice(self.num_colors, len(self.groups), replace=False)

        # assign group / color properties to agents
        for i, agent in zip(self.group_indices, world.agents):
            agent.color = self.colors[color_idxs[i]]
            agent.color_idx = color_idxs[i]
            agent.group = i

        # assign properties to landmarks
        landmarks_ids = np.random.choice(self.num_landmarks, self.num_landmarks, replace=False)
        for group_id, land_id in zip(self.group_indices, landmarks_ids):
            world.landmarks[land_id].color = self.colors[color_idxs[group_id]]
            world.landmarks[land_id].color_idx = color_idxs[group_id]
            world.landmarks[land_id].group = group_id

        # assign each agent a landmark and agent ordering that wil determine order in the agent obs
        # this allows entities to be shuffled per agent while keeping consistent for the episode
        for agent in world.agents:
            if self.shuffle_obs_per_agent:
                agent.landmark_obs_order = np.random.choice(self.num_landmarks, self.num_landmarks, replace=False)
                agent.agent_obs_order = np.random.choice(self.num_agents, self.num_agents, replace=False)
            else:
                agent.landmark_obs_order = np.arange(self.num_landmarks)
                agent.agent_obs_order = np.arange(self.num_agents)

        # set random initial states with positions and velocities of agents and landmarks
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-3, +3, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                for a in world.agents if a.group == l.group
            ]
            min_dists += min(dists)
            rew -= min(dists)
        dists_landmarks = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        if agent.group == world.landmarks[np.argmin(dists_landmarks)].group and min(dists_landmarks) < 0.2:
            occupied_landmarks = 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark in group, penalized for collisions
        rew = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1

        if self.reward_per_group:
            for i, l in enumerate(world.landmarks):
                # consider only agents in same group as landmark in distance calculation
                if l.group == agent.group:
                    dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents if a.group == l.group]
                    rew -= min(dists)

        return rew

    def global_reward(self, world):
        rew = 0
        if not self.reward_per_group:
            for i, l in enumerate(world.landmarks):
                # consider only agents in same group as landmark in distance calculation
                group = self.group_indices[i]
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for j, a in enumerate(world.agents) if self.group_indices[j] == group]
                rew -= min(dists)
        return rew

    def observation(self, agent, world):
        # own position, velocity, color
        color = [self.color_config.get_color_obs_value(self.colors, agent.color_idx)]
        entity_features = [agent.state.p_pos] + [agent.state.p_vel] + color

        # positions and color of all landmarks
        landmark_pos_colors = []
        for i in agent.landmark_obs_order:
            # relative position of landmark to agent
            landmark_pos_colors.append(world.landmarks[i].state.p_pos - agent.state.p_pos)
            # color of landmark
            landmark_pos_colors.append(self.color_config.get_color_obs_value(self.colors, world.landmarks[i].color_idx))

        # positions and color all other agents
        other_agents_pos_colors = []
        for i in agent.agent_obs_order:
            if world.agents[i] is agent:
                # skip self
                continue
            # relative position of other agent to agent
            other_agents_pos_colors.append(world.agents[i].state.p_pos - agent.state.p_pos)
            other_agents_pos_colors.append(self.color_config.get_color_obs_value(self.colors, world.agents[i].color_idx))

        return np.concatenate(entity_features + landmark_pos_colors + other_agents_pos_colors)