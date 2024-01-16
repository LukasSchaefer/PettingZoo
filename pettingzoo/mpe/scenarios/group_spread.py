import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario
import random

class Scenario(BaseScenario):
    def make_world(self, groups, num_colors=None, reward_per_group=True, randomise_all_colors=False, obs_onehot_colors=True, shuffle_obs_per_agent=False):
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
        self.randomise_all_colors = randomise_all_colors
        self.shuffle_obs_per_agent = shuffle_obs_per_agent

        self.num_colors = len(self.groups) if num_colors is None else num_colors
        self.obs_onehot_colors = obs_onehot_colors

        if not self.randomise_all_colors:
            # generate color per group evenly spaced out if not randomised
            self.colors = [np.array([i / (self.num_colors - 1)] * 3) for i in range(self.num_colors)]

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
        if self.randomise_all_colors:
            # randomly generate new colors for each episode
            self.colors = [np.array([i / (self.num_colors - 1)] * 3) for i in range(self.num_colors)]

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
        color = [np.eye(self.num_colors)[agent.color_idx]] if self.obs_onehot_colors else [agent.color]
        entity_features = [agent.state.p_pos] + [agent.state.p_vel] + color

        # positions and color of all landmarks
        landmark_pos_colors = []
        for i in agent.landmark_obs_order:
            # relative position of landmark to agent
            landmark_pos_colors.append(world.landmarks[i].state.p_pos - agent.state.p_pos)
            # color of landmark
            landmark_pos_colors.append(np.eye(self.num_colors)[world.landmarks[i].color_idx] if self.obs_onehot_colors else world.landmarks[i].color)

        # positions and color all other agents
        other_agents_pos_colors = []
        for i in agent.agent_obs_order:
            if world.agents[i] is agent:
                # skip self
                continue
            # relative position of other agent to agent
            other_agents_pos_colors.append(world.agents[i].state.p_pos - agent.state.p_pos)
            other_agents_pos_colors.append(np.eye(self.num_colors)[world.agents[i].color_idx] if self.obs_onehot_colors else world.agents[i].color)

        return np.concatenate(entity_features + landmark_pos_colors + other_agents_pos_colors)