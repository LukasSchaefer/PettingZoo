import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario
import random

class Scenario(BaseScenario):
    def make_world(self, groups, num_colors=None, reward_per_group=True, randomise_all_colors=False, shuffle_obs_per_agent=False):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = sum(groups)
        num_landmarks = sum(groups)
        world.collaborative = True

        self.groups = groups
        self.group_indices = [a * [i] for i, a in enumerate(self.groups)]
        self.group_indices = [
            item for sublist in self.group_indices for item in sublist
        ]
        self.reward_per_group = reward_per_group
        self.randomise_all_colors = randomise_all_colors
        self.shuffle_obs_per_agent = shuffle_obs_per_agent

        if num_colors == None:
            self.num_colors = len(self.groups)
        else:
            self.num_colors = num_colors

        if not self.randomise_all_colors:
            # generate color per group
            self.colors = [np.random.random(3) for _ in range(self.num_colors)]

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent_{}".format(i)
            agent.collide = False
            agent.silent = True
            agent.size = 0.15

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        if self.randomise_all_colors:
            # generate colors if they are randomised for each episode
            self.colors = [np.random.random(3) for _ in self.groups]
            color_idxs = np.arange(len(self.groups))
        else:
            # assign one of the available colors to a group
            color_idxs = np.random.choice(self.num_colors, len(self.groups), replace=False)

        # random properties for agents
        for i, agent in zip(self.group_indices, world.agents):
            agent.color = self.colors[color_idxs[i]]
            agent.color_idx = color_idxs[i]
            agent.group = i

        # random properties for landmarks
        if self.shuffle_obs_per_agent:
            for i, landmark in zip(self.group_indices, world.landmarks):
                landmark.color = self.colors[color_idxs[i]]
                landmark.color_idx = color_idxs[i]
                landmark.group = i
            # assign each agent a landmark and agent ordering that wil determine order in the agent obs
            # this allows entities to be shuffled per agent while keeping consistent for the episode
            for agent in world.agents:
                landmark_obs_order = np.arange(len(world.landmarks))
                random.shuffle(landmark_obs_order)
                agent.landmark_obs_order = landmark_obs_order
                agent_obs_order = np.arange(len(world.agents))
                random.shuffle(agent_obs_order)
                agent.agent_obs_order = agent_obs_order
        else:
            # assign landmark groups randomly as landmarks won't be shuffled in obs
            landmarks_ids = np.arange(len(world.landmarks))
            np_random.shuffle(landmarks_ids)
            for group_id, land_id in zip(self.group_indices, landmarks_ids):
                world.landmarks[land_id].color = self.colors[color_idxs[group_id]]
                world.landmarks[land_id].color_idx = color_idxs[group_id]
                world.landmarks[land_id].group = group_id

        # set random initial states
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
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if agent.group == l.group and min(dists) < 0.1:
                occupied_landmarks += 1
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
        # positions and color of all landmarks
        entity_pos_color = []
        if self.shuffle_obs_per_agent:
            landmark_idxs = agent.landmark_obs_order
        else:
            landmark_idxs = np.arange(len(world.landmarks))
        for i in landmark_idxs:
            entity_pos_color.append(world.landmarks[i].state.p_pos - agent.state.p_pos)
            if self.randomise_all_colors:
                entity_pos_color.append(world.landmarks[i].color)
            else:
                # one hot encoding
                entity_pos_color.append(np.eye(self.num_colors)[world.landmarks[i].color_idx])

        # positions and color all other agents
        other_pos_color = []
        if self.shuffle_obs_per_agent:
            other_idxs = agent.agent_obs_order
        else:
            other_idxs = np.arange(len(world.agents))
        for i in other_idxs:
            if world.agents[i] is agent:
                continue
            other_pos_color.append(world.agents[i].state.p_pos - agent.state.p_pos)
            if self.randomise_all_colors:
                other_pos_color.append(world.agents[i].color)
            else:
                # one hot encoding
                other_pos_color.append(np.eye(self.num_colors)[world.agents[i].color_idx])

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [np.eye(self.num_colors)[agent.group]] + entity_pos_color + other_pos_color)