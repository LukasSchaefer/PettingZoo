import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario
import random

class Scenario(BaseScenario):
    def make_world(self, groups):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = sum(groups)
        num_landmarks = sum(groups) #len(groups)
        world.collaborative = True

        self.groups = groups
        self.group_indices = [a * [i] for i, a in enumerate(self.groups)]
        self.group_indices = [
            item for sublist in self.group_indices for item in sublist
        ]
        # generate colors:
        # self.colors = [np.random.random(3) for _ in groups]

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
        # random properties for agents
        # generate colors:
        self.colors = [np.random.random(3) for _ in self.groups]

        for i, agent in zip(self.group_indices, world.agents):
            agent.color = self.colors[i]

        # random properties for landmarks
        for i, landmark in zip(self.group_indices, world.landmarks):
            landmark.color = self.colors[i]

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
            if min(dists) < 0.1:
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
        return rew

    def global_reward(self, world):
        rew = 0
        for i, l in enumerate(world.landmarks):
            # consider only agents in same group as landmark in distance calculation
            group = self.group_indices[i]
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for j, a in enumerate(world.agents) if self.group_indices[j] == group]
            rew -= min(dists)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos_color = []
        for entity in world.entities:
            if entity is agent:
                continue
            entity_pos_color.append(entity.state.p_pos - agent.state.p_pos)
            entity_pos_color.append(entity.color)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [agent.color] + entity_pos_color)

