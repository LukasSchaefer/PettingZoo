import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario
import random

def generate_distinct_rgb_colors(n):
    colors = set()
    
    while len(colors) < n:
        color = tuple(np.round(np.random.random(3), decimals=2))
        colors.add(color)

    return list(colors)

class Scenario(BaseScenario):
    def make_world(self, groups, colour_count):
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
        self.colour_count = colour_count
        self.colors_rgb = generate_distinct_rgb_colors(colour_count)
        self.colors_rgb_to_one_hot = {str(rgb):one_hot for rgb, one_hot in zip(self.colors_rgb, np.eye(colour_count))}

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
        # generate colors by randmoly sampling distinct colors from the pool
        group_colors_ids = np_random.choice(self.colour_count, len(self.groups), replace=False)
        self.colors = [self.colors_rgb[color_id] for color_id in group_colors_ids]

        for group_id, agent in zip(self.group_indices, world.agents):
            agent.color = self.colors[group_id]

        # attribute colors to landmarks randomly
        landmarks_ids = np.arange(len(world.landmarks))
        np_random.shuffle(landmarks_ids)
        for group_id, land_id in zip(self.group_indices, landmarks_ids):
            world.landmarks[land_id].color = self.colors[group_id]

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
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for j, a in enumerate(world.agents) if str(a.color) == str(l.color)]
            rew -= min(dists)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos_color = []
        for entity in world.entities:
            if entity is agent:
                continue
            entity_pos_color.append(entity.state.p_pos - agent.state.p_pos)
            entity_pos_color.append(self.colors_rgb_to_one_hot[str(entity.color)])
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [self.colors_rgb_to_one_hot[str(agent.color)]] + entity_pos_color)