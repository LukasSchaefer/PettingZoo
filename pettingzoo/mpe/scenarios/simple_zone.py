import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario
import random

class Scenario(BaseScenario):
    def make_world(
        self,
        num_agents,
        landmark_x_range,
        landmark_y_range,
    ):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = num_agents
        world.collaborative = True

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent_{}".format(i)
            agent.collide = False
            agent.silent = True
            agent.size = 0.15
            agent.color = [0.75, 0.25, 0.25]

        # add landmark
        world.landmarks = [Landmark()]
        world.landmarks[0].name = "landmark 0"
        world.landmarks[0].collide = False
        world.landmarks[0].movable = False
        world.landmarks[0].color = [0.25, 0.75, 0.25]
        world.landmarks[0].x_range = np.array(landmark_x_range)
        world.landmarks[0].y_range = np.array(landmark_y_range)

        return world

    def reset_world(self, world, np_random):
        # random properties for agents

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            min_x, max_x = landmark.x_range
            pos_x = np_random.uniform(min_x, max_x)
            min_y, max_y = landmark.y_range
            pos_y = np_random.uniform(min_y, max_y)
            landmark.state.p_pos = np.array([pos_x, pos_y])
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # Agents are rewarded based on minimum distance to landmark
        return -np.sqrt(
            np.sum(
                np.square(
                    agent.state.p_pos
                    - world.landmarks[0].state.p_pos
                )
            )
        )

    def global_reward(self, world):
        return sum([self.reward(a, world) for a in world.agents])

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        x = np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + entity_pos)
        return x

