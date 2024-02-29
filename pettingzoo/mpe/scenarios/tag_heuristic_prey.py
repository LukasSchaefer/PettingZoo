import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario

PREY_SPEED_MULTIPLIERS = [0.1, 0.3, 0.6, 1]

class Scenario(BaseScenario):
    def make_world(self, num_preys=1, num_predators=3, num_obstacles=2, max_speed_prey = 1.0, vary_prey_speed = False, prey_speed_observation = True, all_prey_captured_bonus = True):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_preys_agents = num_preys
        num_predators = num_predators
        num_agents = num_predators + num_preys_agents
        num_landmarks = num_obstacles

        self.max_speed_prey = max_speed_prey
        self.vary_prey_speed = vary_prey_speed
        self.all_prey_captured_bonus = all_prey_captured_bonus
        self.prey_speed_observation = prey_speed_observation

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        world.num_predators = num_predators
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_predators else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_predators else i - num_predators
            agent.name = '{}_{}'.format(base_name, base_index)
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else self.max_speed_prey
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        return world

    def reset_world(self, world, np_random):
        if self.vary_prey_speed:
            self.new_prey_speed = np.random.choice(PREY_SPEED_MULTIPLIERS) * self.max_speed_prey #np.random.uniform(0, self.max_speed_prey)
            for agent in world.agents:
                if not agent.adversary:
                    agent.max_speed = self.new_prey_speed
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if not agent.adversary:
            collisions = 0
            for a in self.adversaries(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
            all_agents_captured = all([any([self.is_collision(ag, adv) for adv in adversaries]) for ag in agents])
            if self.all_prey_captured_bonus and all_agents_captured:
                 rew += 50 * len(adversaries)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        observation_self = [agent.state.p_pos] + [agent.state.p_vel]
        if self.prey_speed_observation:
            observation_self.append([self.new_prey_speed])
        return np.concatenate( observation_self + entity_pos + other_pos + other_vel)
