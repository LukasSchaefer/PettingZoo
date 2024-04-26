from collections.abc import Iterable

import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario


DEFAULT_PREY_SPEED = 1.0
DEFAULT_PREDATOR_SPEED = 1.0
DEFAULT_PREY_SIZE = 0.05
DEFAULT_PREDATOR_SIZE = 0.075
DEFAULT_PREY_ACCEL = 3.0
DEFAULT_PREDATOR_ACCEL = 3.0
PREY_COLOR = np.array([0.35, 0.85, 0.35])
PREDATOR_COLOR = np.array([0.85, 0.35, 0.35])
LANDMARK_COLOR = np.array([0.25, 0.25, 0.25])

PREY_REWARD_CAPTURE = -10
PREY_REWARD_DIST_COEF = 0.1
PREDATOR_REWARD_CAPTURE = 10
PREDATOR_REWARD_ALL_CAPTURED = 100


class Scenario(BaseScenario):
    def make_world(
        self,
        num_preys=1,
        num_predators=3,
        num_obstacles=0,
        prey_base_speed=DEFAULT_PREY_SPEED,
        predator_base_speed=DEFAULT_PREDATOR_SPEED,
        discrete_speeds=True,
        prey_speed_multipliers=[1.0],
        predator_speed_multipliers=[1.0],
        prey_min_max_speed=(1.0, 1.0),
        predator_min_max_speed=(1.0, 1.0),
        individual_agent_speeds=False,
        observe_predator_speed=False,
        observe_prey_speed=False,
        disable_agent_collisions=False,
        prey_size=DEFAULT_PREY_SIZE,
        predator_size=DEFAULT_PREDATOR_SIZE,
    ):
        """
        num_preys: number of prey agents
        num_predators: number of predator agents
        num_obstacles: number of obstacle landmarks in environment
        prey_base_speed: base speed of prey agents
        predator_base_speed: base speed of predator agents
        discrete_speeds: whether to use discrete speed values or sample from a range
        prey_speed_multipliers: list of speed multipliers for prey agents (only used if discrete_speeds is True)
        predator_speed_multipliers: list of speed multipliers for predator agents (only used if discrete_speeds is True)
        prey_min_max_speed: range of speeds to sample from for prey agents (only used if discrete_speeds is False)
        predator_min_max_speed: range of speeds to sample from for predator agents (only used if discrete_speeds is False)
        individual_agent_speeds: whether to assign each agent a different speed or all prey/ predator agents have same speed
        observe_predator_speed: whether to include predator speed in agent observations
        observe_prey_speed: whether to include prey speed in agent observations
        disable_agent_collisions: whether to disable collisions between agents
        prey_size: size of prey agents
        predator_size: size of predator agents
        """
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = num_predators + num_preys
        self.discrete_speeds = discrete_speeds
        self.prey_base_speed = prey_base_speed
        self.prey_speed_multipliers = (
            prey_speed_multipliers
            if isinstance(prey_speed_multipliers, Iterable)
            else [prey_speed_multipliers]
        )
        self.prey_min_max_speed = prey_min_max_speed
        self.predator_base_speed = predator_base_speed
        self.predator_speed_multipliers = (
            predator_speed_multipliers
            if isinstance(predator_speed_multipliers, Iterable)
            else [predator_speed_multipliers]
        )
        self.predator_min_max_speed = predator_min_max_speed
        self.individual_agent_speeds = individual_agent_speeds

        self.observe_predator_speed = observe_predator_speed
        self.observe_prey_speed = observe_prey_speed

        # add agents
        world.agents = [Agent() for _ in range(num_agents)]
        world.num_predators = num_predators
        world.num_preys = num_preys
        world.disable_agent_collisions = disable_agent_collisions
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_predators else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_predators else i - num_predators
            agent.name = "{}_{}".format(base_name, base_index)
            agent.collide = True
            agent.silent = True
            agent.size = predator_size if agent.adversary else prey_size
            agent.accel = (
                DEFAULT_PREDATOR_ACCEL if agent.adversary else DEFAULT_PREY_ACCEL
            )
            agent.max_speed = (
                prey_base_speed if agent.adversary else predator_base_speed
            )

        # add landmarks as obstacles
        world.landmarks = [Landmark() for _ in range(num_obstacles)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        return world

    def _assign_agent_speeds(
        self, agents, base_speed, speed_multipliers, speed_range, np_random
    ):
        """
        Assigns speed to each agent based on base speed and speed multipliers
        :param agents: list of agents
        :param base_speed: base speed of agents
        :param speed_multipliers: list of speed multipliers to sample from
        :param speed_range: range of speeds to sample from
        :param np_random: random number generator
        """
        # determine speeds for preys
        if self.individual_agent_speeds:
            if self.discrete_speeds:
                multipliers = np_random.choice(speed_multipliers, len(agents))
            else:
                min_speed_multiplier, max_speed_multiplier = speed_range
                multipliers = np_random.uniform(
                    min_speed_multiplier, max_speed_multiplier, len(agents)
                )
            speeds = base_speed * multipliers
        else:
            if self.discrete_speeds:
                multiplier = np_random.choice(speed_multipliers)
            else:
                min_speed_multiplier, max_speed_multiplier = speed_range
                multiplier = np_random.uniform(
                    min_speed_multiplier, max_speed_multiplier
                )
            speeds = [base_speed * multiplier] * len(agents)
        for agent, speed in zip(agents, speeds):
            agent.max_speed = speed

    def reset_world(self, world, np_random):
        # assign speeds to prey agents
        self._assign_agent_speeds(
            [agent for agent in world.agents if not agent.adversary],
            self.prey_base_speed,
            self.prey_speed_multipliers,
            self.prey_min_max_speed,
            np_random,
        )
        # assign speeds to predator agents
        self._assign_agent_speeds(
            [agent for agent in world.agents if agent.adversary],
            self.predator_base_speed,
            self.predator_speed_multipliers,
            self.predator_min_max_speed,
            np_random,
        )

        for agent in world.agents:
            agent.color = PREY_COLOR if not agent.adversary else PREDATOR_COLOR
        for landmark in world.landmarks:
            landmark.color = LANDMARK_COLOR
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for landmark in world.landmarks:
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
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += PREY_REWARD_DIST_COEF * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew += PREY_REWARD_CAPTURE

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
        preys = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for pr in preys:
                rew -= min(
                    [
                        np.sqrt(np.sum(np.square(pr.state.p_pos - adv.state.p_pos)))
                        for adv in adversaries
                    ]
                )
        if agent.collide:
            for pr in preys:
                for adv in adversaries:
                    if self.is_collision(pr, adv):
                        rew += PREDATOR_REWARD_CAPTURE
                        break
            all_preys_captured = all(
                [
                    any([self.is_collision(pr, adv) for adv in adversaries])
                    for pr in preys
                ]
            )
            if all_preys_captured:
                rew += PREDATOR_REWARD_ALL_CAPTURED
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # get relative position, velocity, and potentially max speeds of other agents
        other_pos = []
        other_vel = []
        other_max_speed = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
            if self.observe_prey_speed and not other.adversary:
                other_max_speed.append([other.max_speed])
            if self.observe_predator_speed and other.adversary:
                other_max_speed.append([other.max_speed])
        # get own position and velocity
        observation_self = [agent.state.p_pos] + [agent.state.p_vel]
        return np.concatenate(
            observation_self + entity_pos + other_pos + other_vel + other_max_speed
        )
