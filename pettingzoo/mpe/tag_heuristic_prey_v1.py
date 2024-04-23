from ._mpe_utils.simple_env import SimpleEnv, make_env
from .scenarios.tag_heuristic_prey import Scenario
from pettingzoo.utils.to_parallel import parallel_wrapper_fn


class raw_env(SimpleEnv):
    def __init__(self, num_preys=1, num_predators=3, num_obstacles=2, max_frames=25, max_speed_prey = 1.0, vary_predator_speed = False, vary_prey_speed = False, predator_speed_observation = False, prey_speed_observation = False, all_prey_captured_bonus = True, disable_agent_collisions=False, prey_size=0.05, discrete_speed=True):
        scenario = Scenario()
        world = scenario.make_world(num_preys, num_predators, num_obstacles, max_speed_prey, vary_prey_speed, vary_predator_speed, prey_speed_observation, predator_speed_observation, all_prey_captured_bonus, disable_agent_collisions, prey_size, discrete_speed)
        super().__init__(scenario, world, max_frames)


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
