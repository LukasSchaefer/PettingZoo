from ._mpe_utils.simple_env import SimpleEnv, make_env
from .scenarios.tag_heuristic_prey import Scenario
from pettingzoo.utils.to_parallel import parallel_wrapper_fn


class raw_env(SimpleEnv):
    def __init__(self, num_preys=1, num_predators=3, num_obstacles=2, max_frames=25, max_speed_prey = 1.0, vary_prey_speed = False, prey_speed_observation = True, all_prey_captured_bonus = True):
        scenario = Scenario()
        world = scenario.make_world(num_preys, num_predators, num_obstacles, max_speed_prey, vary_prey_speed, prey_speed_observation, all_prey_captured_bonus)
        super().__init__(scenario, world, max_frames)


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
