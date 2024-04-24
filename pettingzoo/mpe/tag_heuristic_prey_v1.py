from ._mpe_utils.simple_env import SimpleEnv, make_env
from .scenarios.tag_heuristic_prey import Scenario
from pettingzoo.utils.to_parallel import parallel_wrapper_fn


class raw_env(SimpleEnv):
    def __init__(self, max_frames=100, **env_args):
        scenario = Scenario()
        world = scenario.make_world(**env_args)
        super().__init__(scenario, world, max_frames)


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
