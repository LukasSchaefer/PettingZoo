from pettingzoo.utils.env import AECEnv
import pettingzoo.utils
import inspect
from pathlib import Path
import os
from gym.envs.registration import EnvSpec, register, registry
from pettingzoo.env_wrappers import PettingZooWrapper, HeuristicPreyWrapper
__version__ = "1.3.5"


envs = Path(os.path.dirname(os.path.realpath(__file__))).glob("**/*_v?.py")
for e in envs:
    name = e.stem.replace("_", "-")
    lib = e.parent.stem
    filename = e.stem

    gymkey = f"pz-{lib}-{name}"
    entry_point = "pettingzoo:PettingZooWrapper"
    if "heuristic-prey" in name:
        entry_point = "pettingzoo:HeuristicPreyWrapper"
    register(
        gymkey,
        entry_point=entry_point,
        kwargs={"lib_name": lib, "env_name": filename,},
    )
    registry.spec(gymkey).gymma_wrappers = tuple()