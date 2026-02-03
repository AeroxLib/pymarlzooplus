# Needed for the imports
REGISTRY_availability = [
    "sc2",
    "hallway",
    "hallway_join1",  # <--- 新增: Join1Env环境
    "gymma",
    "pettingzoo",
    "overcooked",
    "pressureplate",
    "capturetarget",
    "boxpushing",
]

from functools import partial  # noqa: E402

from pymarlzooplus.envs.multiagentenv import MultiAgentEnv  # noqa: E402
from pymarlzooplus.envs.gym_wrapper import _GymmaWrapper  # noqa: E402
from pymarlzooplus.envs.pettingzoo_wrapper import _PettingZooWrapper  # noqa: E402
from pymarlzooplus.envs.overcooked_wrapper import _OvercookedWrapper  # noqa: E402
from pymarlzooplus.envs.pressureplate_wrapper import _PressurePlateWrapper  # noqa: E402
from pymarlzooplus.envs.capturetarget_wrapper import _CaptureTargetWrapper  # noqa: E402
from pymarlzooplus.envs.boxpushing_wrapper import _BoxPushingWrapper  # noqa: E402

# ====================================================
# [新增 1] 导入 Hallway (自定义环境)
# ====================================================
try:
    from .hallway import HallwayEnv
except ImportError:
    print("Warning: HallwayEnv not found in envs/hallway.py. Skipping.")
    HallwayEnv = None

# ====================================================
# [新增 2] 导入 Join1Env (自定义环境)
# ====================================================
try:
    from .hallway_join1 import Join1Env
except ImportError:
    print("Warning: Join1Env not found in envs/hallway_join1.py. Skipping.")
    Join1Env = None

# ====================================================
# [新增 3] 导入 StarCraft2 (SMAC)
# ====================================================
try:
    from .starcraft2 import SC2 as StarCraft2Env
except ImportError:
    print("Warning: StarCraft2Env not found. Make sure starcraft2.py is in src/envs/")
    StarCraft2Env = None


# Gymnasium registrations
import pymarlzooplus.envs.lbf_registration_v2  # noqa: E402
import pymarlzooplus.envs.lbf_registration  # noqa: E402
import pymarlzooplus.envs.mpe_registration  # noqa: E402
import pymarlzooplus.envs.rware_v1_registration  # noqa: E402


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    "gymma": partial(env_fn, env=_GymmaWrapper),
    "pettingzoo": partial(env_fn, env=_PettingZooWrapper),
    "overcooked": partial(env_fn, env=_OvercookedWrapper),
    "pressureplate": partial(env_fn, env=_PressurePlateWrapper),
    "capturetarget": partial(env_fn, env=_CaptureTargetWrapper),
    "boxpushing": partial(env_fn, env=_BoxPushingWrapper),
    
    # [新增注册]
    "sc2": partial(env_fn, env=StarCraft2Env),
    "hallway": partial(env_fn, env=HallwayEnv),
    "hallway_join1": partial(env_fn, env=Join1Env),  # <--- 新增
}
