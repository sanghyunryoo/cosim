from envs.flamingo_light_v1.flamingo_light_v1 import FlamingoLightV1
from envs.flamingo_p_v3.flamingo_p_v3 import FlamingoPV3
from envs.w4_p_v2.w4_p_v2 import W4PV2
from envs.humanoid_p_v0.humanoid_p_v0 import HumanoidPV0
from envs.wrappers import StateBuildWrapper, TimeLimitWrapper, CommandWrapper


def build_env(config):
    if config["env"]['id'] == "flamingo_light_v1":
      env = FlamingoLightV1(config)
    elif config["env"]['id'] == "flamingo_p_v3":
      env = FlamingoPV3(config)
    elif config["env"]['id'] == "w4_p_v2":
      env = W4PV2(config)
    elif config["env"]['id'] == "humanoid_p_v0":
      env = HumanoidPV0(config)
    else:
      raise NameError(f"Please select a valid environment id. Received '{config['env']['id']}'.")

    env = StateBuildWrapper(env, config)
    env = TimeLimitWrapper(env, config)
    env = CommandWrapper(env, config)

    return env
