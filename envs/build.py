from envs.flamingo_p_v0.flamingo_p_v0 import FlamingoPV0
from envs.flamingo_light_p_v0.flamingo_light_p_v0 import FlamingoLightPV0
from envs.humanoid_p_v0.humanoid_p_v0 import HumanoidPV0
from envs.wrappers import StateBuildWrapper, TimeLimitWrapper, CommandWrapper

def build_env(config):
    if config["env"]['id'] == "flamingo_p_v0":
      env = FlamingoPV0(config)
    elif config["env"]['id'] == "flamingo_light_p_v0":
      env = FlamingoLightPV0(config)
    elif config["env"]['id'] == "humanoid_p_v0":
      env = HumanoidPV0(config)
    else:
      raise NameError(f"Please select a valid environment id. Received '{config['env']['id']}'.")

    env = StateBuildWrapper(env, config)
    env = TimeLimitWrapper(env, config)
    env = CommandWrapper(env, config)

    return env
