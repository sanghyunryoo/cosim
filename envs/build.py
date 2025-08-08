from envs.flamingo_v1_5_1.flamingo_v1_5_1 import FlamingoV1_5_1
from envs.flamingo_light_proto_v1.flamingo_light_proto_v1 import FlamingoLightProtoV1
from envs.gaia_v1.gaia_v1 import GaiaV1
from envs.wrappers import TimeLimitWrapper, ActionInStateWrapper, StateStackWrapper, ExternalObsWrapper, CommandWrapper

def build_env(config):
    if config["env"]['id'] == "flamingo_v1_5_1":
      env = FlamingoV1_5_1(config)
    elif config["env"]['id'] == "flamingo_light_proto_v1":
      env = FlamingoLightProtoV1(config)
    elif config["env"]['id'] == "gaia_v1":
      env = GaiaV1(config)
    else:
      raise NameError(f"Please select a valid environment id. Received '{config['env']['id']}'.")

    env = TimeLimitWrapper(env, config)

    if config["env"]["action_in_state"]:
        env = ActionInStateWrapper(env, config)

    env = StateStackWrapper(env, config)

    if config["env"]["external_sensors"] != "None":
       env = ExternalObsWrapper(env, config)

    env = CommandWrapper(env, config)

    return env
