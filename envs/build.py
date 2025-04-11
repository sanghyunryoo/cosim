from envs.flamingo_v1_4_2.flamingo_v1_4_2 import FlamingoV1_4_2
from envs.flamingo_v1_5_1.flamingo_v1_5_1 import FlamingoV1_5_1
from envs.flamingo_light_proto_v1.flamingo_light_proto_v1 import FlamingoLightProtoV1
from envs.wrappers import TimeLimitWrapper, ActionInStateWrapper, StateStackWrapper, CommandWrapper, TimeInStateWrapper


def build_env(config):
    if config["env"]['id'] == "flamingo_v1_4_2":
      env = FlamingoV1_4_2(config)
    if config["env"]['id'] == "flamingo_v1_5_1":
      env = FlamingoV1_5_1(config)
    elif config["env"]['id'] == "flamingo_light_proto_v1":
      env = FlamingoLightProtoV1(config)
    else:
      raise NameError("Select a proper environment!")

    env = TimeLimitWrapper(env, config)
    if config["env"]["action_in_state"]:
        env = ActionInStateWrapper(env, config)
    env = StateStackWrapper(env, config)
    if config["env"]["time_in_state"]:
        env = TimeInStateWrapper(env, config)
    env = CommandWrapper(env, config)

    return env
