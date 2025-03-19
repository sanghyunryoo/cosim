from envs.flamingo_v1_3.flamingo_v1_3 import FlamingoV1_3
from envs.flamingo_v1_4.flamingo_v1_4 import FlamingoV1_4
from envs.flamingo_v1_4_1.flamingo_v1_4_1 import FlamingoV1_4_1
from envs.flamingo_v1_4_2.flamingo_v1_4_2 import FlamingoV1_4_2
from envs.flamingo_edu_v1.flamingo_edu_v1 import FlamingoEduV1
from envs.wrappers import TimeLimitWrapper, ActionInStateWrapper, StateStackWrapper, CommandWrapper, TimeInStateWrapper


def build_env(config):
    if config["env"]["id"] == "flamingo_v1_3":
      env = FlamingoV1_3(config)
    elif config["env"]["id"] == "flamingo_v1_4":
      env = FlamingoV1_4(config)
    elif config["env"]["id"] == "flamingo_v1_4_1":
      env = FlamingoV1_4_1(config)
    elif config["env"]['id'] == "flamingo_v1_4_2":
      env = FlamingoV1_4_2(config)
    elif config["env"]['id'] == "flamingo_edu_v1":
      env = FlamingoEduV1(config)
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
