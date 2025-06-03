import numpy as np
from abc import ABC, abstractmethod
from typing import (Tuple, SupportsFloat)


class BaseEnv(ABC):
    """
    Abstract base class for an environment wrapper for the Sim2Sim framework.
    Defines the standard interface that all environment implementations must follow.
    """

    def __init__(self):
        """
        Initializes the environment.
        This constructor does not perform any specific initialization and should be
        overridden by subclasses if necessary.
        """
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment to its initial state.

        :return:
            - observation (np.ndarray): The initial observation after resetting the environment.
            - info (dict): Additional information about the environment's state, which may
              include metadata or debugging information.
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """
        Executes a step in the environment given an action.

        :param action:
            - np.ndarray: The action to be taken by the agent.

        :return:
            - observation (np.ndarray): The new observation after executing the action.
            - terminated (bool): Whether the episode has ended due to reaching a terminal state.
            - truncated (bool): Whether the episode was truncated due to external constraints
              such as time limits.
            - info (dict): Additional information about the environment’s state, which may
              include diagnostic data or auxiliary variables.
        """
        pass

    @abstractmethod
    def event(self, event: str, value):
        """
        Triggers an event.

        :param event: Name of the event (e.g., "push").
        :param value: Associated value to be passed with the event (e.g., velocity vector).
        """
        pass

    @abstractmethod
    def render(self):
        """
        Renders the environment.

        Provides a visual representation of the environment, such as a GUI window
        or textual output. This is useful for debugging and analysis.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes the environment and releases resources.

        Ensures proper cleanup, such as closing windows or stopping background processes.
        Should be called when the environment is no longer needed.
        """
        pass


class TimeLimitWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.id = env.id
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.max_sim_step = int(config["env"]["max_duration"] * 50)
        self.sim_step = 0
        self.reset_flag = False

    def reset(self):
        self.reset_flag = True
        self.sim_step = 0
        init_state, info = self.env.reset()

        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call `reset()` before calling `step()`."
        self.sim_step += 1
        next_state, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.reset_flag = True

        if self.sim_step == self.max_sim_step:
            truncated = True
            self.reset_flag = False

        return next_state, terminated, truncated, info
    
    def event(self, event: str, value):
        return self.env.event(event, value)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class ActionInStateWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.id = env.id
        self.state_dim = env.state_dim + config["env"]["action_dim"]
        self.action_dim = env.action_dim
        self.reset_flag = False

    def reset(self):
        self.reset_flag = True
        init_state, info = self.env.reset()
        init_state = np.concatenate((init_state, np.zeros(self.action_dim)))

        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call `reset()` before calling `step()`."
        next_state, terminated, truncated, info = self.env.step(action)
        next_state = np.concatenate((next_state, action))

        if terminated or truncated:
            self.reset_flag = False

        return next_state, terminated, truncated, info
    
    def event(self, event: str, value):
        return self.env.event(event, value)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class StateStackWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.id = env.id
        self.num_stack = config["env"]["num_stack"]
        self.state_dim = env.state_dim * self.num_stack
        self.action_dim = env.action_dim
        self.state_stack = np.zeros((self.num_stack, env.state_dim))
        self.reset_flag = False

    def reset(self):
        self.reset_flag = True
        init_state, info = self.env.reset()
        for i in range(self.config["env"]["num_stack"] ):
            self.state_stack[i] = init_state
        init_state = self.state_stack.ravel()

        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call `reset()` before calling `step()`."
        next_state, terminated, truncated, info = self.env.step(action)

        for i in range(self.num_stack - 1):
            self.state_stack[self.num_stack -1 - i] = self.state_stack[self.num_stack - 2 - i]
        self.state_stack[0] = next_state
        next_state = self.state_stack.ravel()

        if terminated or truncated:
            self.reset_flag = False

        return next_state, terminated, truncated, info

    def event(self, event: str, value):
        return self.env.event(event, value)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()



class ExternalObsWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.id = env.id
        self.external_sensors = self.config["env"]["external_sensors"] 
        if self.external_sensors == "height_map":
            height_map_dim = self.config["env"]["height_map"]["x_res"] * self.config["env"]["height_map"]["y_res"]
            self.state_dim = env.state_dim + height_map_dim 
        elif self.external_sensors == "scaled_base_lin_vel":
            self.state_dim = env.state_dim + 3
        elif self.external_sensors == "All":
            height_map_dim = self.config["env"]["height_map"]["x_res"] * self.config["env"]["height_map"]["y_res"]         
            self.state_dim = env.state_dim + (height_map_dim + 3)  # combine height_map and scaled_base_lin_vel
        elif self.external_sensors == "None":
            raise NotImplementedError(f"You must specify at least one external_sensors option when wrapping ExternalObsWrapper.")
        else:
            raise NotImplementedError(f"external_sensors option '{self.external_sensors}' is not supported.")
        
        self.action_dim = env.action_dim
        self.reset_flag = False     

    def reset(self):
        self.reset_flag = True
        init_state, info = self.env.reset()

        if self.external_sensors == "height_map":
            external_obs = info["height_map"]
        elif self.external_sensors == "scaled_base_lin_vel":
            external_obs = info["scaled_base_lin_vel"]
        elif self.external_sensors == "All":
            external_obs = np.concatenate((info["scaled_base_lin_vel"], info["height_map"]))  # scaled_base_lin_vel first
        else:
            raise NotImplementedError(f"external_sensors option '{self.external_sensors}' is not supported.")

        init_state = np.concatenate((init_state, external_obs))
        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call `reset()` before calling `step()`."
        next_state, terminated, truncated, info = self.env.step(action)

        if self.external_sensors == "height_map":
            external_obs = info["height_map"]
        elif self.external_sensors == "scaled_base_lin_vel":
            external_obs = info["scaled_base_lin_vel"]
        elif self.external_sensors == "All":
            external_obs = np.concatenate((info["scaled_base_lin_vel"], info["height_map"]))  # scaled_base_lin_vel first
        else:
            raise NotImplementedError(f"external_sensors '{self.external_sensors}' are not supported.")

        next_state = np.concatenate((next_state, external_obs))

        if terminated or truncated:
            self.reset_flag = False

        return next_state, terminated, truncated, info

    def event(self, event: str, value):
        return self.env.event(event, value)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class CommandWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.id = env.id
        self.state_dim = env.state_dim + config["env"]["command_dim"]
        self.action_dim = env.action_dim
        self.command_dim = config["env"]["command_dim"]
        self.user_command = np.zeros(config["env"]["command_dim"])
        self.scaled_command = np.zeros(config["env"]["command_dim"])
        self.reset_flag = False       
        assert self.command_dim > 0, "command_dim must be greater than 0."

    def receive_user_command(self, user_command):
        self.user_command = user_command[:self.command_dim]
        self.scaled_command[:] = user_command
        if self.config["obs_scales"]["scale_commands"]:
            if self.id == "flamingo_light_proto_v1":
                self.scaled_command[0] *= self.config["obs_scales"]["lin_vel"]
                self.scaled_command[1] *= self.config["obs_scales"]["ang_vel"]
            elif self.id == "flamingo_v1_5_1":
                self.scaled_command[0] *= self.config["obs_scales"]["lin_vel"]
                self.scaled_command[1] *= self.config["obs_scales"]["lin_vel"]
                if len(self.user_command) > 2:
                    self.scaled_command[2] *= self.config["obs_scales"]["ang_vel"]
            else:
                raise NotImplementedError(f"Feeding commands to robot: '{self.id}' is not supported.")

    def reset(self):
        self.reset_flag = True
        init_state, info = self.env.reset()
        init_state = np.concatenate((init_state, np.zeros(self.config["env"]["command_dim"])))
        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call `reset()` before calling `step()`."
        next_state, terminated, truncated, info = self.env.step(action)
        next_state = np.concatenate((next_state, self.scaled_command))

        if self.id == "flamingo_light_proto_v1":
            info["lin_vel_x_command"] = self.user_command[0]
            info["ang_vel_z_command"] = self.user_command[1]
        elif self.id == "flamingo_v1_5_1":
            info["lin_vel_x_command"] = self.user_command[0]
            info["lin_vel_y_command"] = self.user_command[1]
            if len(self.user_command) > 2:
                info["ang_vel_z_command"] = self.user_command[2]
            if len(self.user_command) > 3:
                info["pos_z_command"] = self.user_command[3]
            if len(self.user_command) > 4:
                info["ang_roll_command"] = self.user_command[4]
            if len(self.user_command) > 5:
                info["ang_pitch_command"] = self.user_command[5]
        else:
            raise NameError("Choose the correct robot id.")

        if terminated or truncated:
            self.reset_flag = False

        return next_state, terminated, truncated, info

    def event(self, event: str, value):
        return self.env.event(event, value)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


