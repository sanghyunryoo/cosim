import warnings
from abc import ABC, abstractmethod
from typing import (Tuple, SupportsFloat)
import numpy as np


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
    def get_data(self):
        """Retrieve low-level environment data from the wrapped environment."""
        return self.env.get_data()

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


class StateBuildWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.id = env.id
        self.action_dim = env.action_dim
        self.sim_step = 0
        self.reset_flag = False

        # Require env.control_freq (Hz); fail fast if missing/invalid
        if not hasattr(self.env, "control_freq"):
            raise AttributeError(f"Env: {self.id} must define 'control_freq' (Hz).")
        self.control_freq = float(self.env.control_freq)
        if self.control_freq <= 0:
            raise ValueError(f"Invalid env.control_freq: {self.control_freq}. Must be > 0.")

        # Number of frames to stack (i=0 is the most recent frame)
        self.stack_size = int(self.config["observation"]["stack_size"])
        # Ordered observation keys that will be stacked
        self.stacked_obs_order = list(self.config["observation"]["stacked_obs_order"])
        # Ordered observation keys that will NOT be stacked (single-frame)
        self.non_stacked_obs_order = list(self.config["observation"]["non_stacked_obs_order"])

        # Cache dimensions   
        self._stacked_obs_dim = sum(self.env.obs_to_dim[n] for n in self.stacked_obs_order)
        self._non_stacked_obs_dim = sum(self.env.obs_to_dim[n] for n in self.non_stacked_obs_order)
        self.state_dim = self.stack_size * self._stacked_obs_dim + self._non_stacked_obs_dim
  
        # Rolling buffer for stacked observations (shape: [stack_size, stacked_obs_dim])
        self.obs_buffer = np.zeros((self.stack_size, self._stacked_obs_dim), dtype=np.float32)

        # Cache for frequency/scale-applied observation values
        self._freq_cache = {}

    def _concat_obs_with_freq(self, obs, names):
        """
        Concatenate observations following frequency (Hz) and scale rules defined in
        config["observation"][<name>]. For each name:
          - If sim_step == 0: always refresh with the latest 'obs' value.
          - Else: refresh only when (control_freq / freq) step interval elapses; otherwise keep cached value.
          - Always apply 'scale' by multiplication.

        Args:
            obs (dict): {name: np.ndarray}-like observation dictionary from the env.
            names (list[str]): keys to fetch/concatenate in order.

        Returns:
            np.ndarray (float32): concatenated 1D vector with freq/scale applied.
        """
        parts = []
        for n in names:
            n_cfg = self.config["observation"][n]
            update_freq = float(n_cfg["freq"])
            scale = float(n_cfg["scale"])

            if update_freq <= 0:
                raise ValueError(f"Invalid observation update frequency for '{n}': {update_freq}. Must be > 0.")

            # Steps between updates; at least 1 to avoid division artifacts
            update_interval = max(1, int(round(self.control_freq / update_freq)))
            need_update = (self.sim_step == 0) or (self.sim_step % update_interval == 0)

            if need_update or (n not in self._freq_cache):
                val = np.asarray(obs[n], dtype=np.float32) * scale
                self._freq_cache[n] = val

            parts.append(self._freq_cache[n].ravel().astype(np.float32))

        if parts:
            return np.concatenate(parts, axis=0)
        else:
            return np.zeros((0,), dtype=np.float32)

    def _push_stack(self, latest_vec, reset=False):
        """
        Push a new stacked frame into the rolling buffer.

        - If reset=True: fill the whole buffer with 'latest_vec'.
        - If reset=False: shift older frames down by one and put 'latest_vec' at index 0.
          Index 0 is the most recent; index (stack_size - 1) is the oldest.
        """
        if reset:
            self.obs_buffer[:] = latest_vec
        else:
            if self.stack_size > 1:
                self.obs_buffer[1:, :] = self.obs_buffer[:-1, :]
            self.obs_buffer[0, :] = latest_vec

    def _build_state(self, obs, reset: bool):
        """
        Build the final 1D state vector consisting of:
        - Flattened stacked observations (from the rolling buffer).
        - Concatenated non-stacked observations (also subject to freq/scale).

        Args:
            obs (dict): environment observation dict.
            reset (bool): if True, re-initialize the stack with the current observation.

        Returns:
            np.ndarray (float32): state vector of length 'state_dim'.
        """
        # 1) Gather stacked observations (with frequency/scale rules)
        obs_for_stack = self._concat_obs_with_freq(obs, self.stacked_obs_order)

        # 2) Update the rolling buffer
        self._push_stack(obs_for_stack, reset=reset)

        # 3) Flatten stacked frames and append non-stacked observations (with freq/scale)
        stacked_flat = self.obs_buffer.ravel()  # shape: (stack_size * stacked_obs_dim,)
        non_stacked_vec = self._concat_obs_with_freq(obs, self.non_stacked_obs_order)

        state = np.concatenate([stacked_flat, non_stacked_vec], axis=0)
        return state.astype(np.float32)

    def reset(self):
        """
        Reset the underlying environment and reinitialize internal counters and caches.
        Fills the stack with the initial observation.
        """
        self.reset_flag = True
        self.sim_step = 0
        self._freq_cache.clear()
        init_obs, info = self.env.reset()
        init_state = self._build_state(init_obs, reset=True)
        return init_state, info

    def step(self, action: np.ndarray):
        """
        Step through the environment and build the next state.
        """
        assert self.reset_flag is True, "Call 'reset()' before calling 'step()'."
        self.sim_step += 1
        next_obs, terminated, truncated, info = self.env.step(action)
        next_state = self._build_state(next_obs, reset=False)

        if terminated or truncated:
            self.reset_flag = False
        return next_state, terminated, truncated, info

    def event(self, event: str, value):
        """Forward custom events to the wrapped environment."""
        return self.env.event(event, value)

    def get_data(self):
        """Proxy for any data export the wrapped environment supports."""
        return self.env.get_data()

    def render(self):
        """Render via the wrapped environment."""
        self.env.render()

    def close(self):
        """Close the wrapped environment."""
        self.env.close()



class TimeLimitWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.id = env.id
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.sim_step = 0
        self.max_sim_step = int(config["env"]["max_duration"] * self.env.control_freq)
        self.reset_flag = False

    def reset(self):
        self.reset_flag = True
        self.sim_step = 0
        init_state, info = self.env.reset()
        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call 'reset()' before calling 'step()'."
        self.sim_step += 1
        next_state, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.reset_flag = False

        if self.sim_step == self.max_sim_step:
            truncated = True
            self.reset_flag = False

        return next_state, terminated, truncated, info
    
    def event(self, event: str, value):
        return self.env.event(event, value)

    def get_data(self):
        return self.env.get_data()

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
        self.action_dim = env.action_dim
        self.command_dim = config["observation"]["command_dim"]
        self.state_dim = env.state_dim + self.command_dim
        self.user_command = np.zeros(config["observation"]["command_dim"])
        self.applied_command = np.zeros(config["observation"]["command_dim"])
        self.reset_flag = False       
        assert self.command_dim > 0, "command_dim must be greater than 0."

    def receive_user_command(self, user_command):
        self.user_command = user_command[:self.command_dim]
        self.applied_command[:] = user_command

        if self.config["env"]["position_command"] is False:
            for i in range(self.command_dim):
                self.applied_command[i] *= self.config["observation"]["command_scales"][str(i)]        
        else:
            assert self.command_dim == 2, f"Currently, position command only support 2 dimenstion, but got {self.command_dim}."
            warnings.warn("For position commands, 'command_scales' is always treated as 1.0.")

            data = self.get_data()
            robot_px, robot_py = data.qpos[0], data.qpos[1]
            target_x, target_y = self.user_command[0], self.user_command[1]

            delta_world_x = target_x - robot_px
            delta_world_y = target_y - robot_py

            w, x, y, z = data.qpos[3:7].astype(np.float64)
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            cosy, siny = np.cos(-yaw), np.sin(-yaw)

            robot_x = cosy * delta_world_x - siny * delta_world_y
            robot_y = siny * delta_world_x + cosy * delta_world_y

            self.applied_command[0] = robot_x
            self.applied_command[1] = robot_y

    def reset(self):
        self.reset_flag = True
        init_state, info = self.env.reset()
        init_state = np.concatenate((init_state, np.zeros(self.config["observation"]["command_dim"])))
        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call 'reset()' before calling 'step()'."
        next_state, terminated, truncated, info = self.env.step(action)
        next_state = np.concatenate((next_state, self.applied_command))

        if self.command_dim == 2:
            info["user_command_0"] = self.user_command[0]
            info["user_command_1"] = self.user_command[1]
        elif self.command_dim > 2:
            info["user_command_0"] = self.user_command[0]
            info["user_command_1"] = self.user_command[1]
            info["user_command_2"] = self.user_command[2]
        else:
            raise ValueError(f"Invalid 'command_dim': expected 2  or >= 3; but got {self.command_dim}.")

        if terminated or truncated:
            self.reset_flag = False

        return next_state, terminated, truncated, info

    def event(self, event: str, value):
        return self.env.event(event, value)
    
    def get_data(self):
        return self.env.get_data()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


