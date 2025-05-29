from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco
from envs.flamingo_light_proto_v1.manager.control_manager import ControlManager
from envs.flamingo_light_proto_v1.manager.xml_manager import XMLManager
from envs.flamingo_light_proto_v1.utils.math_utils import MathUtils
from envs.flamingo_light_proto_v1.utils.mujoco_utils import MuJoCoUtils
from envs.flamingo_light_proto_v1.utils.noise_generator_utils import truncated_gaussian_noisy_data, uniform_noisy_data
import glfw


class FlamingoLightProtoV1(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    def __init__(self, config, render_flag=True, render_mode='human'):
        # Set Basic Properties
        self.id = "flamingo_light_proto_v1"
        self.config = config
        self.state_dim = config["env"]["observation_dim"]
        self.action_dim = config["env"]["action_dim"]
        self.command_dim = config["env"]["command_dim"]
        self.render_mode = render_mode
        self.render_flag = render_flag

        # PD control parameters
        self.kp_shoulder = config["hardware"]["Kp_shoulder"]
        self.kp_wheel = config["hardware"]["Kp_wheel"]

        self.kd_shoulder = config["hardware"]["Kd_shoulder"]
        self.kd_wheel = config["hardware"]["Kd_wheel"]

        self.action_scaler = [1.25, 1.25, 30.0, 30.0]

        # Set Simulation Properties
        precision_level = self.config["random"]["precision"]
        sensor_noise_level = self.config["random"]["sensor_noise"]
        self.init_noise = self.config["random"]["init_noise"]
        self.dt_ = config["random_table"]["precision"][precision_level]["timestep"]
        self.frame_skip = config["random_table"]["precision"][precision_level]["frame_skip"]
        self.sensor_noise_map = config["random_table"]["sensor_noise"][sensor_noise_level]

        # Set Placeholders
        self.action = np.zeros(self.action_dim)
        self.filtered_action = np.zeros(self.action_dim)
        self.previous_action = np.zeros(self.action_dim)
        self.computed_torques = np.zeros(self.action_dim)
        self.applied_torques = np.zeros(self.action_dim)
        self.obs = None
        self.scaled_obs = None
        self.viewer = None
        self.mode = None

        # Domain Randomization
        self.xml_manager = XMLManager(config)
        self.model_path = self.xml_manager.get_model_path()

        # Set MuJoCo Wrapper
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(
            self,
            model_path=self.model_path,
            frame_skip=self.frame_skip,
            observation_space=Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32,),
            render_mode=self.render_mode if render_flag else None,
        )

        # Set other Managers and Helpers
        self.control_manager = ControlManager(config)
        self.mujoco_utils = MuJoCoUtils(self.model)

        # Set Indices of q and qd
        qpos_joint_names = ["left_shoulder_joint", "right_shoulder_joint"]
        qvel_joint_names = ["left_shoulder_joint", "right_shoulder_joint", "left_wheel_joint", "right_wheel_joint"]
        self.q_indices = self.mujoco_utils.get_qpos_joint_indices_by_name(qpos_joint_names)
        self.qd_indices = self.mujoco_utils.get_qvel_joint_indices_by_name(qvel_joint_names)

    def _get_obs(self):
        q = self.data.qpos[self.q_indices]
        qd = self.data.qvel[self.qd_indices]
        quat = self.data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        if np.all(quat == 0):
            quat = np.array([0, 0, 0, 1])

        omega = self.data.sensor('angular-velocity').data.astype(np.double)
        projected_gravity = MathUtils.quat_to_base_vel(quat, np.array([0, 0, -1], dtype=np.double))

        q = truncated_gaussian_noisy_data(q, mean=self.sensor_noise_map["q"]["mean"], std=self.sensor_noise_map["q"]["std"],
                                          lower=self.sensor_noise_map["q"]["lower"], upper=self.sensor_noise_map["q"]["upper"])
        qd = truncated_gaussian_noisy_data(qd, mean=self.sensor_noise_map["qd"]["mean"], std=self.sensor_noise_map["qd"]["std"],
                                          lower=self.sensor_noise_map["qd"]["lower"], upper=self.sensor_noise_map["qd"]["upper"])
        omega = truncated_gaussian_noisy_data(omega, mean=self.sensor_noise_map["omega"]["mean"], std=self.sensor_noise_map["omega"]["std"],
                                          lower=self.sensor_noise_map["omega"]["lower"], upper=self.sensor_noise_map["omega"]["upper"])
        projected_gravity = truncated_gaussian_noisy_data(projected_gravity, mean=self.sensor_noise_map["projected_gravity"]["mean"], std=self.sensor_noise_map["projected_gravity"]["std"],
                                          lower=self.sensor_noise_map["projected_gravity"]["lower"], upper=self.sensor_noise_map["projected_gravity"]["upper"])
        obs = np.concatenate([q * self.config["obs_scales"]["dof_pos"], qd * self.config["obs_scales"]["dof_vel"], omega * self.config["obs_scales"]["ang_vel"], projected_gravity])
        scaled_obs = np.concatenate([q * self.config["obs_scales"]["dof_pos"], qd * self.config["obs_scales"]["dof_vel"], omega * self.config["obs_scales"]["ang_vel"], projected_gravity])
     
        return obs, scaled_obs

    def step(self, action):
        try:
            self.action = action
            self.filtered_action = self.control_manager.delay_filter(action)

            # Pull the current joint positions and velocities
            q = self.data.qpos[self.q_indices]
            qd = self.data.qvel[self.qd_indices]

            # Extract joint positions and velocities from observation
            pos_shoulder = q[0:2]
            vel_shoulder = qd[0:2]
            vel_wheel = qd[2:4]

            shoulder_action_scaled = self.action[0:2] * self.action_scaler[0:2]
            wheel_action_scaled = self.action[2:4] * self.action_scaler[2:4]

            shoulder_action = self.control_manager.pd_controller(
                self.kp_shoulder, shoulder_action_scaled, pos_shoulder, self.kd_shoulder, 0.0, vel_shoulder
            )
            wheel_action = self.control_manager.pd_controller(
                self.kp_wheel, 0.0, 0.0, self.kd_wheel, wheel_action_scaled, vel_wheel
            )

            self.computed_torques = np.concatenate([shoulder_action, wheel_action])

            shoulder_action_clipped = np.clip(shoulder_action, -self.config['hardware']['joint_max_torque'], self.config['hardware']['joint_max_torque'])
            wheel_action_clipped = np.clip(wheel_action, -self.config['hardware']['wheel_max_torque'], self.config['hardware']['wheel_max_torque'])

            self.applied_torques = np.concatenate([shoulder_action_clipped, wheel_action_clipped])

            self.do_simulation(self.applied_torques, self.frame_skip)

            self.obs, self.scaled_obs = self._get_obs()
            terminated = self._is_done()
            truncated = False
            info = self._get_info()

            self.previous_action = self.action

        except Exception as e:
            print(f"[FlamingoLightProtoV1] ERROR: {e}")
            raise

        return self.scaled_obs, terminated, truncated, info

    def _get_info(self):
        q = self.data.qpos[self.q_indices]
        qd = self.data.qvel[self.qd_indices]
        cur_state = [q[0], q[1], qd[2], qd[3]]

        info = {
            "dt": self.dt_ * self.frame_skip,
            "action": self.action,
            "torque": self.applied_torques,
            "action_diff_RMS": np.linalg.norm(self.action - self.previous_action),
            "lin_vel_x": self.data.sensor('linear-velocity').data.astype(np.float32)[0],
            "lin_vel_y": self.data.sensor('linear-velocity').data.astype(np.float32)[1],
            "ang_vel_z": self.obs[8],
            "pos_z" : self.data.qpos[2],
            "ang_roll" :  self.obs[9],
            "ang_pitch":  self.obs[10],
            "set_points": self.action * self.action_scaler,
            "cur_state": cur_state
        }

        return info

    def _get_reset_info(self):
        return self._get_info()

    def _is_done(self):
        # Get the IDs of the bodies of interest
        body_ids = self.mujoco_utils.get_body_indices_by_name(["base_link", "left_leg_link", "right_leg_link"])

        # Iterate through all active contacts in the simulation
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get the body indices from the geometry indices involved in the contact
            body1_id = self.model.geom_bodyid[contact.geom1]
            body2_id = self.model.geom_bodyid[contact.geom2]

            # Check if either of the bodies in this contact is one of the bodies of interest
            if body1_id in body_ids or body2_id in body_ids:
                # If a contact involves one of the interested bodies, return True to indicate simulation should be reset
                return True

        # If no relevant contact is found, return False
        return False

    def reset_model(self):
        self.action = np.zeros(self.action_dim)
        self.previous_action = np.zeros(self.action_dim)
        self.control_manager.reset()
        self.applied_torques = np.zeros(self.action_dim)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos()
        self.data.qvel[:] = 0
        self.obs, self.scaled_obs = self._get_obs()
        return self.scaled_obs

    def initial_qpos(self):
        qpos = np.zeros(self.model.nq)
        qpos[2] = 0.2607  # Initial height: 0.2607
        qpos[3:7] = np.array([1, 0, 0, 0])  # Initial orientation
        qpos_joint_names = ["left_shoulder_joint", "right_shoulder_joint", "left_wheel_joint", "right_wheel_joint"]
        q_indices = self.mujoco_utils.get_qpos_joint_indices_by_name(qpos_joint_names)

        # Radom init
        qpos[q_indices] = uniform_noisy_data(qpos[q_indices], lower=-self.init_noise, upper=self.init_noise)

        return qpos
    
    def event(self, event: str, value):
        if event == 'push':
            self.data.qvel[:3] = value[:]
        else:
            raise NotImplementedError(f"event:{event} is not supported.")

    
    def close(self):
        if self.viewer is not None:
            if glfw.get_current_context() == self.viewer.window:
                glfw.make_context_current(None)
            glfw.destroy_window(self.viewer.window)
            glfw.terminate()
            self.viewer = None
            print("Viewer closed")
        super().close()  # Call the parent class's close method to ensure everything is properly closed


