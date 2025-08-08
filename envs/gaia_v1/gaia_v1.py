from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco
from envs.gaia_v1.manager.control_manager import ControlManager
from envs.gaia_v1.manager.xml_manager import XMLManager
from envs.gaia_v1.utils.math_utils import MathUtils
from envs.gaia_v1.utils.mujoco_utils import MuJoCoUtils
from envs.gaia_v1.utils.noise_generator_utils import truncated_gaussian_noisy_data, uniform_noisy_data
import glfw


class GaiaV1(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    def __init__(self, config, render_flag=True, render_mode='human'):
        # Set Basic Properties
        self.id = "gaia_v1"
        self.config = config
        self.state_dim = config["env"]["observation_dim"]
        self.action_dim = config["env"]["action_dim"]
        self.command_dim = config["env"]["command_dim"]
        self.render_mode = render_mode
        self.render_flag = render_flag
        if config["env"]["external_sensors"] == "None":
            self.use_external_sensors = False
        else:
            self.use_external_sensors = True

        # PD control parameters
        self.kp_hip_pitch = config["hardware"]["Kp_hip_pitch"]
        self.kp_torso = config["hardware"]["Kp_torso"]
        self.kp_hip_roll = config["hardware"]["Kp_hip_roll"]
        self.kp_shoulder_pitch = config["hardware"]["Kp_shoulder_pitch"]
        self.kp_hip_yaw = config["hardware"]["Kp_hip_yaw"]
        self.kp_shoulder_roll = config["hardware"]["Kp_shoulder_roll"]
        self.kp_knee = config["hardware"]["Kp_knee"]
        self.kp_shoulder_yaw = config["hardware"]["Kp_shoulder_yaw"]
        self.kp_ankle_pitch = config["hardware"]["Kp_ankle_pitch"]
        self.kp_elbow_pitch = config["hardware"]["Kp_elbow_pitch"]    
        self.kp_ankle_roll = config["hardware"]["Kp_ankle_roll"]  
        self.kp_elbow_yaw = config["hardware"]["Kp_elbow_yaw"]  

        self.kd_hip_pitch = config["hardware"]["Kd_hip_pitch"]
        self.kd_torso = config["hardware"]["Kd_torso"]
        self.kd_hip_roll = config["hardware"]["Kd_hip_roll"]
        self.kd_shoulder_pitch = config["hardware"]["Kd_shoulder_pitch"]
        self.kd_hip_yaw = config["hardware"]["Kd_hip_yaw"]
        self.kd_shoulder_roll = config["hardware"]["Kd_shoulder_roll"]
        self.kd_knee = config["hardware"]["Kd_knee"]
        self.kd_shoulder_yaw = config["hardware"]["Kd_shoulder_yaw"]
        self.kd_ankle_pitch = config["hardware"]["Kd_ankle_pitch"]
        self.kd_elbow_pitch = config["hardware"]["Kd_elbow_pitch"]    
        self.kd_ankle_roll = config["hardware"]["Kd_ankle_roll"]  
        self.kd_elbow_yaw = config["hardware"]["Kd_elbow_yaw"]       
   

        self.action_scaler = [0.5, 0.5, 0.5, 0.5,
                              0.5, 0.5, 0.5, 0.5,
                              0.5, 0.5, 0.5, 0.5,
                              0.5, 0.5, 0.5, 0.5,
                              0.5, 0.5, 0.5, 0.5,
                              0.5, 0.5, 0.5]

        # Set Simulation Properties
        precision_level = self.config["random"]["precision"]
        sensor_noise_level = self.config["random"]["sensor_noise"]
        self.init_noise = self.config["random"]["init_noise"]
        self.dt_ = config["random_table"]["precision"][precision_level]["timestep"]
        self.frame_skip = config["random_table"]["precision"][precision_level]["frame_skip"]
        self.sensor_noise_map = config["random_table"]["sensor_noise"][sensor_noise_level]
        self.control_freq = 1 / (self.dt_ * self.frame_skip)
        self.local_step = 0

        # Set Placeholders
        self.action = np.zeros(self.action_dim)
        self.filtered_action = np.zeros(self.action_dim)
        self.previous_action = np.zeros(self.action_dim)
        self.applied_torques = np.zeros(self.action_dim)
        self.obs = None
        self.scaled_obs = None
        self.viewer = None
        self.mode = None
        if self.use_external_sensors:
            self.external_obs = {}
            self.external_obs_freq = float(self.config['env']['external_sensors_Hz'])

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

        # Height Map
        if self.use_external_sensors and self.config["env"]["external_sensors"] in ['height_map', 'All']:
            self.x_size = self.config["env"]["height_map"]["x_size"]
            self.y_size = self.config["env"]["height_map"]["y_size"]
            self.x_res = self.config["env"]["height_map"]["x_res"]
            self.y_res = self.config["env"]["height_map"]["y_res"]
            self.mujoco_utils.init_heightmap_visualization(self.x_res, self.y_res)

        # Set Indices of q and qd
        joint_names_in_order = ["left_hip_pitch_joint", "right_hip_pitch_joint",
                            "torso_joint",
                            "left_hip_roll_joint", "right_hip_roll_joint",
                            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                            "left_hip_yaw_joint", "right_hip_yaw_joint",
                            "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                            "left_knee_joint", "right_knee_joint",
                            "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
                            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                            "left_elbow_pitch_joint", "right_elbow_pitch_joint",
                            "left_ankle_roll_joint", "right_ankle_roll_joint",
                            "left_elbow_yaw_joint", "right_elbow_yaw_joint",]

        self.q_indices = self.mujoco_utils.get_qpos_joint_indices_by_name(joint_names_in_order)
        self.qd_indices = self.mujoco_utils.get_qvel_joint_indices_by_name(joint_names_in_order)

        print(self.q_indices)
        print(self.qd_indices)
      

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
        obs = np.concatenate([q, qd, omega, projected_gravity])
        scaled_obs = np.concatenate([q * self.config["obs_scales"]["dof_pos"], qd * self.config["obs_scales"]["dof_vel"], omega * self.config["obs_scales"]["ang_vel"], projected_gravity])

        return obs, scaled_obs
    
    def _get_external_obs(self):
        if not self.use_external_sensors:
            return None

        base_lin_vel = self.data.sensor("linear-velocity").data.astype(np.float32)  # [vx, vy, vz]
        base_lin_vel = truncated_gaussian_noisy_data(base_lin_vel, mean=self.sensor_noise_map["base_lin_vel"]["mean"], std=self.sensor_noise_map["base_lin_vel"]["std"],
                                                      lower=self.sensor_noise_map["base_lin_vel"]["lower"], upper=self.sensor_noise_map["base_lin_vel"]["upper"])
        scaled_base_lin_vel = base_lin_vel * self.config["obs_scales"]["lin_vel"]  # scaled_base_lin_vel

        if self.use_external_sensors and self.config["env"]["external_sensors"] in ['height_map', 'All']:
            height_map = self.mujoco_utils.get_height_map(self.data, self.x_size, self.y_size, self.x_res,self.y_res)  # height_map
            height_map = truncated_gaussian_noisy_data(height_map, mean=self.sensor_noise_map["height_map"]["mean"], std=self.sensor_noise_map["height_map"]["std"],
                                                        lower=self.sensor_noise_map["height_map"]["lower"], upper=self.sensor_noise_map["height_map"]["upper"])
        else:
            height_map = None

        tick = max(1, int(self.control_freq / self.external_obs_freq))
        if self.local_step % tick == 0:     
            self.external_obs["scaled_base_lin_vel"] = scaled_base_lin_vel
            self.external_obs["height_map"] = height_map

        return self.external_obs

    def step(self, action):
        self.action = action
        self.filtered_action = self.control_manager.delay_filter(action)

        # Pull the current joint positions and velocities
        q = self.data.qpos[self.q_indices]
        qd = self.data.qvel[self.qd_indices]

        # Extract joint positions and velocities from observation (order from Isaac Lab)
        pos_hip_pitch, vel_hip_pitch = q[0:2], qd[0:2]
        pos_torso, vel_torso = q[2:3], qd[2:3]
        pos_hip_roll, vel_hip_roll = q[3:5], qd[3:5]
        pos_shoulder_pitch, vel_shoulder_pitch = q[5:7], qd[5:7]
        pos_hip_yaw, vel_hip_yaw = q[7:9], qd[7:9]
        pos_shoulder_roll, vel_shoulder_roll = q[9:11], qd[9:11]
        pos_knee, vel_knee = q[11:13], qd[11:13]
        pos_shoulder_yaw, vel_shoulder_yaw = q[13:15], qd[13:15]
        pos_ankle_pitch, vel_ankle_pitch = q[15:17], qd[15:17]
        pos_elbow_pitch, vel_elbow_pitch = q[17:19], qd[17:19]
        pos_ankle_roll, vel_ankle_roll = q[19:21], qd[19:21]
        pos_elbow_yaw, vel_elbow_yaw = q[21:23], qd[21:23]

        # Get the scaled action
        hip_pitch_action_scaled = self.filtered_action[0:2] * self.action_scaler[0:2]
        torso_action_scaled = self.filtered_action[2:3] * self.action_scaler[2:3]
        hip_roll_action_scaled = self.filtered_action[3:5] * self.action_scaler[3:5]
        shoulder_pitch_action_scaled = self.filtered_action[5:7] * self.action_scaler[5:7]
        hip_yaw_action_scaled = self.filtered_action[7:9] * self.action_scaler[7:9]
        shoulder_roll_action_scaled = self.filtered_action[9:11] * self.action_scaler[9:11]
        knee_action_scaled = self.filtered_action[11:13] * self.action_scaler[11:13]
        shoulder_yaw_action_scaled = self.filtered_action[13:15] * self.action_scaler[13:15]
        ankle_pitch_action_scaled = self.filtered_action[15:17] * self.action_scaler[15:17]
        elbow_pitch_action_scaled = self.filtered_action[17:19] * self.action_scaler[17:19]
        ankle_roll_action_scaled = self.filtered_action[19:21] * self.action_scaler[19:21]
        elbow_yaw_action_scaled = self.filtered_action[21:23] * self.action_scaler[21:23]

        hip_pitch_torques = self.control_manager.pd_controller(self.kp_hip_pitch, hip_pitch_action_scaled, pos_hip_pitch, self.kd_hip_pitch, 0.0, vel_hip_pitch)
        torso_torques = self.control_manager.pd_controller(self.kp_torso, torso_action_scaled, pos_torso, self.kd_torso, 0.0, vel_torso)
        hip_roll_torques = self.control_manager.pd_controller(self.kp_hip_roll, hip_roll_action_scaled, pos_hip_roll, self.kd_hip_roll, 0.0, vel_hip_roll)
        shoulder_pitch_torques = self.control_manager.pd_controller(self.kp_shoulder_pitch, shoulder_pitch_action_scaled, pos_shoulder_pitch, self.kd_shoulder_pitch, 0.0, vel_shoulder_pitch)
        hip_yaw_torques = self.control_manager.pd_controller(self.kp_hip_yaw, hip_yaw_action_scaled, pos_hip_yaw, self.kd_hip_yaw, 0.0, vel_hip_yaw)
        shoulder_roll_torques = self.control_manager.pd_controller(self.kp_shoulder_roll, shoulder_roll_action_scaled, pos_shoulder_roll, self.kd_shoulder_roll, 0.0, vel_shoulder_roll)
        knee_torques = self.control_manager.pd_controller(self.kp_knee, knee_action_scaled, pos_knee, self.kd_knee, 0.0, vel_knee)
        shoulder_yaw_torques = self.control_manager.pd_controller(self.kp_shoulder_yaw, shoulder_yaw_action_scaled, pos_shoulder_yaw, self.kd_shoulder_yaw, 0.0, vel_shoulder_yaw)
        ankle_pitch_torques = self.control_manager.pd_controller(self.kp_ankle_pitch, ankle_pitch_action_scaled, pos_ankle_pitch, self.kd_ankle_pitch, 0.0, vel_ankle_pitch)
        elbow_pitch_torques = self.control_manager.pd_controller(self.kp_elbow_pitch, elbow_pitch_action_scaled, pos_elbow_pitch, self.kd_elbow_pitch, 0.0, vel_elbow_pitch)
        ankle_roll_torques = self.control_manager.pd_controller(self.kp_ankle_roll, ankle_roll_action_scaled, pos_ankle_roll, self.kd_ankle_roll, 0.0, vel_ankle_roll)
        elbow_yaw_torques = self.control_manager.pd_controller(self.kp_elbow_yaw, elbow_yaw_action_scaled, pos_elbow_yaw, self.kd_elbow_yaw, 0.0, vel_elbow_yaw)

        hip_pitch_torques_clipped = np.clip(hip_pitch_torques, -self.config['hardware']['legs_joint_max_torque'], self.config['hardware']['legs_joint_max_torque'])
        torso_torques_clipped = np.clip(torso_torques, -self.config['hardware']['legs_joint_max_torque'], self.config['hardware']['legs_joint_max_torque'])
        hip_roll_torques_clipped = np.clip(hip_roll_torques, -self.config['hardware']['legs_joint_max_torque'], self.config['hardware']['legs_joint_max_torque'])
        shoulder_pitch_torques_clipped = np.clip(shoulder_pitch_torques, -self.config['hardware']['arms_joint_max_torque'], self.config['hardware']['arms_joint_max_torque'])
        hip_yaw_torques_clipped = np.clip(hip_yaw_torques, -self.config['hardware']['legs_joint_max_torque'], self.config['hardware']['legs_joint_max_torque'])
        shoulder_roll_torques_clipped = np.clip(shoulder_roll_torques, -self.config['hardware']['arms_joint_max_torque'], self.config['hardware']['arms_joint_max_torque'])
        knee_torques_clipped = np.clip(knee_torques, -self.config['hardware']['legs_joint_max_torque'], self.config['hardware']['legs_joint_max_torque'])
        shoulder_yaw_torques_clipped = np.clip(shoulder_yaw_torques, -self.config['hardware']['arms_joint_max_torque'], self.config['hardware']['arms_joint_max_torque'])
        ankle_pitch_torques_clipped = np.clip(ankle_pitch_torques, -self.config['hardware']['feet_joint_max_torque'], self.config['hardware']['feet_joint_max_torque'])
        elbow_pitch_torques_clipped = np.clip(elbow_pitch_torques, -self.config['hardware']['arms_joint_max_torque'], self.config['hardware']['arms_joint_max_torque'])
        ankle_roll_torques_clipped = np.clip(ankle_roll_torques, -self.config['hardware']['feet_joint_max_torque'], self.config['hardware']['feet_joint_max_torque'])
        elbow_yaw_torques_clipped = np.clip(elbow_yaw_torques, -self.config['hardware']['arms_joint_max_torque'], self.config['hardware']['arms_joint_max_torque'])

        self.applied_torques = np.concatenate([
            # Torso
            torso_torques_clipped,                     # 0

            # Left Arm
            shoulder_pitch_torques_clipped[0:1],       # 1
            shoulder_roll_torques_clipped[0:1],        # 2
            shoulder_yaw_torques_clipped[0:1],         # 3
            elbow_pitch_torques_clipped[0:1],          # 4
            elbow_yaw_torques_clipped[0:1],            # 5

            # Right Arm
            shoulder_pitch_torques_clipped[1:2],       # 6
            shoulder_roll_torques_clipped[1:2],        # 7
            shoulder_yaw_torques_clipped[1:2],         # 8
            elbow_pitch_torques_clipped[1:2],          # 9
            elbow_yaw_torques_clipped[1:2],            # 10

            # Left Leg
            hip_pitch_torques_clipped[0:1],            # 11
            hip_roll_torques_clipped[0:1],             # 12
            hip_yaw_torques_clipped[0:1],              # 13
            knee_torques_clipped[0:1],                 # 14
            ankle_pitch_torques_clipped[0:1],          # 15
            ankle_roll_torques_clipped[0:1],           # 16

            # Right Leg
            hip_pitch_torques_clipped[1:2],            # 17
            hip_roll_torques_clipped[1:2],             # 18
            hip_yaw_torques_clipped[1:2],              # 19
            knee_torques_clipped[1:2],                 # 20
            ankle_pitch_torques_clipped[1:2],          # 21
            ankle_roll_torques_clipped[1:2],           # 22
        ])
        
        self.do_simulation(self.applied_torques, self.frame_skip)

        self.obs, self.scaled_obs = self._get_obs()
        info = self._get_info()
        terminated = self._is_done()
        truncated = False

        self.previous_action = self.action
        self.local_step += 1  

        return self.scaled_obs, terminated, truncated, info

    def _get_info(self):
        q = self.data.qpos[self.q_indices]
        cur_state = q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], \
                    q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], \
                    q[20], q[21], q[22] 

        info = {
            "dt": self.dt_ * self.frame_skip,
            "action": self.action,
            "torque": self.applied_torques,
            "action_diff_RMS": np.linalg.norm(self.action - self.previous_action),
            "lin_vel_x": self.data.sensor('linear-velocity').data.astype(np.float32)[0],
            "lin_vel_y": self.data.sensor('linear-velocity').data.astype(np.float32)[1],
            "ang_vel_z": self.obs[48],
            "pos_z" : self.data.qpos[2],
            "ang_roll" :  self.obs[49],
            "ang_pitch":  self.obs[50],
            "set_points": self.action * self.action_scaler,
            "cur_state": cur_state
        }

        self.external_obs = self._get_external_obs()
        if self.external_obs is not None:
            info['scaled_base_lin_vel'] = self.external_obs['scaled_base_lin_vel']
            info['height_map'] = self.external_obs['height_map']

        return info

    def _get_reset_info(self):
        info = self._get_info()
        return info

    def _is_done(self):
        return False

    def reset_model(self):
        self.local_step = 0
        self.action = np.zeros(self.action_dim)
        self.previous_action = np.zeros(self.action_dim)
        self.control_manager.reset()
        self.applied_torques = np.zeros(self.action_dim)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos()
        self.data.qvel[:] = 0

        mujoco.mj_forward(self.model, self.data)

        self.obs, self.scaled_obs = self._get_obs()
        
        return self.scaled_obs

    def initial_qpos(self):
        qpos = np.zeros(self.model.nq)
        qpos[2] = 1.105
        qpos[3:7] = np.array([1, 0, 0, 0])
        qpos[7:30] = np.zeros(23)
        qpos[7:30] = uniform_noisy_data(qpos[7:30], lower=-self.init_noise, upper=self.init_noise)
        return qpos
    
    def event(self, event: str, value):
        if event == 'push':
            # Assume value is given in the robot frame (vx, vy, vz)
            # Convert this to world-frame velocity and assign to qvel[:3]
            raw_quat = self.data.qpos[3:7].astype(np.float64)           # [w, x, y, z]
            R = MathUtils.quat_to_rot_matrix(raw_quat)                  # Local-to-world rotation matrix (3×3)
            robot_vel = np.array(value, dtype=np.float64).reshape(3,)   # Velocity vector in robot frame
            world_vel = R.dot(robot_vel)                                # Transform to world-frame velocity
            self.data.qvel[:2] = world_vel[:2]  # xy: robot frame                        
            self.data.qvel[2] = value[2]        #  z: world frame
        else:
            raise NotImplementedError(f"event:{event} is not supported.")

    def get_data(self):
        return self.data

    def close(self):
        if self.viewer is not None:
            if glfw.get_current_context() == self.viewer.window:
                glfw.make_context_current(None)
            glfw.destroy_window(self.viewer.window)
            glfw.terminate()
            self.viewer = None
            print("Viewer closed")
        super().close()  # Call the parent class's close method to ensure everything is properly closed


