"""
PyBullet Gymnasium environment for a color-conditioned KUKA reaching task.

This module defines :class:`ColoredChestKukaEnv`, a custom Gymnasium
environment in which a fixed-base KUKA iiwa robot must move its end effector
toward the top center of a target colored chest placed on a table.

The environment is implemented for PyBullet and supports both headless
rendering (``render_mode="rgb_array"``) and GUI rendering
(``render_mode="human"``).
"""

import math
from typing import Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
import numpy as np
import pybullet as p
import pybullet_data


class ColoredChestKukaEnv(gym.Env):
    """
    Colab-safe Gymnasium environment for a KUKA reach task in PyBullet.

    Overview
    --------
    The task consists of controlling the Cartesian displacement of the KUKA
    end effector so that it reaches the top center of one target chest among
    three colored chests placed on a table. The target chest is selected at
    reset time and encoded directly in the observation as a one-hot target
    indicator.

    Task objective
    --------------
    At each episode, exactly three chests are spawned on the tabletop with
    fixed colors:

    - chest 0: red
    - chest 1: green
    - chest 2: blue

    One of these chests is designated as the target. The agent must move the
    robot end effector close to the top center of that chest and remain within
    the success radius for a configurable number of consecutive simulation
    steps.

    Action space
    ------------
    ``spaces.Box(low=-action_scale, high=action_scale, shape=(3,), dtype=float32)``

    The action is a 3D Cartesian delta applied to the current end-effector
    position:

    - ``action[0]``: delta on the world x-axis
    - ``action[1]``: delta on the world y-axis
    - ``action[2]``: delta on the world z-axis

    The action is clipped elementwise to the configured action bounds, then
    added to the current end-effector position. The resulting target position
    is also clipped to a hard-coded workspace box before inverse kinematics is
    solved.

    Observation space
    -----------------
    ``spaces.Box(low=-inf, high=inf, shape=(10,), dtype=float32)``

    The observation is a 10-dimensional vector with the following layout:

    1. end-effector x position
    2. end-effector y position
    3. end-effector z position
    4. target chest top-center x position
    5. target chest top-center y position
    6. target chest top-center z position
    7. one-hot flag for red target
    8. one-hot flag for green target
    9. one-hot flag for blue target
    10. normalized episode progress, computed as
        ``step_count / max(1, max_steps)``

    Reward types
    ------------
    Two reward formulations are supported through ``reward_type``.

    ``"basic"``
        The reward is the negative Euclidean distance between the current
        end-effector tip position and the top center of the target chest:

        ``reward = -distance_to_target``

    ``"advanced"``
        Starts from the same dense distance reward and adds two shaping terms:

        - a positive shaping bonus when the end effector is within a radius of
          0.20 m from the target, scaled linearly as the distance decreases
        - a penalty if the target chest has moved relative to its previously
          stored position

        In addition, both reward modes receive a success bonus of ``+20.0``
        once the success condition is satisfied.

    Success condition
    -----------------
    A step is counted as "close" when the distance to the target is strictly
    less than ``success_distance``. An episode is considered successful once
    the agent remains close for at least ``success_hold_steps`` consecutive
    steps.

    Episode termination and truncation
    ----------------------------------
    - ``terminated`` is ``True`` when the success condition is met.
    - ``truncated`` is ``True`` when ``step_count >= max_steps``.

    Rendering
    ---------
    Supported render modes are:

    - ``None``: no rendering
    - ``"rgb_array"``: off-screen rendering through PyBullet DIRECT mode
    - ``"human"``: interactive PyBullet GUI rendering for machines with a
      display server

    The :meth:`render` method returns an RGB NumPy array for enabled rendering
    modes and returns ``None`` when ``render_mode`` is ``None``.

    Reset behavior
    --------------
    On every reset, the simulation is rebuilt, the robot is returned to a fixed
    initial joint configuration, the three chests are respawned at non-
    overlapping positions on the table, and a target chest is chosen either
    randomly or from ``options["target_idx"]`` if provided.

    Key Colab adaptation
    --------------------
    This environment never requires an X window unless
    ``render_mode="human"``. In Colab, use:

    ``env = gym.make("ColoredChestKuka-v0", render_mode="rgb_array")``

    Then call:

    ``frame = env.render()``

    Notes
    -----
    The end-effector target orientation is fixed throughout the task, and the
    environment uses inverse kinematics to convert Cartesian motion targets
    into joint-space motor commands.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        num_chests: int = 3,
        reward_type: str = "advanced",
        render_mode: Optional[str] = None,
        max_steps: int = 150,
        action_scale: float = 0.05,
        success_distance: float = 0.06,
        success_hold_steps: int = 5,
        chest_scale: float = 0.12,
        use_plane: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the environment and build the initial PyBullet world.

        Parameters
        ----------
        num_chests:
            Number of chests to spawn. This implementation currently supports
            exactly 3 chests.
        reward_type:
            Reward formulation to use. Must be either ``"basic"`` or
            ``"advanced"``.
        render_mode:
            Rendering backend selection. Allowed values are ``None``,
            ``"human"``, and ``"rgb_array"``.
        max_steps:
            Maximum number of environment steps before the episode is truncated.
        action_scale:
            Per-axis absolute bound used to define the action space.
        success_distance:
            Distance threshold below which the end effector is considered close
            to the target chest.
        success_hold_steps:
            Number of consecutive close steps required to declare success.
        chest_scale:
            Global scale factor passed to the chest URDF objects.
        use_plane:
            Whether to load the default PyBullet plane into the scene.
        seed:
            Optional seed used to initialize the NumPy random generator used for
            chest placement and random target selection.

        Raises
        ------
        ValueError
            If ``num_chests`` is not 3, ``reward_type`` is invalid, or
            ``render_mode`` is unsupported.
        """
        super().__init__()

        if num_chests != 3:
            raise ValueError("This version currently supports exactly 3 chests.")
        if reward_type not in {"basic", "advanced"}:
            raise ValueError("reward_type must be 'basic' or 'advanced'.")
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("render_mode must be one of None, 'human', or 'rgb_array'.")

        self.num_chests = num_chests
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_scale = float(action_scale)
        self.success_distance = float(success_distance)
        self.success_hold_steps = int(success_hold_steps)
        self.chest_scale = float(chest_scale)
        self.use_plane = use_plane

        self.np_random = np.random.default_rng(seed)

        self.chest_colors = [
            [1.0, 0.0, 0.0, 1.0],  # red
            [0.0, 1.0, 0.0, 1.0],  # green
            [0.0, 0.0, 1.0, 1.0],  # blue
        ]

        self.table_position = [1.0, -0.2, 0.0]
        self.table_orientation = [0.0, 0.0, 0.7071, 0.7071]

        self.table_top_z = 0.65
        self.table_x_min = 0.72
        self.table_x_max = 0.98
        self.table_y_min = -0.58
        self.table_y_max = 0.18
        self.min_chest_separation = 0.18

        self.robot_base_position = [1.4, -0.2, 0.6]
        self.end_effector_index = 6
        self.ee_target_orientation = p.getQuaternionFromEuler([0.0, -math.pi, 0.0])
        self.tip_offset = np.array([0.0, 0.0, 0.01], dtype=np.float32)

        self.cam_target_pos = [0.95, -0.2, 0.2]
        self.cam_distance = 2.05
        self.cam_yaw = -50
        self.cam_pitch = -40
        self.cam_width = 640
        self.cam_height = 480

        self.physics_client_id: Optional[int] = None
        self.plane_id: Optional[int] = None
        self.table_id: Optional[int] = None
        self.kuka_id: Optional[int] = None
        self.chest_ids = []
        self.target_idx = 0
        self.step_count = 0
        self.consecutive_close_steps = 0
        self.prev_target_chest_pos = None

        self.action_space = spaces.Box(
            low=-self.action_scale,
            high=self.action_scale,
            shape=(3,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

        self._connect()
        self._build_world()

    def _connect(self) -> None:
        """
        Connect to the PyBullet physics server if no connection exists yet.

        The environment uses GUI mode only when ``render_mode == "human"``.
        Otherwise it uses DIRECT mode, which is suitable for headless execution.
        """
        if self.physics_client_id is not None:
            return

        if self.render_mode == "human":
            connection_mode = p.GUI
        else:
            connection_mode = p.DIRECT

        self.physics_client_id = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def close(self) -> None:
        """
        Disconnect from the active PyBullet physics server.

        This method is safe to call multiple times. After disconnection, the
        cached physics client identifier is reset to ``None``.
        """
        if self.physics_client_id is not None:
            try:
                p.disconnect(self.physics_client_id)
            finally:
                self.physics_client_id = None

    def _build_world(self) -> None:
        """
        Reset and rebuild the full simulation world.

        This method clears the current simulation, applies gravity, optionally
        loads a plane, loads the table and fixed-base KUKA arm, resets the arm
        to its initial joint state, samples non-overlapping chest positions, and
        spawns the three colored chest objects.
        """
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        if self.use_plane:
            self.plane_id = p.loadURDF("plane.urdf")

        self.table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=self.table_position,
            baseOrientation=self.table_orientation,
            useFixedBase=True,
        )

        self.kuka_id = p.loadURDF(
            "kuka_iiwa/model_vr_limits.urdf",
            basePosition=self.robot_base_position,
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
        )

        self.num_joints = p.getNumJoints(self.kuka_id)
        self._reset_robot_arm()

        self.chest_ids = []
        chest_positions = self._sample_non_overlapping_chest_positions(self.num_chests)

        for i, chest_position in enumerate(chest_positions):
            chest_id = p.loadURDF(
                "cube.urdf",
                basePosition=chest_position,
                globalScaling=self.chest_scale,
                useFixedBase=True,
            )
            p.changeVisualShape(chest_id, -1, rgbaColor=self.chest_colors[i])
            self.chest_ids.append(chest_id)

        for _ in range(20):
            p.stepSimulation()

    def _reset_robot_arm(self) -> None:
        """
        Reset the KUKA arm to a fixed initial joint configuration.

        The method both hard-resets the joint states and configures position
        controllers so the robot remains at the chosen pose when the simulation
        advances.
        """
        init_joint_positions = [0.0, 0.0, 0.0, 1.5708, 0.0, -1.0367, 0.0]

        for j in range(self.num_joints):
            pos = init_joint_positions[j]
            p.resetJointState(self.kuka_id, j, pos)
            p.setJointMotorControl2(
                self.kuka_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=pos,
                force=500,
            )

    def _sample_non_overlapping_chest_positions(self, num_positions: int):
        """
        Sample chest positions on the tabletop subject to a minimum separation.

        Parameters
        ----------
        num_positions:
            Number of chest positions to sample.

        Returns
        -------
        list
            A list of ``[x, y, z]`` positions suitable for loading chest bodies.

        Raises
        ------
        RuntimeError
            If valid non-overlapping positions cannot be found within the
            configured number of sampling attempts.
        """
        positions = []
        max_attempts = 200

        for _ in range(num_positions):
            placed = False
            for _ in range(max_attempts):
                x = float(self.np_random.uniform(self.table_x_min, self.table_x_max))
                y = float(self.np_random.uniform(self.table_y_min, self.table_y_max))
                candidate = np.array([x, y], dtype=np.float32)

                if all(
                    np.linalg.norm(candidate - np.array(pos[:2], dtype=np.float32))
                    >= self.min_chest_separation
                    for pos in positions
                ):
                    positions.append([x, y, self.table_top_z + 0.02])
                    placed = True
                    break

            if not placed:
                raise RuntimeError("Could not place all chests without overlap.")

        return positions

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment state and start a new episode.

        Parameters
        ----------
        seed:
            Optional reset-time seed. When provided, the environment's NumPy
            random generator is reinitialized with this value.
        options:
            Optional reset options dictionary. If it contains ``"target_idx"``,
            that chest index is used as the target instead of sampling one
            randomly.

        Returns
        -------
        tuple
            A pair ``(obs, info)`` where ``obs`` is the initial observation and
            ``info`` contains the selected target index and RGBA color.
        """
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._build_world()

        if options is not None and "target_idx" in options:
            self.target_idx = int(options["target_idx"])
        else:
            self.target_idx = int(self.np_random.integers(0, self.num_chests))

        self.step_count = 0
        self.consecutive_close_steps = 0
        self.prev_target_chest_pos = np.array(
            self._get_chest_top_center(self.chest_ids[self.target_idx]),
            dtype=np.float32,
        )

        return self._get_obs(), {
            "target_idx": self.target_idx,
            "target_color_rgba": self.chest_colors[self.target_idx],
        }

    def _get_end_effector_position(self) -> np.ndarray:
        """
        Return the current world-space position of the effective robot tip.

        The PyBullet link state is taken from ``end_effector_index`` and then
        adjusted by ``tip_offset`` expressed in the link frame so that the
        returned point better matches the intended tool-tip location.
        """
        link_state = p.getLinkState(
            self.kuka_id,
            self.end_effector_index,
            computeForwardKinematics=True,
        )
        link_position = np.array(link_state[0], dtype=np.float32)
        link_orientation = link_state[1]
        rotation_matrix = np.array(
            p.getMatrixFromQuaternion(link_orientation),
            dtype=np.float32,
        ).reshape(3, 3)
        return (link_position + rotation_matrix @ self.tip_offset).astype(np.float32)

    def _get_chest_top_center(self, chest_id: int) -> np.ndarray:
        """
        Compute the world-space top-center point of a chest body.

        Parameters
        ----------
        chest_id:
            PyBullet body identifier of the chest.

        Returns
        -------
        numpy.ndarray
            A ``float32`` vector of shape ``(3,)`` giving the top-center point.
            If visual shape data is unavailable, the chest base position is
            returned instead.
        """
        chest_pos, _ = p.getBasePositionAndOrientation(chest_id)
        chest_pos = np.array(chest_pos, dtype=np.float32)

        visual_data = p.getVisualShapeData(chest_id)
        if not visual_data:
            return chest_pos

        chest_height = float(visual_data[0][3][2])
        top_center = chest_pos.copy()
        top_center[2] += chest_height / 2.0
        return top_center.astype(np.float32)

    def _distance_to_target(self) -> float:
        """
        Compute the Euclidean distance from the end effector to the target chest.

        Returns
        -------
        float
            Distance between the effective tool-tip position and the target
            chest top center.
        """
        ee_pos = self._get_end_effector_position()
        target_pos = self._get_chest_top_center(self.chest_ids[self.target_idx])
        return float(np.linalg.norm(target_pos - ee_pos))

    def _get_obs(self) -> np.ndarray:
        """
        Build the current observation vector.

        Returns
        -------
        numpy.ndarray
            A 10-dimensional ``float32`` observation containing end-effector
            position, target position, one-hot target identity, and normalized
            episode progress.
        """
        ee_pos = self._get_end_effector_position()
        target_pos = self._get_chest_top_center(self.chest_ids[self.target_idx])

        target_one_hot = np.zeros(self.num_chests, dtype=np.float32)
        target_one_hot[self.target_idx] = 1.0

        progress = np.array([self.step_count / max(1, self.max_steps)], dtype=np.float32)

        return np.concatenate([ee_pos, target_pos, target_one_hot, progress]).astype(np.float32)

    def _compute_reward_and_success(self):
        """
        Compute the scalar reward, success flag, and current target distance.

        Returns
        -------
        tuple
            A tuple ``(reward, success, distance)`` where:

            - ``reward`` is the per-step scalar reward
            - ``success`` indicates whether the success hold condition is met
            - ``distance`` is the current Euclidean distance to the target
        """
        distance = self._distance_to_target()
        reward = -distance

        if self.reward_type == "advanced":
            shaping_radius = 0.20
            if distance <= shaping_radius:
                bonus_scale = 1.0 - (distance / shaping_radius)
                reward += 10.0 * bonus_scale

            current_target_pos = self._get_chest_top_center(self.chest_ids[self.target_idx])
            chest_move_dist = float(np.linalg.norm(current_target_pos - self.prev_target_chest_pos))
            if chest_move_dist > 1e-6:
                reward -= 10.0 * chest_move_dist
            self.prev_target_chest_pos = current_target_pos.copy()

        if distance < self.success_distance:
            self.consecutive_close_steps += 1
        else:
            self.consecutive_close_steps = 0

        success = self.consecutive_close_steps >= self.success_hold_steps
        if success:
            reward += 20.0

        return float(reward), bool(success), float(distance)

    def step(self, action: np.ndarray):
        """
        Advance the simulation by one environment step.

        Parameters
        ----------
        action:
            A 3D Cartesian delta action. It is reshaped to ``(3,)``, clipped to
            the action-space limits, converted into a target end-effector
            position, and executed through inverse kinematics plus PyBullet
            position control.

        Returns
        -------
        tuple
            A standard Gymnasium five-tuple
            ``(obs, reward, terminated, truncated, info)``.
        """
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        current_ee_pos = self._get_end_effector_position()
        target_ee_pos = current_ee_pos + action

        target_ee_pos[0] = np.clip(target_ee_pos[0], 0.65, 1.15)
        target_ee_pos[1] = np.clip(target_ee_pos[1], -0.75, 0.30)
        target_ee_pos[2] = np.clip(target_ee_pos[2], 0.65, 1.25)

        joint_poses = p.calculateInverseKinematics(
            self.kuka_id,
            self.end_effector_index,
            target_ee_pos,
            self.ee_target_orientation,
        )

        for j in range(self.num_joints):
            p.setJointMotorControl2(
                self.kuka_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[j],
                force=500,
            )

        p.stepSimulation()

        obs = self._get_obs()
        reward, success, distance = self._compute_reward_and_success()

        terminated = success
        truncated = self.step_count >= self.max_steps

        info = {
            "is_success": success,
            "distance_to_target": distance,
            "target_idx": self.target_idx,
            "target_color_rgba": self.chest_colors[self.target_idx],
            "consecutive_close_steps": self.consecutive_close_steps,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the current scene from the fixed task camera.

        Returns
        -------
        numpy.ndarray or None
            Returns an RGB array of shape ``(cam_height, cam_width, 3)`` when
            rendering is enabled, otherwise returns ``None``.
        """
        if self.render_mode is None:
            return None

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.cam_target_pos,
            distance=self.cam_distance,
            yaw=self.cam_yaw,
            pitch=self.cam_pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.cam_width) / self.cam_height,
            nearVal=0.1,
            farVal=100.0,
        )
        img = p.getCameraImage(
            width=self.cam_width,
            height=self.cam_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb_array = np.reshape(img[2], (self.cam_height, self.cam_width, 4))[:, :, :3]
        return rgb_array


ENV_ID = "ColoredChestKuka-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point=__name__ + ":ColoredChestKukaEnv",
        max_episode_steps=150,
    )
