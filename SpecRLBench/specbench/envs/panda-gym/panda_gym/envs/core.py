from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from panda_gym.pybullet import PyBullet
from specbench.utils.ltl.logic import Assignment


class PyBulletRobot(ABC):
    """Base class for robot env.

    Args:
        sim (PyBullet): Simulation instance.
        body_name (str): The name of the robot within the simulation.
        file_name (str): Path of the urdf file.
        base_position (np.ndarray): Position of the base of the robot as (x, y, z).
    """

    def __init__(
        self,
        sim: PyBullet,
        body_name: str,
        file_name: str,
        base_position: np.ndarray,
        action_space: spaces.Space,
        joint_indices: np.ndarray,
        joint_forces: np.ndarray,
    ) -> None:
        self.sim = sim
        self.body_name = body_name
        with self.sim.no_rendering():
            self._load_robot(file_name, base_position)
            self.setup()
        self.action_space = action_space
        self.joint_indices = joint_indices
        self.joint_forces = joint_forces

    def _load_robot(self, file_name: str, base_position: np.ndarray) -> None:
        """Load the robot.

        Args:
            file_name (str): The URDF file name of the robot.
            base_position (np.ndarray): The position of the robot, as (x, y, z).
        """
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            useFixedBase=True,
            collision_group=1,
            collision_mask=2,
        )
        # # iterate over each link in the robot and print out their names
        # # do not fabricate new functions that does not exist
        # body_id = self.sim._bodies_idx[self.body_name]
        # num_joints = self.sim.physics_client.getNumJoints(body_id)
        # for link_index in range(num_joints):
        #     link_name = self.sim.physics_client.getJointInfo(body_id, link_index)[12].decode('utf-8')
        #     print(f"Link {link_index}: {link_name}")
        #     # print out the collisionshape detials of each link
        #     collision_shapes = self.sim.physics_client.getCollisionShapeData(body_id, link_index)
        #     print(f"Collision shapes for Link {link_index} ({link_name}): {collision_shapes}")

        #     # if the linke_name is panda_hand, then display the axis for that link in smulation.
        #     # if link_name == "panda_hand":
        #     #     link_position = self.sim.get_link_position(self.body_name, link_index)
        #     #     link_orientation = self.sim.get_link_orientation(self.body_name, link_index)
        #     #     import pybullet as p
        #     #     rot_matrix = p.getMatrixFromQuaternion(link_orientation)
        #     #     rot_matrix = [rot_matrix[i:i+3] for i in range(0, len(rot_matrix), 3)]
        #     #     axis_length = 0.1
        #     #     x_axis = [link_position[i] + axis_length * rot_matrix[i][0] for i in range(3)]
        #     #     y_axis = [link_position[i] + axis_length * rot_matrix[i][1] for i in range(3)]
        #     #     z_axis = [link_position[i] + axis_length * rot_matrix[i][2] for i in range(3)]
        #     #     p.addUserDebugLine(link_position, x_axis, [1, 0, 0], 2, 0.1)
        #     #     p.addUserDebugLine(link_position, y_axis, [0, 1, 0], 2, 0.1)
        #     #     p.addUserDebugLine(link_position, z_axis, [0, 0, 1], 2, 0.1)

    def setup(self) -> None:
        """Called after robot loading."""
        pass

    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Set the action. Must be called just before sim.step().

        Args:
            action (np.ndarray): The action.
        """

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the robot.

        Returns:
            np.ndarray: The observation.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the robot and return the observation."""

    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the position of a link as (x, y, z)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Position as (x, y, z)
        """
        return self.sim.get_link_position(self.body_name, link)

    def get_link_orientation(self, link: int) -> np.ndarray:
        """Returns the orientation of a link as (x, y, z, w)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Orientation as (x, y, z, w)
        """
        return self.sim.get_link_orientation(self.body_name, link)

    def get_link_velocity(self, link: int) -> np.ndarray:
        """Returns the velocity of a link as (vx, vy, vz)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Velocity as (vx, vy, vz)
        """
        return self.sim.get_link_velocity(self.body_name, link)

    def get_joint_angle(self, joint: int) -> float:
        """Returns the angle of a joint

        Args:
            joint (int): The joint index.

        Returns:
            float: Joint angle
        """
        return self.sim.get_joint_angle(self.body_name, joint)

    def get_joint_velocity(self, joint: int) -> float:
        """Returns the velocity of a joint as (wx, wy, wz)

        Args:
            joint (int): The joint index.

        Returns:
            np.ndarray: Joint velocity as (wx, wy, wz)
        """
        return self.sim.get_joint_velocity(self.body_name, joint)

    def control_joints(self, target_angles: np.ndarray) -> None:
        """Control the joints of the robot.

        Args:
            target_angles (np.ndarray): The target angles. The length of the array must equal to the number of joints.
        """
        self.sim.control_joints(
            body=self.body_name,
            joints=self.joint_indices,
            target_angles=target_angles,
            forces=self.joint_forces,
        )

    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(self.body_name, joints=self.joint_indices, angles=angles)

    def inverse_kinematics(self, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint values.

        Args:
            link (int): The link.
            position (x, y, z): Desired position of the link.
            orientation (x, y, z, w): Desired orientation of the link.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(self.body_name, link=link, position=position, orientation=orientation)
        return inverse_kinematics


class Task(ABC):
    """Base class for tasks.
    Args:
        sim (PyBullet): Simulation instance.
    """

    COLORS = {
        # "blue": np.array([0, 0, 1, 0.5]),
        # "green": np.array([0, 1, 0, 0.5]),
        # "red": np.array([1, 0, 0, 0.5]),
        # "yellow": np.array([1, 1, 0, 0.5]),

        ### the a shold be 1.0 otherwise they will be ignored when generating the depth image.
        "blue": np.array([0, 0, 1, 1.0]),
        "green": np.array([0, 1, 0, 1.0]),
        "red": np.array([1, 0, 0, 1.0]),
        "magenta": np.array([1, 0, 1, 1.0]),
        "yellow": np.array([1, 1, 0, 1.0]),
    }

    def __init__(self, sim: PyBullet) -> None:
        self.sim = sim
        self.goal = None

    @abstractmethod
    def reset(self) -> None:
        """Reset the task: sample a new goal."""

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the task."""

    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        """Return the achieved goal."""

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        if self.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.goal.copy()

    @abstractmethod
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """Returns whether the achieved goal match the desired goal."""

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """Compute reward associated to the achieved and the desired goal."""


class RobotTaskEnv(gym.Env):
    """Robotic task goal env, as the junction of a task and a robot.

    Args:
        robot (PyBulletRobot): The robot.
        task (Task): The task.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        robot: PyBulletRobot,
        task: Task,
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robot.sim
        self.render_mode = self.sim.render_mode
        self.metadata["render_fps"] = 1 / self.sim.dt
        self.robot = robot
        self.task = task
        observation, _ = self.reset()  # required for init; seed can be changed later
        observation_shape = observation["observation"].shape
        achieved_goal_shape = observation["achieved_goal"].shape
        desired_goal_shape = observation["desired_goal"].shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                desired_goal=spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
                achieved_goal=spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
            )
        )
        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward
        self._saved_goal = dict()  # For state saving and restoring

        self.render_width = render_width
        self.render_height = render_height
        self.render_target_position = (
            render_target_position if render_target_position is not None else np.array([0.0, 0.0, 0.0])
        )
        self.render_distance = render_distance
        self.render_yaw = render_yaw
        self.render_pitch = render_pitch
        self.render_roll = render_roll
        with self.sim.no_rendering():
            self.sim.place_visualizer(
                target_position=self.render_target_position,
                distance=self.render_distance,
                yaw=self.render_yaw,
                pitch=self.render_pitch,
            )

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs().astype(np.float32)  # robot state
        task_obs = self.task.get_obs().astype(np.float32)  # object position, velocity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal().astype(np.float32),
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.task.np_random = self.np_random
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        observation = self._get_obs()
        info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal())}
        return observation, info

    def save_state(self) -> int:
        """Save the current state of the environment. Restore with `restore_state`.

        Returns:
            int: State unique identifier.
        """
        state_id = self.sim.save_state()
        self._saved_goal[state_id] = self.task.goal
        return state_id

    def restore_state(self, state_id: int) -> None:
        """Restore the state associated with the unique identifier.

        Args:
            state_id (int): State unique identifier.
        """
        self.sim.restore_state(state_id)
        self.task.goal = self._saved_goal[state_id]

    def remove_state(self, state_id: int) -> None:
        """Remove a saved state.

        Args:
            state_id (int): State unique identifier.
        """
        self._saved_goal.pop(state_id)
        self.sim.remove_state(state_id)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()
        # An episode is terminated iff the agent has reached the target
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        truncated = False
        info = {"is_success": terminated}
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info))
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.sim.close()

    def render(self) -> Optional[np.ndarray]:
        """Render.

        If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        return self.sim.render(
            width=self.render_width,
            height=self.render_height,
            target_position=self.render_target_position,
            distance=self.render_distance,
            yaw=self.render_yaw,
            pitch=self.render_pitch,
            roll=self.render_roll,
        )


# TODO: step function and observation function, the observation should contain the closest distance of each link to each AP, and the ego-state of the agent
class RobotLTLTaskEnv(gym.Env):
    """Robotic task goal env, as the junction of a task and a robot.

    Args:
        robot (PyBulletRobot): The robot.
        task (Task): The task.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        robot: PyBulletRobot,
        task: Task,
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robot.sim
        self.render_mode = self.sim.render_mode
        self.metadata["render_fps"] = 1 / self.sim.dt
        self.robot = robot
        self.task = task
        observation, _ = self.reset()  # required for init; seed can be changed later
        observation_shape = observation["ego_state"].shape
        # achieved_goal_shape = observation["achieved_goal"].shape
        # desired_goal_shape = observation["desired_goal"].shape
        self.observation_space = spaces.Dict(
            dict(
                ego_state=spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                # desired_goal=spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
                # achieved_goal=spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
            )
        )
        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward
        self._saved_goal = dict()  # For state saving and restoring

        self.render_width = render_width
        self.render_height = render_height
        self.render_target_position = (
            render_target_position if render_target_position is not None else np.array([0.0, 0.0, 0.0])
        )
        self.render_distance = render_distance
        self.render_yaw = render_yaw
        self.render_pitch = render_pitch
        self.render_roll = render_roll
        with self.sim.no_rendering():
            self.sim.place_visualizer(
                target_position=self.render_target_position,
                distance=self.render_distance,
                yaw=self.render_yaw,
                pitch=self.render_pitch,
            )

    def _get_obs(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        robot_obs = self.robot.get_obs().astype(np.float32)  # robot state, joints position and velocity
        # task_obs = self.task.get_obs().astype(np.float32)  # object position, velocity, etc...
        task_obs, info = self._get_distance_obs_and_info()
        # observation = np.concatenate([robot_obs, task_obs])
        # achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        obs = {"ego_state": robot_obs, 
            #    "ee_position": self.robot.get_ee_position(),
            #    "target_position": self.sim.get_base_position_by_name("object_green_0")
               }
        

        if self.task.observe_vision:
            left_rgb, left_depth = self.sim.get_rgbd_camera(
                width=256,
                height=256,
                target_position=np.array([0.0, -0.1, 0.3]),
                distance=0.9,
                yaw=40,
                pitch=-10,
                roll=0,
            )
            right_rgb, right_depth = self.sim.get_rgbd_camera(
                width=256,
                height=256,
                target_position=np.array([0.0, 0.1, 0.3]),
                distance=0.9,
                yaw=140,
                pitch=-10,
                roll=0,
            )
            task_obs = {"left_rgb": left_rgb, "left_depth": left_depth,
                        "right_rgb": right_rgb, "right_depth": right_depth}
        
        obs.update(task_obs)
        """ This view is limited in range and if the end effector grabs and object, it will block most of the view
        # TODO: tune the view of the RGBD camera
        # Attach RGBD camera to end effector using logic similar to PyBullet.render
        ee_pos = self.robot.get_ee_position()
        ee_ori = self.robot.get_ee_orientation()  # quaternion (x, y, z, w)

        import pybullet as p
        rot_matrix = np.array(p.getMatrixFromQuaternion(ee_ori)).reshape(3, 3)
        # print(rot_matrix)

        # visualize the axis of the end effector
        axis_length = 0.1
        x_axis = rot_matrix[:, 0]
        y_axis = rot_matrix[:, 1]
        z_axis = rot_matrix[:, 2]
        p.addUserDebugLine(ee_pos, ee_pos + axis_length * x_axis, [1, 0, 0], 2, 0.1)
        p.addUserDebugLine(ee_pos, ee_pos + axis_length * y_axis, [0, 1, 0], 2, 0.1)
        p.addUserDebugLine(ee_pos, ee_pos + axis_length * z_axis, [0, 0, 1], 2, 0.1)

        import pybullet as p
        rot_matrix = np.array(p.getMatrixFromQuaternion(ee_ori)).reshape(3, 3)
        # print(rot_matrix)
        # x, y, z: 0, 1, 2
        ### DO NOT REMOVE THIS COMMENT: in the ee's local frame the, 
        # x axis is pointing upward, y axis is pointing to the right, z is pointing forward
        forward_vector = rot_matrix[:, 2]  # z is forward
        up_vector = rot_matrix[:, 0]       # x is up
        camera_eye = ee_pos + 0.02 * up_vector
        camera_target = ee_pos + 0.5* forward_vector# + 0.04 * up_vector + 0.04 * forward_vector  # look forward from above
        view_matrix = self.sim.physics_client.computeViewMatrix(
            cameraEyePosition=camera_eye.tolist(),
            cameraTargetPosition=camera_target.tolist(),
            cameraUpVector=up_vector.tolist(),
        )
        proj_matrix = self.sim.physics_client.computeProjectionMatrixFOV(
            fov=150, aspect=1.0, 
            # nearVal=0.1, farVal=100.0
            nearVal=0.001, farVal=10.0
        )
        width, height = 256, 256
        _, _, rgba, depth, _ = self.sim.physics_client.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            shadow=True,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))[..., :3]
        depth = np.array(depth).reshape((height, width))
        # print(rgb)
        info["rgbd"] = (rgb, depth)
        obs.update({"rgb": rgb, "depth": depth})
        """

        # print(obs)
        return obs, info

    def get_propositions(self):
        """
        Get the list of color propositions for the task.
        """
        return self.task.colors
    
    def get_possible_assignments(self) -> list[Assignment]:
        return Assignment.zero_or_one_propositions(set(self.get_propositions()))

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.task.np_random = self.np_random
        success = False
        with self.sim.no_rendering():
            for i in range(10000):
                # print(f"Resetting environment, attempt {i+1}/1000")
                self.robot.reset()
                self.task.reset()
                if not self._check_collision_and_distance():
                    success = True
                    break
        if not success:
            # print("Failed to reset environment after 10000 attempts.")
            raise RuntimeError("Failed to reset environment without collision after 10000 attempts.")
        observation, info = self._get_obs()
        # info = {"propositions": self._get_active_propositions()}
        # info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal())}
        return observation, info

    def _check_collision_and_distance(self) -> bool:
        """Check if the robotic arm's body collides with any layout regions or violates the minimum keepout distance.

        Returns:
            bool: True if there is a collision or the distance is less than the keepout threshold, False otherwise.
        """
        # robot_id = self.sim._bodies_idx[self.robot.body_name]

        # # 0-6, 8 (hand) are robotic arm and 9,10 are the grippers
        # robot_link_index_list = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10] # len = 10

        # for color in self.task.colors:
        #     layout_ids = [self.sim._bodies_idx[f"{color}_{n}"] for n in range(self.task.region_num)]
            
        #     for i, robot_link_index in enumerate(robot_link_index_list):
                
        #         for layout_id in layout_ids:
        #             closest_points = self.sim.physics_client.getClosestPoints(
        #                 bodyA=robot_id, bodyB=layout_id, linkIndexA=robot_link_index, distance=10.0)
        #             assert len(closest_points) == 1, f"Expected ONLY 1 closest point, got {len(closest_points)}"
        #             closest_point = closest_points[0]
        #             dist = closest_point[8]
        #             if dist <= self.task.keepout:
        #                 return True
        # return False

        robot_id = self.sim._bodies_idx[self.robot.body_name]  # Robot ID
        layout_ids = [self.sim._bodies_idx[layout_name] for layout_name in self.task.region_names]  # Layout IDs

        # Note that link7 and link11 (end effector) do not have a collision shape.
        for link_index in range(-1, 12):  # Iterate over all links of the robotic arm, -1 is the base
            for layout_id in layout_ids:
                # # Check for collisions using getContactPoints
                # contact_points = self.sim.physics_client.getContactPoints(
                #     bodyA=robot_id, bodyB=layout_id, linkIndexA=link_index
                # )
                # if len(contact_points) > 0:
                #     # print(f"Collision detected between robot link {link_index} and layout {layout_id}")
                #     return True  # Collision detected

                # Check for minimum distance using getClosestPoints
                closest_points = self.sim.physics_client.getClosestPoints(
                    bodyA=robot_id, bodyB=layout_id, linkIndexA=link_index, distance=10.0
                )

                if closest_points:
                    for point in closest_points:
                        # print(
                        #     f"Closest point info: Robot link {link_index}, Layout {layout_id}, Distance: {point[8]}"
                        # )
                        if point[8] <= self.task.keepout:
                            # print(f"Keepout violation detected: Distance {point[8]} <= Keepout {self.task.keepout}")
                            return True  # Distance is too small
                else:
                    # get link position and the layout position first.
                    link_position = self.sim.get_link_position(self.robot.body_name, link_index)
                    layout_position = self.sim.get_base_position_by_id(layout_id)
                    if np.linalg.norm(link_position - layout_position) <= self.task.keepout + self.task.region_radius:
                        return True  # Keepout violation detected

                # else:
                #     print(f"No closest points found between robot link {link_index} and layout {layout_id}")

            # # Log details of link 7
            # if link_index == 7:
            #     link_position = self.sim.get_link_position(self.robot.body_name, link_index)
            #     link_orientation = self.sim.get_link_orientation(self.robot.body_name, link_index)
            #     print(f"Link 7 Position: {link_position}, Orientation: {link_orientation}")

            #     # Log collision shapes of link 7
            #     collision_shapes = self.sim.physics_client.getCollisionShapeData(robot_id, link_index)
            #     print(f"Collision shapes for link 7: {collision_shapes}")

            # if link_index == 11:
            #     collision_shapes = self.sim.physics_client.getCollisionShapeData(robot_id, link_index)
            #     print(f"Collision shapes for link 11: {collision_shapes}")

        # # Log positions of layouts
        # for layout_name in self.task.region_names:
        #     layout_position = self.sim.get_base_position(layout_name)
        #     layout_orientation = self.sim.get_base_orientation(layout_name)
        #     print(f"Layout {layout_name} Position: {layout_position}, Orientation: {layout_orientation}")

        # # Log collision shapes of layouts
        # for layout_id in layout_ids:
        #     layout_collision_shapes = self.sim.physics_client.getCollisionShapeData(layout_id, -1)
        #     print(f"Collision shapes for layout ID {layout_id}: {layout_collision_shapes}")

        return False

    # def _get_active_propositions(self) -> list[str]:
    #     return [f"dist_to_{color}" for color in self.task.colors]

    # def _get_distance_obs_and_info(self) -> dict:
    #     robot_id = self.sim._bodies_idx[self.robot.body_name]  # Robot ID
    #     distances = {}; info = {"propositions": []}

    #     # Iterate over APs, i.e., colors and compute distances
    #     for color in self.task.colors:
    #         closest_distances = []
    #         layout_ids = [self.sim._bodies_idx[f"{color}_{n}"] for n in range(self.task.region_num)]
    #         for robot_link_index in range(12):
    #             dist = []
    #             if robot_link_index in {7, 11}: 
    #                 link_position = self.sim.get_link_position(self.robot.body_name, robot_link_index)
    #                 dist = [np.linalg.norm(link_position - self.sim.get_base_position_by_id(layout_id))
    #                         for layout_id in layout_ids]
    #                 if any(d <= self.task.region_radius for d in dist):
    #                     print(f"Link {robot_link_index} is within region {color}")
    #                     info["propositions"].append(color)
    #             else:
    #                 for layout_id in layout_ids:
    #                     closest_points = self.sim.physics_client.getClosestPoints(
    #                         bodyA=robot_id, bodyB=layout_id, linkIndexA=robot_link_index, distance=10.0)
    #                     assert len(closest_points) > 0
    #                     dist.extend([point[8] for point in closest_points])
    #                 if any(d <= 0 for d in dist):
    #                     print(f"Link {robot_link_index} is within region {color}")
    #                     info["propositions"].append(color)
    #             closest_distances.append(np.min(dist))
    #         closest_distances = np.array(closest_distances, dtype=np.float32)
    #         if self.task.max_dist is None:
    #             closest_distances = np.exp(-self.task.exp_gain * closest_distances)
    #         else:
    #             closest_distances = np.maximum(0, self.task.max_dist - closest_distances) / self.task.max_dist
    #         distances[f"dist_to_{color}"] = closest_distances
    #     # remove duplicate element in "propositions"
    #     info["propositions"] = list(set(info["propositions"]))
    #     # print(distances)
    #     return distances, info
    
    # def _get_distance_obs_and_info(self) -> dict:
    #     """
    #     Get distance observations between the end effector and the regions and propositions information.
    #     """
    #     robot_id = self.sim._bodies_idx[self.robot.body_name]  # Robot ID
    #     distances = {}; info = {"propositions": []}

    #     # Iterate over APs, i.e., colors and compute distances
    #     for color in self.task.colors:
    #         layout_ids = [self.sim._bodies_idx[f"{color}_{n}"] for n in range(self.task.region_num)]
    #         ee_position = self.robot.get_ee_position()
                
    #         dist = [np.linalg.norm(ee_position - self.sim.get_base_position_by_id(layout_id))
    #                 for layout_id in layout_ids]
    #         if any(d <= self.task.region_radius for d in dist):
    #             # print(f"Robot's end effector is within region {color}")
    #             info["propositions"].append(color)

    #         closest_distance = np.min(dist).reshape(1,).astype(np.float32) - self.task.region_radius
    #         if self.task.max_dist is None:
    #             closest_distance = np.exp(-self.task.exp_gain * closest_distance)
    #         else:
    #             closest_distance = np.maximum(0, self.task.max_dist - closest_distance) / self.task.max_dist
    #         distances[f"dist_to_{color}"] = closest_distance
    #     # remove duplicate element in "propositions"
    #     info["propositions"] = list(set(info["propositions"]))
    #     # print(distances)
    #     return distances, info

    def _get_distance_obs_and_info(self) -> dict:
        """
        Get distance observations between the end effector and the regions and propositions information.
        """
        raise NotImplementedError

    def save_state(self) -> int:
        """Save the current state of the environment. Restore with `restore_state`.

        Returns:
            int: State unique identifier.
        """
        state_id = self.sim.save_state()
        self._saved_goal[state_id] = self.task.goal
        return state_id

    def restore_state(self, state_id: int) -> None:
        """Restore the state associated with the unique identifier.

        Args:
            state_id (int): State unique identifier.
        """
        self.sim.restore_state(state_id)
        self.task.goal = self._saved_goal[state_id]

    def remove_state(self, state_id: int) -> None:
        """Remove a saved state.

        Args:
            state_id (int): State unique identifier.
        """
        self._saved_goal.pop(state_id)
        self.sim.remove_state(state_id)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        observation, info = self._get_obs()
        # An episode is terminated iff the agent has reached the target
        # terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        terminated = False
        truncated = False
        # info = {"is_success": terminated}
        # reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info))
        reward = 0.0
        # print(observation, info)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.sim.close()

    def render(self) -> Optional[np.ndarray]:
        """Render.

        If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        return self.sim.render(
            width=self.render_width,
            height=self.render_height,
            target_position=self.render_target_position,
            distance=self.render_distance,
            yaw=self.render_yaw,
            pitch=self.render_pitch,
            roll=self.render_roll,
        )
