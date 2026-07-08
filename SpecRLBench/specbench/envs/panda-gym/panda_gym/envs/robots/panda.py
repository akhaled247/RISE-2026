from typing import Optional

import numpy as np
from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet


class Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
        obs_use_ee_only: bool = True,
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        self.obs_use_ee_only = obs_use_ee_only
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )
        # link_details = self.get_link_details("panda")
        # print(link_details)
        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    # def get_link_details(self, body_name: str) -> list:
    #     """Get details of all links in a body.

    #     Args:
    #         body_name (str): The name of the body.

    #     Returns:
    #         list: A list of dictionaries containing link details.
    #     """
    #     body_id = self.sim._bodies_idx[body_name]
    #     num_joints = self.sim.physics_client.getNumJoints(body_id)
    #     link_details = []

    #     for link_index in range(num_joints):
    #         joint_info = self.sim.physics_client.getJointInfo(body_id, link_index)
    #         link_name = joint_info[12].decode("utf-8")  # Link name
    #         parent_index = joint_info[16]  # Parent link index
    #         joint_name = joint_info[1].decode("utf-8")  # Joint name
    #         joint_type = joint_info[2]  # Joint type
    #         link_details.append({
    #             "link_index": link_index,
    #             "link_name": link_name,
    #             "parent_index": parent_index,
    #             "joint_name": joint_name,
    #             "joint_type": joint_type,
    #         })

    #     return link_details

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    # def get_obs(self) -> np.ndarray:
    #     # joints position and velocity
    #     joint_positions = np.array([self.get_joint_angle(joint=i) for i in range(7)])
    #     joint_velocities = np.array([self.get_joint_velocity(joint=i) for i in range(7)])
    #     # end-effector position and velocity
    #     ee_position = np.array(self.get_ee_position())
    #     ee_velocity = np.array(self.get_ee_velocity())
    #     # concatenate end-effector position and velocity
    #     all_positions = np.concatenate((joint_positions, ee_position))
    #     all_velocities = np.concatenate((joint_velocities, ee_velocity))
    #     # fingers opening
    #     if not self.block_gripper:
    #         fingers_width = self.get_fingers_width()
    #         observation = np.concatenate((all_positions, all_velocities, [fingers_width]))
    #     else:
    #         observation = np.concatenate((all_positions, all_velocities))
    #     return observation

    def get_obs(self) -> np.ndarray:
        if self.obs_use_ee_only:
            # end-effector position and velocity
            ee_position = np.array(self.get_ee_position())
            ee_velocity = np.array(self.get_ee_velocity())
            # ee_orientation = np.array(self.get_ee_orientation())
            # fingers opening
            if not self.block_gripper:
                fingers_width = self.get_fingers_width()
                # observation = np.concatenate((ee_position, ee_orientation, ee_velocity, [fingers_width]))
                observation = np.concatenate((ee_position, ee_velocity, [fingers_width]))
            else:
                # observation = np.concatenate((ee_position, ee_orientation, ee_velocity))
                observation = np.concatenate((ee_position, ee_velocity))
        else:
            robot_link_index_list = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]
            observation = []
            for index in robot_link_index_list:
                position = np.array(self.sim.get_link_position("panda", index))
                orientation = np.array(self.sim.get_link_orientation("panda", index))
                # print(f"position: {position.shape}, orientation: {orientation.shape}")
                observation.append(position)
                observation.append(orientation)
            observation = np.concatenate(observation, axis=0)
            if not self.block_gripper:
                fingers_width = self.get_fingers_width()
                observation = np.concatenate((observation, [fingers_width]))
        return observation

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)
    
    def get_ee_orientation(self) -> np.ndarray:
        """Returns the orientation of the end-effector as (x, y, z, w)"""
        return self.get_link_orientation(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
