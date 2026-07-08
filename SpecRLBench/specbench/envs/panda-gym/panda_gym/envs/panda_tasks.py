from typing import Optional

import numpy as np

from panda_gym.envs.core import RobotTaskEnv, RobotLTLTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.flip import Flip
from panda_gym.envs.tasks.pick_and_place import PickAndPlace
from panda_gym.envs.tasks.LTL_pick_and_place import LTLPickAndPlace
from panda_gym.envs.tasks.push import Push
from panda_gym.envs.tasks.reach import Reach
from panda_gym.envs.tasks.LTL_reach import LTLReach
from panda_gym.envs.tasks.slide import Slide
from panda_gym.envs.tasks.stack import Stack
from panda_gym.pybullet import PyBullet


class PandaFlipEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.

    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Flip(sim, reward_type=reward_type)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )


class PandaPickAndPlaceEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlace(sim, reward_type=reward_type)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )


class PandaLTLPickAndPlaceEnv(RobotLTLTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        partial_observability: bool = False,
        observe_vision: bool = False,
        obs_use_ee_only: bool = True,
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type, obs_use_ee_only=obs_use_ee_only)
        task = LTLPickAndPlace(sim, reward_type=reward_type,
                               partial_observability=partial_observability, observe_vision=observe_vision, obs_use_ee_only=obs_use_ee_only)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

    def _get_distance_obs_and_info(self) -> dict:
        """
        Get distance observations between the end effector and the regions and propositions information.
        """
        obs = {}; info = {"propositions": set(), "touch": set()}
        ee_position = self.robot.get_ee_position()

        # Iterate over APs, i.e., colors and compute distances
        for item_type in ["object", "region"]:
            for color in self.task.colors:
                item_name = f"{item_type}_{color}"
                layout_ids = [self.sim._bodies_idx[name] for name in self.task.region_names if item_name in name]

                ray = [self.sim.get_base_position_by_id(layout_id) - ee_position for layout_id in layout_ids]
                dist = [np.linalg.norm(r) for r in ray]
                dist_threshold = self.task.object_size * np.sqrt(2)/2 if item_type == "object" else self.task.region_radius

                # update active propositions such as pick_<object>
                if item_type == "object":
                    # if any(d <= dist_threshold for d in dist) and self._is_grasp(layout_ids):
                    if self._is_grasp(layout_ids):
                        # print(f"Robot's end effector is within object {color}")
                        info["propositions"].add(f"pick_{color}")
                
                # check if the ee is close enough to the object/region
                if any(d <= dist_threshold for d in dist):
                    info["touch"].add(item_name)

                # update observations
                ray_idx = np.argmin(dist)
                min_dist = np.min(dist).reshape(1,).astype(np.float32) - dist_threshold
                min_dist = np.maximum(0, min_dist)  # if the ee is inside the region, then distance is zeros.
                if self.task.max_dist is None:
                    # if the ee is outside of the region, then normalize the direction, otherwise dir is zeros since the region is reached
                    if min_dist > 0:
                        obs_dir = ray[ray_idx] / np.linalg.norm(ray[ray_idx])  
                    else:
                        obs_dir = np.zeros_like(ray[ray_idx])
                    obs_dist = np.exp(-self.task.exp_gain * min_dist)
                    
                else:
                    # if the ee is outside of the region, then normalize the direction, otherwise dir is zeros since the region is reached
                    # additionally, if the distance is beyond the max_dist, then direction is also zeros since it is not observable.
                    if min_dist > 0 and min_dist <= self.task.max_dist:
                        obs_dir = ray[ray_idx] / np.linalg.norm(ray[ray_idx])
                    else:
                        obs_dir = np.zeros_like(ray[ray_idx])
                    obs_dist = np.maximum(0, self.task.max_dist - min_dist) / self.task.max_dist
                obs[f"dist_to_{item_name}"] = np.concatenate([obs_dir, obs_dist])

        # update active propositions such as place_<object>_<region>
        # iterate over objects and items and check if any object is within 
        # the range of regions and its velocity is close to 0
        object_names, object_pos, object_vel = [], [], []
        region_names, region_pos = [], []
        for color in self.task.colors:
            for i in range(self.task.object_num):
                object_name = f"object_{color}_{i}"
                object_names.append(object_name)
                object_pos.append(self.sim.get_base_position_by_name(object_name))
                vel = self.sim.get_base_velocity(object_name)
                vel = np.linalg.norm(vel)
                object_vel.append(vel)
            for i in range(self.task.region_num):
                region_name = f"region_{color}_{i}"
                region_names.append(region_name)
                region_pos.append(self.sim.get_base_position_by_name(region_name))
        
        obj_pos = np.asarray(object_pos, dtype=float)
        reg_pos = np.asarray(region_pos, dtype=float)
        N, M = len(obj_pos), len(reg_pos)

        # Extract just the color token
        obj_colors = np.array([n.split('_')[1] for n in object_names])
        reg_colors = np.array([n.split('_')[1] for n in region_names])

        dists = np.linalg.norm(obj_pos[:, None, :] - reg_pos[None, :, :], axis=2)
        dist_mask = dists < self.task.region_radius + self.task.object_size / 2

        if np.any(dist_mask):
            obj_speed = np.asarray(object_vel, dtype=float)
            slow_obj = obj_speed < 1e-3
            mask_ds = dist_mask & slow_obj[:, None]

            if np.any(mask_ds):
                z_mask = (obj_pos[:, 2][:, None] <= self.task.object_size / 2 + 1e-2)
                final_mask = mask_ds & z_mask
                
                if np.any(final_mask):
                    i_idx, j_idx = np.nonzero(final_mask)
                    props = {f"place_{obj_colors[i]}_{reg_colors[j]}" for i, j in zip(i_idx, j_idx)}
                    info["propositions"].update(props)

        return obs, info

    def _is_grasp(self, object_ids: list, force_threshold=5.0) -> bool:
        """
        Check if the robot's end effector is grasping any of the objects with the given IDs.

        Args:
            object_ids (list): List of object IDs to check for grasping.

        Returns:
            bool: True if the end effector is grasping any of the objects, False otherwise.
        """
        robot_id = self.sim._bodies_idx[self.robot.body_name]  # Robot ID

        for object_id in object_ids:
            # contact_points_left_finger
            cp_left = self.sim.physics_client.getContactPoints(
                bodyA=robot_id, bodyB=object_id, linkIndexA=self.robot.fingers_indices[0]
            )
            # contact_points_right_finger
            cp_right = self.sim.physics_client.getContactPoints(
                bodyA=robot_id, bodyB=object_id, linkIndexA=self.robot.fingers_indices[1]
            )

            # both fingers should contact the object
            if not cp_left or not cp_right:
                continue

            # check normal forces
            fn = sum(c[9] for c in cp_left) + sum(c[9] for c in cp_right)
            if fn <= force_threshold:
                continue

            # check if contact points are on opposite sides
            left_normals = [np.array(c[7]) for c in cp_left]  # Contact normals for left finger
            right_normals = [np.array(c[7]) for c in cp_right]  # Contact normals for right finger

            # Calculate the average normal for both fingers
            avg_left_normal = np.mean(left_normals, axis=0)
            avg_right_normal = np.mean(right_normals, axis=0)

            # Ensure the average normals are roughly in opposite directions
            dot_product = np.dot(avg_left_normal, avg_right_normal)
            # print(f"dot_product = {dot_product}")
            if dot_product < -0.8:  # Threshold for opposite direction (cosine similarity close to -1)
                return True

        return False


class PandaPushEnv(RobotTaskEnv):
    """Push task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Push(sim, reward_type=reward_type)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )


class PandaReachEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Reach(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )


class PandaLTLReachEnv(RobotLTLTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        partial_observability: bool = False,
        observe_vision: bool = False,
        obs_use_ee_only: bool = True,
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        print(f"partial_observability: {partial_observability}, control_type: {'grippers-only' if obs_use_ee_only else 'grippers-arm'}")
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type, obs_use_ee_only=obs_use_ee_only)
        task = LTLReach(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position, 
                        partial_observability=partial_observability, observe_vision=observe_vision, obs_use_ee_only=obs_use_ee_only)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

    # def _get_distance_obs_and_info(self) -> dict:
    #     """
    #     Get distance observations between the end effector and the regions and propositions information.
    #     """
    #     obs = {}; info = {"propositions": set()}
    #     ee_position = self.robot.get_ee_position()

    #     # Iterate over APs, i.e., colors and compute distances
    #     for color in self.task.colors:
    #         layout_ids = [self.sim._bodies_idx[name] for name in self.task.region_names if color in name]

    #         ray = [self.sim.get_base_position_by_id(layout_id) - ee_position for layout_id in layout_ids]
    #         dist = [np.linalg.norm(r) for r in ray]
    #         # dist = [np.linalg.norm(ee_position - self.sim.get_base_position_by_id(layout_id))
    #         #         for layout_id in layout_ids]
    #         if any(d <= self.task.region_radius for d in dist):
    #             # print(f"Robot's end effector is within region {color}")
    #             info["propositions"].add(color)

    #         ray_idx = np.argmin(dist)
    #         min_dist = np.min(dist).reshape(1,).astype(np.float32) - self.task.region_radius
    #         min_dist = np.maximum(0, min_dist)  # if the ee is inside the region, then distance is zeros.
    #         if self.task.max_dist is None:
    #             # if the ee is outside of the region, then normalize the direction, otherwise dir is zeros since the region is reached
    #             if min_dist > 0:
    #                 obs_dir = ray[ray_idx] / np.linalg.norm(ray[ray_idx])  
    #             else:
    #                 obs_dir = np.zeros_like(ray[ray_idx])
    #             obs_dist = np.exp(-self.task.exp_gain * min_dist)
                
    #         else:
    #             # if the ee is outside of the region, then normalize the direction, otherwise dir is zeros since the region is reached
    #             # additionally, if the distance is beyond the max_dist, then direction is also zeros since it is not observable.
    #             if min_dist > 0 and min_dist <= self.task.max_dist:
    #                 obs_dir = ray[ray_idx] / np.linalg.norm(ray[ray_idx])
    #             else:
    #                 obs_dir = np.zeros_like(ray[ray_idx])
    #             obs_dist = np.maximum(0, self.task.max_dist - min_dist) / self.task.max_dist
    #         obs[f"dist_to_{color}"] = np.concatenate([obs_dir, obs_dist])

    #     return obs, info

    def _get_distance_obs_and_info(self) -> dict:
        obs = {}; info = {"propositions": set()}
        robot_id = self.sim._bodies_idx[self.robot.body_name]

        # the observation only contains distances between the end-effector and the regions
        if self.task.obs_use_ee_only:
            ee_position = self.robot.get_ee_position()
            # Iterate over APs, i.e., colors and compute distances
            for color in self.task.colors:
                layout_ids = [self.sim._bodies_idx[name] for name in self.task.region_names if color in name]

                ray = [self.sim.get_base_position_by_id(layout_id) - ee_position for layout_id in layout_ids]
                dist = [np.linalg.norm(r) for r in ray]
                # dist = [np.linalg.norm(ee_position - self.sim.get_base_position_by_id(layout_id))
                #         for layout_id in layout_ids]
                if any(d <= self.task.region_radius for d in dist):
                    # print(f"Robot's end effector is within region {color}")
                    info["propositions"].add(color)

                ray_idx = np.argmin(dist)
                min_dist = np.min(dist).reshape(1,).astype(np.float32) - self.task.region_radius
                min_dist = np.maximum(0, min_dist)  # if the ee is inside the region, then distance is zeros.
                if self.task.max_dist is None:
                    # if the ee is outside of the region, then normalize the direction, otherwise dir is zeros since the region is reached
                    if min_dist > 0:
                        obs_dir = ray[ray_idx] / np.linalg.norm(ray[ray_idx])  
                    else:
                        obs_dir = np.zeros_like(ray[ray_idx])
                    obs_dist = np.exp(-self.task.exp_gain * min_dist)
                    
                else:
                    # if the ee is outside of the region, then normalize the direction, otherwise dir is zeros since the region is reached
                    # additionally, if the distance is beyond the max_dist, then direction is also zeros since it is not observable.
                    if min_dist > 0 and min_dist <= self.task.max_dist:
                        obs_dir = ray[ray_idx] / np.linalg.norm(ray[ray_idx])
                    else:
                        obs_dir = np.zeros_like(ray[ray_idx])
                    obs_dist = np.maximum(0, self.task.max_dist - min_dist) / self.task.max_dist
                obs[f"dist_to_{color}"] = np.concatenate([obs_dir, obs_dist])
        
        # the observation contains distances between all robot links and the regions
        else:
            # 0-6, 8 (hand) are robotic arm and 9,10 are the grippers
            robot_link_index_list = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10] # len = 10

            for color in self.task.colors:
                layout_ids = [self.sim._bodies_idx[f"{color}_{n}"] for n in range(self.task.region_num)]
                dirs_dists = np.zeros((len(robot_link_index_list), 4), dtype=np.float32)
                
                for i, robot_link_index in enumerate(robot_link_index_list):
                    min_dist = float('inf'); corresponding_dir = None
                    for layout_id in layout_ids:
                        closest_points = self.sim.physics_client.getClosestPoints(
                            bodyA=robot_id, bodyB=layout_id, linkIndexA=robot_link_index, distance=10.0)
                        assert len(closest_points) == 1, f"Expected ONLY 1 closest point, got {len(closest_points)}"
                        closest_point = closest_points[0]
                        dist = closest_point[8]
                        position_on_arm = np.array(closest_point[5])
                        position_on_region = np.array(closest_point[6])
                        dir = position_on_region - position_on_arm
                        # print(f"robot_link_index: {robot_link_index}, layout_id: {layout_id}, dist: {dist}, dir: {dir}")
                        dir = dir / np.linalg.norm(dir) if np.linalg.norm(dir) > 0 else dir
                        if dist < min_dist:
                            min_dist = dist
                            corresponding_dir = dir
                    if self.task.max_dist is None:
                        min_dist = np.exp(-self.task.exp_gain * min_dist)
                    else:
                        min_dist = np.maximum(0, self.task.max_dist - min_dist) / self.task.max_dist
                        if min_dist == 0.0:
                            corresponding_dir = np.zeros_like(corresponding_dir)
                    dirs_dists[i, :3] = corresponding_dir
                    dirs_dists[i, -1] = min_dist
                    obs[f"dist_to_arm_{color}"] = dirs_dists[:8, :]
                    obs[f"dist_to_grippers_{color}"] = dirs_dists[8:, :]
                # dist is normalized, 1 means close to the region, 0 means far away
                if np.any(dirs_dists[:8, -1] >= 1):
                    info["propositions"].add(f"arm_{color}")
                if np.all(dirs_dists[8:, -1] >= 1):
                    info["propositions"].add(f"grippers_{color}")

        return obs, info

class PandaSlideEnv(RobotTaskEnv):
    """Slide task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Slide(sim, reward_type=reward_type)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )


class PandaStackEnv(RobotTaskEnv):
    """Stack task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Stack(sim, reward_type=reward_type)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )
