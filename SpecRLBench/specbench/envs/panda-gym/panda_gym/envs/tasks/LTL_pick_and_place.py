from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


# TODO: 
# 1. Define the APs as ["pick_<color>", "place_<color>", "reach_<color>"].
#    There are objects with different colors, different colored regions on the table, and different colored spheres.
#    Thus the observation should include dist_to_<obj>, dist_to_<region>, and dist_to_<sphere>.
#    Also add another built-in safety constraints such that the objects should not collide between other during tasks.

class LTLPickAndPlace(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        partial_observability=False,
        observe_vision=False,
        obs_use_ee_only=True,
        # goal_xy_range: float = 0.3,
        # goal_z_range: float = 0.2,
        # obj_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold

        self.colors = ["magenta", "yellow", "blue", "green"]

        self.object_size = 0.04
        self.region_radius = 0.05; self.region_height = 1e-3
        self.sphere_radius = 0.05
        self.object_num = self.region_num = 1 # 
        # self.sphere_num = 2

        self.keepout = 0.06
        self.region_names = []

        # partial observability
        self.max_dist = None if not partial_observability else 0.2
        self.exp_gain = 2.0
        # vision observation
        self.observe_vision = observe_vision

        # ee only observation
        self.obs_use_ee_only = obs_use_ee_only

        self.object_range_low = np.array([-0.4, -0.3, self.object_size / 2])
        self.object_range_high = np.array([0.2, 0.3, self.object_size / 2])

        self.region_range_low = np.array([-0.4, -0.3, self.region_height])
        self.region_range_high = np.array([0.2, 0.3, self.region_height])

        # self.sphere_range_low = np.array([-0.5, -0.3, self.sphere_radius + self.object_size + 0.02])
        # self.sphere_range_high = np.array([0.2, 0.3, self.sphere_radius + 0.5 + self.object_size + 0.02])

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        for color in self.colors:
            for i in range(self.object_num):
                body_name = f"object_{color}_{i}"
                self.region_names.append(body_name)
                self.sim.create_box(
                    body_name=body_name,
                    half_extents=np.ones(3) * self.object_size / 2,
                    mass=2.0,
                    position=np.array([0.0, 0.0, self.object_size / 2]),
                    rgba_color=self.COLORS[color],
                    collision_group=2,
                    collision_mask=3,
                )
            # for i in range(self.region_num):
            #     body_name = f"region_{color}_{i}"
            #     self.region_names.append(body_name)
            #     self.sim.create_cylinder(
            #         body_name=body_name,
            #         radius=self.region_radius,
            #         height=self.region_height,
            #         mass=0.0,
            #         position=np.array([0.0, 0.0, self.region_height]),
            #         rgba_color=self.COLORS[color] * np.array([1.0, 1.0, 1.0, 0.3]),
            #     )
            for i in range(self.region_num):
                body_name = f"region_{color}_{i}"
                self.region_names.append(body_name)
                self.sim.create_sphere(
                    body_name=body_name,
                    radius=self.sphere_radius,
                    mass=0.0,
                    position=np.array([0.0, 0.0, self.sphere_radius]),
                    rgba_color=self.COLORS[color] * np.array([1.0, 1.0, 1.0, 0.3]),
                    collision_group=4,
                    collision_mask=0,
                )
            # for i in range(self.sphere_num):
            #     body_name = f"sphere_{color}_{i}"
            #     self.region_names.append(body_name)
            #     self.sim.create_sphere(
            #         body_name=body_name,
            #         radius=self.sphere_radius,
            #         mass=0.0,
            #         position=np.array([0.0, 0.0, self.sphere_radius]),
            #         rgba_color=self.COLORS[color] * np.array([1.0, 1.0, 1.0, 0.6]),
            #         collision_group=4,
            #         collision_mask=0,
            #     )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        """Reset the task by building and placing the layout."""
        for _ in range(10000):  # Try up to 10,000 times to find a valid layout    
            layout = self._sample_layout()
            if len(layout) == len(self.region_names):
                for region_name, position in layout.items():
                    self.sim.set_base_pose(region_name, position, np.array([0.0, 0.0, 0.0, 1.0]))
                return
        raise ValueError("Failed to sample a valid layout within constraints after 10,000 attempts.")
    
    def _sample_layout(self) -> dict:
        """Sample positions for regions using a layout sampling approach.

        Returns:
            dict: A dictionary mapping region names to their positions.
        """
        layout = {}
        positions = []

        for region_name in self.region_names:
            for _ in range(10000):  # Try up to 10000 times to find a valid position for each region
                if "object" in region_name:
                    position = self.np_random.uniform(self.object_range_low, self.object_range_high)
                    item_size = self.object_size / 2
                    # total_keepout = self.keepout + self.object_size
                elif "region" in region_name:
                    position = self.np_random.uniform(self.region_range_low, self.region_range_high)
                    item_size = self.region_radius
                    # total_keepout = self.keepout + 2*self.region_radius
                # elif "sphere" in region_name:
                #     position = self.np_random.uniform(self.sphere_range_low, self.sphere_range_high)
                #     # total_keepout = self.keepout + self.sphere_radius
                #     item_size = self.sphere_radius
                # total_keepout = self.keepout
                if self._is_valid_position(position, item_size, positions):
                    layout[region_name] = position
                    positions.append(np.append(position, item_size))
                    break
        return layout

    def _is_valid_position(self, position: np.ndarray, item_size: float, positions: list) -> bool:
        """Check if a position is valid based on existing positions.

        Args:
            position (np.ndarray): The position to validate.
            item_size (float): The size of the item.
            positions (list): List of existing positions.

        Returns:
            bool: True if the position is valid, False otherwise.
        """
        if not positions:
            return True
        distances = np.linalg.norm(np.array(positions)[:, :3] - position, axis=1)
        distances = distances - (np.array(positions)[:, 3] + item_size + self.keepout)
        return np.all(distances >= 0)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
