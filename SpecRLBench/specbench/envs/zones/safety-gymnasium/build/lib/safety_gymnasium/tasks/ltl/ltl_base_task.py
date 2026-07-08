"""LTL tasks"""

from safety_gymnasium.assets.geoms import LtlWalls
from safety_gymnasium.bases.base_task import BaseTask


class LTLBaseTask(BaseTask):
    """Base task for LTL tasks."""

    def __init__(self, config, zone_size: float, walls=True) -> None:
        # print(config)
        
        super().__init__(config=config)
        self.zone_size = zone_size
        self.placements_conf.extents = [-2.5, -2.5, 2.5, 2.5]
        self.lidar_conf.num_bins = 16
        # For partial observability, limit the lidar range to half the environment size.
        # This is redundant for camera observation
        self.lidar_conf.max_dist = 1.2 if self.partial_observability else None
        self.lidar_conf.exp_gain = 0.5
        self.lidar_conf.alias = True
        self.cost_conf.constrain_indicator = False
        self.observation_flatten = False
        if walls:
            self._add_geoms(LtlWalls())

    def calculate_reward(self):
        return 0

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        pass

    @property
    def goal_achieved(self):
        return False
