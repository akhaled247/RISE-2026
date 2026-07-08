"""MovingZones."""

from dataclasses import dataclass, field

import numpy as np

from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_object import Mocap


@dataclass
class MovingZones(Mocap):  # pylint: disable=too-many-instance-attributes
    """Colored zones that move in circular patterns like gremlins."""

    COLORS = {
        "black": np.array([0, 0, 0, 1]),
        "blue": np.array([0, 0, 1, 1]),
        "green": np.array([0, 1, 0, 1]),
        "cyan": np.array([0, 1, 1, 1]),
        "red": np.array([1, 0, 0, 1]),
        "magenta": np.array([1, 0, 1, 1]),
        "yellow": np.array([1, 1, 0, 1]),
        "white": np.array([1, 1, 1, 1]),
        "orange": np.array([1.0, 0.5, 0.0, 1.0]),
        "purple": np.array([0.5, 0.0, 0.5, 1.0]),
        "lime": np.array([0.5, 1.0, 0.0, 1.0]),
        "teal": np.array([0.0, 0.5, 0.5, 1.0]),
        "pink": np.array([1.0, 0.75, 0.8, 1.0]),
        "brown": np.array([0.6, 0.3, 0.0, 1.0]),
        "navy": np.array([0.0, 0.0, 0.5, 1.0]),
        "gray": np.array([0.5, 0.5, 0.5, 1.0]),
    }

    # Cost parameters
    contact_cost: float = 1.0  # Cost for being inside a moving zone
    dist_threshold: float = 0.2  # Threshold for cost for being too close
    dist_cost: float = 1.0  # Cost for being within distance threshold
    
    # Technical parameters
    density: float = 0.001
    is_meshed: bool = False

    def __init__(self, color: str, size: float, num: int, moving_num: int, locations=None, keepout=0.55, travel=0.3):
        self.color_name = color
        self.name = f'{color}_zones'  # Match regular Zones naming
        # self.name = f'gremlins'
        self.num = num
        self.moving_num = moving_num
        assert self.moving_num <= self.num, "moving_num must be less than or equal to num"
        self.size: float = size
        # self.size = 0.1
        self.placements: list = None
        self.locations: list = locations if locations else []
        self.keepout: float = keepout
        self.travel: float = travel
        self.alpha: float = 0.25
        self.color: np.array = self.COLORS[self.color_name]
        # self.group: int = self.calculate_group()
        self.group: np.array = GROUP['gremlin']
        self.is_lidar_observed: bool = True
        self.is_constrained: bool = True

        self.mesh_name = self.name[:-1]
        self.rotation_directions = []
        self.phase_offsets = []
        self._randomness_initialized = False

        # for i in range(self.moving_num):
        #     random_value = self.random_generator.uniform(0, 1)
        #     print(f"random_value: {random_value}")
        #     direction = 1 if random_value >= 0.5 else -1
        #     self.rotation_directions.append(direction)

    def _initialize_randomness(self):
        """Initialize random rotation directions and phase offsets."""
        if self._randomness_initialized:
            return
        
        # Create a deterministic seed based on zone properties
        # This ensures reproducible behavior while giving variety between zones
        base_seed = hash(f"{self.color_name}_{self.num}_{self.moving_num}_{self.travel}") % (2**31)
        local_rng = np.random.RandomState(base_seed)
        
        for i in range(self.moving_num):
            # Sample random value for rotation direction (50/50 chance)
            random_value = local_rng.uniform(0, 1)
            direction = 1 if random_value >= 0.5 else -1
            self.rotation_directions.append(direction)
            
            # Add random phase offset so zones don't all start at same position
            phase_offset = local_rng.uniform(0, 2 * np.pi)
            self.phase_offsets.append(phase_offset)
            
        self._randomness_initialized = True

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object"""
        return {'obj': self.get_obj(xy_pos, rot), 'mocap': self.get_mocap(xy_pos, rot)}

    def get_obj(self, xy_pos, rot):
        """To facilitate get objects config for this object"""
        body = {
            'name': self.name,
            'pos': np.r_[xy_pos, 1e-2],
            'rot': rot,
            'geoms': [
                {
                    'name': self.name,
                    'size': [self.size, 1e-2],
                    'type': 'cylinder',
                    'density': self.density,
                    'group': self.group,
                    'rgba': self.color * np.array([1, 1, 1, self.alpha]),
                    'contype': 0,  # No collision generation
                    'conaffinity': 0,  # No collision detection with others
                },
            ],
        }
        return body

    def get_mocap(self, xy_pos, rot):
        """To facilitate get mocaps config for this object"""
        body = {
            'name': self.name,
            'pos': np.r_[xy_pos, 1e-2],
            'rot': rot,
            'geoms': [
                {
                    'name': self.name,
                    'size': [self.size, 1e-2],
                    'type': 'cylinder',
                    'group': self.group,
                    'rgba': self.color * np.array([1, 1, 1, 0]),
                    'contype': 0,  # No collision generation
                    'conaffinity': 0,  # No collision detection with others
                },
            ],
        }
        return body

    def move(self):
        """Set mocap object positions before a physics step is executed."""
        # Initialize randomness on first move if not already done
        if not self._randomness_initialized:
            self._initialize_randomness()

        base_phase = float(self.engine.data.time) * 2.
        for i in range(self.moving_num):
            phase = base_phase * self.rotation_directions[i] + self.phase_offsets[i]
            name = f'{self.name[:-1]}{i}'
            target = np.array([np.sin(phase), np.cos(phase)]) * self.travel
            pos = np.r_[target, 1e-3]
            self.set_mocap_pos(name + 'mocap', pos)

    def cal_cost(self):
        """Zone cost processing - matches regular Zones implementation."""
        cost = {f'cost_zones_{self.color_name}': 0}  # Same key as regular Zones
        
        # print(f"\n=== CAL_COST DEBUG for {self.color_name} ===")
        # print(f"free_geoms positions: {[(name, geom['pos'][:2]) for name, geom in self.engine.free_geoms.items() if self.color_name in name]}")
        # print(f"mocaps positions: {[(name, mocap['pos'][:2]) for name, mocap in self.engine.mocaps.items() if self.color_name in name]}")
        # print(f"actual xpos positions: {[(f'{self.name[:-1]}{i}obj', self.engine.data.body(f'{self.name[:-1]}{i}obj').xpos[:2].copy()) for i in range(self.num)]}")
        
        for i, h_pos in enumerate(self.pos):
            h_dist = self.agent.dist_xy(h_pos)  # Use same method as regular Zones
            # print(f"Zone {i}: pos={h_pos[:2]}, dist_to_agent={h_dist:.3f}, size={self.size}")
            if h_dist <= self.size:
                cost[f'cost_zones_{self.color_name}'] = 1  # Same cost structure
                # print(f"COST TRIGGERED for {self.color_name} zone {i}!")
        
        # print(f"=== CAL_COST COMPLETE ===\n")
        return cost
    
    @property
    def pos(self):
        """Helper to get the current moving zone positions."""
        # pylint: disable-next=no-member
        return [self.engine.data.body(f'{self.name[:-1]}{i}obj').xpos.copy() for i in range(self.num)]