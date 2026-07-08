"""Customized LTL task."""

from safety_gymnasium.assets.mocaps import MovingZones
from safety_gymnasium.tasks.ltl.ltl_base_task import LTLBaseTask


class CustomizedLTL(LTLBaseTask):

    def __init__(self, config) -> None:
        super().__init__(config=config, zone_size=0.0)
        print(f"CustomizedLTL, config = {config}")
        
        required_keys = [
            'atomic_propositions',
            'number_of_zones_per_color',
            'size_of_zones',
            'number_of_moving_zones',
            'keepout_distances'
        ]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required config key: {key}")

        ap_list = config['atomic_propositions']
        for key in required_keys[1:]:
            if set(config[key].keys()) != set(ap_list):
                raise ValueError(
                    f"Config key '{key}' must have the same keys as 'atomic_propositions'. "
                    f"Expected keys: {ap_list}, got: {list(config[key].keys())}"
                )

        ap_list = config['atomic_propositions']
        num_zones_per_color = config['number_of_zones_per_color']
        size_of_zones = config['size_of_zones']
        number_of_moving_zones = config['number_of_moving_zones']
        keepout_distances = config['keepout_distances']

        for ap in ap_list:
            self._add_mocaps(MovingZones(color=ap, size=size_of_zones[ap], keepout=keepout_distances[ap], num=num_zones_per_color[ap], moving_num=number_of_moving_zones[ap]))
