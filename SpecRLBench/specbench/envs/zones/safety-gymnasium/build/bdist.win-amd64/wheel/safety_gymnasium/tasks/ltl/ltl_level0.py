"""LTL level 0. No moving zones"""

from safety_gymnasium.assets.geoms import Zones
from safety_gymnasium.tasks.ltl.ltl_base_task import LTLBaseTask


class LTLLevel0(LTLBaseTask):
    """Two green, two yellow, two red, and two magenta zones."""

    def __init__(self, config) -> None:
        print(f"config = {config}")
        super().__init__(config=config, zone_size=0.4)
        print(f"LTL0, |AP| = 4")
        keepout = 0.55 if not self.allow_overlap else self.zone_size / 2
        self._add_geoms(Zones(color='green', size=self.zone_size, keepout=keepout, num=2))
        self._add_geoms(Zones(color='yellow', size=self.zone_size, keepout=keepout, num=2))
        self._add_geoms(Zones(color='blue', size=self.zone_size, keepout=keepout, num=2))
        self._add_geoms(Zones(color='magenta', size=self.zone_size, keepout=keepout, num=2))
