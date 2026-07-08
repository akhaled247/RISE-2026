"""LTL level 1. One moving zone per color"""

from safety_gymnasium.assets.mocaps import MovingZones, Gremlins
from safety_gymnasium.tasks.ltl.ltl_base_task import LTLBaseTask


class LTLLevel1(LTLBaseTask):
    """Two green, two yellow, two red, and two magenta zones."""

    def __init__(self, config) -> None:
        super().__init__(config=config, zone_size=0.4)
        print(f"LTL1, |AP| = 4, one moving zone per AP")
        keepout = 0.65 if not self.allow_overlap else self.zone_size / 2
        self._add_mocaps(MovingZones(color='green', size=self.zone_size, keepout=keepout, num=2, moving_num=1))
        self._add_mocaps(MovingZones(color='yellow', size=self.zone_size, keepout=keepout, num=2, moving_num=1))
        self._add_mocaps(MovingZones(color='blue', size=self.zone_size, keepout=keepout, num=2, moving_num=1))
        self._add_mocaps(MovingZones(color='magenta', size=self.zone_size, keepout=keepout, num=2, moving_num=1))

