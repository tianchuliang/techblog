##################
# the 2D space class
# the main environment which runs the
# simulation.
#
# The universe file contains
# The space class
# The time class
# The matter class
# and finally the Universe class
#################
import numpy as np
from typing import Tuple, Dict, List


class Matter(object):
    def __init__(
        self,
        name: str,
        degrade_rate: float = 0.1,
        degrade: bool = True,
        total_amount: int = 1000,
        seed: int = 7,
    ):
        self.total_amount = total_amount
        self.seed = seed
        self.name = name
        self.degrade_rate = degrade_rate
        self.degrade = degrade

    def __degrade(self):
        pass

    def __vanish_step(self):
        self.total_amount -= 1

    def __vanish(self):
        self.total_amount = 0


class Time(object):
    def __init__(self):
        print("universe time started")
        self.current_time = 0

    def __increment_time(self):
        self.current_time += 1

    def __decrement_time(self):
        self.current_time -= 1

    def __reset_time(self):
        self.current_time = 0


class Space(object):
    """
    Space has the following attributes:
        1. 2D parameters, cartesian
 
    Space has the following methods:
        1. initialize it self
        2. put objects in it
       
    """

    def __init__(self, xlim: int = 1000, ylim: int = 1000):
        self.xlim = xlim
        self.ylim = ylim

    def __add_matter(self, matter_list: List[Matter]):
        self.matter_map = {}
        for matter in matter_list:
            self.matter_map[matter.name] = matter

    def __disperse_matter(self, randomseed: int = 7):
        self.matter_loc = {}  # dict of matter location in space
        for matter in self.matter_map.keys():
            xpos = int(np.random.random(1) * self.xlim)
            ypos = int(np.random.random(1) * self.ylim)
            self.matter_loc[matter] = (xpos, ypos)


class Universe(Time, Space):
    def __init__(self):
        Time.__init__(self)
        Space.__init__(self)

    def create_matter(self):
        pass
