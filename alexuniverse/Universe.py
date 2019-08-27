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

class Time(object):
    def __init__(self):
	print("universe time started")
	self.current_time = 0
    def __increment_time(self):
	self.current_time += 1
    def __decrement_time(self):
	self.current_time -= 1


class Space(object):
    """
    Space has the following attributes:
        1. 2D parameters, cartesian
 
    Space has the following methods:
        1. initialize it self
        2. put objects in it
       
    """
    def __init__(self,xlim:int =1000,ylim:int = 1000):
	self.xlim = xlim
	self.ylim = ylim

class Matter(object):
    def __init__(self, total_amount:int=1000,seed:int=7):
	self.total_amount = total_amount
	self.seed = seed

class Universe(Time,Space,Matter):
    def __init__(self):
	Time.__init__(self)
	Space.__init__(self)
	Matter.__init__(self)


      
