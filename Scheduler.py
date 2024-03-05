# -*- coding: utf-8 -*-
"""
@author: machadoyang
"""

from mesa.time import RandomActivationByType
from collections import OrderedDict

class AgentTypeScheduler(RandomActivationByType):
    def __init__(self, model, agent_class_order):
        super().__init__(model)
        self.class_order = agent_class_order
        
    def _class_index(self, agent):
        # Returns a low number for classes you want to have come first
        if agent.__class__ in self.class_order:
            return self.class_order.index(agent.__class__)
        else:
           return len(self.class_order)
           
    def step(self):
        # sort the agent list by the class order, then call the BaseScheduler step
        sorted_list_of_agents = sorted(self.agents, key=self._class_index)
        new_queue = OrderedDict(dict((i,j) for i,j in enumerate(sorted_list_of_agents)))
        self._agents = new_queue
        super().step()
        print("finalized")