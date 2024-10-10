import random
from collections import namedtuple

class ExperienceReplay(object):
    Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

    def __init__(self, capacity):  
        self.capacity = capacity
        self.memory = []
        self.position = 0
  
    def push(self, *args): 
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = ExperienceReplay.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):   
        return random.sample(self.memory, batch_size)

    def __len__(self): 
        return len(self.memory)
