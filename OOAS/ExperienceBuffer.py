from collections import deque
import numpy as np
import time

class ExperienceBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)
        print(self.size())
    
    def get(self):
        return list(self.buffer)

    def size(self):
        return len(self.buffer)
    
    def sample(self, batch_size) -> list:
        """
            Get one random batch from experience replay
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]