from collections import deque
import random


random.seed(5)

class ReplayBuffer(object):

    def __init__(self, buffer_size, const=False):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()
        self.const = const
        if const:
            print 'WARNING: replay buffer is not using random batches'

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        if self.const:
            batch = []
            for i in xrange(batch_size):
                batch.append(self.buffer[i])
            return batch

        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, *entry):
        experience = tuple(entry)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0