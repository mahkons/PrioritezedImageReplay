import torch
import numpy as np
import random

from range_tree import RangeTree

DECAY_PARAM = 0.001

# TODO
# make use of multiprocessing like torch.util.data.dataloaders?

class Sampler():
    def __init__(self, trainlen):
        self.trainlen = trainlen

    def sample(self, batch_size):
        pass

    def update(self, sample_ids, priorities):
        pass

    def __len__(self):
        return self.trainlen


class RandomSampler(Sampler):
    def __init__(self, trainlen):
        super().__init__(trainlen)

    def sample(self, batch_size):
        sample_ids = random.sample(range(self.trainlen), batch_size)
        probs = np.ones(batch_size) / self.trainlen
        return sample_ids, probs

    def update(self, sample_ids, priorities):
        pass


class PrioritizedSampler(Sampler):
    def __init__(self, trainlen):
        super().__init__(trainlen)
        self.tree = RangeTree(np.ones(self.trainlen) * 10) # just some big init val
        self.update_steps = 0

    def sample(self, batch_size):
        sum = self.tree.get_sum()
        sample_ids, probs = zip(*[self.tree.get(np.random.uniform(sum)) for _ in range(batch_size)])
        probs = np.array(probs)
        probs /= sum
        return sample_ids, probs

    def update(self, sample_ids, priorities):
        self.update_steps += 1
        self.tree.add_value(DECAY_PARAM) # exploration / exploitation
        for sid, p in zip(sample_ids, priorities):
            self.tree.update(sid, p)

