import numpy as np

# TODO c++ should be faster
# TODO batch get

class RangeTree:
    def __init__(self, values):
        capacity = len(values)
        self.size = 1
        while self.size < capacity:
            self.size *= 2
        self.values = np.zeros(2 * self.size)

        self.values[self.size:self.size + capacity] = values
        for pos in range(self.size - 1, -1, -1):
            self.values[pos] = self.values[2 * pos] + self.values[2 * pos + 1]

        self.capacity = capacity
        self.constant_add = 0

    def update(self, pos, x):
        pos += self.size
        self.values[pos] = x - self.constant_add
        pos //= 2
        while (pos):
            self.values[pos] = self.values[2 * pos] + self.values[2 * pos + 1]
            pos //= 2

    # adds constant to all values in the tree
    def add_value(self, value):
        self.constant_add += value
            
    def get_sum(self):
        return self.values[1] + self.constant_add * self.capacity # ignores elements in [capacity, size)
            
    def get(self, x):
        pos = 1
        cnt = self.size
        while pos < self.size:
            pos *= 2
            cnt /= 2
            if x > self.values[pos] + self.constant_add * cnt:
                x -= self.values[pos] + self.constant_add * cnt
                pos += 1
        return pos - self.size
