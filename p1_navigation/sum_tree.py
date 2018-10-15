import numpy as np


class SumTree:
    ptr = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.count = 0

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        # Update parent node
        self.update(idx, priority)

        self.ptr += 1

        # sum tree is full, reset point back to 0
        # (overwrites previous entries)
        if self.ptr >= self.capacity:
            self.ptr = 0

        if self.count < self.capacity:
            self.count += 1

    def get(self, s):
        # start the search from the root
        index = self.retrieve(0, s)
        priority = self.tree[index]
        data = self.data[index - self.capacity + 1]

        return index, priority, data

    def retrieve(self, index, s):
        left = 2 * index + 1
        right = left + 1

        # idx points to a leaf node
        if left >= len(self.tree):
            return index

        if self.tree[left] >= s:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])

    def update(self, index, priority):
        self.tree[index] = priority
        # propagate the change from the parent node all the way to the root
        change = priority - self.tree[index]
        self.propagate(index, change)

    def propagate(self, index, change):
        parent = (index - 1) // 2
        self.tree[parent] += change

        # if not at the root, recursively propagate change
        if parent != 0:
            self.propagate(parent, change)

