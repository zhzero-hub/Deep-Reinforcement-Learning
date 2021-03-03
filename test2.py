import gym
import numpy as np
from gym import spaces


if __name__ == '__main__':
    a = [1, 2]
    b = [1, 2]
    c = [a, b]
    d = np.asarray(np.asarray(a))
    print(d)
