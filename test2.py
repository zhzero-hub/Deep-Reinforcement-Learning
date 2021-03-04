import gym
import numpy as np
from gym import spaces


if __name__ == '__main__':
    a = [1, 2, 3]
    b = np.asarray(a)
    c = np.repeat(b, 5)
    d = c.reshape((5, 1))

    print(d)
