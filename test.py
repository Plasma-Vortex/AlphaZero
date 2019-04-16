import numpy as np
from Game import *

def foo(s):
    if __debug__:
        print('hi')
    return s

a = np.array([0, 1, 0, -1, 1, -1]).reshape(2, 3)
b = [np.maximum(a, 0), np.maximum(-a, 0), a, a]
c = np.stack(b, axis=-1)
print(a)
print(b)
print(c)
print(c.shape)
# print(a)

# print(1/2)

# t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# a = np.array([t, u, 0])
# b = AddSymmetriesTTT(a)
# for i in b:
#     print(i)