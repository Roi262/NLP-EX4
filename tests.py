import numpy as np
def f(a):
    a = np.copy(a)
    a[3] = 999
    return a

a = np.array([0,1,2,3,4,1,1,7])
l = np.argwhere(a==1)
print(a.take((3,6)))
# print(l.T[0])

b = np.arange(8)
# print(a)
# f(a)
# print(a)
# print(b)
# f(b)
# print(b)
