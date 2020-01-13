import numpy as np
def f(a):
    a = np.copy(a)
    a[3] = 999
    return a

a = [0,1,2,3,4,5,6,7]
b = np.arange(8)
print(a)
f(a)
print(a)
print(b)
f(b)
print(b)
