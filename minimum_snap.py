import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.linalg import block_diag

n = 4
r = 5
Q = block_diag()
T = [0,1,2,3,4]
p = cp.Variable(n*(r+1))
x = [
  [3,3],
  [7,2],
  [4,9],
  [7,6],
  [10,10]
]
x = np.array(x,dtype=np.float32)
for j in range(1,n+1):
  q = np.zeros(shape=(r+1,r+1))
  for i in range(r+1):
    for l in range(r+1):
      if i < 4 or l < 4:
        q[i][l] = 0
      else:
        q[i][l] = i*(i-1)*(i-2)*(i-3) * l*(l-1)*(l-2)*(l-3) / (i+l-7) * (T[j]**(i+l-7) - T[j-1]**(i+l-7))
  if j > 1:
    Q = block_diag(Q,q)
  else:
    Q = q
# print(Q)
# print(p.value)
obj = cp.Minimize(p.T @ Q @ p)
# print(obj)

# constraints
# start point & end point
A_start = []
for k in range(4):
  Ak1 = []
  
  for i in range(r+1):
    if i < k:
      Ak1.append(0)
    else:
      Ak1.append(math.factorial(i) / math.factorial(i-k) * T[0]**(i-k))
  A_start.append(Ak1)
A_start = np.array(A_start)
d_start = np.zeros(4)

A_end = []
for k in range(4):
  Ak2 = []
  for i in range(r+1):
    if i < k:
      Ak2.append(0)
    else:
      Ak2.append(math.factorial(i) / math.factorial(i-k) * T[-1]**(i-k))
  A_end.append(Ak2)
A_end = np.array(A_end)
d_end = np.zeros(4)
print(A_end)

# waypoints
A_waypoints = []
for j in range(1,n-1,1):
  A1 = []
  for i in range(r+1):
    A1.append(T[j]**i)
  A2 = []
  for i in range(r+1):
    A2.append(T[j+1]**i)
  A_waypoints.append(A1)
  A_waypoints.append(A2)
A_waypoints = np.array(A_waypoints)
d_waypoints = []
for j in range(n-2):
  d_waypoints.append(x[j+1])
  d_waypoints.append(x[j+2])
d_waypoints = np.array(d_waypoints)
print(A_waypoints)
print(d_waypoints)

# continuity
