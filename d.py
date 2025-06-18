import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.linalg import block_diag

# f(t) = p0 + p1*t + p2*t^2 +...+pr*t^n
# p = [p0 p1 p2 ... pr]

T = [1,2,3,2]
x = [
  [1,1],
  [3,4],
  [7,5],
  [12,3],
  [6,1]
]
r = 5

np.set_printoptions(linewidth=np.inf,suppress=True)

def getQ(T:list,r:int,opt_dim=4):
  Q = []
  for j in range(0,len(T)):
    q = np.zeros(shape=(r+1,r+1))
    for i in range(r+1):
      for l in range(r+1):
        if i < opt_dim or l < opt_dim:
          q[i][l] = 0
        else:
          q[i][l] = math.factorial(i) / math.factorial(i-opt_dim) * \
                    math.factorial(l) / math.factorial(l-opt_dim) / \
                    (i+l-2*opt_dim+1) * \
                    (T[j]**(i+l-2*opt_dim+1))
    if j > 0:
      Q = block_diag(Q,q)
    else:
      Q = q
  return Q

def getA(waypoints:list,T:list,r:int):
  n = len(T) # number of segments
  x_dim = len(waypoints[0])
  A_start = []
  d_start = []
  for k in range(4):
    Ak1 = np.zeros(n*(r+1))
    start = 0
    for i in range(r+1):
      if i < k:
        Ak1[start+i] = 0
      else:
        Ak1[start+i] = math.factorial(i) / math.factorial(i-k) * 0**(i-k)
    A_start.append(Ak1)
    d_start.append([0,0]) if k > 0 else d_start.append(waypoints[0])
  A_start = np.array(A_start)
  d_start = np.array(d_start)
  print(A_start)
  print(d_start)

  A_end = []
  d_end = []
  for k in range(4):
    Ak2 = np.zeros(n*(r+1))
    start = (n-2)*(r+1) - 1
    for i in range(r+1):
      if i < k:
        Ak2[start+i] = 0
      else:
        Ak2[start+i] = math.factorial(i) / math.factorial(i-k) * T[-1]**(i-k)
    A_end.append(Ak2)
    d_end.append([0,0]) if k > 0 else d_end.append(waypoints[-1])
  A_end = np.array(A_end)
  d_end = np.array(d_end)
  print(A_end)
  print(d_end)
  print(A_end.shape)

  A_wp = []
  d_wp = []
  for j in range(1,n-1,1):
    A1 = np.zeros(n*(r+1))
    start = j*(r+1)
    for i in range(r+1):
      A1[start+i] = 0**i

    A2 = np.zeros(n*(r+1))
    start = (j+1)*(r+1)
    for i in range(r+1):
      A2[start+i] = T[j]**i
    A_wp.append(A1)
    A_wp.append(A2)
    d_wp.append(x[j])
    d_wp.append(x[j+1])
  A_wp = np.array(A_wp)
  d_wp = np.array(d_wp)
  print(A_wp)
  print(d_wp)
  print(A_wp.shape)

  # continuity for x,v,a
  A_con = []
  for j in range(0,n-1,1):
    for k in range(0,3,1):
      A1 = np.zeros(n*(r+1))
      start1 = j*(r+1)
      for i in range(r+1):
        if i < k:
          A1[start1+i] = 0
        else:
          A1[start1+i] = math.factorial(i) / math.factorial(i-k) * T[j]**(i-k)

      A2 = np.zeros(n*(r+1))
      start2 = (j+1)*(r+1)
      for i in range(r+1):
        if i < k:
          A2[start2+i] = 0
        else:
          A2[start2+i] = math.factorial(i) / math.factorial(i-k) * 0**(i-k)
      A_con.append(A1-A2)
  d_con = np.zeros((len(A_con),x_dim))
  A_con = np.array(A_con)
  print(A_con)
  print(d_con)
  print(A_con.shape)

  A_eq = np.concatenate((A_start,A_end,A_wp,A_con),axis=0)
  d_eq = np.concatenate((d_start,d_end,d_wp,d_con),axis=0)
  return A_eq,d_eq

def getMinSnapTraj(waypoints:list,T:list,r:int,opt_dim=4):
  n_traj = len(T)
  x_dim = len(waypoints[0])
  Q = getQ(T,r,opt_dim)
  A,d = getA(waypoints,T,r)
  P = cp.Variable(shape=(n_traj*(r+1),x_dim))
  obj = 0
  print(A.shape,Q.shape,P.shape)
  for d in range(x_dim):
    obj += cp.quad_form(P[:,d],Q)
  constraints = [A @ P == d]
  prob = cp.Problem(cp.Minimize(obj),constraints)
  prob.solve()
  print(prob.status)
  print(P.value)
  return P.value

x_dim = len(x[0])
P = getMinSnapTraj(x,T,r)
P = np.reshape(P,newshape=(len(T),r+1,x_dim))
print(P)
