import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.linalg import block_diag

T = [0,1,3,4,7]
x_dim = 2
x = [
  [1,1],
  [3,5],
  [6,10],
  [8,4],
  [5,2]
]
n = len(x) - 1
x = np.array(x,dtype=np.float32)

r = 5
Q = block_diag()
p = cp.Variable((n*(r+1),x_dim))
opt_dim = 4
for j in range(1,n+1):
  q = np.zeros(shape=(r+1,r+1))
  for i in range(r+1):
    for l in range(r+1):
      if i < opt_dim or l < opt_dim:
        q[i][l] = 0
      else:
        q[i][l] = math.factorial(i) / math.factorial(i-opt_dim) * \
                  math.factorial(l) / math.factorial(l-opt_dim) / \
                  (i+l-2*opt_dim+1) * \
                  (T[j]**(i+l-2*opt_dim+1) - T[j-1]**(i+l-2*opt_dim+1))
  if j > 1:
    Q = block_diag(Q,q)
  else:
    Q = q
  
obj = 0
for d in range(x_dim):
  obj += cp.quad_form(p[:,d],Q)

# constraints
# start point & end point
A_start = []
d_start = []
for k in range(4):
  Ak1 = np.zeros(n*(r+1))
  start = 0
  for i in range(r+1):
    if i < k:
      Ak1[start+i] = 0
    else:
      Ak1[start+i] = math.factorial(i) / math.factorial(i-k) * T[0]**(i-k)
  A_start.append(Ak1)
  d_start.append([0,0]) if k > 0 else d_start.append(x[0])
A_start = np.array(A_start)
d_start = np.array(d_start)

A_end = []
d_end = []
for k in range(4):
  Ak2 = np.zeros(n*(r+1))
  start = (n-1)*(r+1) - 1
  for i in range(r+1):
    if i < k:
      Ak2[start+i] = 0
    else:
      Ak2[start+i] = math.factorial(i) / math.factorial(i-k) * T[-1]**(i-k)
  A_end.append(Ak2)
  d_end.append([0,0]) if k > 0 else d_end.append(x[-1])
A_end = np.array(A_end)
d_end = np.array(d_end)

# waypoints
A_waypoints = []
d_waypoints = []
for j in range(1,n,1):
  A1 = np.zeros(n*(r+1))
  start = j*(r+1)
  for i in range(r+1):
    A1[start+i] = T[j]**i
  A2 = np.zeros(n*(r+1))
  for i in range(r+1):
    A2[start+i] = T[j+1]**i
  A_waypoints.append(A1)
  A_waypoints.append(A2)
  d_waypoints.append(x[j])
  d_waypoints.append(x[j+1])
A_waypoints = np.array(A_waypoints)
d_waypoints = np.array(d_waypoints)
np.set_printoptions(linewidth=np.inf,suppress=True)

# continuity for v,a
A_con = []
for j in range(0,n-1,1):
  for k in range(1,3,1):
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
        A2[start2+i] = math.factorial(i) / math.factorial(i-k) * T[j]**(i-k)
    A_con.append(A1-A2)
d_con = np.zeros((len(A_con),x_dim))
A_con = np.array(A_con)
print(np.array2string(A_con))
print(np.array2string(d_con))

A_eq = np.concatenate((A_start,A_end,A_waypoints,A_con),axis=0)
d_eq = np.concatenate((d_start,d_end,d_waypoints,d_con),axis=0)
# print(A_eq.shape)
# print(d_eq.shape)
# print(p.shape)
constraints = [A_eq @ p == d_eq]

prob = cp.Problem(objective=cp.Minimize(obj),constraints=constraints)
prob.solve()
print(prob.status)
# print(prob.value)
# print(p.value)

# 轨迹可视化
plt.figure(figsize=(10, 6))

# 提取优化后的多项式系数
coeffs = p.value.reshape(n, r+1, x_dim)  # shape: (n_segments, r+1, dim)

a = []
for j in range(len(T)-1):
  a_ = np.zeros(x_dim)
  for d in range(x_dim):
    a_[d] = sum(coeffs[j, i, d] * T[j+1]**i for i in range(r+1))
  a.append(a_)
print(a)
# 生成轨迹点
t_plot = np.linspace(T[0], T[-1], 500)
x_plot = np.zeros((len(t_plot), x_dim))
for idx, t in enumerate(t_plot):
    # 确定当前时间属于哪个段
    seg = 0
    while seg < n-1 and t > T[seg+1]:
        seg += 1
    
    # 计算位置 (x和y坐标)
    for dim in range(x_dim):
        x_plot[idx, dim] = sum(coeffs[seg, i, dim] * t**i for i in range(r+1))

# 绘制轨迹
plt.plot(x_plot[:, 0], x_plot[:, 1], 'b-', linewidth=2, label='Minimum Snap Trajectory')

# 标记航路点
plt.scatter(x[:, 0], x[:, 1], c='r', s=100, zorder=5, label='Waypoints')
for i, (xi, yi) in enumerate(x):
    plt.text(xi+0.2, yi+0.2, f'WP{i}', fontsize=10)

plt.show()