import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import block_diag

# 航路点数据
x = np.array([
    [3, 3],
    [7, 2],
    [4, 9],
    [7, 6],
    [10, 10]
], dtype=np.float32)

# 参数设置
n_segments = len(x) - 1  # 轨迹段数
poly_order = 5  # 多项式阶数 (r+1)
min_derivative = 4  # 最小化snap (4阶导数)

# 计算基于路径长度的时间分配
path_lengths = np.sqrt(np.sum(np.diff(x, axis=0)**2, axis=1))
T = np.cumsum(np.insert(path_lengths, 0, 0))  # [0, l1, l1+l2, ...]

# 构造Q矩阵
def create_Q_matrix(n_segments, poly_order, min_derivative, T):
    Q = None
    for seg in range(n_segments):
        q = np.zeros((poly_order+1, poly_order+1))
        for i in range(poly_order+1):
            for l in range(poly_order+1):
                if i < min_derivative or l < min_derivative:
                    q[i, l] = 0
                else:
                    fact_i = math.factorial(i)/math.factorial(i-min_derivative)
                    fact_l = math.factorial(l)/math.factorial(l-min_derivative)
                    exponent = i + l - 2*min_derivative + 1
                    q[i, l] = fact_i * fact_l * \
                             (T[seg+1]**exponent - T[seg]**exponent) / exponent
        Q = block_diag(Q, q) if Q is not None else q
    return Q

Q = create_Q_matrix(n_segments, poly_order, min_derivative, T)
p = cp.Variable((n_segments*(poly_order+1), 2))  # 二维轨迹

# 目标函数：对x和y维度分别优化
obj = 0
for dim in range(2):
    obj += cp.quad_form(p[:, dim], Q)

# 约束条件
constraints = []

# 1. 起点约束 (位置、速度、加速度、加加速度)
for k in range(4):  # 0:位置, 1:速度, 2:加速度, 3:加加速度
    A = np.zeros(n_segments*(poly_order+1))
    for i in range(poly_order+1):
        if i >= k:
            A[i] = math.factorial(i)/math.factorial(i-k) * T[0]**(i-k)
    constraints.append(A @ p[:, 0] == x[0, 0])  # x维度
    constraints.append(A @ p[:, 1] == x[0, 1])  # y维度

# 2. 终点约束
for k in range(4):
    A = np.zeros(n_segments*(poly_order+1))
    start = (n_segments-1)*(poly_order+1)
    for i in range(poly_order+1):
        if i >= k:
            A[start + i] = math.factorial(i)/math.factorial(i-k) * T[-1]**(i-k)
    constraints.append(A @ p[:, 0] == x[-1, 0])  # x维度
    constraints.append(A @ p[:, 1] == x[-1, 1])  # y维度

# 3. 航路点位置约束
for wp in range(1, n_segments):
    # 前一段的终点
    A_prev = np.zeros(n_segments*(poly_order+1))
    start = (wp-1)*(poly_order+1)
    for i in range(poly_order+1):
        A_prev[start + i] = T[wp]**i
    constraints.append(A_prev @ p[:, 0] == x[wp, 0])
    constraints.append(A_prev @ p[:, 1] == x[wp, 1])
    
    # 后一段的起点 (应与前一段终点相同)
    A_next = np.zeros(n_segments*(poly_order+1))
    start = wp*(poly_order+1)
    for i in range(poly_order+1):
        A_next[start + i] = T[wp]**i
    constraints.append(A_next @ p[:, 0] == x[wp, 0])
    constraints.append(A_next @ p[:, 1] == x[wp, 1])

# 4. 连续性约束 (速度、加速度、加加速度)
for wp in range(1, n_segments):
    for k in range(1, 4):  # 1:速度, 2:加速度, 3:加加速度
        # 前一段在航路点处的k阶导数
        A_prev = np.zeros(n_segments*(poly_order+1))
        start = (wp-1)*(poly_order+1)
        for i in range(poly_order+1):
            if i >= k:
                A_prev[start + i] = math.factorial(i)/math.factorial(i-k) * T[wp]**(i-k)
        
        # 后一段在航路点处的k阶导数
        A_next = np.zeros(n_segments*(poly_order+1))
        start = wp*(poly_order+1)
        for i in range(poly_order+1):
            if i >= k:
                A_next[start + i] = math.factorial(i)/math.factorial(i-k) * T[wp]**(i-k)
        
        constraints.append(A_prev @ p[:, 0] == A_next @ p[:, 0])  # x维度
        constraints.append(A_prev @ p[:, 1] == A_next @ p[:, 1])  # y维度

# 求解问题
prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve(verbose=True)

# 可视化结果
plt.figure(figsize=(10, 6))
t_plot = np.linspace(T[0], T[-1], 500)
pos = np.zeros((len(t_plot), 2))

for i, t in enumerate(t_plot):
    seg = np.searchsorted(T, t) - 1
    seg = max(0, min(seg, n_segments-1))
    t_rel = t - T[seg]
    
    for dim in range(2):
        coeffs = p.value[seg*(poly_order+1):(seg+1)*(poly_order+1), dim]
        pos[i, dim] = sum(coeff * t_rel**i for i, coeff in enumerate(coeffs))

# 绘制轨迹
plt.plot(pos[:, 0], pos[:, 1], 'b-', linewidth=2, label='Trajectory')
plt.scatter(x[:, 0], x[:, 1], c='r', s=100, label='Waypoints')

# 标记段边界
for t in T[1:-1]:
    idx = np.argmin(np.abs(t_plot - t))
    plt.scatter(pos[idx, 0], pos[idx, 1], c='g', s=80, marker='s', label='Segment Boundary' if t == T[1] else "")

plt.title('Minimum Snap Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
