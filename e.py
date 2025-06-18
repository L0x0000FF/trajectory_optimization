import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import block_diag

# 设置打印选项
np.set_printoptions(linewidth=np.inf, suppress=True)

def getQ(T: list, r: int, opt_dim=4):
    """
    构造Q矩阵（目标函数的二次型矩阵）
    
    参数:
        T: 各段轨迹的时间长度列表 [T0, T1, ..., T_{n-1}]
        r: 多项式阶数
        opt_dim: 最小化的导数阶数（4对应最小化snap）
    
    返回:
        Q: 块对角矩阵，形状为 (n*(r+1), n*(r+1))
    """
    n_segments = len(T)
    Q = None
    
    for seg in range(n_segments):
        q = np.zeros((r+1, r+1))
        for i in range(r+1):
            for l in range(r+1):
                if i >= opt_dim and l >= opt_dim:
                    # 计算积分项 ∫₀^{T_j} (derivative)^2 dt
                    exponent = i + l - 2 * opt_dim + 1
                    fact = (math.factorial(i) / math.factorial(i - opt_dim)) * \
                           (math.factorial(l) / math.factorial(l - opt_dim))
                    q[i, l] = fact * (T[seg] ** exponent) / exponent
                    
        if Q is None:
            Q = q
        else:
            Q = block_diag(Q, q)
    
    # 添加小的正则化项确保正定性
    Q += 1e-6 * np.eye(Q.shape[0])
    return Q

def getA_eq(waypoints: list, T: list, r: int):
    """
    构造等式约束矩阵A_eq和右侧向量d_eq
    
    参数:
        waypoints: 航路点列表 [[x0,y0], [x1,y1], ..., [xn,yn]]
        T: 各段轨迹的时间长度列表 [T0, T1, ..., T_{n-1}]
        r: 多项式阶数
    
    返回:
        A_eq: 等式约束矩阵，形状为 (n_constraints, n_segments*(r+1))
        d_eq: 等式约束右侧向量，形状为 (n_constraints, x_dim)
    """
    n_segments = len(T)
    n_waypoints = len(waypoints)
    x_dim = len(waypoints[0])
    
    # 验证输入
    if n_segments != n_waypoints - 1:
        raise ValueError(f"轨迹段数({n_segments})必须等于航路点数-1({n_waypoints-1})")
    
    # 初始化约束列表
    A_list = []
    d_list = []
    
    # 1. 起点约束 (位置、速度、加速度、加加速度)
    for k in range(4):  # k=0:位置, k=1:速度, k=2:加速度, k=3:加加速度
        A = np.zeros(n_segments * (r+1))
        # 第一段轨迹的起点 (t=0)
        for i in range(r+1):
            if i >= k:
                A[i] = math.factorial(i) / math.factorial(i-k) * (0**(i-k))
        A_list.append(A)
        
        # 右侧值: 位置约束用航路点，高阶导数约束用0
        d = np.zeros(x_dim)
        if k == 0:
            d = np.array(waypoints[0])
        else:
            d = np.zeros(x_dim)  # 速度、加速度、加加速度设为0
        d_list.append(d)
    
    # 2. 终点约束 (位置、速度、加速度、加加速度)
    for k in range(4):
        A = np.zeros(n_segments * (r+1))
        # 最后一段轨迹的终点 (t=T_last)
        start_idx = (n_segments - 1) * (r+1)
        for i in range(r+1):
            if i >= k:
                A[start_idx + i] = math.factorial(i) / math.factorial(i-k) * (T[-1]**(i-k))
        A_list.append(A)
        
        # 右侧值
        d = np.zeros(x_dim)
        if k == 0:
            d = np.array(waypoints[-1])
        else:
            d = np.zeros(x_dim)  # 速度、加速度、加加速度设为0
        d_list.append(d)
    
    # 3. 航路点位置约束 (中间点)
    for wp_idx in range(1, n_waypoints - 1):
        # 前一段轨迹的终点约束 (t=T_{seg})
        A_prev_end = np.zeros(n_segments * (r+1))
        seg_idx = wp_idx - 1  # 前一段轨迹索引
        start_idx = seg_idx * (r+1)
        for i in range(r+1):
            A_prev_end[start_idx + i] = T[seg_idx] ** i  # 在时间T_seg处取值
        A_list.append(A_prev_end)
        d_list.append(np.array(waypoints[wp_idx]))
        
        # 后一段轨迹的起点约束 (t=0)
        A_next_start = np.zeros(n_segments * (r+1))
        seg_idx = wp_idx  # 后一段轨迹索引
        start_idx = seg_idx * (r+1)
        for i in range(r+1):
            A_next_start[start_idx + i] = 0 ** i  # 在时间0处取值
        A_list.append(A_next_start)
        d_list.append(np.array(waypoints[wp_idx]))
    
    # 4. 连续性约束 (速度、加速度、加加速度)
    for wp_idx in range(1, n_waypoints - 1):
        for k in range(1, 4):  # k=1:速度, k=2:加速度, k=3:加加速度
            A_con = np.zeros(n_segments * (r+1))
            # 前一段轨迹在终点处的导数 (t=T_{seg})
            seg_prev = wp_idx - 1
            start_prev = seg_prev * (r+1)
            for i in range(r+1):
                if i >= k:
                    A_con[start_prev + i] = math.factorial(i) / math.factorial(i-k) * T[seg_prev]**(i-k)
            
            # 后一段轨迹在起点处的导数 (t=0) - 取负号
            seg_next = wp_idx
            start_next = seg_next * (r+1)
            for i in range(r+1):
                if i >= k:
                    A_con[start_next + i] = -math.factorial(i) / math.factorial(i-k) * 0**(i-k)
            
            A_list.append(A_con)
            d_list.append(np.zeros(x_dim))  # 导数连续要求差值为0
    
    # 合并所有约束
    A_eq = np.array(A_list)  # 形状: (n_constraints, n_segments*(r+1))
    d_eq = np.array(d_list)  # 形状: (n_constraints, x_dim)
    
    return A_eq, d_eq

def getMinSnapTraj(waypoints: list, T: list, r: int = 5, opt_dim: int = 4):
    """
    计算Minimum Snap轨迹
    
    参数:
        waypoints: 航路点列表
        T: 各段轨迹的时间长度
        r: 多项式阶数 (默认5)
        opt_dim: 最小化的导数阶数 (默认4-snap)
    
    返回:
        P: 多项式系数矩阵, 形状 (n_segments, r+1, x_dim)
    """
    n_segments = len(T)
    x_dim = len(waypoints[0])
    
    # 构造Q矩阵和约束
    Q = getQ(T, r, opt_dim)
    A_eq, d_eq = getA_eq(waypoints, T, r)
    
    # 打印维度信息用于调试
    print(f"A_eq shape: {A_eq.shape}")
    print(f"d_eq shape: {d_eq.shape}")
    
    # 创建优化变量
    P = cp.Variable((n_segments * (r+1), x_dim))
    
    # 目标函数: 最小化snap (对每个维度独立)
    objective = 0
    for dim in range(x_dim):
        objective += cp.quad_form(P[:, dim], Q)
    
    # 约束条件
    constraints = [A_eq @ P == d_eq]
    
    # 求解问题
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.ECOS)
    
    if problem.status != 'optimal':
        print(f"优化失败! 状态: {problem.status}")
        return None
    
    print("优化成功!")
    # 重构系数矩阵: (n_segments, r+1, x_dim)
    coeffs = P.value.reshape(n_segments, r+1, x_dim)
    return coeffs

def plot_trajectory(waypoints, T, coeffs):
    """可视化轨迹结果"""
    plt.figure(figsize=(10, 6))
    
    # 绘制航路点
    waypoints = np.array(waypoints)
    plt.scatter(waypoints[:, 0], waypoints[:, 1], c='red', s=100, zorder=5, label='Waypoints')
    
    # 绘制轨迹
    n_segments = len(T)
    colors = plt.cm.viridis(np.linspace(0, 1, n_segments))
    
    for seg in range(n_segments):
        # 生成时间点
        t_vals = np.linspace(0, T[seg], 100)
        seg_points = np.zeros((len(t_vals), 2))
        
        # 计算轨迹点
        for i, t in enumerate(t_vals):
            for dim in range(2):
                seg_points[i, dim] = sum(
                    coeffs[seg, j, dim] * t**j for j in range(coeffs.shape[1])
                )
        
        # 绘制轨迹段
        plt.plot(seg_points[:, 0], seg_points[:, 1], 
                 color=colors[seg], linewidth=2, 
                 label=f'Segment {seg+1}')
        
        # 标记段终点
        end_point = seg_points[-1]
        plt.scatter(end_point[0], end_point[1], 
                   color=colors[seg], s=80, marker='s', zorder=4)
    
    # 装饰图形
    plt.title('Minimum Snap Trajectory', fontsize=14)
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ====================== 测试用例 ======================
if __name__ == "__main__":
    # 航路点 (起点, 中间点1, 中间点2, 终点)
    waypoints = [
        [1, 1],   # 起点
        [3, 4],   # 中间点1
        [7, 5],   # 中间点2
        [12, 3],  # 终点
    ]
    
    # 各段轨迹时间长度 (3段轨迹)
    T = [1, 0.5, 2]
    
    # 计算轨迹 (5阶多项式，最小化snap)
    coeffs = getMinSnapTraj(waypoints, T, r=5, opt_dim=4)
    
    if coeffs is not None:
        print("\n多项式系数 (按段、阶数、维度):")
        for seg in range(len(T)):
            print(f"\n段 {seg+1} (时长={T[seg]}):")
            print("X系数:", ["%.6f" % x for x in coeffs[seg, :, 0]])
            print("Y系数:", ["%.6f" % y for y in coeffs[seg, :, 1]])
        
        # 可视化
        plot_trajectory(waypoints, T, coeffs)
