"""
学生模板：平方反比引力场中的运动
文件：inverse_square_law_motion_student.py
作者：[朱思宇]
日期：[2025.6.5]

重要：函数名称、参数名称和返回值的结构必须与参考答案保持一致！
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 常量 (如果需要，学生可以自行定义或从参数传入)
# 例如：GM = 1.0 # 引力常数 * 中心天体质量
GM_val = 1.0

def derivatives(t, state_vector, gm_val):
    """
    计算状态向量 [x, y, vx, vy] 的导数。

    运动方程（直角坐标系）:
    dx/dt = vx
    dy/dt = vy
    dvx/dt = -GM * x / r^3
    dvy/dt = -GM * y / r^3
    其中 r = sqrt(x^2 + y^2)。

    参数:
        t (float): 当前时间 (solve_ivp 需要，但在此自治系统中不直接使用)。
        state_vector (np.ndarray): 一维数组 [x, y, vx, vy]，表示当前状态。
        gm_val (float): 引力常数 G 与中心天体质量 M 的乘积。

    返回:
        np.ndarray: 一维数组，包含导数 [dx/dt, dy/dt, dvx/dt, dvy/dt]。
    
    实现提示:
    1. 从 state_vector 中解包出 x, y, vx, vy。
    2. 计算 r_cubed = (x**2 + y**2)**1.5。
    3. 注意处理 r_cubed 接近零的特殊情况（例如，如果 r 非常小，可以设定一个阈值避免除以零）。
    4. 计算加速度 ax 和 ay。
    5. 返回 [vx, vy, ax, ay]。
    """
    # TODO: 学生在此处实现代码
    x, y, vx, vy = state_vector
    r = np.sqrt(x**2 + y**2)
    r_cubed = (x**2 + y**2 + 1e-6)**1.5  # 避免除零错误的小偏置量
    ax = -gm_val * x / r_cubed
    ay = -gm_val * y / r_cubed
    return np.array([vx, vy, ax, ay])

def solve_orbit(initial_conditions, t_span, t_eval, gm_val):
    """
    使用 scipy.integrate.solve_ivp 求解轨道运动问题。

    参数:
        initial_conditions (list or np.ndarray): 初始状态 [x0, y0, vx0, vy0]。
        t_span (tuple): 积分时间区间 (t_start, t_end)。
        t_eval (np.ndarray): 需要存储解的时间点数组。
        gm_val (float): GM 值 (引力常数 * 中心天体质量)。

    返回:
        scipy.integrate.OdeSolution: solve_ivp 返回的解对象。
                                     可以通过 sol.y 访问解的数组，sol.t 访问时间点。
    
    实现提示:
    1. 调用 solve_ivp 函数。
    2. `fun` 参数应为你的 `derivatives` 函数。
    3. `args` 参数应为一个元组，包含传递给 `derivatives` 函数的额外参数 (gm_val,)。
    4. 可以选择合适的数值方法 (method)，如 'RK45' (默认) 或 'DOP853'。
    5. 设置合理的相对容差 (rtol) 和绝对容差 (atol) 以保证精度，例如 rtol=1e-7, atol=1e-9。
    """
    # TODO: 学生在此处实现代码
    sol = solve_ivp(
        fun=derivatives,
        t_span=t_span,
        y0=initial_conditions,
        t_eval=t_eval,
        args=(gm_val,),
        method='RK45',
        rtol=1e-7,
        atol=1e-9
    )
    return sol

def calculate_energy(state_vector, gm_val, m=1.0):
    """
    计算质点的（比）机械能。
    （比）能量 E/m = 0.5 * v^2 - GM/r

    参数:
        state_vector (np.ndarray): 二维数组，每行是 [x, y, vx, vy]，或单个状态的一维数组。
        gm_val (float): GM 值。
        m (float, optional): 运动质点的质量。默认为 1.0，此时计算的是比能 (E/m)。

    返回:
        np.ndarray or float: （比）机械能。

    实现提示:
    1. 处理 state_vector 可能是一维（单个状态）或二维（多个状态的时间序列）的情况。
    2. 从 state_vector 中提取 x, y, vx, vy。
    3. 计算距离 r = np.sqrt(x**2 + y**2)。注意避免 r=0 导致除以零的错误。
    4. 计算速度的平方 v_squared = vx**2 + vy**2。
    5. 计算比动能 kinetic_energy_per_m = 0.5 * v_squared。
    6. 计算比势能 potential_energy_per_m = -gm_val / r (注意处理 r=0 的情况)。
    7. 比机械能 specific_energy = kinetic_energy_per_m + potential_energy_per_m。
    8. 如果需要总能量，则乘以质量 m。
    """
    # TODO: 学生在此处实现代码
    if len(state_vector.shape) == 1:  # 单个状态
        x, y, vx, vy = state_vector
        r = np.sqrt(x**2 + y**2)
        kinetic_energy = 0.5 * (vx**2 + vy**2)
        potential_energy = -gm_val / r
        return (kinetic_energy + potential_energy)
    else:  # 多个状态
        x = state_vector[:, 0]
        y = state_vector[:, 1]
        vx = state_vector[:, 2]
        vy = state_vector[:, 3]
        r = np.sqrt(x**2 + y**2)
        kinetic_energy = 0.5 * (vx**2 + vy**2)
        potential_energy = -gm_val / r
        return kinetic_energy + potential_energy

def calculate_angular_momentum(state_vector, m=1.0):
    """
    计算质点的（比）角动量 (z分量)。
    （比）角动量 Lz/m = x*vy - y*vx

    参数:
        state_vector (np.ndarray): 二维数组，每行是 [x, y, vx, vy]，或单个状态的一维数组。
        m (float, optional): 运动质点的质量。默认为 1.0，此时计算的是比角动量 (Lz/m)。

    返回:
        np.ndarray or float: （比）角动量。

    实现提示:
    1. 处理 state_vector 可能是一维或二维的情况。
    2. 从 state_vector 中提取 x, y, vx, vy。
    3. 计算比角动量 specific_Lz = x * vy - y * vx。
    4. 如果需要总角动量，则乘以质量 m。
    """
    # TODO: 学生在此处实现代码
    if len(state_vector.shape) == 1:
        x, y, vx, vy = state_vector
        return x * vy - y * vx
    else:
        x = state_vector[:, 0]
        y = state_vector[:, 1]
        vx = state_vector[:, 2]
        vy = state_vector[:, 3]
        return x * vy - y * vx


if __name__ == "__main__":
    # --- 学生可以在此区域编写测试代码或进行实验 ---
    print("平方反比引力场中的运动 - 学生模板")

    # 任务1：实现函数并通过基础测试 (此处不设测试，依赖 tests 文件)
    GM_val = 1.0
    t_start = 0
    t_end = 20
    t_eval = np.linspace(t_start, t_end, 500)
    # 椭圆轨道（E < 0）
    initial_ellipse = [1.0, 0.0, 0.0, 0.8]
    sol_ellipse = solve_orbit(initial_ellipse, (t_start, t_end), t_eval, GM_val)
    x_ellipse = sol_ellipse.y[0]
    y_ellipse = sol_ellipse.y[1]

    # 抛物线轨道（E = 0）
    initial_parabola = [1.0, 0.0, 0.0, np.sqrt(2)]
    sol_parabola = solve_orbit(initial_parabola, (t_start, t_end), t_eval, GM_val)
    x_parabola = sol_parabola.y[0]
    y_parabola = sol_parabola.y[1]

    # 双曲线轨道（E > 0）
    initial_hyperbola = [1.0, 0.0, 0.0, 1.15]
    sol_hyperbola = solve_orbit(initial_hyperbola, (t_start, t_end), t_eval, GM_val)
    x_hyperbola = sol_hyperbola.y[0]
    y_hyperbola = sol_hyperbola.y[1]

    # 绘图
    plt.figure(figsize=(10, 8))
    plt.plot(x_ellipse, y_ellipse, label='椭圆轨道 (E < 0)')
    plt.plot(x_parabola, y_parabola, label='抛物线轨道 (E = 0)')
    plt.plot(x_hyperbola, y_hyperbola, label='双曲线轨道 (E > 0)')
    plt.plot(0, 0, 'ro', markersize=8, label='中心天体')
    plt.title('Orbits with different total energies')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('orbits_E.png')  # 保存为PNG文件
    plt.show()

    # 任务2：不同总能量下的轨道绘制
    GM_val = 1.0
    E = -0.5  # 固定总能量
    r_p = 0.5  # 近心点距离
    v_p = np.sqrt(2*(E + GM_val / r_p))
    L_z_values = [r_p * v_p * 0.8, r_p * v_p, r_p * v_p * 1.2]  # 不同角动量值

    plt.figure(figsize=(10, 8))
    for L_z in L_z_values:
        # 根据能量和角动量计算初始速度分量
        v_p_calculated = L_z / r_p
        initial_conditions = [r_p, 0.0, 0.0, v_p_calculated]
        sol = solve_orbit(initial_conditions, (t_start, t_end), t_eval, GM_val)
        x = sol.y[0]
        y = sol.y[1]
        plt.plot(x, y, label=f'角动量 L_z = {L_z:.2f}')

    plt.plot(0, 0, 'ro', markersize=8, label='中心天体')
    plt.title('Elliptical orbits with different angular momenta')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('orbits_Lz.png')  # 保存为PNG文件
    plt.show()
    # 示例：设置椭圆轨道初始条件 (学生需要根据物理意义自行调整或计算得到)
    # GM_val_demo = 1.0
    # ic_ellipse_demo = [1.0, 0.0, 0.0, 0.8]
    # t_start_demo = 0
    # t_end_demo = 20
    # t_eval_demo = np.linspace(t_start_demo, t_end_demo, 500)
    
    # try:
    #     sol_ellipse = solve_orbit(ic_ellipse_demo, (t_start_demo, t_end_demo), t_eval_demo, gm_val=GM_val_demo)
    #     x_ellipse, y_ellipse = sol_ellipse.y[0], sol_ellipse.y[1]
        
    #     # 计算能量和角动量 (假设学生已实现)
    #     # energy = calculate_energy(sol_ellipse.y.T, GM_val_demo)
    #     # angular_momentum = calculate_angular_momentum(sol_ellipse.y.T)
    #     # print(f"Ellipse Demo: Initial Energy approx {energy[0]:.3f}")

    #     plt.figure(figsize=(8, 6))
    #     plt.plot(x_ellipse, y_ellipse, label='椭圆轨道 (示例)')
    #     plt.plot(0, 0, 'ko', markersize=8, label='中心天体')
    #     plt.title('轨道运动示例')
    #     plt.xlabel('x 坐标')
    #     plt.ylabel('y 坐标')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.axis('equal')
    #     plt.show()
    # except NotImplementedError:
    #     print("请先实现 solve_orbit, calculate_energy, calculate_angular_momentum 函数。")
    # except Exception as e:
    #     print(f"运行示例时发生错误: {e}")

    # 学生需要根据“项目说明.md”完成以下任务：
    # 1. 实现 `derivatives`, `solve_orbit`, `calculate_energy`, `calculate_angular_momentum` 函数。
    # 2. 针对 E > 0, E = 0, E < 0 三种情况设置初始条件，求解并绘制轨道。
    # 3. 针对 E < 0 且固定时，改变角动量，求解并绘制轨道。
    # 4. (可选) 进行坐标转换和对称性分析。

    print("\n请参照 '项目说明.md' 完成各项任务。")
    print("使用 'tests/test_inverse_square_law_motion.py' 文件来测试你的代码实现。")

    pass # 学生代码的主要部分应在函数内实现
