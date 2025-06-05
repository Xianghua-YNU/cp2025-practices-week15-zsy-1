#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 学生代码模板

本项目要求实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1

学生姓名：[朱思宇]
学号：[20221170050]
完成日期：[2025.6.5]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')


def ode_system_shooting(t, y):
    """
    Define the ODE system for shooting method.
    
    Convert the second-order ODE u'' = -π(u+1)/4 into a first-order system:
    y1 = u, y2 = u'
    y1' = y2
    y2' = -π(y1+1)/4
    
    Args:
        t (float): Independent variable (time/position)
        y (array): State vector [y1, y2] where y1=u, y2=u'
    
    Returns:
        list: Derivatives [y1', y2']
    
    TODO: Implement the ODE system conversion
    Hint: Return [y[1], -np.pi*(y[0]+1)/4]
    """
    # TODO: Implement ODE system for shooting method
    # [STUDENT_CODE_HERE]
    y1, y2 = y
    dy1dt = y2
    dy2dt = -np.pi * (y1 + 1) / 4
    return [dy1dt, dy2dt]

def boundary_conditions_scipy(ya, yb):
    """
    Define boundary conditions for scipy.solve_bvp.
    
    Boundary conditions: u(0) = 1, u(1) = 1
    ya[0] should equal 1, yb[0] should equal 1
    
    Args:
        ya (array): Values at left boundary [u(0), u'(0)]
        yb (array): Values at right boundary [u(1), u'(1)]
    
    Returns:
        array: Boundary condition residuals
    
    TODO: Implement boundary conditions
    Hint: Return np.array([ya[0] - 1, yb[0] - 1])
    """
    # TODO: Implement boundary conditions for scipy.solve_bvp
    # [STUDENT_CODE_HERE]
    return np.array([ya[0] - 1, yb[0] - 1])


def ode_system_scipy(x, y):
    """
    Define the ODE system for scipy.solve_bvp.
    
    Note: scipy.solve_bvp uses (x, y) parameter order, different from odeint
    
    Args:
        x (float): Independent variable
        y (array): State vector [y1, y2]
    
    Returns:
        array: Derivatives as column vector
    
    TODO: Implement ODE system for scipy.solve_bvp
    Hint: Use np.vstack to return column vector
    """
    # TODO: Implement ODE system for scipy.solve_bvp
    # [STUDENT_CODE_HERE]
    y1, y2 = y
    dy1dx = y2
    dy2dx = -np.pi * (y1 + 1) / 4
    return np.vstack((dy1dx, dy2dx))

def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    """
    Solve boundary value problem using shooting method.
    
    Algorithm:
    1. Guess initial slope m1
    2. Solve IVP with initial conditions [u(0), m1]
    3. Check if u(1) matches boundary condition
    4. If not, adjust slope using secant method and repeat
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of discretization points
        max_iterations (int): Maximum iterations for shooting
        tolerance (float): Convergence tolerance
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    
    TODO: Implement shooting method algorithm
    Hint: Use secant method to adjust initial slope
    """
    # TODO: Validate input parameters
    
    # TODO: Extract boundary conditions and setup domain
    
    # TODO: Implement shooting method with secant method for slope adjustment
    
    # TODO: Return solution arrays
    # [STUDENT_CODE_HERE]
    u0, u1 = boundary_conditions
    x_start, x_end = x_span
    x = np.linspace(x_start, x_end, n_points)
    
    # 初始猜测斜率
    m0 = 0.0
    m1 = 1.0
    
    def find_correct_slope(m):
        y_initial = [u0, m]
        sol = odeint(ode_system_shooting, y_initial, x)
        return sol[-1, 0] - u1
    
    # 使用fsolve寻找使边界条件满足的初始斜率
    correct_slope = fsolve(find_correct_slope, m0)
    
    # 使用正确的初始斜率求解IVP
    y_initial = [u0, correct_slope[0]]
    sol = odeint(ode_system_shooting, y_initial, x)
    
    # 确保返回的解是正确的形状
    return x, sol[:, 0]

def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    Solve boundary value problem using scipy.solve_bvp.
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of initial mesh points
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    
    TODO: Implement scipy.solve_bvp wrapper
    Hint: Set up initial guess and call solve_bvp
    """
    # TODO: Setup initial mesh and guess
    
    # TODO: Call scipy.solve_bvp
    
    # TODO: Extract and return solution
    # [STUDENT_CODE_HERE]
    x = np.linspace(x_span[0], x_span[1], n_points)
    y_guess = np.ones((2, n_points))
    
    # 调用scipy.solve_bvp求解
    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x, y_guess)
    
    # 确保障收敛
    if not sol.success:
        raise RuntimeError("scipy.solve_bvp failed to converge")
    
    return sol.x, sol.y[0]

def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    Compare shooting method and scipy.solve_bvp, generate comparison plot.
    
    Args:
        x_span (tuple): Domain for the problem
        boundary_conditions (tuple): Boundary values (left, right)
        n_points (int): Number of points for plotting
    
    Returns:
        dict: Dictionary containing solutions and analysis
    
    TODO: Implement method comparison and visualization
    Hint: Call both methods, plot results, calculate differences
    """
    # TODO: Solve using both methods
    
    # TODO: Create comparison plot with English labels
    
    # TODO: Calculate and display differences
    
    # TODO: Return analysis results
    # [STUDENT_CODE_HERE]
    x_shoot, y_shoot = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
    
    # 使用scipy.solve_bvp求解
    x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points)
    
    # 确保两个解在相同的x点上进行比较
    x_common = np.linspace(x_span[0], x_span[1], n_points)
    y_shoot_interp = np.interp(x_common, x_shoot, y_shoot)
    y_scipy_interp = np.interp(x_common, x_scipy, y_scipy)
    
    # 绘制结果对比图
    plt.figure(figsize=(10, 6))
    plt.plot(x_common, y_shoot_interp, label='Shooting Method', linestyle='--')
    plt.plot(x_common, y_scipy_interp, label='scipy.solve_bvp', linestyle='-')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Comparison of Shooting Method and scipy.solve_bvp')
    plt.legend()
    plt.grid(True)
    
    # 保存图像为PNG格式
    plt.savefig('comparison_plot.png')
    plt.show()
    
    # 计算两种方法结果的最大差异
    max_difference = np.max(np.abs(y_shoot_interp - y_scipy_interp))
    print(f"Maximum difference between methods: {max_difference}")
    
    return {
        'shooting_solution': (x_common, y_shoot_interp),
        'scipy_solution': (x_common, y_scipy_interp),
        'max_difference': max_difference
    }


# Test functions for development and debugging
def test_ode_system():
    """
    Test the ODE system implementation.
    """
    print("Testing ODE system...")
    try:
        # Test point
        t_test = 0.5
        y_test = np.array([1.0, 0.5])
        
        # Test shooting method ODE system
        dydt = ode_system_shooting(t_test, y_test)
        print(f"ODE system (shooting): dydt = {dydt}")
        
        # Test scipy ODE system
        dydt_scipy = ode_system_scipy(t_test, y_test)
        print(f"ODE system (scipy): dydt = {dydt_scipy}")
        
    except NotImplementedError:
        print("ODE system functions not yet implemented.")


def test_boundary_conditions():
    """
    Test the boundary conditions implementation.
    """
    print("Testing boundary conditions...")
    try:
        ya = np.array([1.0, 0.5])  # Left boundary
        yb = np.array([1.0, -0.3])  # Right boundary
        
        bc_residual = boundary_conditions_scipy(ya, yb)
        print(f"Boundary condition residuals: {bc_residual}")
        
    except NotImplementedError:
        print("Boundary conditions function not yet implemented.")


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题")
    print("=" * 50)
    
    # Run basic tests
    test_ode_system()
    test_boundary_conditions()
    
    # Try to run comparison (will fail until functions are implemented)
    try:
        print("\nTesting method comparison...")
        results = compare_methods_and_plot()
        print("Method comparison completed successfully!")
    except NotImplementedError as e:
        print(f"Method comparison not yet implemented: {e}")
    
    print("\n请实现所有标记为 TODO 的函数以完成项目。")
