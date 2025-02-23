import numpy as np
from matplotlib import pyplot as plt

from derivative import *


def polyak_step_size(f_x, f_star, grad_f_x):
    """
    Compute the Polyak step size.

    Parameters:
    f_x (float): Current function value f(x).
    f_star (float): Optimal function value (or best known value).
    grad_f_x (numpy array): Gradient of f at x.

    Returns:
    float: Step size alpha.
    """
    grad_norm_sq = np.dot(grad_f_x, grad_f_x)  # Equivalent to sum of squared derivatives
    if grad_norm_sq == 0:  # Avoid division by zero
        return 0
    return (f_x - f_star) / grad_norm_sq


def rmsprop_update(x, grad_x, E_g2, alpha_0, beta, epsilon=1e-8):
    """
    Perform one update step using RMSProp.

    Parameters:
    x (numpy array): Current parameter values.
    grad_x (numpy array): Gradient at x.
    E_g2 (numpy array): Moving average of squared gradients.
    alpha_0 (float): Base learning rate.
    beta (float): Decay rate for moving average.
    epsilon (float): Small constant to prevent division by zero.

    Returns:
    numpy array: Updated parameter values.
    numpy array: Updated moving average E_g2.
    """
    grad_sq = grad_x ** 2  # Compute squared gradient
    E_g2 = beta * E_g2 + (1 - beta) * grad_sq  # Update moving average

    E_g2 = np.array(E_g2, dtype=np.float64)

    step_size = alpha_0 / (np.sqrt(E_g2) + epsilon)  # Compute adaptive step size
    x_new = x - step_size * grad_x  # Update parameters

    return x_new, E_g2


def heavy_ball_step(x, z, grad_x, alpha, beta):
    """
    Computes the next step using the Heavy Ball method.

    Parameters:
    x (numpy array): Current parameter values.
    z (numpy array): Current momentum term.
    grad_x (numpy array): Gradient at x.
    alpha (float): Learning rate.
    beta (float): Momentum coefficient.

    Returns:
    tuple: (Updated x, Updated momentum z)
    """
    z_new = beta * z + alpha * grad_x  # Compute new momentum
    x_new = x - z_new  # Update parameter

    return x_new, z_new


def adam_step(x, m, v, grad_x, alpha, beta1, beta2, epsilon, t):
    """
    Computes the next step using the Adam optimization algorithm.

    Parameters:
    x (numpy array): Current parameter values.
    m (numpy array): First moment (momentum term).
    v (numpy array): Second moment (RMSProp-like term).
    grad_x (numpy array): Gradient at x.
    alpha (float): Learning rate.
    beta1 (float): Exponential decay rate for first moment.
    beta2 (float): Exponential decay rate for second moment.
    epsilon (float): Small constant for numerical stability.
    t (int): Current time step (used for bias correction).

    Returns:
    tuple: (Updated x, Updated m, Updated v)
    """
    m_new = beta1 * m + (1 - beta1) * grad_x  # Update first moment
    v_new = beta2 * v + (1 - beta2) * (grad_x ** 2)  # Update second moment

    # Bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    v_hat = np.array(v_hat, dtype=np.float64)
    # Compute step update
    step_size = alpha * (m_hat / (np.sqrt(v_hat) + epsilon))
    x_new = x - step_size  # Update parameters

    return x_new, m_new, v_new


def polyak_optimization(func, start, f_star, num_iters):
    """
    Run Polyak optimization and track function values over iterations.

    Parameters:
    start (tuple): Initial (x, y) values.
    f_star (float): Optimal function value.
    num_iters (int): Number of iterations.

    Returns:
    list: Function values at each iteration.
    """
    vars = [x, y]  # Variables used in differentiation
    x_t, y_t = start  # Initialize variables
    f_values = []

    for t in range(1, num_iters + 1):
        values = {x: x_t, y: y_t}  # Create dictionary of variable values
        f_x = func.subs(values).evalf()  # Compute function value
        grad_f_x = evaluate_derivative(vars, func, values)  # Compute gradient
        alpha_t = polyak_step_size(f_x, f_star, grad_f_x)  # Compute step size

        # Update parameters
        x_t -= alpha_t * grad_f_x[0]
        y_t -= alpha_t * grad_f_x[1]

        f_values.append(f_x)  # Store function value

    return f_values


def rmsprop_optimization(func, start, alpha_0, beta, num_iters, epsilon=1e-8):
    """
    Run RMSProp optimization and track function values over iterations.

    Parameters:
    start (tuple): Initial (x, y) values.
    num_iters (int): Number of iterations.
    alpha_0 (float): Learning rate.
    beta (float): Decay rate for squared gradient moving average.
    epsilon (float): Small constant to prevent division by zero.

    Returns:
    list: Function values at each iteration.
    """
    vars = [x, y]
    x_t = np.array(start, dtype=np.float64)
    f_values = []
    E_g2 = np.zeros(2)  # Initialize moving average

    for _ in range(num_iters):
        values = {x: x_t[0], y: x_t[1]}
        f_x = func.subs(values).evalf()
        grad_f_x = evaluate_derivative(vars, func, values)

        x_t, E_g2 = rmsprop_update(x_t, grad_f_x, E_g2, alpha_0, beta, epsilon)

        f_values.append(f_x)

    return f_values


def heavy_ball_optimization(func, start, alpha, beta, num_iters):
    """
    Run Heavy Ball optimization and track function values over iterations.

    Parameters:
    start (tuple): Initial (x, y) values.
    num_iters (int): Number of iterations.
    alpha (float): Learning rate.
    beta (float): Momentum coefficient.

    Returns:
    list: Function values at each iteration.
    """
    vars = [x, y]
    x_t = np.array(start, dtype=np.float64)
    z_t = np.zeros(2)  # Initialize momentum term
    f_values = []

    for _ in range(num_iters):
        values = {x: x_t[0], y: x_t[1]}
        f_x = func.subs(values).evalf()
        grad_f_x = evaluate_derivative(vars, func, values)

        x_t, z_t = heavy_ball_step(x_t, z_t, grad_f_x, alpha, beta)

        f_values.append(f_x)

    return f_values


def adam_optimization(func, start, alpha, beta1, beta2, num_iters, epsilon=1e-8):
    """
    Run Adam optimization and track function values over iterations.

    Parameters:
    start (tuple): Initial (x, y) values.
    num_iters (int): Number of iterations.
    alpha (float): Learning rate.
    beta1 (float): Decay rate for first moment.
    beta2 (float): Decay rate for second moment.
    epsilon (float): Small constant for numerical stability.

    Returns:
    list: Function values at each iteration.
    """
    vars = [x, y]
    x_t = np.array(start, dtype=np.float64)
    m_t = np.zeros(2)  # Initialize first moment
    v_t = np.zeros(2)  # Initialize second moment
    f_values = []

    for t in range(1, num_iters + 1):
        values = {x: x_t[0], y: x_t[1]}
        f_x = func.subs(values).evalf()
        grad_f_x = evaluate_derivative(vars, func, values)

        x_t, m_t, v_t = adam_step(x_t, m_t, v_t, grad_f_x, alpha, beta1, beta2, epsilon, t)

        f_values.append(f_x)

    return f_values


# Initialize moving average of squared gradients
E_g2 = 0  # This will be updated across function calls
x, y = sympy.symbols('x y', real=True)
# Define the function f(x, y)
f_expr = 5 * (x - 9) ** 4 + 6 * (y - 4) ** 2
# Run optimization with different initial points
f_star = 0  # Assuming the optimal function value is known

# Define parameter values to test
alpha_values = [0.00001, 0.0001, 0.01, 0.1, 0.5]  # Learning rates
beta_values = [0.25, 0.9]  # Momentum/decay rates where applicable
start = (12,1)  # Initial starting point
num_iters = 50  # Number of iterations

plt.figure(figsize=(10, 6))

for alpha in alpha_values:
    for beta in beta_values:
        polyak_values = polyak_optimization(f_expr, start, f_star, num_iters)
        rmsprop_values = rmsprop_optimization(f_expr, start, alpha, beta, num_iters)
        heavy_ball_values = heavy_ball_optimization(f_expr, start, alpha, beta, num_iters)
        adam_values = adam_optimization(f_expr, start, alpha, beta, 0.999, num_iters)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_iters + 1), polyak_values, label="Polyak", linestyle="--")
        plt.plot(range(1, num_iters + 1), rmsprop_values, label="RMSProp", linestyle="-")
        plt.plot(range(1, num_iters + 1), heavy_ball_values, label="Heavy Ball", linestyle="-.")
        plt.plot(range(1, num_iters + 1), adam_values, label="Adam", linestyle=":")

        plt.xlabel("Iteration Number")
        plt.ylabel("Function Value")
        plt.title(f"Comparison (alpha={alpha}, beta={beta})")
        plt.legend()
        plt.grid()
        plt.ylim(0, 500)
        plt.savefig(f"images/partB/func/Comparison of f(x) (alpha={alpha}, beta={beta}).png")


# Define symbols
x, y = sympy.symbols('x y', real=True)

# Define the new function f1
f_expr = sympy.Max(x - 9, 0) + 6 * sympy.Abs(y - 4)


f_star = 0  # Assuming the optimal function value is known

# Define parameter values to test
alpha_values = [0.00001, 0.0001, 0.01, 0.1, 0.5]  # Learning rates
beta_values = [0.25, 0.9]  # Momentum/decay rates where applicable
start = (12,1)  # Initial starting point
num_iters = 50  # Number of iterations

plt.figure(figsize=(10, 6))

for alpha in alpha_values:
    for beta in beta_values:
        polyak_values = polyak_optimization(f_expr, start, f_star, num_iters)
        rmsprop_values = rmsprop_optimization(f_expr, start, alpha, beta, num_iters)
        heavy_ball_values = heavy_ball_optimization(f_expr, start, alpha, beta, num_iters)
        adam_values = adam_optimization(f_expr, start, alpha, beta, 0.999, num_iters)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_iters + 1), polyak_values, label="Polyak", linestyle="--")
        plt.plot(range(1, num_iters + 1), rmsprop_values, label="RMSProp", linestyle="-")
        plt.plot(range(1, num_iters + 1), heavy_ball_values, label="Heavy Ball", linestyle="-.")
        plt.plot(range(1, num_iters + 1), adam_values, label="Adam", linestyle=":")

        plt.xlabel("Iteration Number")
        plt.ylabel("Function Value")
        plt.title(f"Comparison (alpha={alpha}, beta={beta})")
        plt.legend()
        plt.grid()
        plt.savefig(f"images/partB/ReLu/Comparison of Max (alpha={alpha}, beta={beta}).png")


"""
Part C
"""

x, y = sympy.symbols('x y', real=True)

# Define the new function f1
f_expr = sympy.Max(x - 9, 0) + 6 * sympy.Abs(y - 4)


f_star = 0  # Assuming the optimal function value is known

# Define parameter values to test
alpha_values = [0.1, 0.5]  # Learning rates
beta_values = [0.25, 0.9]  # Momentum/decay rates where applicable
start = (100,1)  # Initial starting point
num_iters = 500 # Number of iterations

plt.figure(figsize=(10, 6))

for alpha in alpha_values:
    for beta in beta_values:
        polyak_values = polyak_optimization(f_expr, start, f_star, num_iters)
        rmsprop_values = rmsprop_optimization(f_expr, start, alpha, beta, num_iters)
        heavy_ball_values = heavy_ball_optimization(f_expr, start, alpha, beta, num_iters)
        adam_values = adam_optimization(f_expr, start, alpha, beta, 0.999, num_iters)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_iters + 1), polyak_values, label="Polyak", linestyle="--")
        plt.plot(range(1, num_iters + 1), rmsprop_values, label="RMSProp", linestyle="-")
        plt.plot(range(1, num_iters + 1), heavy_ball_values, label="Heavy Ball", linestyle="-.")
        plt.plot(range(1, num_iters + 1), adam_values, label="Adam", linestyle=":")

        plt.xlabel("Iteration Number")
        plt.ylabel("Function Value")
        plt.title(f"Comparison (alpha={alpha}, beta={beta})")
        plt.legend()
        plt.grid()
        plt.savefig(f"Comparison of ReLu x=100 (alpha={alpha}, beta={beta}).png")