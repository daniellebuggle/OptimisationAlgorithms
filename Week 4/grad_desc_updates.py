from matplotlib import pyplot as plt
from derivative import *

"""
Part A
"""
def polyak_step_size(f_x, f_star, grad_f_x):
    """
    Calculate Polyak Step Size

    :param f_x: Current function value f(x).
    :param f_star: Optimal function value.
    :param grad_f_x: Array of partial derivatives.
    :return: Step size - alpha.
    """
    grad_norm_sq = np.dot(grad_f_x, grad_f_x)
    if grad_norm_sq == 0:  # Avoid division by zero
        return 0
    return (f_x - f_star) / grad_norm_sq


def rmsprop_update(x, grad_x, moving_avg, alpha_0, beta, epsilon=1e-8):
    """
    Perform update step using RMSProp

    :param x: Current parameter values
    :param grad_x: Gradient at x
    :param moving_avg: Moving average of squared gradients
    :param alpha_0: Base learning rate
    :param beta: Decay rate
    :param epsilon: Small constant to prevent division by zero
    :return: Parameter values with new step and the new moving average.
    """
    grad_sq = grad_x ** 2
    moving_avg = beta * moving_avg + (1 - beta) * grad_sq

    moving_avg = np.array(moving_avg, dtype=np.float64)
    step_size = alpha_0 / (np.sqrt(moving_avg) + epsilon)
    x_new = x - step_size * grad_x
    return x_new, moving_avg, step_size


def heavy_ball_step(x, z, grad_x, alpha, beta):
    """
    Computes next step using Heavy Ball method.

    :param x: Current parameter value.
    :param z: Momentum term.
    :param grad_x: Gradient at x.
    :param alpha: Learning rate.
    :param beta: Momentum coefficient (memory)
    :return: Updated x and momentum z.
    """
    z_new = beta * z + alpha * grad_x
    x_new = x - z_new
    return x_new, z_new


def adam_step(x, m, v, grad_x, alpha, beta1, beta2, epsilon, t):
    """
    Compute next step using Adam algorithm.

    :param x: Current parameter values
    :param m: First momentum term
    :param v: Second momentum term
    :param grad_x: Gradient at x
    :param alpha: Learning rate
    :param beta1: Decay rate for first momentum
    :param beta2: Decay rate for second momentum
    :param epsilon: Small constant to prevent division by zero
    :param t: Current time step
    :return: Updated x with new step, and updated momentum terms - m and v.
    """
    m_new = beta1 * m + (1 - beta1) * grad_x
    v_new = beta2 * v + (1 - beta2) * (grad_x ** 2)

    # Bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    v_hat = np.array(v_hat, dtype=np.float64)

    step_size = alpha * (m_hat / (np.sqrt(v_hat) + epsilon))
    x_new = x - step_size

    return x_new, m_new, v_new, step_size


def polyak_optimization(func, start, f_star, num_iters):
    """
    Run Polyak optimization and track function values over iterations.

    :param func: Function to perform Polyak on.
    :param start: Initial (x, y) values.
    :param f_star: Optimal function value.
    :param num_iters: Number of iterations
    :return: Function values at each iteration.
    """
    vars = [x, y]  # Variables used in differentiation
    x_t, y_t = start
    f_values = []
    step_sizes = []

    for t in range(1, num_iters + 1):
        values = {x: x_t, y: y_t}
        f_x = func.subs(values).evalf()
        grad_f_x = evaluate_derivative(vars, func, values)
        alpha_t = polyak_step_size(f_x, f_star, grad_f_x)
        step_sizes.append(alpha_t)
        x_t -= alpha_t * grad_f_x[0]
        y_t -= alpha_t * grad_f_x[1]
        f_values.append(f_x)

    return f_values, step_sizes


def rmsprop_optimization(func, start, alpha_0, beta, num_iters, epsilon=1e-8):
    """
    Run RMSProp optimization and track function values over iterations.

    :param func: Function to perform RMSProp on.
    :param start: Initial (x, y) values.
    :param alpha_0: Learning rate.
    :param beta: Decay rate.
    :param num_iters: Number of iterations.
    :param epsilon: Small constant to prevent division by zero.
    :return: Function values at each iteration.
    """
    vars = [x, y]
    x_t = np.array(start, dtype=np.float64)
    f_values = []
    E_g2 = np.zeros(2)
    step_sizes = []

    for _ in range(num_iters):
        values = {x: x_t[0], y: x_t[1]}
        f_x = func.subs(values).evalf()
        grad_f_x = evaluate_derivative(vars, func, values)

        x_t, E_g2, step_size = rmsprop_update(x_t, grad_f_x, E_g2, alpha_0, beta, epsilon)

        f_values.append(f_x)
        step_sizes.append(step_size)

    return f_values, step_sizes


def heavy_ball_optimization(func, start, alpha, beta, num_iters):
    """
    Run Heavy Ball optimization and track function values over iterations.

    :param func: Function to perform Heavy Ball on.
    :param start: Initial (x, y) values.
    :param alpha: Learning rate.
    :param beta: Momentum coefficient.
    :param num_iters: Number of iterations.
    :return:
    """
    vars = [x, y]
    x_t = np.array(start, dtype=np.float64)
    z_t = np.zeros(2)  # Initialize momentum term
    f_values = []
    step_sizes = []

    for _ in range(num_iters):
        values = {x: x_t[0], y: x_t[1]}
        f_x = func.subs(values).evalf()
        grad_f_x = evaluate_derivative(vars, func, values)

        x_t, z_t = heavy_ball_step(x_t, z_t, grad_f_x, alpha, beta)

        f_values.append(f_x)
        step_sizes.append(z_t)

    return f_values, step_sizes


def adam_optimization(func, start, alpha, beta1, beta2, num_iters, epsilon=1e-8):
    """
    Run Adam optimization and track function values over iterations.

    :param func: Function to perform Adam algorithm on.
    :param start: Initial (x, y) values.
    :param alpha: Learning rate.
    :param beta1: Decay rate for first momentum.
    :param beta2: Decay rate for second momentum.
    :param num_iters: Number of iterations.
    :param epsilon: Small constant for numerical stability.
    :return: Function values at each iteration.
    """
    vars = [x, y]
    x_t = np.array(start, dtype=np.float64)
    m_t = np.zeros(2)  # Initialize first moment
    v_t = np.zeros(2)  # Initialize second moment
    f_values = []
    step_sizes = []

    for t in range(1, num_iters + 1):
        values = {x: x_t[0], y: x_t[1]}
        f_x = func.subs(values).evalf()
        grad_f_x = evaluate_derivative(vars, func, values)

        x_t, m_t, v_t, step_size = adam_step(x_t, m_t, v_t, grad_f_x, alpha, beta1, beta2, epsilon, t)

        f_values.append(f_x)
        step_sizes.append(step_size)

    return f_values, step_sizes

"""
Part B
"""
x, y = sympy.symbols('x y', real=True)
f_expr = 5 * (x - 9) ** 4 + 6 * (y - 4) ** 2
f_star = 0

# Define parameter values to test
alpha_values = [0.00001, 0.0001, 0.01, 0.1, 0.5]
beta_values = [0.25, 0.9]
beta2_values = [0.25, 0.999]
start = (12, 1)
num_iters = 50

for alpha in alpha_values:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for i, beta in enumerate(beta_values):
        for j, beta2 in enumerate(beta2_values):
            ax = axes[i, j]

            polyak_values, polyak_step_sizes = polyak_optimization(f_expr, start, f_star, num_iters)
            rmsprop_values, rmsprop_step_sizes = rmsprop_optimization(f_expr, start, alpha, beta, num_iters)
            heavy_ball_values, heavyball_step_sizes = heavy_ball_optimization(f_expr, start, alpha, beta, num_iters)
            adam_values, adam_step_sizes = adam_optimization(f_expr, start, alpha, beta, beta2, num_iters)

            # Plot the optimisation results on the current subplot
            ax.plot(range(1, num_iters + 1), polyak_values, label="Polyak", linestyle="--")
            ax.plot(range(1, num_iters + 1), rmsprop_values, label="RMSProp", linestyle="-")
            ax.plot(range(1, num_iters + 1), heavy_ball_values, label="Heavy Ball", linestyle="-.")
            ax.plot(range(1, num_iters + 1), adam_values, label="Adam", linestyle=":")

            # Set labels and title for each subplot
            ax.set_xlabel("Iteration Number")
            ax.set_ylabel("Function Value")
            ax.set_title(f"alpha={alpha}, beta1={beta}, beta2={beta2}")
            ax.legend()
            ax.set_ylim(0, 500)
            ax.grid()

    fig.suptitle(f"Polynomial Function for alpha={alpha}")
    plt.tight_layout()
    plt.savefig(f"images/partB/func/func_values/Comparison_of_{alpha}_beta.png")

f_expr = sympy.Max(x - 9, 0) + 6 * sympy.Abs(y - 4)

for alpha in alpha_values:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for i, beta in enumerate(beta_values):
        for j, beta2 in enumerate(beta2_values):
            ax = axes[i, j]

            polyak_values, polyak_step_sizes = polyak_optimization(f_expr, start, f_star, num_iters)
            rmsprop_values, rmsprop_step_sizes = rmsprop_optimization(f_expr, start, alpha, beta, num_iters)
            heavy_ball_values, heavyball_step_sizes = heavy_ball_optimization(f_expr, start, alpha, beta, num_iters)
            adam_values, adam_step_sizes = adam_optimization(f_expr, start, alpha, beta, beta2, num_iters)

            # Plot the optimisation results on the current subplot
            ax.plot(range(1, num_iters + 1), polyak_values, label="Polyak", linestyle="--")
            ax.plot(range(1, num_iters + 1), rmsprop_values, label="RMSProp", linestyle="-")
            ax.plot(range(1, num_iters + 1), heavy_ball_values, label="Heavy Ball", linestyle="-.")
            ax.plot(range(1, num_iters + 1), adam_values, label="Adam", linestyle=":")

            # Set labels and title for each subplot
            ax.set_xlabel("Iteration Number")
            ax.set_ylabel("Function Value")
            ax.set_title(f"alpha={alpha}, beta1={beta}, beta2={beta2}")
            ax.legend()
            ax.grid()

    fig.suptitle(f"ReLu function for alpha={alpha}")
    plt.tight_layout()
    plt.savefig(f"images/partB/ReLu/ReLu_Comparison_of_{alpha}_beta.png")

"""
Part C
"""
f_expr = sympy.Max(x - 9, 0) + 6 * sympy.Abs(y - 4)
f_star = 0
num_iters = 50
start_values = [(-1, 1), (1, 1), (100, 1)]

for start in start_values:
    polyak_values, polyak_step_sizes = polyak_optimization(f_expr, start, f_star, num_iters)
    rmsprop_values, rmsprop_step_sizes = rmsprop_optimization(f_expr, start, 0.1, 0.9, num_iters)
    heavy_ball_values, heavyball_step_sizes = heavy_ball_optimization(f_expr, start, 0.01, 0.9, num_iters)
    adam_values, adam_step_sizes = adam_optimization(f_expr, start, 0.1, 0.9, 0.999, num_iters)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_iters + 1), polyak_values, label="Polyak", linestyle="--")
    plt.plot(range(1, num_iters + 1), rmsprop_values, label="RMSProp", linestyle="-")
    plt.plot(range(1, num_iters + 1), heavy_ball_values, label="Heavy Ball", linestyle="-.")
    plt.plot(range(1, num_iters + 1), adam_values, label="Adam", linestyle=":")

    plt.xlabel("Iteration Number")
    plt.ylabel("Function Value")
    plt.title(f"ReLu Function x={start[0]}")
    plt.legend()
    plt.grid()
    plt.savefig(f"images/partC/Comparison of ReLu Function x={start[0]}.png")
