import numpy as np


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


def rmsprop_step(grad_x, moving_avg, alpha_0, beta, epsilon=1e-8):
    """
    Calculate step size for RMSProp

    :param x: Current parameter values
    :param grad_x: Gradient at x
    :param moving_avg: Moving average of squared gradients
    :param alpha_0: Base learning rate
    :param beta: Decay rate
    :param epsilon: Small constant to prevent division by zero
    :return: Parameter value with step size, and moving average.
    """
    grad_sq = grad_x ** 2
    moving_avg = beta * moving_avg + (1 - beta) * grad_sq

    moving_avg = np.array(moving_avg, dtype=np.float64)
    step_size = alpha_0 / (np.sqrt(moving_avg) + epsilon)

    return step_size, moving_avg


def heavy_ball_step(z, grad_x, alpha, beta):
    """
    Computes step size using Heavy Ball method.

    :param z: Momentum term.
    :param grad_x: Gradient at x.
    :param alpha: Learning rate.
    :param beta: Momentum coefficient (memory)
    :return: Updated Momentum term
    """
    z_new = beta * z + alpha * grad_x
    return z_new


def adam_step(m, v, grad_x, alpha, beta1, beta2, epsilon, t):
    """
    Computes step size using Adam algorithm.

    :param m: First momentum term
    :param v: Second momentum term
    :param grad_x: Gradient at x
    :param alpha: Learning rate
    :param beta1: Decay rate for first momentum
    :param beta2: Decay rate for second momentum
    :param epsilon: Small constant to prevent division by zero
    :param t: Current time step
    :return: Step size
    """
    m_new = beta1 * m + (1 - beta1) * grad_x
    v_new = beta2 * v + (1 - beta2) * (grad_x ** 2)

    # Bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    v_hat = np.array(v_hat, dtype=np.float64)

    step_size = alpha * (m_hat / (np.sqrt(v_hat) + epsilon))

    return m_new, v_new, step_size
