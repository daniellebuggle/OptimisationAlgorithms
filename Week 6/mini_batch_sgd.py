from matplotlib import pyplot as plt
from loss_functions import *
from step_sizes import *
import sympy as sp

F_STAR = 0
x1, x2 = sp.symbols('x1 x2')
x0 = [3, 3]  # Initial point
X = generate_trainingdata(m=25)
"""
Part A
"""


def approx_gradient(f, x, minibatch):
    grad_x1_total = 0
    grad_x2_total = 0
    for w in minibatch:
        # Compute symbolic derivatives
        grad_x1, grad_x2 = compute_symbolic_derivative([x1, x2], [w])

        # Compute exact gradient at this point
        grad_x1_value, grad_x2_value = numerical_derivative(x[0], x[1], grad_x1, grad_x2)

        grad_x1_total += grad_x1_value
        grad_x2_total += grad_x2_value

    # Average the gradients over the mini-batch
    grad_x1_avg = grad_x1_total / len(minibatch)
    grad_x2_avg = grad_x2_total / len(minibatch)

    return np.array([grad_x1_avg, grad_x2_avg])


"""
(i)
"""


def mini_batch_SGD(X, initial_x0, num_iters, batch_size, step_size_func, alpha0, beta0, beta1, epsilon=1e-8):
    m, n = X.shape
    theta = np.array(initial_x0)

    moving_avg = np.zeros_like(theta)  # RMSPROP
    z_t = np.zeros_like(theta)  # Heavy Ball
    m_t = np.zeros_like(theta)  # Adam - First momentum term
    v_t = np.zeros_like(theta)  # Adam - Second momentum term
    t = 1  # Adam - time step

    theta_history = [theta.copy()]
    f_values = [f(theta, X)]

    epoch_count = 0

    for iter_num in range(num_iters):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]

        epoch_count += 1
        for i in range(0, m, batch_size):
            minibatch = X_shuffled[i:i + batch_size]

            # Calculate the approximate gradient for the current mini-batch
            grad = approx_gradient(f, theta, minibatch)
            if step_size_func == polyak_step_size:
                f_x = f(theta, minibatch)
                alpha = step_size_func(f_x, F_STAR, grad)
            elif step_size_func == rmsprop_step:
                alpha, average = step_size_func(grad, moving_avg, alpha0, beta0)
                moving_avg = average
            elif step_size_func == heavy_ball_step:
                z = heavy_ball_step(z_t, grad, alpha0, beta0)
                alpha = z
            elif step_size_func == adam_step:
                m_t, v_t, alpha = step_size_func(m_t, v_t, grad, alpha0, beta0, beta1, epsilon, t)
                t += 1
            else:
                alpha = alpha0

            theta = theta - (alpha * grad)
            theta_history.append(theta.copy())
            f_values.append(f(theta, X))

    return np.array(theta_history), np.array(f_values), epoch_count


"""
(iii)
"""


def symbolic_f(x, minibatch):
    y = 0
    count = 0
    for w in minibatch:
        z1 = x[0] - w[0] - 1  # x[0] - w[0] - 1
        z2 = x[1] - w[1] - 1  # x[1] - w[1] - 1
        term1 = 37 * (z1 ** 2 + z2 ** 2)
        term2 = (z1 + 5) ** 2 + (z2 + 5) ** 2
        y += sp.Min(term1, term2)
        count += 1
    return y / count


def compute_symbolic_derivative(x, minibatch):
    grad_x1 = sp.diff(symbolic_f(x, minibatch), x[0])
    grad_x2 = sp.diff(symbolic_f(x, minibatch), x[1])
    return grad_x1, grad_x2


def numerical_derivative(x1_val, x2_val, grad_x1, grad_x2):
    grad_x1_value = grad_x1.subs({x1: x1_val, x2: x2_val})
    grad_x2_value = grad_x2.subs({x1: x1_val, x2: x2_val})
    return grad_x1_value, grad_x2_value


"""
(ii)
"""


def a_part_two():
    training_data = generate_trainingdata(m=25)
    X = training_data

    x1_range = np.linspace(-1.5, 3, 100)
    x2_range = np.linspace(-1.5, 3, 100)
    x1, x2 = np.meshgrid(x1_range, x2_range)

    f_values = np.zeros_like(x1)

    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            x = np.array([x1[i, j], x2[i, j]])
            f_values[i, j] = f(x, X)

    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(x1, x2, f_values, cmap='viridis')
    ax.set_title('Wireframe Plot of f(x, T)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x, T)')

    ax2 = fig.add_subplot(122)
    contour = ax2.contour(x1, x2, f_values, 20, cmap='viridis')
    ax2.set_title('Contour Plot of f(x, T)')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    fig.colorbar(contour)

    plt.tight_layout()
    plt.savefig("images/wireframe_contour_all_N.png")


"""
Part B
"""

"""
(i)
"""


def gradient_descent(x0, minibatch, alpha, num_iterations):
    X = np.array([x0])
    symbolic_grad = compute_symbolic_derivative([x1, x2], minibatch)

    x1_value = X[-1][0]
    x2_value = X[-1][1]

    for i in range(num_iterations):
        grad_x1_value, grad_x2_value = numerical_derivative(x1_value, x2_value, symbolic_grad[0], symbolic_grad[1])
        step = alpha * np.array([grad_x1_value, grad_x2_value])
        x0 = X[-1] - step
        X = np.append(X, [x0], axis=0)

        x1_value = x0[0]
        x2_value = x0[1]

    return X


def part_b_one():
    alpha = 0.01
    num_iterations = 100

    # Run gradient descent
    X_history = gradient_descent(x0, X, alpha, num_iterations)
    x1_range = np.linspace(-4, 4, 100)
    x2_range = np.linspace(-4, 4, 100)
    x1_feature, x2_feature = np.meshgrid(x1_range, x2_range)

    f_values = np.zeros_like(x1_feature)

    for i in range(x1_feature.shape[0]):
        for j in range(x1_feature.shape[1]):
            x = np.array([x1_feature[i, j], x2_feature[i, j]])  # Current point (x1, x2)
            f_values[i, j] = f(x, X)  # Calculate the loss for this point

    # Create the contour plot
    plt.figure(figsize=(8, 6))

    # Plot the contour of the loss function
    contour = plt.contour(x1, x2, f_values, 20, cmap='viridis')
    plt.title(f'Contour Plot with Gradient Descent Path with alpha={alpha}')
    plt.xlabel('x1')
    plt.ylabel('x2')

    # Plot the path of gradient descent (as a line)
    plt.plot(X_history[:, 0], X_history[:, 1], 'ro-', label='Gradient Descent Path')
    plt.colorbar(contour)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/grad_descent_alpha_{alpha}.png")


"""
(ii)
"""


def part_b_two():
    alpha = 0.01
    num_iters = 100
    batch_size = 5

    # Run mini-batch SGD with constant step size
    theta_final, func_values, epochs = mini_batch_SGD(X, x0, num_iters, batch_size, None, alpha, None, None)

    x1_range = np.linspace(-4, 4, 100)
    x2_range = np.linspace(-4, 4, 100)
    x1_feature, x2_feature = np.meshgrid(x1_range, x2_range)

    f_values = np.zeros_like(x1_feature)

    for i in range(x1_feature.shape[0]):
        for j in range(x1_feature.shape[1]):
            x = np.array([x1_feature[i, j], x2_feature[i, j]])  # Current point (x1, x2)
            f_values[i, j] = f(x, X)  # Calculate the loss for this point

    # Create the contour plot
    plt.figure(figsize=(8, 6))

    # Plot the contour of the loss function
    contour = plt.contour(x1_feature, x2_feature, f_values, 20, cmap='viridis')
    plt.title(f'Contour Plot with Mini-Batch SGD with alpha={alpha}')
    plt.xlabel('x1')
    plt.ylabel('x2')

    theta_final = np.array(theta_final)
    # Plot the path of gradient descent (as a line)
    plt.plot(theta_final[:, 0], theta_final[:, 1], 'ro-', label='Mini-Batch SGD Path')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.colorbar(contour)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/mini_batch_sgd_alpha_{alpha}.png")


"""
Plot epochs vs function value
"""


def part_b_epochs():
    alpha = 0.01
    num_iters = 25
    batch_size = 5

    num_runs = 5  # Run SGD multiple times
    plt.figure(figsize=(8, 6))

    for run in range(num_runs):
        theta_final, f_values, num_epochs = mini_batch_SGD(X, x0, num_iters, batch_size, None, alpha, None, None)
        f_values_per_epoch = f_values[batch_size - 1::batch_size]  # Every batch_size-th value corresponds to an epoch

        # Plot function values at each epoch
        plt.plot(range(1, num_epochs + 1), f_values_per_epoch, label=f'Run {run + 1}')

    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Function Value f(θ)')
    plt.title('Function Value vs. Epochs for Mini-Batch SGD')
    plt.legend()
    plt.grid()
    plt.savefig(f"images/runs_epoch_vs_function_{alpha}_log.png")


"""
(iii)
"""


def part_b_three():
    alpha = 0.01
    num_iters = 100
    batch_sizes = [5, 15, 20, 25]

    x1_range = np.linspace(-4, 4, 100)
    x2_range = np.linspace(-4, 4, 100)
    x1_feature, x2_feature = np.meshgrid(x1_range, x2_range)

    f_values = np.zeros_like(x1_feature)

    for i in range(x1_feature.shape[0]):
        for j in range(x1_feature.shape[1]):
            x = np.array([x1_feature[i, j], x2_feature[i, j]])  # Current point (x1, x2)
            f_values[i, j] = f(x, X)  # Calculate the loss for this point

    for batch_size in batch_sizes:
        print(f"Running for batch size: {batch_size}")
        theta_final, f_vals, num_epochs = mini_batch_SGD(X, x0, num_iters, batch_size, None, alpha, None, None)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[1]
        contour = ax1.contour(x1_feature, x2_feature, f_values, 20, cmap='viridis')
        ax1.plot(theta_final[:, 0], theta_final[:, 1], 'ro-', label='Mini-Batch SGD Path')
        ax1.set_title(f'Contour Plot (Batch={batch_size})')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.legend()
        fig.colorbar(contour, ax=ax1)

        ax2 = axes[0]
        num_updates_per_epoch = np.ceil(len(X) / batch_size).astype(int)  # How many updates per epoch?
        epoch_indices = np.arange(num_updates_per_epoch - 1, len(f_vals), num_updates_per_epoch)
        f_values_per_epoch = f_vals[epoch_indices]
        ax2.plot(range(1, len(f_values_per_epoch) + 1), f_values_per_epoch, label=f'Batch={batch_size}')

        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Function Value f(θ)')
        ax2.set_title(f'Function Value vs. Epochs (Batch={batch_size})')
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        plt.savefig(f"images/batch_sizes/sgd_subplots_alpha_{alpha}_batch_{batch_size}.png")
        plt.clf()


"""
(iv)
"""


def part_b_four():
    alpha_values = [0.0001, 0.01, 0.1, 0.5]
    num_iters = 100
    batch_size = 5

    x1_range = np.linspace(-4, 4, 100)
    x2_range = np.linspace(-4, 4, 100)
    x1_feature, x2_feature = np.meshgrid(x1_range, x2_range)

    f_values = np.zeros_like(x1_feature)

    for i in range(x1_feature.shape[0]):
        for j in range(x1_feature.shape[1]):
            x = np.array([x1_feature[i, j], x2_feature[i, j]])  # Current point (x1, x2)
            f_values[i, j] = f(x, X)  # Calculate the loss for this point

    """Subplots"""
    for alpha in alpha_values:
        print(f"Running for alpha: {alpha}")
        theta_final, f_vals, num_epochs = mini_batch_SGD(X, x0, num_iters, batch_size, None, alpha, None, None)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[1]
        contour = ax1.contour(x1_feature, x2_feature, f_values, 20, cmap='viridis')
        ax1.plot(theta_final[:, 0], theta_final[:, 1], 'ro-', label='Mini-Batch SGD Path')
        ax1.set_title(f'Contour Plot (Alpha={alpha})')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.legend()
        fig.colorbar(contour, ax=ax1)

        ax2 = axes[0]
        num_updates_per_epoch = np.ceil(len(X) / batch_size).astype(int)
        epoch_indices = np.arange(num_updates_per_epoch - 1, len(f_vals), num_updates_per_epoch)
        f_values_per_epoch = f_vals[epoch_indices]
        ax2.plot(range(1, len(f_values_per_epoch) + 1), f_values_per_epoch, label=f'Alpha={alpha}')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Function Value f(θ)')
        ax2.set_title(f'Function Value vs. Epochs (Alpha={alpha})')
        ax2.legend()
        ax2.grid()

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f"images/alpha/sgd_subplots_alpha_{alpha}.png")
        plt.clf()


"""
Part C
"""


def polyak_sgd_contour():
    alpha_values = [0.0001, 0.01, 0.1, 0.5]
    num_iters = 50
    batch_size = 5

    x1_range = np.linspace(-4, 4, 100)
    x2_range = np.linspace(-4, 4, 100)
    x1_feature, x2_feature = np.meshgrid(x1_range, x2_range)

    f_values = np.zeros_like(x1_feature)

    for i in range(x1_feature.shape[0]):
        for j in range(x1_feature.shape[1]):
            x = np.array([x1_feature[i, j], x2_feature[i, j]])  # Current point (x1, x2)
            f_values[i, j] = f(x, X)  # Calculate the loss for this point

    # Create the contour plot
    plt.figure(figsize=(8, 6))

    # Plot the contour of the loss function
    contour = plt.contour(x1_feature, x2_feature, f_values, 20, cmap='viridis')
    plt.title(f'Contour Plot with Mini-Batch SGD with Polyak')
    plt.xlabel('x1')
    plt.ylabel('x2')
    for alpha in alpha_values:
        print(f"Running for alpha: {alpha}")
        theta_final, f_vals, num_epochs = mini_batch_SGD(X, x0, num_iters, batch_size, None, alpha, None, None)

        plt.plot(theta_final[:, 0], theta_final[:, 1], label=f'Alpha = {alpha}')

    plt.colorbar(contour)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/polyak/contour.png")


def rms_sgd_contour():
    alpha_values = [0.0001, 0.01, 0.1, 0.5]
    num_iters = 50
    batch_size = 5

    x1_range = np.linspace(-4, 4, 100)
    x2_range = np.linspace(-4, 4, 100)
    x1_feature, x2_feature = np.meshgrid(x1_range, x2_range)

    f_values = np.zeros_like(x1_feature)

    for i in range(x1_feature.shape[0]):
        for j in range(x1_feature.shape[1]):
            x = np.array([x1_feature[i, j], x2_feature[i, j]])  # Current point (x1, x2)
            f_values[i, j] = f(x, X)  # Calculate the loss for this point

    # Create the contour plot
    plt.figure(figsize=(8, 6))

    # Plot the contour of the loss function
    contour = plt.contour(x1_feature, x2_feature, f_values, 20, cmap='viridis')
    plt.title(f'Contour Plot with Mini-Batch SGD with RMSProp')
    plt.xlabel('x1')
    plt.ylabel('x2')

    beta_values = [0.25, 0.9]

    for beta in beta_values:
        for alpha in alpha_values:
            theta_final, f_values, num_epochs = mini_batch_SGD(X, x0, num_iters, batch_size, polyak_step_size, alpha,
                                                               beta, None)
            plt.plot(theta_final[:, 0], theta_final[:, 1], label=f'Alpha = {alpha}, Beta = {beta}')

    plt.colorbar(contour)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/rmsprop/contour.png")


def rms_sgd():
    alpha_values = [0.0001, 0.01, 0.1, 0.5]
    beta_values = [0.25, 0.9]
    num_iters = 50
    batch_size = 5

    for beta in beta_values:
        plt.clf()
        for alpha in alpha_values:
            theta_final, f_values, num_epochs = mini_batch_SGD(X, x0, num_iters, batch_size, rmsprop_step, alpha,
                                                               beta, None)
            f_values_per_epoch = f_values[
                                 batch_size - 1::batch_size]  # Every batch_size-th value corresponds to an epoch

            # Plot function values at each epoch
            plt.plot(range(1, num_epochs + 1), f_values_per_epoch, label=f'Alpha: {alpha}')

        plt.xlabel('Epochs')
        plt.ylabel('Function Value f(θ)')
        plt.title(f'RMSProp: Function Value vs. Epochs - Beta = {beta} for varying Alpha values')
        plt.legend()
        plt.grid()
        plt.savefig(f"images/rmsprop/beta_{beta}_no_log.png")


def heavyball_sgd():
    alpha_values = [0.0001, 0.01, 0.1, 0.5]
    beta_values = [0.25, 0.9]
    num_iters = 50
    batch_size = 5

    for beta in beta_values:
        plt.clf()
        for alpha in alpha_values:
            theta_final, f_values, num_epochs = mini_batch_SGD(X, x0, num_iters, batch_size, heavy_ball_step, alpha,
                                                               beta, None)
            f_values_per_epoch = f_values[
                                 batch_size - 1::batch_size]  # Every batch_size-th value corresponds to an epoch

            safe_values = np.where(f_values_per_epoch > 0, f_values_per_epoch, np.nan)
            plt.plot(range(1, num_epochs + 1), safe_values, label=f'Alpha: {alpha}')

        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Function Value f(θ)')
        plt.title(f'HeavyBall: Function Value vs. Epochs - Beta = {beta} for varying Alpha values')
        plt.legend()
        plt.grid()
        plt.savefig(f"images/heavyball/beta_{beta}.png")


def adam_sgd():
    alpha_values = [0.0001, 0.01, 0.1, 0.5]
    beta_values = [0.25, 0.9]
    beta1_values = [0.25, 0.99]
    num_iters = 50
    batch_size = 5

    for alpha in alpha_values:
        plt.clf()
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'Adam: Function Value vs. Epochs for Different Beta Values', fontsize=14)
        for i, beta0 in enumerate(beta_values):
            for j, beta1 in enumerate(beta1_values):
                ax = axes[i, j]

                theta_final, f_values, num_epochs = mini_batch_SGD(X, x0, num_iters, batch_size, adam_step,
                                                                   alpha,
                                                                   beta0, beta1)
                f_values_per_epoch = f_values[batch_size - 1::batch_size]  # Sample every batch_size-th step

                ax.plot(range(1, num_epochs + 1), f_values_per_epoch, label=f'Alpha: {alpha}')
                ax.set_yscale('log')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Function Value f(θ)')
                ax.set_title(f'Beta0 = {beta0}, Beta1 = {beta1}')
                ax.grid()

        plt.tight_layout()  # Adjust layout to fit title
        plt.savefig(f"images/adam_beta_comparison_{alpha}.png")


def adam_sgd_alpha():
    alpha_values = [0.0001, 0.01, 0.1, 0.5]
    beta_values = [0.25, 0.9]
    beta1_values = [0.25, 0.99]
    num_iters = 50
    batch_size = 5

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Adam: Function Value vs. Epochs for Different Alpha and Beta Values', fontsize=14)

    for idx, alpha in enumerate(alpha_values):
        ax = axes[idx // 2, idx % 2]

        for beta0 in beta_values:
            for beta1 in beta1_values:
                theta_final, f_values, num_epochs = mini_batch_SGD(X, x0, num_iters, batch_size, adam_step,
                                                                   alpha, beta0, beta1)
                f_values_per_epoch = f_values[batch_size - 1::batch_size]  # Sample every batch_size-th step
                ax.plot(range(1, num_epochs + 1), f_values_per_epoch, label=f'Beta0={beta0}, Beta1={beta1}')

        ax.set_yscale('log')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Function Value f(θ)')
        ax.set_title(f'Alpha = {alpha}')
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f"images/adam/adam_alpha_comparison.png")
    plt.show()


"""
Part C
"""


def run_comparison():
    num_iters = 50
    batch_sizes = [5, 15, 25]
    alpha = 0.01

    x1_range = np.linspace(-4, 4, 100)
    x2_range = np.linspace(-4, 4, 100)
    x1_feature, x2_feature = np.meshgrid(x1_range, x2_range)

    f_values = np.zeros_like(x1_feature)

    for i in range(x1_feature.shape[0]):
        for j in range(x1_feature.shape[1]):
            x = np.array([x1_feature[i, j], x2_feature[i, j]])  # Current point (x1, x2)
            f_values[i, j] = f(x, X)  # Calculate the loss for this point

    for batch_size in batch_sizes:
        print(f"Running for batch size: {batch_size}")
        # Baseline
        theta_final, f_vals, num_epochs = mini_batch_SGD(X, x0, num_iters, batch_size, None, alpha, None, None)
        # Polyak
        polyak_final, polyak_vals, _ = mini_batch_SGD(X, x0, num_iters, batch_size, polyak_step_size, alpha, None, None)
        # RMSProp
        rmsprop_final, rmsprop_vals, _ = mini_batch_SGD(X, x0, num_iters, batch_size, rmsprop_step, 0.5, 0.25, None)
        # HeavyBall
        heavyball_final, heavyball_vals, _ = mini_batch_SGD(X, x0, num_iters, batch_size, heavy_ball_step, 0.01, 0.9,
                                                            None)
        # Adam
        adam_final, adam_vals, _ = mini_batch_SGD(X, x0, num_iters, batch_size, adam_step, 0.01, 0.25,
                                                  0.25)

        # Create a new figure with 1 row, 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Contour Plot + SGD Path (Left subplot)
        ax1 = axes[1]
        contour = ax1.contour(x1_feature, x2_feature, f_values, 20, cmap='viridis')
        ax1.plot(theta_final[:, 0], theta_final[:, 1], label='Constant α= 0.01')
        ax1.plot(polyak_final[:, 0], polyak_final[:, 1], label='Polyak')
        ax1.plot(rmsprop_final[:, 0], rmsprop_final[:, 1], label='RMSProp')
        ax1.plot(heavyball_final[:, 0], heavyball_final[:, 1], label='HeavyBall')
        ax1.plot(adam_final[:, 0], adam_final[:, 1], label='Adam')

        ax1.set_title(f'Contour Plot (Batch={batch_size})')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.legend()
        fig.colorbar(contour, ax=ax1)

        ax2 = axes[0]
        num_updates_per_epoch = np.ceil(len(X) / batch_size).astype(int)
        epoch_indices = np.arange(num_updates_per_epoch - 1, len(f_vals), num_updates_per_epoch)
        f_values_per_epoch = f_vals[epoch_indices]
        polyak_values_per_epoch = polyak_vals[epoch_indices]
        rmsprop_values_per_epoch = rmsprop_vals[epoch_indices]
        heavyball_values_per_epoch = heavyball_vals[epoch_indices]
        adam_values_per_epoch = adam_vals[epoch_indices]
        ax2.plot(range(1, len(f_values_per_epoch) + 1), f_values_per_epoch, label=f'Constant α= 0.01')
        ax2.plot(range(1, len(f_values_per_epoch) + 1), polyak_values_per_epoch, label=f'Polyak')
        ax2.plot(range(1, len(f_values_per_epoch) + 1), rmsprop_values_per_epoch, label=f'RMSProp')
        ax2.plot(range(1, len(f_values_per_epoch) + 1), heavyball_values_per_epoch, label=f'HeavyBall')
        ax2.plot(range(1, len(f_values_per_epoch) + 1), adam_values_per_epoch, label=f'Adam')

        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Function Value f(θ)')
        ax2.set_title(f'Function Value vs. Epochs (Batch={batch_size})')
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        plt.savefig(f"images/partC/sgd_comparison_batch_{batch_size}.png")
        plt.clf()
