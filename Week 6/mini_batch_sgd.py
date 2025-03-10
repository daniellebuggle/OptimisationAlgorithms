import numpy as np
from matplotlib import pyplot as plt
from loss_functions import *
from step_sizes import *
import sympy as sp

F_STAR = 0

"""
Part A
"""


def approx_gradient(f, x, minibatch, epsilon=1e-5):
    grad = np.zeros_like(x)  # Initialise gradient vector

    for i in range(len(x)):  # Iterate over each dimension of x
        x_plus = x.copy()  # Perturb x[i] by +epsilon
        x_minus = x.copy()  # Perturb x[i] by -epsilon

        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        # Approximate the derivative using central difference formula
        grad[i] = (f(x_plus, minibatch) - f(x_minus, minibatch)) / (2 * epsilon)

    return grad


"""
(i)
"""


def mini_batch_SGD(X, num_iters, batch_size, step_size_func, alpha0, beta0, beta1, epsilon):
    m, n = X.shape  # m = number of training data points, n = number of features
    theta = np.zeros(n)  # Initialize theta (parameters)

    moving_avg = np.zeros_like(theta)  # RMSPROP
    z_t = np.zeros_like(theta)  # Heavy Ball
    m_t = np.zeros_like(theta)  # Adam - First momentum term
    v_t = np.zeros_like(theta)  # Adam - Second momentum term
    t = 1  # Adam - time step

    for iter_num in range(num_iters):
        # Shuffle the training data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]

        # Loop through mini-batches
        for i in range(0, m, batch_size):
            # Create mini-batch
            minibatch_X = X_shuffled[i:i + batch_size]

            # Calculate the approximate gradient for the current mini-batch
            grad = approx_gradient(f, theta, minibatch_X)

            if step_size_func == polyak_step_size:
                f_x = f(theta, minibatch_X)
                alpha = step_size_func(f_x, F_STAR, grad)
            elif step_size_func == rmsprop_step:
                alpha, average = step_size_func(grad, moving_avg, alpha0, beta0)
                moving_avg = average
            elif step_size_func == heavy_ball_step:
                z = heavy_ball_step(z_t, grad, alpha0, beta0)
                alpha = z
            elif step_size_func == adam_step:
                m_t, v_t, alpha = step_size_func(m_t, v_t, grad, beta0, beta1, epsilon, t)
                t += 1

            theta = theta - alpha * grad  # Update parameters

    return theta


def symbolic_f(x, minibatch):
    y = 0
    count = 0
    for w in minibatch:
        # Make symbolic z and use symbolic Min
        z1 = x[0] - w[0] - 1  # x[0] - w[0] - 1
        z2 = x[1] - w[1] - 1  # x[1] - w[1] - 1
        term1 = 37 * (z1 ** 2 + z2 ** 2)
        term2 = (z1 + 5) ** 2 + (z2 + 5) ** 2
        y += sp.Min(term1, term2)
        count += 1
    return y / count


# Compute symbolic derivative
def compute_symbolic_derivative(x, minibatch):
    grad_x1 = sp.diff(symbolic_f(x, minibatch), x[0])  # Differentiate with respect to x1
    return grad_x1


def numerical_derivative(x1_val, x2_val, symbolic_grad):
    return symbolic_grad.subs({x1: x1_val, x2: x2_val})


# Example usage
np.random.seed(42)

"""
(ii)
"""
training_data = generate_trainingdata(m=25)
X = training_data  # Feature matrix (25 points, 2 features)

x1_range = np.linspace(-1.5, 3, 100)
x2_range = np.linspace(-1.5, 3, 100)
x1, x2 = np.meshgrid(x1_range, x2_range)

f_values = np.zeros_like(x1)

for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
        x = np.array([x1[i, j], x2[i, j]])  # Current point (x1, x2)
        f_values[i, j] = f(x, X)  # Calculate the loss for this point

# Step 4: Plot wireframe and contour plots
fig = plt.figure(figsize=(12, 6))

# Wireframe plot
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(x1, x2, f_values, cmap='viridis')
ax.set_title('Wireframe Plot of f(x, T)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x, T)')

# Contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contour(x1, x2, f_values, 20, cmap='viridis')
ax2.set_title('Contour Plot of f(x, T)')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
fig.colorbar(contour)

plt.tight_layout()
plt.savefig("images/wireframe_contour_all_N.png")

"""
(iii)
"""
x1, x2 = sp.symbols('x1 x2')
minibatch = [(0, 0), (1, 1), (2, 2)]

x = [x1, x2]

grad_x1 = compute_symbolic_derivative(x, minibatch)

print(f"Symbolic Gradient with respect to x1: {grad_x1}")

x1_val = 1.0
x2_val = 2.0

# Evaluate the symbolic derivative at the point (x1_val, x2_val)
grad_x1_evaluated = numerical_derivative(x1_val, x2_val, grad_x1)

# Output the evaluated derivative
print(f"Symbolic Gradient with respect to x1 at (x1={x1_val}, x2={x2_val}): {grad_x1_evaluated}")

"""
Part B
"""


def gradient_descent(x0, minibatch, alpha, num_iterations):
    X = np.array([x0])  # Ensure X is an array, with x0 as the first element
    symbolic_grad = compute_symbolic_derivative([x1, x2], minibatch)

    x1_value = X[-1][0]
    x2_value = X[-1][1]

    for i in range(num_iterations):
        step = alpha * np.array(numerical_derivative(x1_value, x2_value, symbolic_grad))
        x0 = X[-1] - step
        X = np.append(X, [x0], axis=0)

        x1_value = x0[0]
        x2_value = x0[1]

    return X


x0 = [3, 3]  # Initial point
minibatch = generate_trainingdata(m=25)  # Your training data
alpha = 0.01  # Step size (learning rate)
num_iterations = 100  # Number of iterations

# Run gradient descent
X_history = gradient_descent(x0, minibatch, alpha, num_iterations)
print(X_history)

# Plot the path on the contour plot
x1_range = np.linspace(-4, 4, 100)
x2_range = np.linspace(-4, 4, 100)
x1, x2 = np.meshgrid(x1_range, x2_range)

f_values = np.zeros_like(x1)

for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
        x = np.array([x1[i, j], x2[i, j]])  # Current point (x1, x2)
        f_values[i, j] = f(x, minibatch)  # Calculate the loss for this point

# Create the contour plot
plt.figure(figsize=(8, 6))

# Plot the contour of the loss function
contour = plt.contour(x1, x2, f_values, 20, cmap='viridis')
plt.title(f'Contour Plot with Gradient Descent Path with alpha={alpha}')
plt.xlabel('x1')
plt.ylabel('x2')

# Plot the path of gradient descent (as a line)
plt.plot(X_history[:, 0], X_history[:, 1], 'ro-', label='Gradient Descent Path')

# Add a colorbar for the contour plot
plt.colorbar(contour)

# Show the legend
plt.legend()

# Adjust layout for better display
plt.tight_layout()

# Display the plot
plt.savefig(f"images/grad_descent_alpha_{alpha}.png")
