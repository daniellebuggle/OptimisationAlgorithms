import numpy as np
from matplotlib import pyplot as plt


class QuadraticFn:
    def f(self, x):
        return x ** 4  # Function value f(x)

    def df(self, x):
        return 4 * x ** 3  # Derivative of f(x)


class PolynomialFn:
    def __init__(self, a):
        self.a = a  # Coefficient for x^2

    def f(self, x):
        return self.a * x ** 2

    def df(self, x):
        return 2 * self.a * x


class AbsoluteFn:
    def __init__(self, gamma):
        self.gamma = gamma

    def f(self, x):
        return self.gamma * np.abs(x)  # Function value y(x) = γ|x|

    def df(self, x):
        # Derivative of γ|x| is γ for x > 0 and -γ for x < 0
        # Handle the derivative at x=0 as 0 or a small number to avoid issues
        if x > 0:
            return self.gamma
        elif x < 0:
            return -self.gamma
        else:
            return 0


# (b) (i)
def grad_descent(fn, x0, alpha, num_iters, max_val=1e6):
    """
    Implements gradient descent with step size alpha.
    :param fn: Function to implement gradient descent on.
    :param x0: Initial starting point.
    :param alpha: Step size.
    :param num_iters: Number of iterations.
    :param max_val: Stopping threshold.
    :return: Arrays of X values and function values.
    """
    x = x0  # starting point
    X = np.array([x])  # convert starting point into array object
    F = np.array(fn.f(x))  # convert function into array object
    for i in range(num_iters):
        step = alpha * (fn.df(x))
        x = x - step
        # Stop if x explodes
        if abs(x) > max_val:
            print(f"Diverged at iteration {i} with x = {x}, stopping early.")
            break

        X = np.append(X, [x], axis=0)
        F = np.append(F, fn.f(x))
    return X, F


# (ii)
func = QuadraticFn()

(X, F) = grad_descent(func, x0=1, alpha=0.1, num_iters=150)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(F, label="y(x) values", color='r')
ax[0].set_xlabel('#Iterations')
ax[0].set_ylabel('Function Value - y(x)')
ax[0].set_title('Change in y(x) over iterations')
ax[0].legend()

ax[1].plot(X, label="x values", color='b')
ax[1].set_xlabel('#Iterations')
ax[1].set_ylabel('x')
ax[1].set_title('Change in x values over iterations')
ax[1].legend()

plt.tight_layout()
plt.savefig(f'images/grad_descent_subplot.png')


# (iii)
fig, ax = plt.subplots(2, 3, figsize=(12, 5))
initial_values = [8, 2, -1]
alpha_values = [0.001, 0.01, 1]
for i in initial_values:
    for k in alpha_values:
        (X, F) = grad_descent(func, x0=i, alpha=k, num_iters=150)

        alpha_index = alpha_values.index(k)
        ax[0, alpha_index].clear()
        ax[1, alpha_index].clear()

        ax[0, alpha_index].plot(F, label="y(x) values", color='r')
        ax[0, alpha_index].set_xlabel('#Iterations')
        ax[0, alpha_index].set_ylabel('Function Value - y(x)')
        ax[0, alpha_index].set_title(f'Change in y(x) for x0={i}, α={k}')
        ax[0, alpha_index].legend()

        ax[1, alpha_index].plot(X, label="x values", color='b')
        ax[1, alpha_index].set_xlabel('#Iterations')
        ax[1, alpha_index].set_ylabel('x')
        ax[1, alpha_index].set_title(f'Change in x for x0={i}, α={k}')
        ax[1, alpha_index].legend()

    plt.tight_layout()
    plt.savefig(f'images/part2/subplot_x0={i}.png')

plt.figure()
# (c) (i)
poly1 = PolynomialFn(1)
poly2 = PolynomialFn(2)
poly3 = PolynomialFn(0.5)

(X, F) = grad_descent(poly1, x0=1, alpha=0.1, num_iters=150)
(X2, F2) = grad_descent(poly2, x0=1, alpha=0.1, num_iters=150)
(X3, F3) = grad_descent(poly3, x0=1, alpha=0.1, num_iters=150)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(F, label="gamma=1")
ax[0].plot(F2, label="gamma=2")
ax[0].plot(F3, label="gamma=0.5")
ax[0].set_xlabel('#Iterations')
ax[0].set_ylabel('Function Value - y(x)')
ax[0].set_title('Change in y(x) over iterations')
ax[0].set_yscale('log')
ax[0].legend()

# Plot the gradient descent steps for each function
ax[1].step(X, poly1.f(X))
ax[1].step(X2, poly2.f(X2))
ax[1].step(X3, poly3.f(X3))

# Plot the actual functions
xx = np.arange(-1, 1.1, 0.1)
ax[1].plot(xx, poly1.f(xx), label="gamma=1")
ax[1].plot(xx, poly2.f(xx), label="gamma=2")
ax[1].plot(xx, poly3.f(xx), label="gamma=0.5")
ax[1].set_xlabel('x')
ax[1].set_ylabel('f(x)')
ax[1].set_title('Gradient Descent Steps on Function γx^2')
ax[1].legend()
plt.savefig(f'images/gamma_comparison_subplot.png')

# (c)(ii)
plt.figure()
abs_fn1 = AbsoluteFn(1)
abs_fn2 = AbsoluteFn(2)
abs_fn3 = AbsoluteFn(0.5)

(X, F) = grad_descent(abs_fn1, x0=1, alpha=0.1, num_iters=150)
(X2, F2) = grad_descent(abs_fn2, x0=1, alpha=0.1, num_iters=150)
(X3, F3) = grad_descent(abs_fn3, x0=1, alpha=0.1, num_iters=150)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(F, label="gamma=1")
ax[0].plot(F2, label="gamma=2")
ax[0].plot(F3, label="gamma=0.5")
ax[0].set_xlabel('#Iterations')
ax[0].set_ylabel('Function Value - y(x)')
ax[0].set_title('Change in y(x) over iterations')
ax[0].set_yscale('log')
ax[0].legend()

# Plot the gradient descent steps for each function
ax[1].step(X, abs_fn1.f(X))
ax[1].step(X2, abs_fn2.f(X2))
ax[1].step(X3, abs_fn3.f(X3))

# Plot the actual functions
xx = np.arange(-1, 1.1, 0.1)
ax[1].plot(xx, abs_fn1.f(xx), label="gamma=1")
ax[1].plot(xx, abs_fn2.f(xx), label="gamma=2")
ax[1].plot(xx, abs_fn3.f(xx), label="gamma=0.5")
ax[1].set_xlabel('x')
ax[1].set_ylabel('f(x)')
ax[1].set_title('Gradient Descent Steps on Function y(x) = γ|x|')
ax[1].legend()
plt.savefig(f'images/absolute_comparison_subplot.png')
