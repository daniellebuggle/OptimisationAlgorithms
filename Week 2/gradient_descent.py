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
        return self.a * x ** 2  # Function value f(x)

    def df(self, x):
        return 2 * self.a * x  # Derivative of f(x)


# (b) (i)
def grad_descent(fn, x0, alpha, num_iters, max_val=1e6):
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

plt.plot(F, label="y(x) values", color='r')
plt.xlabel('#Iterations')
plt.ylabel('Function Value - y(x)')
plt.legend()
plt.title('Change in y(x) over iterations for x0=1, α=0.1')
plt.show()

plt.plot(X, label="x values")
plt.xlabel('#Iterations')
plt.ylabel('x')
plt.title('Change in x values over iterations for x0=1, α=0.1')
plt.legend()
plt.show()

# (iii)
initial_values = [1, 2, -1]
alpha_values = [0.001, 0.01, 1]
for i in initial_values:
    for k in alpha_values:
        plt.figure()
        (X, F) = grad_descent(func, x0=i, alpha=k, num_iters=150)

        plt.plot(F, label="y(x) values", color='r')
        plt.xlabel('#Iterations')
        plt.ylabel('Function Value - y(x)')
        plt.legend()
        plt.title(f'Change in y(x) over iterations for x0={i}, α={k}')
        plt.savefig(f'images/part2/y_values_x0={i}_α={k}.png')

        plt.figure()

        plt.plot(X, label="x values")
        plt.xlabel('#Iterations')
        plt.ylabel('x')
        plt.title(f'Change in x values over iterations for x0={i}, α={k}')
        plt.legend()
        plt.savefig(f'images/part2/x_values_x0={i}_α={k}.png')


plt.figure()
#(c) (i)
poly1 = PolynomialFn(1)
poly2 = PolynomialFn(2)
poly3 = PolynomialFn(0.5)

(X, F) = grad_descent(poly1, x0=1, alpha=0.1, num_iters=150)
(X2, F2) = grad_descent(poly2, x0=1, alpha=0.1, num_iters=150)
(X3, F3) = grad_descent(poly3, x0=1, alpha=0.1, num_iters=150)

plt.step(X, poly1.f(X))
plt.step(X2, poly2.f(X2))
plt.step(X3, poly3.f(X3))

xx = np.arange(-1, 1.1, 0.1)
plt.plot(xx, poly1.f(xx), label="gamma=1")
plt.plot(xx, poly2.f(xx), label="gamma=2")
plt.plot(xx, poly3.f(xx), label="gamma=0.5")


plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Steps on Function γ(x)')
plt.legend()
plt.savefig(f'images/gamma_comparison.png')