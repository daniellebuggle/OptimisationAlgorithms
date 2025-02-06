import numpy as np
from matplotlib import pyplot as plt


class QuadraticFn:
    def f(self, x):
        return x ** 4  # Function value f(x)

    def df(self, x):
        return 4 * x ** 3  # Derivative of f(x)


def grad_descent(fn, x0, alpha, num_iters):
    x = x0  # starting point
    X = np.array([x])  # convert starting point into array object
    F = np.array(fn.f(x))  # convert function into array object
    for i in range(num_iters):
        step = alpha * (fn.df(x))
        x = x - step
        X = np.append(X, [x], axis=0)
        F = np.append(F, fn.f(x))
    return X, F


func = QuadraticFn()
(X, F) = grad_descent(func, x0=1, alpha=0.1, num_iters=150)


xx = np.arange(-1, 1.1, 0.1)
plt.plot(xx, func.f(xx))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function f(x) = x⁴')
plt.show()


plt.plot(F)
plt.xlabel('#iteration')
plt.ylabel('function value')
plt.title('Gradient Descent: Function Value Over Iterations')
plt.show()

plt.semilogy(F)
plt.xlabel('#iteration')
plt.ylabel('function value')
plt.title('Gradient Descent: Function Value (Log Scale)')
plt.show()

plt.plot(X)
plt.xlabel('#iteration')
plt.ylabel('x')
plt.title('Gradient Descent: x Values Over Iterations')
plt.show()

plt.step(X, func.f(X))
xx = np.arange(-1, 1.1, 0.1)
plt.plot(xx, func.f(xx))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Steps on Function f(x)')
plt.show()
