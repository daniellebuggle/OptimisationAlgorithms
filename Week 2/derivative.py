import sympy
import numpy as np


def derivative_expr(var, func):
    """
    Function to get the derivative of an expression.
    :param var: Variable with respect to which differentiation is performed.
    :param func: Mathematical expression to differentiate.
    :return: Derivative of the expression.
    """
    return sympy.diff(func, var)


def evaluate_derivative(var, func, values):
    """
    Compute the derivative of a function and evaluate.
    :param var: Variable with respect to which differentiation is performed.
    :param func: Mathematical expression to differentiate.
    :param values: Values to evaluate function with.
    :return: Evaluated function.
    """
    derivative = derivative_expr(var, func)
    derivative_numeric = sympy.lambdify(var, derivative, 'numpy')
    return derivative_numeric(values)


def finite_difference(func, var, delta, x_values):
    func_numeric = sympy.lambdify(var, func, 'numpy')
    return (func_numeric(x_values + delta) - func_numeric(x_values)) / delta


# (a)(i)
x = sympy.symbols('x', real=True)
f = x ** 4
dfdx = derivative_expr(x, f)
print(dfdx)

# (ii)
x_values = np.arange(1, 101)
calculations = evaluate_derivative(x, f, x_values)
print(calculations)


delta = 0.01
finite_derivatives = finite_difference(f, x, delta, x_values)

print("x-values | Approximate Derivative")
print("-------------------------------")
for x_val, d_val in zip(x_values, finite_derivatives):
    print(f"{x_val:>7.2f} | {d_val:>9.3f}")
print(finite_difference(f, x, delta, x_values))


