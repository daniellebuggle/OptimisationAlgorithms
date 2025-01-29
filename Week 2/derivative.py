import matplotlib.pyplot as plt
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


def plot_comparison(x, y1, y2, xlim_range, ylim_range, y1_label, y2_label, xaxis_label, yaxis_label, title, file_name):
    plt.plot(x, y1, label=y1_label, color='b')
    plt.plot(x, y2, label=y2_label, color='r')

    if xlim_range or ylim_range:
        plt.xlim(xlim_range)
        plt.ylim(ylim_range)

    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title)
    plt.savefig(file_name)


plot_comparison(x_values, calculations, finite_derivatives, None, None, 'Derivative Evaluations',
                'Finite Difference', 'x', 'y',
                'Comparing Derivative Evaluation to Finite Difference', 'Normal.png')


plot_comparison(x_values, np.log(calculations), np.log(finite_derivatives), [1, 2], [1,4], 'Derivative Evaluations',
                'Finite Difference', 'x', 'log(y)',
                'Logarithmic Scale: Comparing Derivative Evaluation to Finite Difference', 'log.png')

#(iii)
delta = 1.0
finite_derivatives_1 = finite_difference(f, x, delta, x_values)

plot_comparison(x_values, calculations, finite_derivatives_1, None, None, 'Derivative Evaluations',
                'Finite Difference', 'x', 'y',
                'Comparing Derivative Evaluation to Finite Difference', '1.png')
