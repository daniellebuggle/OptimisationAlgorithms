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
    """
    Compute the finite difference estimate of a function.
    :param func: Mathematical expression to differentiate.
    :param var: Variable with respect to which differentiation is performed.
    :param delta: Value used to calculate the finite difference - step size.
    :param x_values: Values used to evaluate the finite difference.
    :return: The calculated finite difference estimate.
    """
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

delta = 0.01
finite_derivatives = finite_difference(f, x, delta, x_values)


def plot_comparison(x, y1, y2, xlim_range, ylim_range, y1_label, y2_label, xaxis_label, yaxis_label, title, file_name):
    plt.clf()
    plt.figure()
    plt.plot(x, y1, label=y1_label, color='b')
    plt.plot(x, y2, label=y2_label, color='r')

    if xlim_range or ylim_range:
        plt.xlim(xlim_range)
        plt.ylim(ylim_range)

    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title)
    plt.legend()
    plt.savefig("images/" + file_name)


plot_comparison(x_values, calculations, finite_derivatives, None, None, 'Derivative Evaluations',
                'Finite Difference', 'x', 'y',
                'Comparing Derivative Evaluation to Finite Difference', 'part_2.png')


x_values = np.arange(1, 101)
calculations = evaluate_derivative(x, f, x_values)

# (v) Vary the size of delta and calculate the error for each
# (iii)
finite_derivatives = finite_difference(f, x, 0.001, x_values)
error = finite_derivatives - calculations  # Error between finite difference and exact derivative
errors_0001 = error  # Calculate the mean error for this delta value

finite_derivatives = finite_difference(f, x, 1, x_values)
error = finite_derivatives - calculations  # Error between finite difference and exact derivative
errors_1 = error # Calculate the mean error for this delta value

# (vi) Plot the errors for different delta values
plt.figure()
plt.plot(x_values, errors_0001, label='Error for δ = 0.001', color='b')  # Plot for δ = 0.001
plt.plot(x_values, errors_1, label='Error for δ = 1', color='r')  # Plot for δ = 1

plt.xlabel('x values')
plt.ylabel('Error (Finite Difference - Exact Derivative)')
plt.title('Effect of Perturbation δ on Finite Difference Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('images/delta_error_comparison_p3.png')