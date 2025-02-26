import numpy as np
import sympy


def derivative_expr(var, func):
    return sympy.diff(func, var)


def evaluate_derivative(vars, func, values):
    derivatives = [derivative_expr(var, func).simplify() for var in vars]  # Compute partial derivatives
    evaluated_gradient = []
    for d in derivatives:
        # Check if the derivative contains a Piecewise expression
        if isinstance(d, sympy.Piecewise):
            evaluated_value = d.subs(values).evalf()
        else:
            # Convert to numerical function and evaluate
            func_numeric = sympy.lambdify(vars, d, 'numpy')
            evaluated_value = func_numeric(*values.values())

        evaluated_gradient.append(evaluated_value)

    return np.array(evaluated_gradient)
