import sympy
import numpy as np
from sympy.abc import x


def derivative_expr(x, f):
    """
    Function to get the derivative of an expression.
    :param x: Variable with respect to which differentiation is performed.
    :param f: Mathematical expression to differentiate
    :return: Derivative of the expression
    """
    return sympy.diff(f, x)


var_x = sympy.symbols('x', real=True)
function = x ** 4
dfdx = derivative_expr(var_x, function)
print(dfdx)




