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
    return sympy.diff(func, var)


# (a)(i)
x= sympy.symbols('x', real=True)
f = x ** 4
dfdx = derivative_expr(x, f)
print(dfdx)



