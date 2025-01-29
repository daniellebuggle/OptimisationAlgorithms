import sympy
import numpy as np
from sympy.abc import x


def derivative_expr(var, func):
    """
    Function to get the derivative of an expression.
    :param var: Variable with respect to which differentiation is performed.
    :param func: Mathematical expression to differentiate
    :return: Derivative of the expression
    """
    return sympy.diff(func, var)


# (a)(i)
x= sympy.symbols('x', real=True)
f = x ** 4
dfdx = derivative_expr(x, f)
print(dfdx)



