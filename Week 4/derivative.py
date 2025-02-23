import numpy as np
import sympy

# Define symbols
x, y = sympy.symbols('x y', real=True)

# Define the functions
f = 5 * (x - 9) ** 4 + 6 * (y - 4) ** 2
f1 = sympy.Max(x - 9, 0) + 6 * sympy.Abs(y - 4)


# Define differentiation function
def derivative_expr(var, func):
    """Compute the derivative of an expression."""
    return sympy.diff(func, var)


# Function to evaluate derivatives safely
def evaluate_derivative(vars, func, values):
    """
    Compute and evaluate the gradient (partial derivatives) of a function,
    safely handling Piecewise expressions.

    Parameters:
    vars (list): List of variables to differentiate with respect to.
    func (sympy expression): The function to differentiate.
    values (dict): Dictionary of variable values for evaluation (e.g., {x: 1, y: 2}).

    Returns:
    list: Evaluated gradient as a list of numerical values.
    """
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


"""# Compute derivatives for f1
dfdx = derivative_expr(x, f1).simplify()
dfdy = derivative_expr(y, f1).simplify()

print("∂f1/∂x:", dfdx)
print("∂f1/∂y:", dfdy)

# Test evaluation for f1
test_points = [{x: 1, y: 1}, {x: 10, y: 1}]

for point in test_points:
    grad = evaluate_derivative([x, y], f1, point)
    print(f"Gradient at {point}: {grad}")

# Compute derivatives for f
dfdx_f = derivative_expr(x, f).simplify()
dfdy_f = derivative_expr(y, f).simplify()

print("∂f/∂x:", dfdx_f)
print("∂f/∂y:", dfdy_f)

# Test evaluation for f
for point in test_points:
    grad = evaluate_derivative([x, y], f, point)
    print(f"Gradient at {point}: {grad}")"""
