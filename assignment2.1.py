# Question 1

def neville(x, y, target_x):
    n = len(x)
    Q = [[0] * n for _ in range(n)]

    for i in range(n):
        Q[i][0] = y[i]

    for i in range(1, n):
        for j in range(1, i + 1):
            Q[i][j] = ((target_x - x[i - j]) * Q[i][j - 1] - (target_x - x[i]) * Q[i - 1][j - 1]) / (x[i] - x[i - j])

    return Q[n - 1][n - 1]

x = [3.6, 3.8, 3.9]
y = [1.675, 1.436, 1.318]
target_x = 3.7

result = neville(x, y, target_x)
print(f"{result}")
print("\n")

# Question 2

def newton_forward_coefficients(x, fx):
    n = len(x)
    coefficients = [fx[0]]
    for i in range(1, n):
        coefficients.append(divided_difference(x[:i+1], fx[:i+1]))
    return coefficients

def divided_difference(x, fx):
    if len(x) == 1:
        return fx[0]
    return (divided_difference(x[1:], fx[1:]) - divided_difference(x[:-1], fx[:-1])) / (x[-1] - x[0])

def newton_forward_polynomial(x, fx, degree):
    coefficients = newton_forward_coefficients(x, fx)
    result = coefficients[0]
    term = 1
    for i in range(1, degree + 1):
        term *= (degree - i + 1) / i * (7.4 - 7.2)
        result += term * coefficients[i]
    return result

x = [7.2, 7.4, 7.5, 7.6]
fx = [23.5492, 25.3913, 26.8224, 27.4589]

# printing and calculating
for degree in range(1, 4):
    p = newton_forward_polynomial(x, fx, degree)
    print(p)
print("\n")
    
# Question 3

import math

def divided_difference(x, fx):
    if len(x) == 1:
        return fx[0]
    return (divided_difference(x[1:], fx[1:]) - divided_difference(x[:-1], fx[:-1])) / (x[-1] - x[0])

def newton_forward_polynomial(x, fx, degree, table):
    h = x[1] - x[0]
    u = (7.3 - x[0]) / h
    result = fx[0]

    for i in range(1, degree + 1):
        term = table[0][i] / math.factorial(i)
        for j in range(i):
            term *= (u - j)
        result += term

    return result

x = [7.2, 7.4, 7.5, 7.6]
fx = [23.5492, 25.3913, 26.8224, 27.4589]

# forward table
n = len(x)
table = [[0 for _ in range(n)] for _ in range(n)]
for i in range(n):
    table[i][0] = fx[i]

for i in range(1, n):
    for j in range(n - i):
        table[j][i] = table[j + 1][i - 1] - table[j][i - 1]

# approximating 7.3
degree = 1
f7 = newton_forward_polynomial(x, fx, degree, table)
print(f7)
print("\n")


# Question 4

import numpy as np

x = np.array([3.6, 3.8, 3.9])
f_x = np.array([1.675, 1.436, 1.318])
f_prime_x = np.array([-1.195, -1.188, -1.182])

# Initializing matrix
n = len(x) * 2
matrix = np.zeros((n, n + 1))

# filling columns
for i in range(len(x)):
    matrix[i * 2, 0] = x[i]
    matrix[i * 2 + 1, 0] = x[i]
    matrix[i * 2, 1] = f_x[i]
    matrix[i * 2 + 1, 1] = f_x[i]
    matrix[i * 2, 2] = f_prime_x[i]
    matrix[i * 2 + 1, 2] = f_prime_x[i]

# divided difference method, filling table
for i in range(3, n + 1):
    for j in range(n - i):
        if matrix[j + 1, 0] == matrix[j, 0]:
            matrix[j, i] = matrix[j + 1, i - 1]
        else:
            matrix[j, i] = (matrix[j + 1, i - 1] - matrix[j, i - 1]) / (matrix[j + 1, 0] - matrix[j, 0])

# matrix printout
np.set_printoptions(precision=8, suppress=True)
print(matrix)
print("\n")

# Question 5

x = np.array([2, 5, 8, 10])
f_x = np.array([3, 5, 7, 9])

# data points
n = len(x)

# initializing matrix and vector
A = np.zeros((n, n))
b = np.zeros(n)


# cubic spline eqn
A[0, 0] = 1
A[n - 1, n - 1] = 1

for i in range(1, n - 1):
    h_i = x[i] - x[i - 1]
    h_i_plus_1 = x[i + 1] - x[i]
    
    A[i, i - 1] = h_i
    A[i, i] = 2 * (h_i + h_i_plus_1)
    A[i, i + 1] = h_i_plus_1
    
    b[i] = 6 * ((f_x[i + 1] - f_x[i]) / h_i_plus_1 - (f_x[i] - f_x[i - 1]) / h_i)

# system of eqns
c = np.linalg.solve(A, b)

# coeff's
a = f_x.copy()
b = np.zeros(n - 1)
d = np.zeros(n - 1)

for i in range(n - 1):
    h_i = x[i + 1] - x[i]
    b[i] = (a[i + 1] - a[i]) / h_i - h_i * (2 * c[i] + c[i + 1]) / 6
    d[i] = (c[i + 1] - c[i]) / (6 * h_i)

# print matrix and vectors
np.set_printoptions(precision=4, suppress=True)

print(A)
print(b)
print(c)


