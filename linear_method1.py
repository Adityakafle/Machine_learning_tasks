import numpy as np
# Since we have to represent equation in the form of Ax=b using numpy
A = np.array([[1, 3],[3, -1],[2, 2]])
b = np.array([-2, 4, 1])

# This is the function i use from numpy library to solve over-determined System of liner Equation
x = np.linalg.lstsq(A, b, rcond=None)[0]

print("Solution using SSRE method:", x)