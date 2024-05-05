import numpy as np

# Since we have to represent equation in the form of Ax=b
A = np.array([[1, 3],[3, -1],[2, 2]])
b = np.array([-2, 4, 1])


# Using function from numpy A inverse is calculated and @ is used for matrix multiplication
x = np.linalg.pinv(A) @ b

# Print solutions
print("Solution using Pseudo-inverse method:", x)
