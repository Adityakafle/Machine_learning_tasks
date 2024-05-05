import numpy as np

#Matrix from the given equation
A = np.array([[40, -20, 0], [-20, 40, -20], [0, -20, 40]])

def power_iteration_method(iterations,x):

    #initialize previous eigenvalue
    pre_eigenvalue = 0

    for i in range(iterations):
        Ax = np.dot(A, x)
        cur_eigenvalue = np.dot(np.dot(A, x), x) / np.dot(x, x)
        x = Ax / np.linalg.norm(Ax)
        error_estimate = abs(cur_eigenvalue - pre_eigenvalue)
        pre_eigenvalue = cur_eigenvalue

    return cur_eigenvalue, error_estimate

#let eigenvector be:
x = np.array([1, 0, 0])
iterations =10

largest_eigenvalue, error_estimate = power_iteration_method(iterations , x)

print("The largest Eigenvalue is ", largest_eigenvalue)
print("The error estimates between the current and previous estimate of eigenvalue is ", error_estimate)
