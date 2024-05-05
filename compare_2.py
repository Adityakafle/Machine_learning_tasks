import numpy as np
import matplotlib.pyplot as plt
import linear_method1, linear_method2

from linear_method1 import x as x_ssre
print(x_ssre)
from linear_method2 import x as x_pinv
print(x_pinv)

tolerance = 1e-10  # Set a small tolerance for comparison
difference = np.abs(x_ssre - x_pinv)
if np.all(difference < tolerance):
    print("The solution is unique.")
else:
    print("The solution is not unique.")

# Plotting the solutions using matplotlib for visualization
plt.scatter(x_pinv[0], x_pinv[1], color='red', label='Pseudo-inverse Solution')
plt.scatter(x_ssre[0], x_ssre[1], color='blue', label='SSE Solution')
plt.xlabel('x_ssre')
plt.ylabel('x_pinv')
plt.title('Comparison of Solutions from SSRE and Pseudo Inverse')
plt.legend()
plt.show()

