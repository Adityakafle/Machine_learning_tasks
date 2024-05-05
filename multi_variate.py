#This is the range which is given in question
z_range = (0, 2) 
x_range = (lambda z: 0, lambda z: z**0.5)
y_range = (lambda x, z: 0, lambda x, z: x * z)

#This is the value i calculated using normal triple integration
theoretical_value_calculated = 3.123809524
# Number of partition for riemann integration
num_partitions = 800

#This is Given Function
def function(x, y, z):
    return z * (x**2 + y**2)

def riemann_integral(f, x_range, y_range, z_range, num_partitions):
    dy = (y_range[1](x_range[1](z_range[1]), z_range[1]) - y_range[0](x_range[0](z_range[0]), z_range[0])) / num_partitions
    dx = (x_range[1](z_range[1]) - x_range[0](z_range[0])) / num_partitions
    dz = (z_range[1] - z_range[0]) / num_partitions
    
    integral = 0.0

    for i in range(num_partitions):
        # Current z value
        z = z_range[0] + i * dz
        
        # Determining width of each partition along the x-axis for the current z value
        dx = (x_range[1](z) - x_range[0](z)) / num_partitions
        
        for i in range(num_partitions):
            x = x_range[0](z) + i * dx
            
            # Calculate the width of each partition along the y-axis for the current z and x values
            dy = (y_range[1](x, z) - y_range[0](x, z)) / num_partitions
            
            for j in range(num_partitions):
                # Current y value
                y = y_range[0](x, z) + j * dy
                
                # Compute the function value at (x, y, z) and multiply it by the volume of the partition
                # Then add it to the integral
                integral += f(x, y, z) * dx * dy * dz
    
    return integral

# Compute the value from riemann method 
integral_value = riemann_integral(function, x_range, y_range, z_range, num_partitions)
precision = abs(theoretical_value_calculated - integral_value) / abs(theoretical_value_calculated) * 100
print("Integral value:", integral_value)
print("Relative error:", precision, "%")


