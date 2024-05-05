import numpy as np
import matplotlib.pyplot as plt


# Define parameters
zeta = float(input('Enter Damping ratio or ζ value:'))  # Damping ratio
omega_n = float(input('Enter value of natural frequency or ω: '))  # Natural frequency

def f(t, y, v):
    dy_by_dt = v
    dv_by_dt = -2 * zeta * omega_n * v - omega_n**2 * y
    return dy_by_dt, dv_by_dt

# Euler method solver for first-order ODEs
def euler_first_order(f, t0, y0, v0, tf, dt):
    t_values = [t0]
    y_values = [y0]
    v_values = [v0]

    t = t0
    y = y0
    v = v0

    while t < tf:
        dy_by_dt, dv_by_dt = f(t, y, v)
        y += dy_by_dt * dt
        v += dv_by_dt * dt
        t += dt

        t_values.append(t)
        y_values.append(y)
        v_values.append(v)

    return np.array(t_values), np.array(y_values)

# Define initial conditions
t0 = 0
y0 = 1  # Initial displacement
v0 = 0  # Initial velocity
tf = 20  # Final time
dt = 0.01  # Time step

# Solve the system of first-order ODEs using Euler's method
t_values, y_values = euler_first_order(f, t0, y0, v0, tf, dt)

# Plot the displacement over time
plt.plot(t_values, y_values)
plt.xlabel('Time')
plt.ylabel('Displacement (y)')
plt.title('Damped Harmonic Oscillator: Displacement vs Time')
plt.grid(True)
plt.show()