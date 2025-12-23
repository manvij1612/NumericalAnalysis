import numpy as np
import matplotlib.pyplot as plt

# grid
N = 400
x = np.linspace(0, 1, N, endpoint=False)
dx = 1/N

# initial condition
u = np.sin(2*np.pi*x)

# heat equation
u_heat = u.copy()
dt_heat = 0.4* dx**2
for n in range(400):
    u_heat = u_heat + dt_heat*(np.roll(u_heat,-1)-2*u_heat+np.roll(u_heat,1))/dx**2

plt.figure()
plt.plot(x, u, label="Initial")
plt.plot(x, u_heat, label="Heat (stable)")
plt.xlabel("x")
plt.ylabel("u")
plt.title("Heat Equation")
plt.legend()
plt.savefig("heat.png")

#advection 

u_adv = u.copy()
dt_adv = 0.5 * dx
for n in range(400):
    u_adv = u_adv + dt_adv*(np.roll(u_adv,-1) - np.roll(u_adv,1))/(2 *dx)

plt.figure()
plt.plot(x, u, label="Initial")
plt.plot(x, u_adv, label="advection (unstable)")
plt.xlabel("x")
plt.ylabel("u")
plt.title("advection Equation")
plt.legend()
plt.savefig("advection.png")
