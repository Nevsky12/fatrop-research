from visualize_robot_beam import show_robot_with_beam, animate_stationary_robot_with_beam
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

q0 = np.array([0, -np.pi/4, 0, -3*np.pi/4, -np.pi/2, np.pi/2, np.pi/4]).T
th_eq = 0.0
wn = 18
zeta = 0.005
L = 0.3

# test beam behaviour
def simulate_beam(wn, zeta, L, th0):
    ddtheta_func = ca.Function.load("panda/panda_beam_model.casadi")
    th_vals = [0]
    thd = 0.0
    thdd = 0.0
    dt = 0.001
    for i in range(100):
        thdd = ddtheta_func(q0, np.zeros(7), np.zeros(7), wn, zeta, L, th_vals[-1], thd).full().item()
        thd += thdd * dt
        th_vals.append(th_vals[-1] + thd * dt)
    plt.plot(np.arange(len(th_vals))*dt, th_vals, lw=2, label=f"wn={wn}, zeta={zeta}")

plt.figure()
for wn in [5, 10, 18, 25, 50, 100]:
    zeta = 0.005
    simulate_beam(wn, zeta, L, th_eq)
plt.legend()

plt.figure()
for zeta in [0.001, 0.005, 0.01, 0.1]:
    wn = 18
    simulate_beam(wn, zeta, L, th_eq)
plt.legend()
plt.show()

exit()

plt.figure()
ax = plt.gcf().add_subplot(111, projection='3d')
# show_robot_with_beam(ax, q0, th_eq, L)
ani = animate_stationary_robot_with_beam(ax, q0, 0.0, 0.0, wn, zeta, L)

plt.show()
