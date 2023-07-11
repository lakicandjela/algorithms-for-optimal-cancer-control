import numpy as np
from scipy.integrate import solve_ivp, cumtrapz, trapz, solve_bvp
from scipy.optimize import Bounds
from gekko import GEKKO
import matplotlib.pyplot as plt
import random




# Definisanje parametara
# slucaj 1
a1 = 0.2
a2 = 0.3
a3 = 0.1
b1 = 1
b2 = 1
c1 = 1
c2 = 0.5
c3 = 1
c4 = 1
d1 = 0.2
r1 = 1.5
r2 = 1
s = 0.33
alpha = 0.3
rho = 0.01


# slucaj 2

# s = 0.3


# slucaj 3

# rho = 0.02


def system(t, y, u):
    N, T, I = y
    dNdt = r2 * N * (1 - b2 * N) - c4 * T * N - a3 * u
    dTdt = r1 * T * (1 - b1 * T) - c2 * I * T - c3 * T * N - a2 * u
    dIdt = s + ((rho * I * T)/(alpha + T)) - c1 * I * T - d1 * I - a1 * u

    return [dNdt, dTdt, dIdt]

def input_func(u, t):
    index = int(t / 10) 

    # index = int(np.floor(t)) #promijeniti u slucaju modifikovanog algoritma sa vremenima promjene

    return u[index]

# N0 = 0.9
# T0 = 0.25
# I0 = 0.25

# t_span = np.linspace(0, 149, num=150)
# t_span = [0, 149]
# step_size = 10
# t_eval = np.arange(t_span[0], t_span[1]+1, step_size)

# def boundary_conditions(ya, yb):
#     return np.array([ya[1] - T0, yb[1]])


# guess = np.zeros((3, 150)) 

# u = [random.uniform(0, 1) for _ in range(15)]
# u = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# sol = solve_ivp(lambda t, y: system(t, y, input_func(u, t)), [0, 149], [N0, T0, I0], method='RK45', t_eval=t_eval, dense_output=True, rtol = 1)
# # sol = solve_bvp(lambda t, y: system(t, y, input_func(u, t)), boundary_conditions, np.linspace(0, 149, num=150), guess)




# t = sol.t
# y = sol.y

# plt.plot(t, y[1])
# plt.xlabel('t')
# plt.ylabel('y')
# plt.title('RK45 Solution')
# plt.grid(True)
# plt.show()

# expanded_u = np.repeat(u, 10)

# # print(expanded_u)

# # integrated_u = cumtrapz(expanded_u, t_span, initial=0)
# integrated_u = trapz(expanded_u, t_span)


# # print(integrated_u)

# # plt.plot(t_span, integrated_u)
# # plt.xlabel('t')
# # plt.ylabel('Integrated u')
# # plt.title('Integration of u')
# # plt.grid(True)
# # plt.show()





 
