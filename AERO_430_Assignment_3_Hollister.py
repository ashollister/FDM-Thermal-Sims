"""
AERO 430 Assignment 3
Andrew Hollister
UIN: 127008398
"""

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import tabulate as tbl


"""
Data Structure
"""


class DataStructure:
    def __init__(self):
        self.data = []

    def add_data(self, nodes, k, x_vals, y_vals, temp_mesh):
        self.data.append((nodes, k, x_vals, y_vals, temp_mesh))

    def return_data(self, nodes, k):
        x_vals = next(item[2] for item in self.data if item[0] == nodes and item[1] == k)
        y_vals = next(item[3] for item in self.data if item[0] == nodes and item[1] == k)
        temp_mesh = next(item[4] for item in self.data if item[0] == nodes and item[1] == k)
        return x_vals, y_vals, temp_mesh


"""
Analytical Solution
"""


def exact_temp(x, y, k):
    temp = 100*np.sinh(k*np.pi*y)*np.sin(np.pi*x)/np.sinh(k*np.pi)
    return temp


"""
2nd Order FDM Solution
"""


def bc_temp_func(x):
    return -100*np.sin(np.pi*x)  # deg c


def FDM_2nd_order(n_nodes, k):

    # Generating A Matrix
    dim = n_nodes**2
    A = np.zeros([dim, dim])
    A += (-2*(k**2+1))*np.eye(dim)
    A += np.eye(dim, k=-n_nodes)
    A += np.eye(dim, k=n_nodes)
    A += k**2*np.eye(dim, k=1)
    A += k**2*np.eye(dim, k=-1)
    for i in range(dim):
        if i % n_nodes == 0 and i != 0:
            A[i-1][i] = 0
            A[i][i-1] = 0

    # Generating B Matrix
    B = np.zeros([dim, 1])
    dx = length/(n_nodes+1)
    for i in range(n_nodes):
        x = dx*(i+1)
        B[i][0] = bc_temp_func(x)

    # Solving Matrix
    temps = np.reshape(np.linalg.solve(A, B), [n_nodes, n_nodes])
    temps = np.row_stack((-bc_temp_func(np.linspace(0, length, n_nodes+2))[1:-1], temps))
    temps = np.row_stack((temps, np.zeros(n_nodes)))
    temps = np.column_stack((temps, np.zeros(n_nodes+2)))
    temps = np.column_stack((np.zeros(n_nodes+2), temps))

    return np.flip(temps)


"""
Heat Flux Functions
"""


# Exact Heat Transfer Function
def exact_heat_transfer(k):
    return -200*k/np.tanh(k*np.pi)


# Approximate Heat Flux Function
def approx_heat_transfer(lower, upper, nodes, fdm_temp, k):
    dTdy = []
    dx = length/nodes
    for i in range(len(fdm_temp)):
        col = fdm_temp[:, i]
        dTdy.append((col[-3]-4*col[-2]+3*col[-1])/(2*dx))

    sum = 0
    for i in range(nodes + 1):
        summand = dTdy[i]
        if (i != 0) and (i != nodes):
            summand *= (2 + (2 * (i % 2)))
        sum += summand

    val = ((upper - lower) / (3 * nodes)) * sum
    return -val


"""
Richardson Extrapolation Function
"""


def q_rich_extr(q, q2, q4):
    q_extr = (q2**2 - q*q4)/(2*q2-q-q4)
    beta = abs(np.log((q_extr - q2)/(q_extr-q4))/np.log(2))
    perc_err = abs(abs(q_extr-q)/q_extr)*100
    return q_extr, beta, perc_err


"""
Convergence Rate Function
"""


def get_beta(exact, approx, approx_2, h, h_2):
    A = np.log(abs(exact - approx))
    B = np.log(abs(exact - approx_2))
    C = np.log(h) - np.log(h_2)
    return -(A - B) / C


"""
Data Generation
"""

# FDM Solution
length = 1
FDM_Data = DataStructure()
node_list = range(2, 7)
k_list = [1, 0.5, 0.75, 1, 2, 5, 10]
re_log_err = []
re_dx_err = []
hf_log_err = []
hf_dx_err = []
tc_log_err = []
tc_dx_err = []
for k in k_list:
    for n in node_list:

        n_nodes = 2**n-1
        x_vals = np.linspace(0, length, n_nodes+2)
        y_vals = np.linspace(0, length, n_nodes+2)

        sol = FDM_2nd_order(n_nodes, k)
        FDM_Data.add_data(n_nodes+2, k, x_vals, y_vals, sol)

    """
    Post-Processing
    """

    # Exact Solution Plotting
    # x_vals = np.linspace(0, length, 100)
    # y_vals = np.linspace(0, length, 100)
    # X1, Y1, = np.meshgrid(x_vals, y_vals)
    # Z1 = exact_temp(X1, Y1, k)
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1,
    #                 cmap='inferno', edgecolor='none')
    # plt.title('Exact Solution 3-Dimensional Heatmap\nK = '+str(k))
    # ax.set_xlabel('X (cm)')
    # ax.set_ylabel('Y (cm)')
    # ax.set_zlabel(r'T(x, y)$\degree$C')
    # plt.show()

    # 2-Dimensional Heat Maps
    # ax = sns.heatmap(Z1, xticklabels=False, yticklabels=False, cbar_kws={'label': u'Temperature \N{DEGREE SIGN}C'})
    # plt.xticks(100 * np.array([0, 0.25, 0.5, 0.75, 1]), labels=[0, 0.25, 0.5, 0.75, 1])
    # ax.set_xlabel('x (cm)')
    # plt.yticks(100 * np.array([0, 0.25, 0.5, 0.75, 1]), labels=[0, 0.25, 0.5, 0.75, 1])
    # ax.set_ylabel('y (cm)')
    # ax.set_title('Exact Solution Heatmap')
    # plt.gca().invert_yaxis()
    # plt.show()

    # Additional Post-Processing
    heat_flux_data = []
    temp_error_data = []
    for n in node_list:
        n_nodes = 2**n-1
        x_vals, y_vals, Z = FDM_Data.return_data(n_nodes+2, k)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Heat Flux Calculations and Error Calculations
        q_exact = exact_heat_transfer(k)
        q_approx = approx_heat_transfer(0, 1, n_nodes + 1, Z, k)
        error_q = abs((q_exact-q_approx)/q_exact*100)
        heat_flux_data.append([n_nodes + 1, length / (n_nodes + 1), q_exact, q_approx, error_q])

        # Temperature Error Calculations
        midpoint_index = int(len(Z) / 2)
        midpoint = length / (n_nodes + 1) * midpoint_index
        approx_t = Z[midpoint_index][midpoint_index]
        exact_t = exact_temp(midpoint, midpoint, k)
        error_t = abs(exact_t-approx_t)/exact_t*100
        temp_error_data.append([n_nodes + 1, length / (n_nodes + 1), exact_t, approx_t, error_t])

        # 2-Dimensional Heat Maps
        # ax = sns.heatmap(Z, xticklabels=False, yticklabels=False, cbar_kws={'label': u'Temperature \N{DEGREE SIGN}C'})
        # plt.xticks((n_nodes+2)*np.array([0, 0.25, 0.5, 0.75, 1]), labels=[0, 0.25, 0.5, 0.75, 1])
        # ax.set_xlabel('x (cm)')
        # plt.yticks((n_nodes+2)*np.array([0, 0.25, 0.5, 0.75, 1]), labels=[0, 0.25, 0.5, 0.75, 1])
        # ax.set_ylabel('y (cm)')
        # ax.set_title('Heatmap of cross-section of bar\n'+str(n_nodes+2)+' Nodes'+'\nK = '+str(k))
        # plt.gca().invert_yaxis()
        # plt.show()

        # 3-Dimensional Heat Maps
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
        #                 cmap='inferno', edgecolor='none')
        # plt.title(r'2nd order FDM Output T(x, y)$\degree$ C'+'\nNodes = '+str(n_nodes+2)+'\nK = '+str(k))
        # ax.set_xlabel('X (cm)')
        # ax.set_ylabel('Y (cm)')
        # ax.set_zlabel(r'T(x, y)$\degree$C')
        # plt.show()

    # Richardson Extrapolation and Convergence
    heat_flux_rc_data = []
    for i in range(len(heat_flux_data)-2):
        q_extr, q_b, perc_err = q_rich_extr(heat_flux_data[i][3], heat_flux_data[i+1][3], heat_flux_data[i+2][3])
        heat_flux_rc_data.append([heat_flux_data[i][0], heat_flux_data[i][1], q_extr, heat_flux_data[i][3],
                                  perc_err, q_b])
    print('\nConvergence of Extrapolated Heat Flux (K = '+str(k)+'):')
    print(tbl.tabulate(heat_flux_rc_data, headers=['Num. Elements', 'dx', 'Extrapolated Heat Flux', 'Approx. Heat Loss',
                                                   'Percent Error', 'Beta']))

    re_log_err.append([-np.log10(item[4]/100) for item in heat_flux_rc_data])
    re_dx_err.append([-np.log10(item[1]) for item in heat_flux_rc_data])

    # Heat Flux Convergence
    heat_flux_data[0].append('n/a')
    for i in range(len(heat_flux_data)-1):
        heat_flux_data[i+1].append(get_beta(heat_flux_data[i][2], heat_flux_data[i][3], heat_flux_data[i+1][3],
                                            heat_flux_data[i][0], heat_flux_data[i+1][0]))
    print('\nConvergence of Heat Flux (K = '+str(k)+'):')
    print(tbl.tabulate(heat_flux_data, headers=['Num. Elements', 'dx', 'Exact Heat Flux', 'Approx. Heat Loss',
                                                'Percent Error', 'Beta']))
    hf_log_err.append([-np.log10(item[4]/100) for item in heat_flux_data])
    hf_dx_err.append([-np.log10(item[1]) for item in heat_flux_data])

    # Temperature Convergence
    temp_error_data[0].append('n/a')
    for i in range(len(temp_error_data)-1):
        temp_error_data[i+1].append(get_beta(temp_error_data[i][2], temp_error_data[i][3], temp_error_data[i+1][3],
                                             temp_error_data[i][0], temp_error_data[i+1][0]))
    print('\nConvergence of 2nd Order FDM Temperature Solution (K = '+str(k)+'):')
    print(tbl.tabulate(temp_error_data, headers=['Num. Elements', 'dx', 'Exact Midpoint Temp', 'Approx. Midpoint Temp',
                                                 'Percent Error', 'Beta']))
    tc_log_err.append([-np.log10(item[4]/100) for item in temp_error_data])
    tc_dx_err.append([-np.log10(item[1]) for item in temp_error_data])

# Plotting Convergence Graphs
for i in range(len(k_list)):
    plt.plot(re_dx_err[i], re_log_err[i], label='K = '+str(k_list[i]))
plt.title('Extrapolated Heat Flux\n' + r'-$log_{10}$($\Delta$x) vs -$log_{10}$(Relative Error)')
plt.xlabel(r'$-log_{10}$($\Delta$x)')
plt.ylabel(r'$-log_{10}$(Relative Error)')
plt.grid()
plt.legend()
plt.show()

for i in range(len(k_list)):
    plt.plot(hf_dx_err[i], hf_log_err[i], label='K = '+str(k_list[i]))
plt.title('1/3 Simpson Integration Heat Flux\n '
          + r'-$log_{10}$($\Delta$x) vs -$log_{10}$(Relative Error)')
plt.xlabel(r'$-log_{10}$($\Delta$x)')
plt.ylabel(r'$-log_{10}$(Relative Error)')
plt.grid()
plt.legend()
plt.show()

for i in range(len(k_list)):
    plt.plot(tc_dx_err[i], tc_log_err[i], label='K = '+str(k_list[i]))
plt.title('2nd order FDM\n' + r'-$log_{10}$($\Delta$x) vs -$log_{10}$(Relative Error)')
plt.xlabel(r'$-log_{10}$($\Delta$x)')
plt.ylabel(r'$-log_{10}$(Relative Error)')
plt.grid()
plt.legend()
plt.show()
