"""
AERO 430
Exam 2 Code

Andrew Hollister
127008398
"""

import tabulate as tbl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import seaborn as sns
import numpy as np
import math
import time

"""
Data Class Structure
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


def exact_temp(x, y, k1, k2, interface):

    # Heat flow constant across y for material 1 and 2
    kyy1 = kxx/k1**2
    kyy2 = kxx/k2**2

    # Y bar function
    y_bar = 100/(np.sinh(k2*np.pi*(1-interface))*((kyy1*k1*(1/np.tanh(interface*np.pi*k1)))
                                                  / (kyy2*k2)+(1/np.tanh(k2*np.pi*(1-interface)))))

    # Material 1 Exact Solution
    temp1 = y_bar*np.sin(np.pi*x)*np.sinh(k1*np.pi*y)/np.sinh(interface*np.pi*k1)

    # Material 2 Exact Solution
    temp2 = np.sin(np.pi*x)*((100-y_bar*np.cosh(k2*np.pi*(1-interface)))*np.sinh(k2*np.pi*(y-interface))
                             / np.sinh(k2*np.pi*(1-interface))+y_bar*np.cosh(k2*np.pi*(y-interface)))

    temp = np.where(y < interface, temp1, temp2)

    return temp


"""
2nd Order FDM Solution 
"""


def bc_temp_func(x):
    return -100*np.sin(np.pi*x)  # deg c


def approx_temp(x, y, interface, k1, k2):

    # Heat Conduction Variables
    kyy1 = kxx/k1**2
    kyy2 = kxx/k2**2

    # Sizing mesh
    n_nodes = len(x)
    dim = n_nodes**2

    # Creating Meshes for reference
    y_mesh = np.meshgrid(x, y)[1]

    # Defining dx and dy
    dx = x[1]-x[0]
    dy1 = y[1]-y[0]
    dy2 = y[-1]-y[-2]

    # Building A matrix
    a = np.zeros([dim, dim])

    count = 0
    mesh_row = 0
    for row_i, row in enumerate(a):

        # Need to figure out a way to properly figure out the y value of the previous and next node
        if count == n_nodes:
            mesh_row += 1
            count = 0
        count += 1

        # Heat Conduction Variable
        if np.reshape(y_mesh, dim)[row_i] < interface:  # Material Behind Interface

            # Adding Diagonal Terms
            a[row_i][row_i] = 2*(k1**2*dy1**2/dx**2+1)

            # Adding Off-Diagonal Terms
            if row_i - 1 < 0:
                pass
            else:
                a[row_i][row_i - 1] = -k1**2*dy1**2/dx**2
            try:
                a[row_i][row_i + 1] = -k1**2*dy1**2/dx**2
            except IndexError:
                pass
            try:
                a[row_i][row_i + n_nodes] = -1
            except IndexError:
                pass
            if row_i - n_nodes < 0:
                pass
            else:
                a[row_i][row_i - n_nodes] = -1

        elif np.reshape(y_mesh, dim)[row_i] > interface:  # Material Ahead of Interface

            # Adding Diagonal Terms
            a[row_i][row_i] = 2*(k2**2*dy2**2/dx**2+1)

            # Adding Off-Diagonal Terms
            if row_i - 1 < 0:
                pass
            else:
                a[row_i][row_i - 1] = -k2**2*dy2**2/dx**2
            try:
                a[row_i][row_i + 1] = -k2**2*dy2**2/dx**2
            except IndexError:
                pass
            try:
                a[row_i][row_i + n_nodes] = -1
            except IndexError:
                pass
            if row_i - n_nodes < 0:
                pass
            else:
                a[row_i][row_i - n_nodes] = -1

        elif np.reshape(y_mesh, dim)[row_i] == interface:  # Material Along Interface

            # Adding Diagonal Terms
            a[row_i][row_i] = (kyy1/dy1+kxx*dy1/dx**2+kxx*dy2/dx**2+kyy2/dy2)

            # Adding Off-Diagonal Terms
            a[row_i][row_i - 1] = -(0.5*kxx*dy1/dx**2+0.5*kxx*dy2/dx**2)
            a[row_i][row_i + 1] = -(0.5*kxx*dy1/dx**2+0.5*kxx*dy2/dx**2)
            a[row_i][row_i - n_nodes] = -kyy1/dy1
            a[row_i][row_i + n_nodes] = -kyy2/dy2

    # Removing Extraneous Terms
    for i in range(dim):
        if i % n_nodes == 0 and i != 0:
            a[i-1][i] = 0
            a[i][i-1] = 0

    # Generating B Matrix
    b = np.zeros([dim, 1])
    dxl = length/(n_nodes+1)
    for i in range(n_nodes):
        xl = dxl*(i+1)
        b[-i-1][0] = -bc_temp_func(xl)

    # Solving Matrix
    temps = np.reshape(np.linalg.solve(a, b), [n_nodes, n_nodes])
    temps = np.flip(temps)
    temps = np.row_stack((-bc_temp_func(np.linspace(0, length, n_nodes+2))[1:-1], temps))
    temps = np.row_stack((temps, np.zeros(n_nodes)))
    temps = np.column_stack((temps, np.zeros(n_nodes+2)))
    temps = np.column_stack((np.zeros(n_nodes+2), temps))

    return np.flip(temps)


"""
Heat Flux Functions
"""


# Exact Heat Transfer Function
def exact_heat_transfer(k1, k2, interface):

    # Heat flow constant across y for material 1 and 2
    kyy1 = kxx / k1 ** 2
    kyy2 = kxx / k2 ** 2

    # Y bar function
    y_bar = 100 / (np.sinh(k2 * np.pi * (1 - interface)) * ((kyy1 * k1 * (1 / np.tanh(interface * np.pi * k1)))
                                                            / (kyy2 * k2) + (
                                                                        1 / np.tanh(k2 * np.pi * (1 - interface)))))

    # Heat FLux Through Top Boundary
    q_ex = ((100-y_bar*np.cosh(k2*np.pi*(1-interface))) * np.cosh(k2*np.pi*(length-interface))
            / np.sinh(k2*np.pi*(length-interface)) + y_bar*np.sinh(k2*np.pi*(length-interface)))

    return -q_ex


# Approximate Heat Flux Function
def approx_heat_transfer(lower, upper, nodes, fdm_temp, k, y):
    dtdy = []
    dy = y[-1]-y[-2]
    for i in range(len(fdm_temp)):
        col = fdm_temp[:, i]
        dtdy.append((col[-3]-4*col[-2]+3*col[-1])/(2*dy))

    summation = 0
    for i in range(nodes + 1):
        summand = dtdy[i]
        if (i != 0) and (i != nodes):
            summand *= (2 + (2 * (i % 2)))
        summation += summand

    val = ((upper - lower) / (3 * nodes)) * summation
    return -2*k*val


"""
Richardson Extrapolation Function
"""


def q_rich_extr(q, q2, q4):
    q_extr = (q2**2 - q*q4)/(2*q2-q-q4)
    perc_err = abs(abs(q_extr-q)/q_extr)*100
    return q_extr, perc_err


"""
Convergence Rate Function
"""


def get_beta(exact, approx, approx_2, h, h_2):
    a = np.log(abs(exact - approx))
    b = np.log(abs(exact - approx_2))
    c = np.log(h) - np.log(h_2)
    return -(a - b) / c


"""
Main Function
"""


def main():

    # Material (1, 2) Heat Conductivity
    k_list = [(0.005, 0.5),
              (0.05, 0.5),
              (0.5, 0.5),
              (1, 0.5),
              (2, 0.5),
              (3, 0.5)]
    global kxx
    kxx = 1  # Heat Conductivity of both materials in x flow

    # Interface location list
    inter_list = [np.pi/6]

    # Defining mesh sizes to loop through
    node_list = range(3, 51, 2)

    # Bar properties
    global length
    length = 1

    # Data Structures
    nonc_fdm_data = DataStructure()
    conf_fdm_data = DataStructure()

    """
    Data Generation
    """

    for inter in inter_list:
        for k in k_list:
            k1, k2 = k
            for n in node_list:
                n_nodes = n

                # Non-Conformal Mesh
                x_vals = np.linspace(0, length, n_nodes)
                y_vals = np.linspace(0, length, n_nodes)

                # Conformal Mesh
                y_vals_a = np.linspace(inter, length, 1+math.trunc(n_nodes / 2))
                y_vals_b = np.linspace(0, inter, math.ceil(n_nodes / 2))[:-1]
                y_vals_c = np.append(y_vals_b, y_vals_a)

                # Generating Data
                conf_sol = approx_temp(x_vals, y_vals_c, inter, k1, k2)
                nonc_sol = approx_temp(x_vals, y_vals, 1/3, k1, k2)

                # Resizing Meshes
                x_vals = np.linspace(0, length, n_nodes+2)
                y_vals = np.linspace(0, length, n_nodes+2)
                y_vals_a = np.linspace(inter, length, 1+math.trunc((n_nodes + 2) / 2))
                y_vals_b = np.linspace(0, inter, math.ceil((n_nodes + 2) / 2))[:-1]
                y_vals_c = np.append(y_vals_b, y_vals_a)

                # Appending Data to Data Structures
                conf_fdm_data.add_data(n_nodes+2, k, x_vals, y_vals_c, conf_sol)
                nonc_fdm_data.add_data(n_nodes+2, k, x_vals, y_vals, nonc_sol)

    """
    Conformal Mesh FDM Heat Flux and Error Calculations
    """

    # Loop Lists
    re_log_err = []
    re_dx_err = []
    hf_log_err = []
    hf_dx_err = []
    tc_log_err = []
    tc_dx_err = []

    for inter in inter_list:
        for k in k_list:

            k1, k2 = k

            heat_flux_data = []
            temp_error_data = []
            for n in node_list:
                n_nodes = n
                x_vals, y_vals, z = conf_fdm_data.return_data(n_nodes + 2, k)

                # Heat Flux Calculations and Error Calculations
                q_exact = exact_heat_transfer(k1, k2, inter)
                q_approx = approx_heat_transfer(0, 1, n_nodes + 1, z, k2, y_vals)
                error_q = abs((q_exact - q_approx) / q_exact * 100)
                heat_flux_data.append([n_nodes + 1, length / (n_nodes + 1), q_exact, q_approx, error_q])

                # Temperature Error Calculations
                inter_index = int(len(z) / 2)
                midpoint = length / (n_nodes + 1) * inter_index
                approx_t = z[inter_index][inter_index]
                exact_t = exact_temp(midpoint, inter, k1, k2, inter)
                error_t = abs(exact_t - approx_t) / exact_t * 100
                temp_error_data.append([n_nodes + 1, length / (n_nodes + 1), exact_t, approx_t, error_t])

            # Richardson Extrapolation and Convergence
            heat_flux_rc_data = []
            for i in range(len(heat_flux_data) - 2):
                q_extr, perc_err = q_rich_extr(heat_flux_data[i][3], heat_flux_data[i + 1][3],
                                               heat_flux_data[i + 2][3])
                beta = get_beta(q_extr, heat_flux_data[i][3], heat_flux_data[i + 1][3],
                                heat_flux_data[i][0], heat_flux_data[i + 1][0])
                heat_flux_rc_data.append([heat_flux_data[i][0], heat_flux_data[i][1], q_extr, heat_flux_data[i][3],
                                          perc_err, beta])
            print(f'\nConformal Convergence of Extrapolated Heat Flux (k1 = {k1}, k2 = {k2}):')
            print(tbl.tabulate(heat_flux_rc_data,
                               headers=['Num. Elements', 'dx', 'Extrapolated Heat Flux', 'Approx. Heat Loss',
                                        'Percent Error', 'Beta']))

            re_log_err.append([-np.log10(item[4] / 100) for item in heat_flux_rc_data])
            re_dx_err.append([-np.log10(item[1]) for item in heat_flux_rc_data])

            # Heat Flux Convergence
            heat_flux_data[0].append('n/a')
            for i in range(len(heat_flux_data) - 1):
                heat_flux_data[i + 1].append(
                    get_beta(heat_flux_data[i][2], heat_flux_data[i][3], heat_flux_data[i + 1][3],
                             heat_flux_data[i][0], heat_flux_data[i + 1][0]))
            print(f'\nConformal Convergence of Heat Flux (k1 = {k1}, k2 = {k2}):')
            print(tbl.tabulate(heat_flux_data, headers=['Num. Elements', 'dx', 'Exact Heat Flux', 'Approx. Heat Loss',
                                                        'Percent Error', 'Beta']))
            hf_log_err.append([-np.log10(item[4] / 100) for item in heat_flux_data])
            hf_dx_err.append([-np.log10(item[1]) for item in heat_flux_data])

            # Temperature Convergence
            temp_error_data[0].append('n/a')
            for i in range(len(temp_error_data) - 1):
                temp_error_data[i + 1].append(
                    get_beta(temp_error_data[i][2], temp_error_data[i][3], temp_error_data[i + 1][3],
                             temp_error_data[i][0], temp_error_data[i + 1][0]))
            print(f'\nConformal Convergence of 2nd Order FDM Temperature Solution (k1 = {k1}, k2 = {k2}):')
            print(tbl.tabulate(temp_error_data,
                               headers=['Num. Elements', 'dx', 'Exact Midpoint Temp', 'Approx. Midpoint Temp',
                                        'Percent Error', 'Beta']))
            tc_log_err.append([-np.log10(item[4] / 100) for item in temp_error_data])
            tc_dx_err.append([-np.log10(item[1]) for item in temp_error_data])

        # Plotting Convergence Graphs
        for i in range(len(k_list)):
            plt.plot(re_dx_err[i], re_log_err[i], label=f'k1 = {k_list[i][0]}, k2 = {k_list[i][1]}')
        plt.title('Conformal Extrapolated Heat Flux\n' + r'-$log_{10}$($\Delta$x) vs -$log_{10}$(Relative Error)')
        plt.xlabel(r'$-log_{10}$($\Delta$x)')
        plt.ylabel(r'$-log_{10}$(Relative Error)')
        plt.grid()
        plt.legend()
        plt.show()

        for i in range(len(k_list)):
            plt.plot(hf_dx_err[i], hf_log_err[i], label=f'k1 = {k_list[i][0]}, k2 = {k_list[i][1]}')
        plt.title('Conformal 1/3 Simpson Integration Heat Flux\n '
                  + r'-$log_{10}$($\Delta$x) vs -$log_{10}$(Relative Error)')
        plt.xlabel(r'$-log_{10}$($\Delta$x)')
        plt.ylabel(r'$-log_{10}$(Relative Error)')
        plt.grid()
        plt.legend()
        plt.show()

        for i in range(len(k_list)):
            plt.plot(tc_dx_err[i], tc_log_err[i], label=f'k1 = {k_list[i][0]}, k2 = {k_list[i][1]}')
        plt.title('Conformal 2nd order FDM\n' + r'-$log_{10}$($\Delta$x) vs -$log_{10}$(Relative Error)')
        plt.xlabel(r'$-log_{10}$($\Delta$x)')
        plt.ylabel(r'$-log_{10}$(Relative Error)')
        plt.grid()
        plt.legend()
        plt.show()

    """
    Non-Conformal Mesh FDM Heat Flux and Error Calculations
    """

    # Loop Lists
    re_log_err = []
    re_dx_err = []
    hf_log_err = []
    hf_dx_err = []
    tc_log_err = []
    tc_dx_err = []

    for inter in inter_list:
        for k in k_list:

            k1, k2 = k

            heat_flux_data = []
            temp_error_data = []
            for n in node_list:
                n_nodes = n
                x_vals, y_vals, z = nonc_fdm_data.return_data(n_nodes + 2, k)

                # Heat Flux Calculations and Error Calculations
                q_exact = exact_heat_transfer(k1, k2, 1/3)
                q_approx = approx_heat_transfer(0, 1, n_nodes + 1, z, k2, y_vals)
                error_q = abs((q_exact - q_approx) / q_exact * 100)
                heat_flux_data.append([n_nodes + 1, length / (n_nodes + 1), q_exact, q_approx, error_q])

                # Temperature Error Calculations
                inter_index = int(len(z) / 2)
                midpoint = length / (n_nodes + 1) * inter_index
                approx_t = z[inter_index][inter_index]
                exact_t = exact_temp(midpoint, midpoint, k1, k2, 1/3)
                error_t = abs(exact_t - approx_t) / exact_t * 100
                temp_error_data.append([n_nodes + 1, length / (n_nodes + 1), exact_t, approx_t, error_t])

            # Richardson Extrapolation and Convergence
            heat_flux_rc_data = []
            for i in range(len(heat_flux_data) - 2):
                q_extr, perc_err = q_rich_extr(heat_flux_data[i][3], heat_flux_data[i + 1][3],
                                               heat_flux_data[i + 2][3])
                beta = get_beta(q_extr, heat_flux_data[i][3], heat_flux_data[i + 1][3],
                                heat_flux_data[i][0], heat_flux_data[i + 1][0])
                heat_flux_rc_data.append([heat_flux_data[i][0], heat_flux_data[i][1], q_extr, heat_flux_data[i][3],
                                          perc_err, beta])
            print(f'\nNon-Conformal Convergence of Extrapolated Heat Flux (k1 = {k1}, k2 = {k2}):')
            print(tbl.tabulate(heat_flux_rc_data,
                               headers=['Num. Elements', 'dx', 'Extrapolated Heat Flux', 'Approx. Heat Loss',
                                        'Percent Error', 'Beta']))

            re_log_err.append([-np.log10(item[4] / 100) for item in heat_flux_rc_data])
            re_dx_err.append([-np.log10(item[1]) for item in heat_flux_rc_data])

            # Heat Flux Convergence
            heat_flux_data[0].append('n/a')
            for i in range(len(heat_flux_data) - 1):
                heat_flux_data[i + 1].append(
                    get_beta(heat_flux_data[i][2], heat_flux_data[i][3], heat_flux_data[i + 1][3],
                             heat_flux_data[i][0], heat_flux_data[i + 1][0]))
            print(f'\nNon-Conformal Convergence of Heat Flux (k1 = {k1}, k2 = {k2}):')
            print(tbl.tabulate(heat_flux_data, headers=['Num. Elements', 'dx', 'Exact Heat Flux', 'Approx. Heat Loss',
                                                        'Percent Error', 'Beta']))
            hf_log_err.append([-np.log10(item[4] / 100) for item in heat_flux_data])
            hf_dx_err.append([-np.log10(item[1]) for item in heat_flux_data])

            # Temperature Convergence
            temp_error_data[0].append('n/a')
            for i in range(len(temp_error_data) - 1):
                temp_error_data[i + 1].append(
                    get_beta(temp_error_data[i][2], temp_error_data[i][3], temp_error_data[i + 1][3],
                             temp_error_data[i][0], temp_error_data[i + 1][0]))
            print(f'\nNon-Conformal Convergence of 2nd Order FDM Temperature Solution (k1 = {k1}, k2 = {k2}):')
            print(tbl.tabulate(temp_error_data,
                               headers=['Num. Elements', 'dx', 'Exact Midpoint Temp', 'Approx. Midpoint Temp',
                                        'Percent Error', 'Beta']))
            tc_log_err.append([-np.log10(item[4] / 100) for item in temp_error_data])
            tc_dx_err.append([-np.log10(item[1]) for item in temp_error_data])

            # Plotting Convergence Graphs
        for i in range(len(k_list)):
            plt.plot(re_dx_err[i], re_log_err[i], label=f'k1 = {k_list[i][0]}, k2 = {k_list[i][1]}')
        plt.title('Non-Conformal Extrapolated Heat Flux\n' + r'-$log_{10}$($\Delta$x) vs -$log_{10}$(Relative Error)')
        plt.xlabel(r'$-log_{10}$($\Delta$x)')
        plt.ylabel(r'$-log_{10}$(Relative Error)')
        plt.grid()
        plt.legend()
        plt.show()

        for i in range(len(k_list)):
            plt.plot(hf_dx_err[i], hf_log_err[i], label=f'k1 = {k_list[i][0]}, k2 = {k_list[i][1]}')
        plt.title('Non-Conformal 1/3 Simpson Integration Heat Flux\n '
                  + r'-$log_{10}$($\Delta$x) vs -$log_{10}$(Relative Error)')
        plt.xlabel(r'$-log_{10}$($\Delta$x)')
        plt.ylabel(r'$-log_{10}$(Relative Error)')
        plt.grid()
        plt.legend()
        plt.show()

        for i in range(len(k_list)):
            plt.plot(tc_dx_err[i], tc_log_err[i], label=f'k1 = {k_list[i][0]}, k2 = {k_list[i][1]}')
        plt.title('Non-Conformal 2nd order FDM\n' + r'-$log_{10}$($\Delta$x) vs -$log_{10}$(Relative Error)')
        plt.xlabel(r'$-log_{10}$($\Delta$x)')
        plt.ylabel(r'$-log_{10}$(Relative Error)')
        plt.grid()
        plt.legend()
        plt.show()

    for inter in inter_list:
        for k in k_list:
            k1, k2 = k
            for n_nodes in [49]:

                """
                Conformal Mesh Related Plotting
                """

                # Extracting values
                x_vals, y_vals, z = conf_fdm_data.return_data(n_nodes+2, k)
                x, y = np.meshgrid(x_vals, y_vals)
                x1, y1, = np.meshgrid(x_vals, y_vals)
                z1 = exact_temp(x1, y1, k1, k2, inter)

                """
                Exact Solution Plotting
                """

                # 3-Dimensional Heat Maps
                ax = plt.axes(projection='3d')
                ax.plot_surface(x1, y1, z1, rstride=1, cstride=1,
                                cmap='inferno', edgecolor='none')
                plt.title(f'Conformal Mesh Exact Solution 3-Dimensional Heatmap\nk1 = {k1} |'
                          f' k2 = {k2}\nInterface: y = {round(inter, 3)}')
                ax.set_xlabel('X (cm)')
                ax.set_ylabel('Y (cm)')
                ax.set_zlabel(r'T(x, y)$\degree$C')
                plt.show()

                # 2-Dimensional Heat Maps
                ax = sns.heatmap(z1, xticklabels=False, yticklabels=False,
                                 cbar_kws={'label': u'Temperature \N{DEGREE SIGN}C'})
                plt.xticks((n_nodes + 2) * np.array([0, 0.25, 0.5, 0.75, 1]), labels=[0, 0.25, 0.5, 0.75, 1])
                ax.set_xlabel('x (cm)')
                plt.yticks((n_nodes + 2) * np.array([0, 0.25, 0.5, 0.75, 1]), labels=[0, 0.25, 0.5, 0.75, 1])
                ax.set_ylabel('y (cm)')
                ax.set_title(f'Conformal Mesh Exact Solution Heatmap\nk1 = {k1} '
                             f'| k2 = {k2}\nInterface: y = {round(inter, 3)}')
                plt.gca().invert_yaxis()
                plt.show()

                """
                Conformal Mesh FDM Plotting
                """

                # 3-Dimensional Heat Maps
                ax = plt.axes(projection='3d')
                ax.plot_surface(x, y, z, rstride=1, cstride=1,
                                cmap='inferno', edgecolor='none')
                plt.title(
                    r'Conformal 2nd order FDM Output T(x, y)$\degree$ '
                    r'C' + '\nNodes = ' + str(n_nodes+2) + '\nK = ' + str(k))
                ax.set_xlabel('X (cm)')
                ax.set_ylabel('Y (cm)')
                ax.set_zlabel(r'T(x, y)$\degree$C')
                plt.show()

                # 2-Dimensional Heat Maps
                ax = sns.heatmap(z, xticklabels=False, yticklabels=False,
                                 cbar_kws={'label': u'Temperature \N{DEGREE SIGN}C'})
                plt.xticks((n_nodes + 2) * np.array([0, 0.25, 0.5, 0.75, 1]), labels=[0, 0.25, 0.5, 0.75, 1])
                ax.set_xlabel('x (cm)')
                plt.yticks((n_nodes + 2) * np.array([0, 0.25, 0.5, 0.75, 1]), labels=[0, 0.25, 0.5, 0.75, 1])
                ax.set_ylabel('y (cm)')
                ax.set_title('Conformal FDM Heatmap of cross-section'
                             ' of bar\n' + str(n_nodes + 2) + ' Nodes' + '\nK = ' + str(k))
                plt.gca().invert_yaxis()
                plt.show()

                """
                Non-Conformal Mesh Related Plotting
                """

                # Extracting values
                x_vals, y_vals, z = nonc_fdm_data.return_data(n_nodes + 2, k)
                x, y = np.meshgrid(x_vals, y_vals)
                x1, y1, = np.meshgrid(x_vals, y_vals)
                z1 = exact_temp(x1, y1, k1, k2, inter)

                """
                Exact Solution Plotting
                """

                # 3-Dimensional Heat Maps
                ax = plt.axes(projection='3d')
                ax.plot_surface(x1, y1, z1, rstride=1, cstride=1,
                                cmap='inferno', edgecolor='none')
                plt.title(f'Non-Conformal Mesh Exact Solution 3-Dimensional Heatmap\nk1 = {k1} |'
                          f' k2 = {k2}\nInterface: y = {round(inter, 3)}')
                ax.set_xlabel('X (cm)')
                ax.set_ylabel('Y (cm)')
                ax.set_zlabel(r'T(x, y)$\degree$C')
                plt.show()

                # 2-Dimensional Heat Maps
                ax = sns.heatmap(z1, xticklabels=False, yticklabels=False,
                                 cbar_kws={'label': u'Temperature \N{DEGREE SIGN}C'})
                plt.xticks((n_nodes + 2) * np.array([0, 0.25, 0.5, 0.75, 1]), labels=[0, 0.25, 0.5, 0.75, 1])
                ax.set_xlabel('x (cm)')
                plt.yticks((n_nodes + 2) * np.array([0, 0.25, 0.5, 0.75, 1]), labels=[0, 0.25, 0.5, 0.75, 1])
                ax.set_ylabel('y (cm)')
                ax.set_title(f'Non-Conformal Mesh Exact Solution Heatmap\nk1 = {k1}'
                             f' | k2 = {k2}\nInterface: y = {round(inter, 3)}')
                plt.gca().invert_yaxis()
                plt.show()

                """
                Non-Conformal Mesh FDM Plotting
                """

                # 3-Dimensional Heat Maps
                ax = plt.axes(projection='3d')
                ax.plot_surface(x, y, z, rstride=1, cstride=1,
                                cmap='inferno', edgecolor='none')
                plt.title(r'Non-Conformal 2nd order FDM Output T(x, y)$\degree$ C'
                          r'' + '\nNodes = ' + str(n_nodes + 2) + '\nK = ' + str(k))
                ax.set_xlabel('X (cm)')
                ax.set_ylabel('Y (cm)')
                ax.set_zlabel(r'T(x, y)$\degree$C')
                plt.show()

                # 2-Dimensional Heat Maps
                ax = sns.heatmap(z, xticklabels=False, yticklabels=False,
                                 cbar_kws={'label': u'Temperature \N{DEGREE SIGN}C'})
                plt.xticks((n_nodes + 2) * np.array([0, 0.25, 0.5, 0.75, 1]), labels=[0, 0.25, 0.5, 0.75, 1])
                ax.set_xlabel('x (cm)')
                plt.yticks((n_nodes + 2) * np.array([0, 0.25, 0.5, 0.75, 1]), labels=[0, 0.25, 0.5, 0.75, 1])
                ax.set_ylabel('y (cm)')
                ax.set_title('Non-Conformal Heatmap of cross-section'
                             ' of bar\n' + str(n_nodes + 2) + ' Nodes' + '\nK = ' + str(k))
                plt.gca().invert_yaxis()
                plt.show()


if __name__ == "__main__":
    print('\nNote: This program may take a minute to generate all data...')
    main()
