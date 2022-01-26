"""
AERO 430
Exam 1 Code

Andrew Hollister
127008398
"""

# Imports and constants
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema as min
import tabulate as tbl
import math
pi = np.pi


# Creating a data structure to more easily keep track of all data generated
class DataStructure:
    # Class initialization
    def __init__(self):
        self.data = []

    # Function for adding data to data structure
    def add_data(self, interface, g_ss, g_cs, b, x, t, n_elem):
        self.data.append({"Interface": interface, "Beta": b, "x": x, "Temperature": t, "Nodes": n_elem,
                          "Gamma SS": g_ss, "Gamma CS": g_cs})

    # Function for returning data for graphing
    def return_data(self, interface, g_ss, g_cs, b, n_elem):
        y_data = [item["Temperature"] for item in self.data if item['Interface'] == interface
                  and item['Beta'] == b and item["Nodes"] == n_elem and item['Gamma CS'] == g_cs
                  and item["Gamma SS"] == g_ss]

        x_data = [item["x"] for item in self.data if item['Interface'] == interface
                  and item['Beta'] == b and item["Nodes"] == n_elem and item['Gamma CS'] == g_cs
                  and item["Gamma SS"] == g_ss]

        return x_data, y_data

    def return_temp(self, interface, g_ss, g_cs, b, n_elem, x):
        return [item["Temperature"] for item in self.data if item['Interface'] == interface
                and item['Beta'] == b and item["Nodes"] == n_elem and item['Gamma CS'] == g_cs
                and item["Gamma SS"] == g_ss and item['x'] == x]


class HeatData:
    # Class Initialization
    def __init__(self):
        self.data = []

    def add_data(self, T, vals, dx, inter):
        self.data.append({'T': T, 'vals': vals, 'dx': np.log(dx), 'inter': inter})

    def return_data(self, vals, inter):
        y_data = [item['T'] for item in self.data if item['vals'] == vals and item['inter'] == inter]
        x_data = [item['dx'] for item in self.data if item['vals'] == vals and item['inter'] == inter]

        return x_data, y_data


# Post processing
def get_error(exact: float, approx: float):
    return abs((exact-approx)/exact)


# Case 1: Analytical Solution
# Bar Properties
x1 = 3  # cm
x3 = 6.5  # cm
Beta = [0.4, 0.55]
gamma_ss = [0.67, 0.90]
gamma_cs = [1.50, 2.10]

min_case = [Beta[1], gamma_ss[0], gamma_cs[0]]
max_case = [Beta[0], gamma_ss[1], gamma_cs[1]]

# Interface Location
interface = np.array([x1 + (x3 - x1) / 7, x1 + (x3 - x1) / (2 * pi)])

# Temperatures
T0 = 500  # deg C
T1 = 20  # deg C


def get_beta(exact, approx, approx_2, h, h_2):
    A = np.log(abs(exact - approx))
    B = np.log(abs(exact - approx_2))
    C = np.log(h) - np.log(h_2)
    return -(A - B) / C


def rich_extra(q, q2, q4):
    q_extr = (q2**2 - q*q4)/(2*q2-q-q4)
    beta = np.log((q_extr - q2)/(q_extr-q4))/np.log(2)
    return q_extr, beta

# Heat Loss to the environment
def heat_loss_bc(T3, B):
    return 2*pi*B*x3*(T3-T1)


# Heat Loss to the environment
def heat_loss_der(T3, Tn_1, g, dx):
    return -2*pi*g*x3*(T3-Tn_1)/dx


def temp_exact(x, x2, g_ss, g_cs, B):
        del_t = T1-T0
        c_coefficient = (1/B/x3) + (1/g_cs*np.log(x3/x2)) + (1/g_ss*np.log(x2/x1))
        c = del_t/c_coefficient

        c1 = c
        c2 = T0-(c/g_ss*np.log(x1))
        c3 = c1
        c4 = c / g_ss * np.log(x2) - c / g_cs * np.log(x2) + c2

        if x < x2:
            return c1/g_ss*np.log(x) + c2
        elif x >= x2:
            return c3/g_cs*np.log(x) + c4


def temp_fdm(x2, mesh, g_ss, g_cs, B):
    n_nodes = len(mesh)
    for node in range(n_nodes):

        # Creating A matrix
        A = np.zeros((n_nodes, n_nodes))

        for node in range(1, len(mesh)-1):
            dx_l = (mesh[node]-mesh[node-1])
            dx_r = (mesh[node+1]-mesh[node])

            x_mid_l = (mesh[node]+mesh[node-1])/2
            x_mid_r = (mesh[node+1]+mesh[node])/2

            if x_mid_l < x2:
                g_mid_l = g_ss
            else:
                g_mid_l = g_cs

            if x_mid_r < x2:
                g_mid_r = g_ss
            else:
                g_mid_r = g_cs

            k_l = (x_mid_l*g_mid_l/dx_l)
            k_r = (x_mid_r*g_mid_r/dx_r)

            A[node][node - 1] = -k_l

            A[node][node] = k_l+k_r

            A[node][node + 1] = -k_r

        dx = mesh[-1] - mesh[-2]
        x_mid = (mesh[-1]+mesh[-2])/2
        A[-1][-2] = -x_mid*g_cs/(dx*B*x3)
        A[-1][-1] = x_mid*g_cs/(dx*B*x3) + 1
        A[0][0] = 1

        # Creating B matrix
        B_mat = np.zeros(n_nodes)
        B_mat[0] = T0  # deg C
        B_mat[-1] = T1

        # Solving
        T = np.linalg.solve(A, B_mat)

        return T, mesh


Q_der_con = HeatData()
Q_bc_con = HeatData()
Q_der_nonc = HeatData()
Q_bc_nonc = HeatData()
Q_bc_exact = HeatData()
Q_der_exact = HeatData()

# data collection loop
nodes = [3 + interval for interval in range(97)]+[99 + 2**interval for interval in range(11)]
exact_con_data = DataStructure()
exact_nonc_data = DataStructure()
con_fdm_data = DataStructure()
nonc_fdm_data = DataStructure()
for inter in interface:
    for vals in [max_case, min_case]:

        B = vals[0]
        g_ss = vals[1]
        g_cs = vals[2]

        for num_elem in nodes:

            # Building non-conformal mesh
            nonc_mesh = np.linspace(x1, x3, num_elem)

            # Building conformal mesh
            con_mesh_A = np.linspace(inter, x3, 1 + math.trunc(num_elem / 2))
            con_mesh_B = np.linspace(x1, inter, math.ceil(num_elem / 2))[:-1]
            con_mesh = np.append(con_mesh_B, con_mesh_A)

            # Deriving data for non conformal and conformal meshes
            con_fdm_temp, con_fdm_mesh = temp_fdm(inter, con_mesh, g_ss, g_cs, B)
            nonc_fdm_temp, nonc_fdm_mesh = temp_fdm(inter, nonc_mesh, g_ss, g_cs, B)

            # Saving conformal mesh data to data structure and creating corresponding exact data
            for i in range(len(con_fdm_mesh)):
                x = con_fdm_mesh[i]
                exact_temp = temp_exact(x, inter, g_ss, g_cs, B)
                exact_con_data.add_data(inter, g_ss, g_cs, B, x, exact_temp, num_elem)
                con_fdm_data.add_data(inter, g_ss, g_cs, B, x, con_fdm_temp[i], num_elem)

            # Saving non-conformal mesh data to data structure and creating corresponding exact data
            for i in range(len(nonc_fdm_mesh)):
                x = nonc_fdm_mesh[i]
                exact_temp = temp_exact(x, inter, g_ss, g_cs, B)
                exact_nonc_data.add_data(inter, g_ss, g_cs, B, x, exact_temp, num_elem)
                nonc_fdm_data.add_data(inter, g_ss, g_cs, B, x, nonc_fdm_temp[i], num_elem)


# Creating Table
if input('Create Temperature Tables? (y/n): ') == 'y':
    for inter in interface:
        for vals in [min_case, max_case]:
            B = vals[0]
            g_ss = vals[1]
            g_cs = vals[2]

            if vals == min_case:
                title = 'Min Case'
            else:
                title = 'Max Case'

            table = []
            print('\nConformal Mesh FDM Compared to Exact Solution')
            print(title+', Interface Lccated at x = '+str(inter)+' cm')
            headers = ['x (cm)', 'FDM Temp (\u00B0C)', 'Exact Temp (\u00B0C)', 'Percent Error']
            x_data, y_data = con_fdm_data.return_data(inter, g_ss, g_cs, B, 50)
            exact_y_data = exact_con_data.return_data(inter, g_ss, g_cs, B, 50)[1]

            err = []
            for i in range(len(exact_y_data)):
                err.append(get_error(exact_y_data[i], y_data[i])*100)

            table.append(x_data)
            table.append(y_data)
            table.append(exact_y_data)
            table.append(err)
            print(tbl.tabulate(np.transpose(np.array(table)), headers=headers))

            table = []
            print('\nNon-Conformal Mesh FDM Compared to Exact Solution')
            print(title+', Interface Located at x = '+str(inter)+' cm')
            x_data, y_data = nonc_fdm_data.return_data(inter, g_ss, g_cs, B, 50)
            exact_y_data = exact_nonc_data.return_data(inter, g_ss, g_cs, B, 50)[1]

            err = []
            for i in range(len(exact_y_data)):
                err.append(get_error(exact_y_data[i], y_data[i])*100)

            table.append(x_data)
            table.append(y_data)
            table.append(exact_y_data)
            table.append(err)
            print(tbl.tabulate(np.transpose(np.array(table)), headers=headers))

            input('\nProceed?')

# Computing the heat loss
hl_nodes = [3 + interval for interval in range(97)][::2]+[99 + 2**interval for interval in range(1, 11)]
if input('Heat Transfer and Convergence? (y/n): ') == 'y':
    for inter in interface:
        for vals in [min_case, max_case]:
            B = vals[0]
            g_ss = vals[1]
            g_cs = vals[2]

            for num_elem in hl_nodes:

                # Heat Transfer Calculations
                T = con_fdm_data.return_data(inter, g_ss, g_cs, B, num_elem)[1]
                dx = con_fdm_data.return_data(inter, g_ss, g_cs, B, num_elem)[0][-1] - con_fdm_data.return_data(inter, g_ss, g_cs, B, num_elem)[0][-2]
                Q_bc_con.add_data(heat_loss_der(T[-1], T[-2], g_cs, dx), vals, dx, inter)
                Q_der_con.add_data(heat_loss_bc(T[-1], B), vals, dx, inter)

                T = nonc_fdm_data.return_data(inter, g_ss, g_cs, B, num_elem)[1]
                dx = nonc_fdm_data.return_data(inter, g_ss, g_cs, B, num_elem)[0][-1] - nonc_fdm_data.return_data(inter, g_ss, g_cs, B, num_elem)[0][-2]
                Q_bc_nonc.add_data(heat_loss_der(T[-1], T[-2], g_cs, dx), vals, dx, inter)
                Q_der_nonc.add_data(heat_loss_bc(T[-1], B), vals, dx, inter)

                T3 = temp_exact(x3, inter, g_ss, g_cs, B)
                dx = (x3-x1)/num_elem
                TN_1 = temp_exact(x3 - dx, inter, g_ss, g_cs, B)
                Q_bc_exact.add_data(heat_loss_der(T3, TN_1, g_cs, dx), vals, dx, inter)
                Q_der_exact.add_data(heat_loss_bc(T3, B), vals, dx, inter)

            Q_con_der_err = []
            Q_con_bc_err = []
            Q_nonc_der_err = []
            Q_nonc_bc_err = []

            for i in range(len(Q_bc_exact.return_data(vals, inter)[1])):
                Q_con_der_err.append(get_error(Q_der_exact.return_data(vals, inter)[1][i], Q_der_con.return_data(vals, inter)[1][i]))
                Q_con_bc_err.append(get_error(Q_bc_exact.return_data(vals, inter)[1][i], Q_bc_con.return_data(vals, inter)[1][i]))
                Q_nonc_der_err.append(get_error(Q_der_exact.return_data(vals, inter)[1][i], Q_der_nonc.return_data(vals, inter)[1][i]))
                Q_nonc_bc_err.append(get_error(Q_bc_exact.return_data(vals, inter)[1][i], Q_bc_nonc.return_data(vals, inter)[1][i]))

            title = 'Interface = '+str(inter)+' cm'
            if vals == min_case:
                title += ', Min Case'
            else:
                title += ', Max Case'
            plt.plot(hl_nodes[:50], Q_bc_con.return_data(vals, inter)[1][:50], label='Conformal')
            plt.plot(hl_nodes[:50], Q_bc_nonc.return_data(vals, inter)[1][:50], label='Non-Conformal')
            plt.plot(hl_nodes[:50], Q_bc_exact.return_data(vals, inter)[1][:50], label='Exact')
            plt.title('Heat Loss Using Boundary Conditions Vs. Number of Nodes\n'+title)
            plt.xlabel('Number of Nodes')
            plt.ylabel('Heat Loss')
            plt.grid()
            plt.legend()
            plt.show()

            plt.plot(hl_nodes[:50], Q_der_con.return_data(vals, inter)[1][:50], label='Conformal')
            plt.plot(hl_nodes[:50], Q_der_nonc.return_data(vals, inter)[1][:50], label='Non-Conformal')
            plt.plot(hl_nodes[:50], Q_der_exact.return_data(vals, inter)[1][:50], label='Exact')
            plt.title('Heat Loss Using the Constitutive Equation Vs. Number of Nodes\n'+title)
            plt.xlabel('Numbers of Nodes')
            plt.ylabel('Heat Loss')
            plt.grid()
            plt.legend()
            plt.show()

            plt.plot(Q_bc_con.return_data(vals, inter)[0], Q_con_bc_err, label='Boundary Condition\nMethod')
            plt.plot(Q_der_con.return_data(vals, inter)[0], Q_con_der_err, label='Constitutive Equation\nMethod')
            plt.yscale('log')
            plt.title('Convergence of Heat Loss Using Constitutive Equation\nVs. Boundary Condition\n' + title)
            plt.xlabel('ln(dx)')
            plt.ylabel('Relative Error')
            plt.grid()
            plt.legend()
            plt.gca().invert_xaxis()
            plt.show()

    # Beta convergence
    if input('Derive Beta Values? (y/n): ') == 'y':
        for inter in interface:
            for vals in [min_case, max_case]:

                B = vals[0]
                g_ss = vals[1]
                g_cs = vals[2]

                Q_bc_con_data = Q_bc_con.return_data(vals, inter)[1]
                Q_bc_exact_data = Q_bc_exact.return_data(vals, inter)[1]
                Q_der_con_data = Q_der_con.return_data(vals, inter)[1]
                Q_der_exact_data = Q_der_exact.return_data(vals, inter)[1]

                beta_der = []
                beta_bc = []
                lg_dx = []

                for i in range(len(hl_nodes)-1):
                    beta_bc.append(get_beta(Q_bc_exact_data[i], Q_bc_con_data[i], Q_bc_con_data[i+1], hl_nodes[i], hl_nodes[i+1]))
                    beta_der.append(get_beta(Q_der_exact_data[i], Q_der_con_data[i], Q_der_con_data[i+1], hl_nodes[i], hl_nodes[i+1]))
                    lg_dx.append(con_fdm_data.return_data(inter, g_ss, g_cs, B, hl_nodes[i])[0][-1] -
                                 con_fdm_data.return_data(inter, g_ss, g_cs, B, hl_nodes[i])[0][-2])

            print('Interface located at x = '+str(inter))
            print(tbl.tabulate(np.transpose(np.array([lg_dx, beta_der, beta_bc])),
                               headers=['dx (cm)', 'Beta (Constitutive)', 'Beta (Boundary Condition)']))

if input('Temperature Tables without analytical comparison? (y/n): ') == 'y':
    for inter in interface:
        table1 = []
        table2 = []
        table3 = []

        headers = ['x (cm)', 'Min Case Temp (\u00B0C)', 'Max Case Temp (\u00B0C)']

        for vals in [min_case, max_case]:
            B = vals[0]
            g_ss = vals[1]
            g_cs = vals[2]

            x_data1, y_data1 = con_fdm_data.return_data(inter, g_ss, g_cs, B, 50)
            x_data2, y_data2 = nonc_fdm_data.return_data(inter, g_ss, g_cs, B, 50)
            x_data3, y_data3 = exact_nonc_data.return_data(inter, g_ss, g_cs, B, 50)

            if vals == min_case:
                label = 'Min Case'
                table1.append(x_data1)
                table2.append(x_data2)
                table3.append(x_data3)
            else:
                label = 'Max Case'

            table1.append(y_data1)
            table2.append(y_data2)
            table3.append(y_data3)

        print('Conformal Mesh FDM')
        print('Interface located at x = '+str(inter)+' cm')
        print(tbl.tabulate(np.transpose(np.array(table1)), headers=headers))

        print('Non-Conformal Mesh FDM')
        print('Interface located at x = ' + str(inter) + ' cm')
        print(tbl.tabulate(np.transpose(np.array(table2)), headers=headers))

        print('Exact Solution')
        print('Interface located at x = ' + str(inter) + ' cm')
        print(tbl.tabulate(np.transpose(np.array(table3)), headers=headers))

# Plotting Temperature Graphs for 50 nodes
if input('Plot Temperature Distribution? (y/n): ') == 'y':
    for inter in interface:
        for vals in [min_case, max_case]:
            B = vals[0]
            g_ss = vals[1]
            g_cs = vals[2]

            if vals == min_case:
                label = 'Min Case'
            else:
                label = 'Max Case'

            plt.figure(2)
            x_data, y_data = con_fdm_data.return_data(inter, g_ss, g_cs, B, 50)
            plt.plot(x_data, y_data, label=label)

            plt.figure(3)
            x_data, y_data = nonc_fdm_data.return_data(inter, g_ss, g_cs, B, 50)
            plt.plot(x_data, y_data, label=label)

            plt.figure(1)
            x_data, y_data = exact_nonc_data.return_data(inter, g_ss, g_cs, B, 50)
            plt.plot(x_data, y_data, label=label)

        for i in range(1, 4):
            plt.figure(i)
            plt.grid()
            plt.legend()
            plt.xlabel('Position (cm)')
            plt.ylabel('Temperature (\u00B0C)')
            plt.axvline(x=inter, color='grey', linestyle='--')

        plt.figure(1)
        plt.title('Exact Solution')

        plt.figure(2)
        plt.title('FDM Solution with Conformal Mesh')

        plt.figure(3)
        plt.title('FDM Solution with Non-Conformal Mesh')

        plt.show()


# Plotting Convergence Graphs of the Temperature
if input('Plot Convergence Graphs of the Temperature Distribution? (y/n): ') == 'y':
    for inter in interface:
        for vals in [min_case, max_case]:
            err_data = []
            dx_data = []
            for num_elem in nodes:
                    B = vals[0]
                    g_ss = vals[1]
                    g_cs = vals[2]
                    exact = exact_nonc_data.return_data(inter, g_ss, g_cs, B, num_elem)[1][-1]
                    approx = nonc_fdm_data.return_data(inter, g_ss, g_cs, B, num_elem)[1][-1]
                    test = get_error(exact, approx)
                    err_data.append(get_error(exact, approx))
                    dx_data.append(np.log((x3-x1)/num_elem))

            local_mins = (min(np.array(err_data[:100]), np.less))
            mins_points = [[], []]
            for i in local_mins[0]:
                mins_points[0].append(dx_data[i])
                mins_points[1].append(err_data[i])

            plt.ylabel('Relative Error')
            plt.xlabel('Ln(dx)')
            plt.yscale('log')
            plt.gca().invert_xaxis()
            if inter == interface[0]:
                plt.plot(mins_points[0], mins_points[1], color='black', linestyle='dashed', marker='o',
                         markerfacecolor='red')
            plt.plot(dx_data[:100], err_data[:100])
            if vals == min_case:
                title = 'Min Case'
            else:
                title = 'Max Case'
            plt.title('Relative Error of Non-Conformal FDM\n'+title)
            plt.grid()
            plt.show()

    for inter in interface:
        for vals in [min_case, max_case]:
            err_data = []
            dx_data = []
            for num_elem in nodes:
                    B = vals[0]
                    g_ss = vals[1]
                    g_cs = vals[2]
                    exact = exact_con_data.return_data(inter, g_ss, g_cs, B, num_elem)[1]
                    approx = con_fdm_data.return_data(inter, g_ss, g_cs, B, num_elem)[1]
                    test = get_error(exact[-1], approx[-1])
                    err_data.append(get_error(exact[-1], approx[-1]))
                    dx_data.append(np.log((x3-x1)/num_elem))
            plt.ylabel('Relative Error')
            plt.xlabel('Log dx')
            plt.yscale('log')
            plt.gca().invert_xaxis()
            plt.grid()
            if vals == min_case:
                title = 'Min Case'
            else:
                title = 'Max Case'
            plt.title('Relative Error of Conformal FDM\n' + title)
            plt.plot(dx_data[:100], err_data[:100])
            plt.show()

