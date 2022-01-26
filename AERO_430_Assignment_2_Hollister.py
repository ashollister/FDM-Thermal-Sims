from numpy import *
import matplotlib.pyplot as plt
import tabulate as tbl

"""
AERO 430 Homework Assignment 2

Andrew Hollister
UIN: 127008398
Date: 09/14/21

Assignment Description:

Repeat assignment 1 for a bi-material bar as in the report of Caleb Bryan

Assignment 1:
Write a program which does the computations in the Valentina Musu report
which solves all 3 cases of boundary conditions at x = 0
"""

with open('output_txt.txt', 'w') as file:

    # Defining bar properties
    T_0 = 0  # temperature of bar at x=0 in deg C
    T_L = 100  # temperature of bar at x=L deg C
    L = 1  # length of bar cm
    radius = 0.1  # radius of bar cm
    area = pi*radius**2  # cm^2
    P = 2*pi*radius
    k_0 = 0.5  # Thermal conductivity of the rod
    h = 0.4

    inter_locs = [L/2, 3*L/4]
    Alpha = [4]
    C = [0.01, 0.1, 1, 10, 100]


    def richardsons(Q):
        i = len(Q) - 1
        Qe = (Q[i-1]**2 - Q[i-2]*Q[i]) / (2*Q[i-1]-Q[i-2]-Q[i])
        beta = (np.log((Qe-Q[i-2])/(Qe-Q[i-1])))/np.log(2)
        return [Qe, beta]


    def rich_extra(q, q2, q4):
        q_extr = (q2**2 - q*q4)/(2*q2-q-q4)
        beta = log((q_extr - q2)/(q_extr-q4))/log(2)
        return q_extr, beta

    def get_error(exact, approx):
        return abs((exact-approx)/exact)*100


    def get_beta(exact, approx, approx_2, h, h_2):
        A = log(exact-approx)
        B = log(exact-approx_2)
        C = log(h) - log(h_2)
        return -(A-B)/C


    # For 0 <= x < 1/2
    def get_T_left(x, T_inter, a, inter):
        A = -0*sinh(a*(x-inter)) / sinh(a*inter)
        B = (T_inter*sinh(a*x)/sinh(a*inter))
        return A + B


    # For 1/2 < x <= 1
    def get_T_right(x, T_inter, a, inter):
        A = -T_inter*sinh(a*(x-1))/sinh(a*(1-inter))
        B = 100*sinh(a*(x-inter))/sinh(a*(1-inter))
        return A + B


    # Function for T at x = 1/2
    def get_T_inter(a_0, k_0, a_1, k_1, inter):
        A = (a_0*cosh(a_0*inter)/sinh(a_0*inter))*k_0
        B = (a_1*cosh(a_1*(inter-1))/sinh(a_1*(1-inter)))*k_1
        C = (100*a_1/sinh(a_1*(1-inter)))*k_1
        return C/(A+B)


    # Function for determining alpha 1
    def get_a_1(k_1):
        return sqrt(2*0.4/(radius*k_1))


    def get_q_dot_exact(a_1, k_1, area, T_inter, inter):
        A = -T_inter*a_1/sinh(a_1*(1-inter))
        B = 100*a_1*cosh(a_1*(1-inter))/sinh(a_1*(1-inter))
        return -k_1*area*(A+B)


    def get_q_dot_FDM(k_1, area, T_N, T_N_1, h, P, dx):
        return -((-k_1*area*T_N_1)+(k_1*area+((h*P)*dx**2)/2)*T_N)/dx


    # Analytical Solution
    Q_exact_data = []
    exact_data = []
    for interface in inter_locs:
        # print(inter)
        for a in Alpha:
            for k in C:
                k_1 = 0.5*k
                a_1 = get_a_1(k_1)
                T_inter = get_T_inter(a, k_0, a_1, k_1, interface)
                for i in range(3, 11):
                    n_elem = 2 ** i
                    dx = L / n_elem

                    for node in range(n_elem+1):
                        x = dx*node
                        if x < interface:
                            T = get_T_left(x, T_inter, a, interface)
                            exact_data.append({"inter": interface, "alpha": a, "x": x, "Temperature": T, "Nodes": n_elem, "k": k})
                        elif x > interface:
                            T = get_T_right(x, T_inter, a_1, interface)
                            exact_data.append({"inter": interface, "alpha": a, "x": x, "Temperature": T, "Nodes": n_elem, "k": k})
                        elif x == interface:
                            exact_data.append({"inter": interface, "alpha": a, "x": x, "Temperature": T_inter, "Nodes": n_elem, "k": k})

                    q_dot = get_q_dot_exact(a_1, k_1, area, T_inter, interface)
                    Q_exact_data.append({"inter": interface, "alpha": a, "Q dot": q_dot, "k": k, "Nodes": n_elem})
    # FDM
    Q_FDM_data = []
    FDM_data = []
    for interface in inter_locs:
        for a in Alpha:
            for k in C:
                for i in range(3, 11):
                    n_elem = 2 ** i
                    dx = L / n_elem
                    k_1 = 0.5 * k

                    omega_0 = h*P/(k_0*area)
                    omega_1 = h*P/(k_1*area)
                    kappa_0 = 2 + omega_0 * dx ** 2
                    kappa_1 = 2 + omega_1 * dx ** 2

                    # Creating A matrix
                    A = zeros((n_elem-1, n_elem-1))-eye(n_elem-1, k=-1)-eye(n_elem-1, k=1)
                    for node in range(len(A)):
                        if (node+1)*dx < interface:
                            A[node][node] = kappa_0
                        elif (node+1)*dx == interface:
                            A[node][node-1] = -k_0*area/dx
                            A[node][node] = (k_0*area/dx) + k_1*area/dx + h*P*dx
                            A[node][node+1] = -k_1*area/dx
                        elif (1+node)*dx > interface:
                            A[node][node] = kappa_1

                    # Creating B matrix
                    B = zeros(n_elem-1)
                    B[0] = 0  # deg C
                    B[-1] = 100  # deg C

                    # Solving
                    T = linalg.solve(A, B)
                    T = concatenate([[0], T])
                    T = append(T, 100)
                    for i in range(len(T)):
                        x = dx*i
                        FDM_data.append({"inter": interface, "alpha": a, "x": x, "Temperature": T[i], "Nodes": n_elem, "k": k})

                    q_dot_FDM = get_q_dot_FDM(k_1, area, T[-1], T[-2], h, P, dx)
                    Q_FDM_data.append({"inter": interface, "alpha": a, "Q dot": q_dot_FDM, "k": k, "Nodes": n_elem})

    # Convergence
    for interface in inter_locs:
        for a in Alpha:
            for k in C:

                exact_con = []
                rich_con = []
                q_exact = [item["Q dot"] for item in Q_exact_data if item["alpha"] == a
                           and item["inter"] == interface and item["k"] == k]
                q_approx = [item["Q dot"] for item in Q_FDM_data if item["alpha"] == a
                            and item["inter"] == interface and item["k"] == k]
                nodes = [item["Nodes"] for item in Q_exact_data if item["alpha"] == a
                         and item["inter"] == interface and item["k"] == k]

                dx = L / nodes[0]
                error = get_error(q_exact[0], q_approx[0])
                exact_con.append([dx, q_approx[0], q_exact[0], error, 'Nan'])

                for i in range(len(nodes)-1):
                    dx = L / nodes[i]
                    error = get_error(q_exact[i], q_approx[i])
                    B_exact = get_beta(q_exact[i], q_approx[i], q_approx[i+1], nodes[i], nodes[i+1])
                    exact_con.append([dx, q_approx[i], q_exact[i], error, B_exact])

                for i in range(len(nodes)-2):
                    dx = L/nodes[i+2]
                    q_extr, beta = rich_extra(q_approx[i], q_approx[i+1], q_approx[i+2])
                    error = get_error(q_extr, q_approx[i])
                    rich_con.append([dx, q_approx[i], q_extr, error, beta])

                print('\nConvergence using Richardson extrapolation with k1 = '+str(k_0*k))
                print(tbl.tabulate(rich_con, headers=['dx', 'q_dot(1)', 'q_dot(1)_extra', '% Error', 'Beta']))
                print('\nConvergence using exact solution with k1 = '+str(k_0*k))
                print(tbl.tabulate(exact_con, headers=['dx', 'q_dot(1)', 'q_dot(1)_exact', '% Error', 'Beta']))

                file.write('\nConvergence using Richardson extrapolation with k1 = '+str(k_0*k))
                file.write(tbl.tabulate(rich_con, headers=['dx', 'q_dot(1)', 'q_dot(1)_extra', '% Error', 'Beta']))
                file.write('\nConvergence using exact solution with k1 = '+str(k_0*k))
                file.write(tbl.tabulate(exact_con, headers=['dx', 'q_dot(1)', 'q_dot(1)_exact', '% Error', 'Beta']))

                plt.figure(1)
                x_data = [item[0] for item in rich_con]
                y_data = [item[3] for item in rich_con]
                plt.plot(x_data, y_data, label='k1='+str(k_0*k))

                plt.figure(2)
                x_data = [item[0] for item in exact_con]
                y_data = [item[3] for item in exact_con]
                plt.plot(x_data, y_data, label='k1='+str(k_0*k))

            for i in range(2):
                if i == 0:
                    plt.title('FDM Error Convergence Vs. Extrapolated Solution\nInterface Located at L*'+str(interface))
                else:
                    plt.title('FDM Error Convergence Vs. Exact Solution\nInterface Located at L*'+str(interface))
                plt.legend()
                plt.figure(i+1)
                plt.grid()
                plt.ylabel('Log % Error')
                plt.xlabel('Log dx')
                plt.xscale('log')
                plt.yscale('log')
                plt.gca().invert_xaxis()
            plt.show()

    # Plotting omega_data
    table_data = []
    for interface in inter_locs:
        for i in range(3, 11):
            n_elem = 2 ** i
            for a in Alpha:
                k_table1 = []
                k_table0 = []
                x_data_exact = []
                x_data_FDM = []
                k_headers = ['Position (cm)', 'T, k1 = 0.05', 'T, k1 = 0.5', 'T, k1 = 5', 'T, k1 = 50', 'T, k1 = 500']
                for k in C:

                    # Collecting Data
                    y_data_exact = [item["Temperature"] for item in exact_data if item["alpha"] == a
                                    and item["inter"] == interface and item["k"] == k and
                                    item["Nodes"] == n_elem]

                    x_data_exact = [item["x"] for item in exact_data if item["alpha"] == a
                                    and item["inter"] == interface and item["k"] == k and
                                    item["Nodes"] == n_elem]

                    y_data_FDM = [item["Temperature"] for item in FDM_data if item["alpha"] == a
                                  and item["inter"] == interface and item["k"] == k and
                                  item["Nodes"] == n_elem]

                    x_data_FDM = [item["x"] for item in FDM_data if item["alpha"] == a
                                  and item["inter"] == interface and item["k"] == k and
                                  item["Nodes"] == n_elem]

                    # Printing Tables
                    if i < 5:

                        k_table0.append(y_data_exact)
                        k_table1.append(y_data_FDM)

                        print('\nTemperature Vs. Position Tables for Exact and FDM Analysis')
                        print('k1 = '+str(k_0*k)+' Nodes = '+str(n_elem))
                        print(tbl.tabulate(transpose(array([x_data_exact, y_data_exact, x_data_FDM, y_data_FDM])),
                                           headers=['Position (cm)', 'T Exact', 'Position (cm)', 'T FDM']))
                    file.write('\nTemperature Vs. Position Tables for Exact and FDM Analysis\nk1 = '+str(k_0*k)
                               + ' Nodes = '+str(n_elem)+tbl.tabulate(transpose(array([x_data_exact, y_data_exact, x_data_FDM, y_data_FDM])),
                                                                      headers=['Position (cm)', 'T Exact', 'Position (cm)', 'T FDM']))

                    # Plotting Data
                    for i in range(1, 3):
                        plt.figure(i)
                        plt.plot(x_data_FDM, y_data_FDM, '-o', label="FDM: k1 = " + str(0.5 * k))
                        plt.legend()
                    for i in range(2, 4):
                        plt.figure(i)
                        plt.plot(x_data_exact, y_data_exact, '-o', label="Exact: k1 = " + str(0.5 * k))
                        plt.legend()

                k_table0.insert(0, x_data_exact)
                k_table1.insert(0, x_data_FDM)
                if i < 5:
                    print('\nTemperature Vs. Position Tables for Exact Analysis')
                    print('Nodes = ' + str(n_elem))
                    print(tbl.tabulate(transpose(array(k_table0)), headers=k_headers))

                    print('\nTemperature Vs. Position Tables for FDM Analysis')
                    print('Nodes = ' + str(n_elem))
                    print(tbl.tabulate(transpose(array(k_table1)), headers=k_headers))

                for i in range(1, 4):
                    plt.figure(i)
                    plt.axvline(x=interface, color='grey', linestyle='--')
                    plt.title('Temperature Vs. Position of Bi-metallic Bar\nAlpha='+str(a)+', Interface='+str(interface)
                              + ', Nodes='+str(n_elem))
                    plt.xlabel('Position (cm)')
                    plt.ylabel('Temperature (\u00B0C)')

                plt.show()
