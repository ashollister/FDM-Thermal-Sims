"""
AERO 430
Final Project Code

The purpose of this code is to evaluate the forced vibrations upon a bar by utilizing the
exact solution and 2nd order finite difference method schemes

Andrew Hollister
127008398
"""

import numpy as np
import matplotlib.pyplot as plt
import tabulate as tbl

"""
Data Class Structure
"""


class DataStructure:
    def __init__(self):
        self.data = []

    def add_data(self, nodes, k, x, vals):
        self.data.append((nodes, k, x, vals))

    def return_data(self, nodes, k):
        x = next(item[2] for item in self.data if item[0] == nodes and item[1] == k)
        vals = next(item[3] for item in self.data if item[0] == nodes and item[1] == k)
        return x, vals


"""
Analytical Solution
"""


def exact_vibe(x, k):
    return v_l/np.sin(k)*np.sin(k*x)


"""
2nd Order FDM Solution 
"""


def approx_vibe(k, nodes):

    # Defining Size of mesh
    dim = nodes-2

    # Defining dx
    dx = 1 / (nodes-1)

    # Defining kappa
    kappa = 2 - k**2*dx**2

    # Generating A Matrix
    a = np.zeros([dim, dim])
    a += kappa*np.eye(dim)
    a -= np.eye(dim, k=-1)
    a -= np.eye(dim, k=1)

    for i in range(dim):
        if i % nodes == 0 and i != 0:
            a[i-1][i] = 0
            a[i][i-1] = 0

    # Generating B Matrix
    b = np.zeros(dim)
    b[-1] = v_l

    vibes = np.linalg.solve(a, b)
    vibes = np.concatenate([[0], vibes])
    vibes = np.append(vibes, v_l)

    return vibes


"""
Analytical Derivative Solution
"""


def exact_dev(k):
    return abs(k*v_l/np.sin(k)*np.cos(k*length))


"""
2nd Order FDM Derivative Solution 
"""


def approx_dev(k, nodes, vibes):
    dx = 1/(nodes-1)
    u_prime = (vibes[-1] - vibes[-2])/dx - dx/2*k**2*vibes[-1]
    return abs(u_prime)


"""
Richardson Extrapolation Function
"""


def q_rich_extr(q, q2, q4):
    q_extr = (q2**2 - q*q4)/(2*q2-q-q4)
    perc_err = abs(q_extr-q)
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

    # Vibration Boundary Condition at Wall
    global v_l
    v_l = 100

    # Length of Bar
    global length
    length = 1

    # Vibration material property list
    k_list = [2**val for val in range(0, 7)]

    # Defining Number of Nodes List
    node_list = range(2, 11)

    """
    FDM Displacement Data Generation
    """

    # Initializing data structure
    fdm_vibe_data = DataStructure()

    for ki in k_list:
        for n in node_list:

            # Creating mesh
            n_nodes = 2**n
            x_vals = np.linspace(0, length, n_nodes)

            # Generating Data
            fdm_vibe_sol = approx_vibe(ki, n_nodes)

            # Storing Data
            fdm_vibe_data.add_data(n_nodes, ki, x_vals, fdm_vibe_sol)

    """
    FDM Resonance Data Generation
    """

    pass

    """
    Displacement Absolute Error Calculations
    """

    # Initializing variables
    x_vals = None
    err = None
    fig, axs = plt.subplots(len(k_list), len(node_list))

    for k_index, ki in enumerate(k_list):
        for node_index, n in enumerate(node_list):

            # Error tables
            vibe_err_tables = []

            # Recreating mesh
            n_nodes = 2**n

            # Displacement absolute error
            x_vals, approx_vals = fdm_vibe_data.return_data(n_nodes, ki)
            exact_vals = exact_vibe(x_vals, ki)
            err = exact_vals - approx_vals

            # Writing data to tables
            if n_nodes < 70:
                vibe_err_tables = np.transpose(np.array([x_vals, exact_vals, approx_vals, err]))
                print(f'\nError of 2nd Order FDM Solution, k = {ki}, nodes = {n_nodes}')
                print(tbl.tabulate(vibe_err_tables, headers=['Position (cm)', 'Exact', 'FDM', 'Abs. Error']))

            # Plotting data
            plt.figure(1)
            axs[k_index, node_index].plot(x_vals, err, label='FDM')
            axs[k_index, node_index].set_title(f'k={round(ki, 3)} | nodes={n_nodes}')
            axs[k_index, node_index].grid()

            # Plotting for highest number of nodes
        plt.figure(2)
        plt.plot(x_vals, err, label=f'k = {ki}')

    # Configuring FDM plot
    plt.figure(2)
    plt.title('Absolute Error of FDM Solution vs. x (cm)')
    plt.ylabel('Absolute Error (cm)')
    plt.xlabel('x (cm)')
    plt.grid()
    plt.legend()

    # Adds x and y labels
    plt.figure(1)
    for ax in axs.flat:
        ax.set(xlabel='x (cm)', ylabel='Displacement (cm)')

    # Keeps labels that are on the outer parts of the assembly
    for ax in axs.flat:
        ax.label_outer()

    # Displaying FDM plots
    plt.show()

    """
    Resonance Error and Convergence
    """

    pass

    """
    Exact Displacement Plotting
    """

    # Points for Exact Solution
    x_vals = np.linspace(0, length, 1000)

    # Looping through material property list
    for ki in k_list:

        # Vibration values from exact solution
        v_vals = exact_vibe(x_vals, ki)

        # Plotting
        plt.plot(x_vals, v_vals, label=f'k = {ki}')

    # Configuring and displaying plot
    plt.title('Exact Solution - Displacement (cm) vs. x (cm)')
    plt.ylabel('Displacement (cm)')
    plt.xlabel('x (cm)')
    plt.grid()
    plt.legend()
    plt.show()

    """
    FDM Displacement Plotting
    """

    fig, axs = plt.subplots(len(k_list), len(node_list))
    for k_index, ki in enumerate(k_list):
        vibes = 'n/a'
        for node_index, n in enumerate(node_list):

            # Creating mesh
            n_nodes = 2**n

            # Extracting data
            x_vals, vibes = fdm_vibe_data.return_data(n_nodes, ki)
            e_x_vals = np.linspace(0, length, 1000)
            v_vals = exact_vibe(e_x_vals, ki)

            # Plotting data
            plt.figure(1)
            axs[k_index, node_index].plot(x_vals, vibes, label='FDM')
            axs[k_index, node_index].plot(e_x_vals, v_vals, label='Exact')
            axs[k_index, node_index].set_title(f'k={round(ki, 3)} | nodes={n_nodes}')
            axs[k_index, node_index].grid()

        # Plotting for highest number of nodes
        plt.figure(2)
        plt.plot(x_vals, vibes, label=f'k = {ki}')

    # Configuring FDM plot
    plt.figure(2)
    plt.title('FDM Solution - Displacement (cm) vs. x (cm)')
    plt.ylabel('Displacement (cm)')
    plt.xlabel('x (cm)')
    plt.grid()
    plt.legend()

    # Adds x and y labels
    plt.figure(1)
    for ax in axs.flat:
        ax.set(xlabel='x (cm)', ylabel='Displacement (cm)')

    # Keeps labels that are on the outer parts of the assembly
    for ax in axs.flat:
        ax.label_outer()

    # Adding legend to multi-plot
    axs[0, 0].legend()

    # Displaying FDM plots
    plt.show()

    """
    Exact Derivative Function
    """

    # Resonance Frequency Data
    rf_vals = []

    # Looping through material property list
    for ki in k_list:

        # Vibration values from exact solution
        rf_vals.append(exact_dev(np.sqrt(ki)))

    """
    Convergence and Richardson Extrapolation
    """

    rf_log_err = []
    rf_dx_err = []

    re_rf_log_err = []
    re_rf_dx_err = []

    for k_index, ki in enumerate(k_list):

        rf_error_data = []
        re_rf_error_data = []

        for node_index, n in enumerate(node_list):

            # Recreating mesh
            n_nodes = 2**n

            # Displacement absolute error
            x_vals, vibe_vals = fdm_vibe_data.return_data(n_nodes, ki)
            approx_vals = approx_dev(ki, n_nodes, vibe_vals)
            exact_vals = exact_dev(ki)
            err = abs(exact_vals - approx_vals)
            rf_error_data.append([n_nodes, length / n_nodes, exact_vals, approx_vals, err])

        # Resonance Frequency Convergence
        rf_error_data[0].append('n/a')
        for i in range(len(rf_error_data) - 1):
            rf_error_data[i + 1].append(
                get_beta(rf_error_data[i][2], rf_error_data[i][3], rf_error_data[i + 1][3],
                         rf_error_data[i][0], rf_error_data[i + 1][0]))
        print(f'\nConvergence of Derivative Function for k = {ki}:')
        print(tbl.tabulate(rf_error_data, headers=['Num. Elements', 'dx', 'Exact Resonance Freq.',
                                                   'Approx. Resonance Freq.', 'Absolute Error', 'Beta']))
        rf_log_err.append([np.log10(item[4] / 100) for item in rf_error_data])
        rf_dx_err.append([-np.log10(item[1]) for item in rf_error_data])

        # Richardson Extrapolation and Convergence
        for i in range(len(rf_error_data) - 2):
            q_extr, perc_err = q_rich_extr(rf_error_data[i][3], rf_error_data[i + 1][3],
                                           rf_error_data[i + 2][3])
            beta = get_beta(q_extr, rf_error_data[i][3], rf_error_data[i + 1][3],
                            rf_error_data[i][0], rf_error_data[i + 1][0])
            re_rf_error_data.append([rf_error_data[i][0], rf_error_data[i][1], q_extr, rf_error_data[i][3],
                                     perc_err, beta])
        print(f'\nConvergence of Extrapolated Derivative Function k = {ki}:')
        print(tbl.tabulate(re_rf_error_data,
                           headers=['Num. Elements', 'dx', 'Extrapolated Resonance Freq.',
                                    'Approx. Resonance Freq.', 'Absolute Error', 'Beta']))
        re_rf_log_err.append([np.log10(item[4] / 100) for item in re_rf_error_data])
        re_rf_dx_err.append([-np.log10(item[1]) for item in re_rf_error_data])

    # Plotting Convergence Graphs
    for i in range(len(k_list)):
        plt.plot(rf_dx_err[i], rf_log_err[i], label=f'k = {k_list[i]}')
    plt.title('Convergence of Derivative Function\n' + r'-$log_{10}$($\Delta$x) vs -$log_{10}$(Absolute Error)')
    plt.xlabel(r'$-log_{10}$($\Delta$x)')
    plt.ylabel(r'$-log_{10}$(Absolute Error)')
    plt.grid()
    plt.legend()
    plt.show()

    for i in range(len(k_list)):
        plt.plot(re_rf_dx_err[i], re_rf_log_err[i], label=f'k = {k_list[i]}')
    plt.title('Convergence of Extrapolated Derivative Function\n' +
              r'-$log_{10}$($\Delta$x) vs -$log_{10}$(Absolute Error)')
    plt.xlabel(r'$-log_{10}$($\Delta$x)')
    plt.ylabel(r'$-log_{10}$(Absolute Error)')
    plt.grid()
    plt.legend()
    plt.show()

    """
    Exact Resonance Frequency Plotting
    """

    # Redefining Material Property List
    k_list = np.linspace(1, 100, 1000)

    # Resonance Frequency Data
    rf_vals = []

    # Looping through material property list
    for ki in k_list:

        # Vibration values from exact solution
        rf_vals.append(exact_dev(ki))

    # Plotting
    plt.plot(k_list, rf_vals)
    plt.title(f'Exact Derivative Function vs. k')
    plt.ylabel('Derivative Value')
    plt.xlabel('k value')
    plt.ylim(0, 1e5)
    plt.grid()
    plt.show()

    """
    FDM Resonance Frequency Plotting 
    """

    # Looping through material property list
    for n in node_list:
        rf_fdm_vals = []
        rf_vals = []
        n_nodes = 2**n
        for ki in k_list:

            # Resonance freq. values from exact solution
            rf_vals.append(exact_dev(ki))

            # Resonance freq. values from approx. solution
            vibe_vals = approx_vibe(ki, n_nodes)
            rf_fdm_vals.append(approx_dev(ki, n_nodes, vibe_vals))

        # Plotting
        plt.plot(k_list, rf_fdm_vals)
        plt.plot(k_list, rf_vals)
        plt.title(f'Derivative Function vs. k\nn = {n_nodes}')
        plt.ylabel('Derivative Value')
        plt.xlabel('k value')
        plt.ylim(0, 1e5)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()
