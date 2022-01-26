import numpy as np
import matplotlib.pyplot as plt
import tabulate as tbl

"""
AERO 430 Homework Assignment 1

Andrew Hollister
UIN: 127008398
Date: 09/14/21

Assignment Description:

Write a_0 program which does the computations in the Valentina Musu report
which solves all 3 cases of boundary conditions at x = 0
"""


# Defining Functions
# Heat Loss Function
def heat_loss(k, A, a, L, C, D):
    return -k*A*a*(C*np.cosh(a*L)+D*np.sinh(a*L))


# Approximate Heat Loss Function
def approx_heat_loss(kappa, area, T_n, T_n_1, alpha, dx):
    q_dot_c = -kappa*area*((T_n_1-T_n)/dx + (alpha**2*dx*T_n)/2)
    return q_dot_c


def percent_err(val, exp_val):
    return abs((val-exp_val)/val)*100


# Values of Alpha
Alpha = [0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

# Defining bar properties
T_0 = 0  # temperature of bar at x=0 in deg C
T_L = 100  # temperature of bar at x=L deg C
length = 1  # length of bar cm
radius = 0.1  # radius of bar cm
area = np.pi*radius**2  # cm^2
k = 0.5  # Thermal conductivity of the rod

# Problem 1
print('\nProblem 1: Analytical Solution')
print('Case 1: Zero Temperature')
nodes = 20
dx = length/nodes  # distance between nodes in cm

# Case 1 (Zero Ambient Temperature)
Ta = 0  # Ambient temperature in deg D

# Creating Data Structures
case_1_data = {}
case_1_heat = {}
case_1_dx = {}
case_1_data['x(cm)'] = [i*dx for i in range(0, nodes+1)]

# Loops through each value of Alpha
for a in Alpha:
    if a == 6:
        iters = 10
    else:
        iters = 1

    for i in range(iters):
        nodes = 20 + i*10
        dx = length / nodes
        if i == 0:
            case_1_data[a] = []  # Creates a_0 exact_con_data set for each value of alpha

        C = 100 / np.sinh(a * length)  # Value of constant 'C' for case 1
        D = 0  # Value of constant 'D' for case 1

        # Loops through each node of the rod
        for node in range(nodes+1):
            x = dx*node
            temp = C * np.sinh(a * x) + D * np.cosh(a * x) + Ta
            if i == 0:
                case_1_data[a].append(temp)

        # Appending heat loss to exact_con_data
        if i == 0:
            case_1_heat[a] = heat_loss(k, area, a, length, C, D)

        # Appending heat loss to exact_con_data
        if a == 6:
            case_1_dx[round(dx, 3)] = heat_loss(k, area, a, length, C, D)

# Printing Data
print('\n', tbl.tabulate(case_1_data, headers='keys'))

# Plotting Case 1 Data
x_data = []
for i in range(len(case_1_data[0.25])):
    x_data.append(i*dx)
for i in range(len(case_1_data)-1):
    plt.plot(x_data, case_1_data[Alpha[i]], label=str(Alpha[i]))
plt.title('Temperature (\u00B0C) Vs. Position (cm) for Various Alpha')
plt.ylabel('Temperature (\u00B0C)')
plt.xlabel('Position (cm)')
plt.grid()
plt.legend()
plt.show()

# Case 2 (Insulated at left end)
print('Case 2: Insulated at the Left End')

# Creating Data Structures
case_2_data = {}
case_2_dx = {}
case_2_heat = {}
case_2_data['x(cm)'] = [i*dx for i in range(0, nodes+1)]

# Looping Through Each alpha Value
for a in Alpha:
    if a == 6:
        iters = 10
    else:
        iters = 1

    for i in range(iters):
        nodes = 20 + i*10
        dx = length / nodes
        if i == 0:
            case_2_data[a] = []  # Creates a_0 exact_con_data set for each value of alpha

        C = 0  # Value of constant 'C' for case 2
        D = 100 / np.cosh(a * length)  # Value of constant 'D' for case 2

        # Loops through each node of the rod
        for node in range(nodes+1):
            x = dx*node
            temp = C * np.sinh(a * x) + D * np.cosh(a * x)
            if i == 0:
                case_2_data[a].append(temp)

        # Appending heat loss to exact_con_data
        if i == 0:
            case_2_heat[a] = heat_loss(k, area, a, length, C, D)

        # Appending heat loss to exact_con_data
        if a == 6:
            case_2_dx[round(dx, 3)] = heat_loss(k, area, a, length, C, D)

# Printing Case 2 Data
print('\n', tbl.tabulate(case_2_data, headers='keys'))
print('\n', tbl.tabulate([case_2_dx], headers='keys'))

# Plotting Case 2 Data
x_data = []
for i in range(len(case_2_data[0.25])):
    x_data.append(i*dx)
for i in range(len(case_2_data)-1):
    plt.plot(x_data, case_2_data[Alpha[i]], label=str(Alpha[i]))

plt.title('Temperature (\u00B0C) Vs. Position (cm) for Various Alpha')
plt.ylabel('Temperature (\u00B0C)')
plt.xlabel('Position (cm)')
plt.legend()
plt.grid()
plt.show()

# Case 3 (Newton's Cooling)
print('Case 3: Newton\'s Cooling')

# Creating Data Structures
case_3_data = {}
case_3_heat = {}
case_3_dx = {}
case_3_data['x(cm)'] = [i*dx for i in range(0, nodes+1)]

for a in Alpha:
    if a == 6:
        iters = 10
    else:
        iters = 1

    for i in range(iters):
        nodes = 20 + i*10
        dx = length / nodes
        if i == 0:
            case_3_data[a] = []  # Creates a_0 exact_con_data set for each value of alpha

        # Defining parameters dependent on alpha
        h = (a**2*k*radius)/2
        C = (h/k*a)*(100/((h/k*a)*np.sinh(a*length)+np.cosh(a*length)))  # Value of constant 'C' for case 3
        D = 100/((h/k*a)*np.sinh(a*length)+np.cosh(a*length))  # Value of constant 'D' for case 3

        # Loops through each node of the rod
        for node in range(nodes+1):
            x = dx*node
            temp = C * np.sinh(a * x) + D * np.cosh(a * x)
            if i == 0:
                case_3_data[a].append(temp)

        # Appending heat loss to exact_con_data
        if i == 0:
            case_3_heat[a] = heat_loss(k, area, a, length, C, D)

        # Appending heat loss to exact_con_data
        if a == 6:
            case_3_dx[round(dx, 3)] = heat_loss(k, area, a, length, C, D)


# Printing Case 3 Data
print('\n', tbl.tabulate(case_3_data, headers='keys'))
print('\n', tbl.tabulate([case_3_dx], headers='keys'))


# Plotting Case 3 Data
x_data = []
for i in range(len(case_3_data[0.25])):
    x_data.append(i*dx)
for i in range(len(case_3_data)-1):
    plt.plot(x_data, case_3_data[Alpha[i]], label=str(Alpha[i]))
plt.title('Temperature (\u00B0C) Vs. Position (cm) for Various Alpha')
plt.ylabel('Temperature (\u00B0C)')
plt.xlabel('Position (cm)')
plt.legend()
plt.grid()
plt.show()


# Heat Loss for Problem 1
header = ['Alpha', 'Case 1', 'Case 2', 'Case 3']
table_vals = []
for alpha in Alpha:
    table_vals.append([alpha, case_1_heat[alpha], case_2_heat[alpha], case_3_heat[alpha]])
print('\n', tbl.tabulate(table_vals, headers=header))


# Plot Comparison Graphs
for alpha in Alpha:
    line1, = plt.plot(x_data, case_1_data[alpha])
    line2, = plt.plot(x_data, case_2_data[alpha])
    line3, = plt.plot(x_data, case_3_data[alpha])
    plt.title('Temperature (\u00B0C) Vs. Position (cm), Alpha = ' + str(alpha))
    plt.ylabel('Temperature (\u00B0C)')
    plt.xlabel('Position (cm)')
    plt.legend([line1, line2, line3], ['Case 1', 'Case 2', 'Case 3'])
    plt.grid()
    plt.show()


# Problem 2
print('\nProblem 2: Finite Differences Method')
# Case 1 (Zero Ambient Temperature)
print('Case 1: Zero Ambient Temperature')

case_1_data_2 = {}
case_1_dx_2 = {}
case_1_heat_2 = {}
A = []
case_1_data_2['x(cm)'] = [i*dx for i in range(0, nodes+1)]
for alpha in Alpha:
    if alpha == 6:
        iters = 10
    else:
        iters = 1

    for i in range(iters):
        nodes = 20 + i*10
        dx = length / nodes

        # Creating A matrix
        kappa = 2 + alpha**2 * dx**2
        A = np.zeros((nodes-1, nodes-1))+kappa*np.eye(nodes-1)-np.eye(nodes-1, k=-1)-np.eye(nodes-1, k=1)

        # Creating B matrix
        B = np.zeros(nodes-1)
        B[0] = 0  # deg C
        B[-1] = 100  # deg C

        # Solving
        T = np.linalg.solve(A, B)
        T = np.insert(T, 0, 0, 0)
        T = np.append(T, 100)
        if i == 0:
            case_1_data_2[alpha] = T

        if alpha == 6:
            case_1_heat_2[round(dx, 3)] = approx_heat_loss(0.5, area, T[-2], T[-1], alpha, dx)

# Printing Case 1 Data
print('\n', tbl.tabulate(case_1_data_2, headers='keys'))
print('\n', tbl.tabulate([case_1_heat_2], headers='keys'))


# Plotting Problem 2 Case 1 Data
x_data = []
for i in range(len(case_1_data_2[0.25])):
    x_data.append(i*dx)
for i in range(len(case_1_data_2)-1):
    plt.plot(x_data, case_1_data_2[Alpha[i]], label=str(Alpha[i]))
plt.title('Temperature (\u00B0C) Vs. Position (cm) for Various Alpha')
plt.ylabel('Temperature (\u00B0C)')
plt.xlabel('Position (cm)')
plt.grid()
plt.legend()
plt.show()

# Case 2 (Insulated at left end)
print('Case 2: Insulated at Left End')
dx = length/nodes
case_2_data_2 = {}
case_2_heat_2 = {}
A = []
case_2_data_2['x(cm)'] = [i*dx for i in range(0, nodes+1)]
for alpha in Alpha:
    if alpha == 6:
        iters = 10
    else:
        iters = 1

    for i in range(iters):
        nodes = 20 + i * 10
        dx = length / nodes

        # Creating A matrix
        kappa = 2 + alpha**2 * dx**2
        k_p = kappa/2
        A = np.zeros((nodes, nodes))+kappa*np.eye(nodes)-np.eye(nodes, k=-1)-np.eye(nodes, k=1)
        A[0][0] = k_p

        # Creating B matrix
        B = np.zeros(nodes)
        B[0] = 0  # deg C
        B[-1] = 100  # deg C
        T = np.linalg.solve(A, B)
        T = np.append(T, 100)

        if i == 0:
            case_2_data_2[alpha] = T

        if alpha == 6:
            case_2_heat_2[round(dx, 3)] = approx_heat_loss(0.5, area, T[-2], T[-1], alpha, dx)

# Printing Case 2 Data
print('\n', tbl.tabulate(case_2_data_2, headers='keys'))
print('\n', tbl.tabulate([case_2_heat_2], headers='keys'))

# Plotting Problem 2 Case 1 Data
x_data = []
for i in range(len(case_2_data_2[0.25])):
    x_data.append(i*dx)
for i in range(len(case_2_data_2)-1):
    plt.plot(x_data, case_2_data_2[Alpha[i]], label=str(Alpha[i]))
plt.title('Temperature (\u00B0C) Vs. Position (cm) for Various Alpha')
plt.ylabel('Temperature (\u00B0C)')
plt.xlabel('Position (cm)')
plt.grid()
plt.legend()
plt.show()

# Case 3 (Newton's Cooling)
print('Case 3: Newton\'s Cooling')
dx = length/nodes
case_3_data_2 = {}
case_3_heat_2 = {}
A = []
case_3_data_2['x(cm)'] = [i*dx for i in range(0, nodes+1)]
for alpha in Alpha:
    if alpha == 6:
        iters = 10
    else:
        iters = 1

    for i in range(iters):
        nodes = 20 + i * 10
        dx = length / nodes

        # Creating A matrix
        kappa = 2 + alpha**2 * dx**2
        h = alpha**2 * kappa * radius/2
        k_p = kappa/2 + dx*h/k
        A = np.zeros((nodes, nodes))+kappa*np.eye(nodes)-np.eye(nodes, k=-1)-np.eye(nodes, k=1)
        A[0][0] = k_p

        # Creating B matrix
        B = np.zeros(nodes)
        B[0] = 0  # deg C
        B[-1] = 100  # deg C

        # Solving
        T = np.linalg.solve(A, B)
        T = np.append(T, 100)

        if i == 0:
            case_3_data_2[alpha] = T

        if alpha == 6:
            case_3_heat_2[round(dx, 3)] = approx_heat_loss(0.5, area, T[-2], T[-1], alpha, dx)

# Printing Case 3 Data
print('\n', tbl.tabulate(case_3_data_2, headers='keys'))
print('\n', tbl.tabulate([case_3_heat_2], headers='keys'))

# Plotting Problem 2 Case 3 Data
x_data = []
for i in range(len(case_3_data_2[0.25])):
    x_data.append(i*dx)
for i in range(len(case_3_data_2)-1):
    plt.plot(x_data, case_3_data_2[Alpha[i]], label=str(Alpha[i]))
plt.title('Temperature (\u00B0C) Vs. Position (cm) for Various Alpha')
plt.ylabel('Temperature (\u00B0C)')
plt.xlabel('Position (cm)')
plt.grid()
plt.legend()
plt.show()

case_1_err = []
case_2_err = []
case_3_err = []
for i in range(len(case_1_dx)):
    case_1_err.append(percent_err(list(case_1_dx.values())[i], list(case_1_heat_2.values())[i]))
    case_2_err.append(percent_err(list(case_2_dx.values())[i], list(case_2_heat_2.values())[i]))
    case_3_err.append(percent_err(list(case_3_dx.values())[i], list(case_3_heat_2.values())[i]))


# Post Processing
header = ['dx', 'q_dot_exact', 'q_dot', '% error']
row = []
for i in range(len(case_1_err)):
    row.append([list(case_1_dx.keys())[i], list(case_1_dx.values())[i], list(case_1_heat_2.values())[i], case_1_err[i]])
print('\n', tbl.tabulate(row, headers=header))

row = []
for i in range(len(case_2_err)):
    row.append([list(case_2_dx.keys())[i], list(case_2_dx.values())[i], list(case_2_heat_2.values())[i], case_2_err[i]])
print('\n', tbl.tabulate(row, headers=header))

row = []
for i in range(len(case_3_err)):
    row.append([list(case_3_dx.keys())[i], list(case_3_dx.values())[i], list(case_3_heat_2.values())[i], case_3_err[i]])
print('\n', tbl.tabulate(row, headers=header))
