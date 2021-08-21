import numpy as np
import random as rnd

# north-west corner allocation
def TP_NWCA(T):
    T_cost = np.copy(T)
    m = T_cost.shape[0]
    n = T_cost.shape[1]

    total_supply = np.sum(T_cost[:m - 1, -1])
    total_demand = np.sum(T_cost[-1, :n - 1])

    if total_supply == total_demand: # balanced
        pass
    elif total_supply < total_demand: # unbalanced + infeasible
        print("Demand > Supply - Solution not possible")
        return None, None
    else: # unbalanced
        dummy_col = np.zeros([m]) # create dummy demand (all cost = 0)
        dummy_col[-1] = total_supply - total_demand

        T_cost = np.insert(T_cost, n - 1, dummy_col, 1) # inserted into tableau
        n += 1

    T_aug = np.copy(T_cost)
    T_aug = T_aug.astype('float64')
    T_sol = np.zeros([m - 1, n - 1]) # solution matrix

    i, j = 0, 0 # start at top left corner of matrix (north west corner)

    while i < m - 1 and j < n - 1:
        di, dj = 0, 0

        if (T_aug[-1, j] > T_aug[i, -1]): # if demand j > supply i
            amnt = T_aug[i, -1] # use all of supply i to demand j
            di = 1 # move down to next supply
        elif (T_aug[-1, j] < T_aug[i, -1]): # else if, demand j < supply i, satisfy all of demand j
            amnt = T_aug[-1, j]
            dj = 1 # move across to next demand
        else: # else , demand j = supply i, satisfy all of demand j using all of supply i
            amnt = T_aug[-1, j]
            di = 1 # move down to next supply
            dj = 1 # move across to next demand

        T_sol[i, j] = amnt # assign amount to solution table

        T_aug[-1, j] -= amnt
        T_aug[i, -1] -= amnt

        i += di
        j += dj

    t_cost = 0

    for i in range(0, m - 1):
        for j in range(0, n - 1):
            t_cost += T_cost[i, j] * T_sol[i, j] # multiply cell cost by cell amount and add to total

    return T_sol, t_cost


# least-cost method
def TP_LC(T):
    T_cost = np.copy(T)
    m = T_cost.shape[0]
    n = T_cost.shape[1]

    total_supply = np.sum(T_cost[:m - 1, -1])
    total_demand = np.sum(T_cost[-1, :n - 1])

    if total_supply == total_demand: # balanced
        pass
    elif total_supply < total_demand: # unbalanced + infeasible
        print("Demand > Supply - Solution not possible")
        return None, None
    else: # unbalanced
        dummy_col = np.zeros([m]) # create dummy demand (all cost = 0)
        dummy_col[-1] = total_supply - total_demand

        T_cost = np.insert(T_cost, n - 1, dummy_col, 1) # inserted into tableau
        n += 1

    T_aug = np.copy(T_cost)
    T_aug = T_aug.astype('float64')
    T_sol = np.zeros([m - 1, n - 1]) # solution matrix

    cost_table = np.zeros([(m - 1) * (n - 1), 3])

    for i in range(0, m - 1): # scan transportation tableau and build table of cells and costs
        for j in range(0, n - 1):
            indx = (i * (n - 1)) + j
            cost_table[indx, 0] = T_cost[i, j]
            cost_table[indx, 1] = i
            cost_table[indx, 2] = j

    cost_table = cost_table[cost_table[:, 0].argsort()] # sort by cost (lowest to highest)
    p = 0

    while np.sum(T_aug[:m - 1, -1]) > 0: # total supply not used up
        min_i, min_j = int(cost_table[p, 1]), int(cost_table[p, 2]) # select next (p'th) lowest cost  cell

        if (T_aug[-1, min_j] > T_aug[min_i, -1]): # if demand j > supply i, use all of supply i to demand j
            amnt = T_aug[min_i, -1]
        elif (T_aug[-1, min_j] < T_aug[min_i, -1]): # else, if demand j < supply i, satisfy all of demand j
            amnt = T_aug[-1, min_j]
        else: # else, demand j = supply i (eliminate both)
            amnt = T_aug[-1, min_j]

        T_sol[min_i, min_j] = amnt # assign amount to solution table

        T_aug[-1, min_j] -= amnt # decrement demand by amount
        T_aug[min_i, -1] -= amnt # decrement supply by amount

        p += 1

    t_cost = 0 # total cost

    for i in range(0, m - 1):
        for j in range(0, n - 1):
            t_cost += T_cost[i, j] * T_sol[i, j] # multiply cell cost by cell amount and add to total cost

    return T_sol, t_cost

# vogel's approximation method
def TP_VAM(T):
    T_cost = np.copy(T)
    m = T_cost.shape[0]
    n = T_cost.shape[1]

    total_supply = np.sum(T_cost[:m - 1, -1])
    total_demand = np.sum(T_cost[-1, :n - 1])

    if total_supply == total_demand:
        pass
    elif total_supply < total_demand:
        print("Demand > Supply - Solution not possible")
        return None, None
    else:
        dummy_col = np.zeros([m]) # create dummy demand (all cost = 0)
        dummy_col[-1] = total_supply - total_demand

        T_cost = np.insert(T_cost, n - 1, dummy_col, 1) # inserted into tableau
        n += 1

    pen_col = np.zeros([m])
    pen_row = np.zeros([n + 1])

    T_aug = np.copy(T_cost)

    T_aug = np.insert(T_aug, n - 1, pen_col, 1) # insert penalty rows
    T_aug = np.insert(T_aug, m - 1, pen_row, 0)

    T_aug = T_aug.astype('float64')
    T_sol = np.zeros([m - 1, n - 1]) # solution matrix

    while np.sum(T_aug[:m - 1, -1]) > 0: # total supply not used up

        for i in range(0, m - 1): # get all penalties for supplies (rows)
            sorted_vals = np.sort(T_aug[i, :n - 1].flatten())

            if sorted_vals[0] == float('inf') and sorted_vals[1] == float('inf'): # if smallest/2nd smallest values inf (row fully done)
                pen = -1
            elif sorted_vals[1] == float('inf'): # if 2nd smallest value 'inf' - only 1 cell left
                pen = 1
            else: # neither smallest/2nd smallest values inf - calc penalty as normal
                pen = sorted_vals[1] - sorted_vals[0]

            T_aug[i, -2] = pen

        for j in range(0, n - 1): # get all penalties for demands (columns)
            sorted_vals = np.sort(T_aug[:m - 1, j].flatten())

            if sorted_vals[0] == float('inf') and sorted_vals[1] == float('inf'): # if smallest/2nd smallest values inf (column fully done)
                pen = -1
            elif sorted_vals[1] == float('inf'): # if 2nd smallest value 'inf' - only 1 cell left
                pen = 1
            else: # neither smallest/2nd smallest values inf - calc penalty as normal
                pen = sorted_vals[1] - sorted_vals[0]

            T_aug[-2, j] = pen

        max_row_penalty = max(T_aug[:-2, -2]) # max row and column penalties
        max_col_penalty = max(T_aug[-2, :-2])

        if max_row_penalty > max_col_penalty: # if max penalty in one of rows, select cell in supply row with min cost
            max_i_indx = np.where(T_aug[:-2, -2] == max_row_penalty)[0][0]

            s_costs = T_aug[max_i_indx, :n - 1]
            min_cost = min(s_costs)
            max_j_indx = np.where(s_costs == min_cost)[0][0]
        elif max_row_penalty < max_col_penalty: # if max penalty in one of cols, select cell in demand col with min cost
            max_j_indx = np.where(T_aug[-2, :-2] == max_col_penalty)[0][0]

            d_costs = T_aug[:m - 1, max_j_indx]
            min_cost = min(d_costs)
            max_i_indx = np.where(d_costs == min_cost)[0][0]
        else: # if penalties equal, select whichever cell can have more assigned
            max_i_indx = np.where(T_aug[:-2, -2] == max_row_penalty)[0][0]
            max_j_indx = np.where(T_aug[-2, :-2] == max_col_penalty)[0][0]

            s_costs = T_aug[max_i_indx, :n - 1] # row of penalised supply costs
            d_costs = T_aug[:m - 1, max_j_indx] # col of penalised demand costs

            min_cost_s = min(s_costs)
            min_cost_d = min(d_costs)

            min_cost_i_indx = np.where(d_costs == min_cost_d)[0][0] # find indices of lowest costs
            min_cost_j_indx = np.where(s_costs == min_cost_s)[0][0]

            min_cost_i_supply = T_aug[min_cost_i_indx, -1] # check demand vs supply (penalised row)
            min_cost_i_demand = T_aug[-1, max_j_indx]

            min_cost_j_supply = T_aug[max_i_indx, -1] # check demand vs supply (penalised col)
            min_cost_j_demand = T_aug[-1, min_cost_j_indx]

            if min(min_cost_j_supply, min_cost_j_demand) > min(min_cost_i_supply, min_cost_i_demand):
                max_i_indx = max_i_indx
                max_j_indx = min_cost_j_indx
            else:
                max_i_indx = min_cost_i_indx
                max_j_indx = max_j_indx

        if (T_aug[-1, max_j_indx] > T_aug[max_i_indx, -1]): # if demand j > supply i, use entire supply
            amnt = T_aug[max_i_indx, -1]

            for k in range(0, n - 1): # eliminate all costs from supply row from further consideration
                T_aug[max_i_indx, k] = 'inf'
        elif (T_aug[-1, max_j_indx] < T_aug[max_i_indx, -1]): # if demand j < supply i, satisfy entire demand
            amnt = T_aug[-1, max_j_indx]

            for k in range(0, m - 1): # eliminate all costs from demand col from further consideration
                T_aug[k, max_j_indx] = 'inf'
        else:
            amnt = T_aug[-1, max_j_indx]

            for k in range(0, n - 1): # eliminate all costs from supply row from further consideration
                T_aug[max_i_indx, k] = 'inf'

            for k in range(0, m - 1): # eliminate all costs from demand col from further consideration
                T_aug[k, max_j_indx] = 'inf'

        T_sol[max_i_indx, max_j_indx] = amnt # assign amount to solution table

        T_aug[max_i_indx, -1] -= amnt
        T_aug[-1, max_j_indx] -= amnt

    t_cost = 0 # total cost

    for i in range(0, m - 1):
        for j in range(0, n - 1):
            t_cost += T_cost[i, j] * T_sol[i, j] # multiply cell cost by cell amount and add to total cost

    return T_sol, t_cost


def TP_PrintSolution(table):
    m = table.shape[0]
    n = table.shape[1]

    header = "\nSol"

    for j in range(0, n):
        header += "\tD{0}".format(j + 1)

    print(header)

    for i in range(0, m):
        row = "S{0}".format(i + 1)

        for j in range(0, n):
            row += "\t{0}".format(int(table[i, j]))

        print(row)


def TP_FillTable(T_test):

    m = T_test.shape[0] # no. of rows
    n = T_test.shape[1] # no. of cols

    sum1 = rnd.randint(1000, 5000) # total supply/demand = random

    a = rnd.sample(range(1, sum1), m - 2) + [0, sum1] # generate random sample of m size
    list.sort(a) # sort
    supply = [a[i + 1] - a[i] for i in range(len(a) - 1)] # take differences to get m supplies that add to sum1 (total)

    a = rnd.sample(range(1, sum1), n - 2) + [0, sum1] # generate random sample of n size
    list.sort(a) # sort
    demand = [a[i + 1] - a[i] for i in range(len(a) - 1)] # take differences to get n demands that add to sum1 (total)

    for i in range(0, m):
        for j in range(0, n):
            c = rnd.randint(1, 100) # random cost for given cell

            if i == m - 1 and j == n - 1:
                T_test[i, j] = 0 # dummy cell
            elif i == m - 1: # last row
                T_test[i, j] = demand[j] # assign demand
            elif j == n - 1: # last col
                T_test[i, j] = supply[i] # assign supply
            else:
                T_test[i, j] = c # assign cell cost

    return T_test

# Modified Distribution method for optimality
def TP_OptimalCheck(T_cost, T_sol, cost):
    m = T_cost.shape[0] - 1
    n = T_cost.shape[1] - 1

    optimality = ""

    count_fill = np.count_nonzero(T_sol) # no. of assigned entries

    if m + n - 1 != count_fill: # if S + D - 1 != no assigned cells, solution is degenerate (cant check optimality)
        optimality = "Degenerate"
        return optimality

    vec_A = np.zeros([count_fill + 1, m + n])
    vec_b = np.zeros([count_fill + 1])

    indx = 1

    vec_A[0, 0] = 1 # set extra condition (u_i = 0)
    vec_b[0] = 0

    for i in range(0, m): # set rest of conditions based on assigned cells
        for j in range(0, n):
            if T_sol[i, j] > 0:
                vec_A[indx, i] = 1 # set coefficient of ui in current row of A = 1
                vec_A[indx, m + j] = 1 # set coefficient of vj in current row of A = 1
                vec_b[indx] = T_cost[i, j] # set value in b vector to cost C_ij

                indx += 1

    vec_uv = np.linalg.solve(vec_A, vec_b) # solve for values of ui & vj
    vec_delta = []

    for i in range(0, m):
        for j in range(0, n):
            if T_sol[i, j] == 0: # unoccupied cell
                vec_delta.append(T_cost[i, j] - (vec_uv[i] + vec_uv[j + m]))

    if min(vec_delta) >= 0: # if minimum is non-negative, then solution is optimal
        optimality = "Optimal"
    else:
        optimality = "Not Optimal"

    return optimality

# run Monte Carlo comparison of methods
def TP_Comparison(iter, min_row, max_row, max_col, fb=True):

    if fb:
        print("n\tSize\t\tNW\t\tLC\t\tVA\t\tLowest")

    n_nwca = 0 # counts
    n_lc = 0
    n_vam = 0

    for i in range(0, iter): # run n iterations

        # m = 3
        # n = 4

        m = rnd.randint(min_row, max_row)
        n = rnd.randint(m + 1, max_col)

        T_test = np.zeros([m + 1, n + 1]) # create and randomly fill table
        T_test = TP_FillTable(T_test)

        sol1, cost1 = TP_NWCA(T_test) # solve using 3 methods
        sol2, cost2 = TP_LC(T_test)
        sol3, cost3 = TP_VAM(T_test)

        optimal = ""

        min_cost = min([cost1, cost2, cost3]) # figure out which of 3 give best result (can be > 1 method)

        if min_cost == cost1: # NWCA found lowest cost
            if not optimal:
                optimal += "NW"
            else:
                optimal += "/NW"
            n_nwca += 1

        if min_cost == cost2: # LC found lowest cost
            if not optimal:
                optimal += "LC"
            else:
                optimal += "/LC"
            n_lc += 1

        if min_cost == cost3: # VAM found lowest cost
            if not optimal:
                optimal += "VA"
            else:
                optimal += "/VA"
            n_vam += 1

        s_string = "{0} x {1}".format(m, n)

        if fb:
            print("{0}\t{1}\t\t{2}\t\t{3}\t\t{4}\t\t{5}".format(i + 1, s_string, int(cost1), int(cost2), int(cost3), optimal))

    print("\nResults\t\t\tNW\t\tLC\t\tVA")
    print("{0}\t\t\t{1:.4f}\t\t{2:.4f}\t\t{3:.4f}".format("%", n_nwca / iter, n_lc / iter, n_vam / iter))


# examples 1
# TP_Comparison(1000, 5, 10, 15, True)

# TP_Comparison(5000, 25, 50, 75, True)

# testing 1 - from presentation
T1 = np.array([[6, 7, 8, 10, 100], [4, 7, 13, 5, 200], [7, 8, 7, 8, 300], [150, 100, 275, 75, 0]])
sol1a, cost1a = TP_NWCA(T1)
sol1b, cost1b = TP_LC(T1)
sol1c, cost1c = TP_VAM(T1)

check1a = TP_OptimalCheck(T1, sol1a, cost1a)
check1b = TP_OptimalCheck(T1, sol1b, cost1b)
check1c = TP_OptimalCheck(T1, sol1c, cost1c)

TP_PrintSolution(sol1a)
print(cost1a)
print(check1a)

TP_PrintSolution(sol1b)
print(cost1b)
print(check1b)

TP_PrintSolution(sol1c)
print(cost1c)
print(check1c)


# testing 2 - https://arts.brainkart.com/article/initial-basic-feasible-solution---northwest-corner-method---transportation-problem-1128/
'''T2 = np.array([[8, 6, 10, 9, 35], [9, 12, 13, 7, 50], [14, 9, 16, 5, 40], [45, 20, 30, 30, 0]])
sol2a, cost2a = TP_NWCA(T2)
sol2b, cost2b = TP_LC(T2)
sol2c, cost2c = TP_VAM(T2)

TP_PrintSolution(sol2a)
print(cost2a)

TP_PrintSolution(sol2b)
print(cost2b)

TP_PrintSolution(sol2c)
print(cost2c)
'''

# testing 3 - https://cbom.atozmath.com/example/CBOM/Transportation.aspx?he=e&q=nwcm
'''T3 = np.array([[19, 30, 50, 10, 7], [70, 30, 40, 60, 9], [40, 8, 70, 20, 18], [5, 8, 7, 14, 0]])
sol3a, cost3a = TP_NWCA(T3)
sol3b, cost3b = TP_LC(T3)
sol3c, cost3c = TP_VAM(T3)

TP_PrintSolution(sol3a)
print(cost3a)

TP_PrintSolution(sol3b)
print(cost3b)

TP_PrintSolution(sol3c)
print(cost3c)
'''

# testing 4 - https://www.engineeringenotes.com/project-management-2/operations-research/testing-the-optimality-of-transportation-solution-operations-research/15526
'''T4 = np.array([[21, 16, 25, 13, 11], [17, 18, 14, 22, 13], [32, 22, 13, 41, 19], [6, 10, 12, 15, 0]])
sol4a, cost4a = TP_NWCA(T4)
sol4b, cost4b = TP_LC(T4)
sol4c, cost4c = TP_VAM(T4)

check4a = TP_OptimalCheck(T4, sol4a, cost4a)
check4b = TP_OptimalCheck(T4, sol4b, cost4b)
check4c = TP_OptimalCheck(T4, sol4c, cost4c)

TP_PrintSolution(sol4a)
print(cost4a)
print(check4a)

TP_PrintSolution(sol4b)
print(cost4b)
print(check4b)

TP_PrintSolution(sol4c)
print(cost4b)
print(check4c)
'''

# testing 5 - unbalanced problems - https://cbom.atozmath.com/example/CBOM/Transportation.aspx?he=e&q=nwcm&ex=2
'''T5 = np.array([[4, 8, 8, 76], [16, 24, 16, 82], [8, 16, 24, 77], [72, 102, 41, 0]])
sol5a, cost5a = TP_NWCA(T5)
sol5b, cost5b = TP_LC(T5)
sol5c, cost5c = TP_VAM(T5)

TP_PrintSolution(sol5a)
print(cost5a)

TP_PrintSolution(sol5b)
print(cost5b)

TP_PrintSolution(sol5c)
print(cost5c)
'''

# testing 6 - 30 x 40 random table
'''T6 = TP_FillTable(np.zeros([30 + 1, 40 + 1]))
sol6a, cost6a = TP_NWCA(T6)
sol6b, cost6b = TP_LC(T6)
sol6c, cost6c = TP_VAM(T6)

check6a = TP_OptimalCheck(T6, sol6a, cost6a)
check6b = TP_OptimalCheck(T6, sol6b, cost6b)
check6c = TP_OptimalCheck(T6, sol6c, cost6c)

print(cost6a)
print(check6a)

print(cost6b)
print(check6b)

print(cost6b)
print(check6c)
'''
