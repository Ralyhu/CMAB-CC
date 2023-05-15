import numpy as np
import random
import math

# for repr.
np.random.seed(50)

# sets some pairs to zero (to be used with sparse graphs)
def generate_weight_pairs_m(n_pairs, target_ratio, lb=0.0, ub=1.0, zero_pairs=0):

    w_plus = [random.uniform(lb, ub) for i in range(n_pairs)]
    w_minus = [random.uniform(lb, ub) for i in range(n_pairs)]

    tot_pairs = n_pairs + zero_pairs

    # sums are used to incrementally compute avgs
    sum_plus = sum(w_plus)
    sum_minus = sum(w_minus)
    avg_plus = sum_plus / tot_pairs
    avg_minus = sum_minus / tot_pairs
    gaps = [ abs(x-y) for (x,y) in zip(w_plus, w_minus)]
    delta = max(gaps)
    #print("Initial delta = " + str(delta))
    max_indeces = [i for i, j in enumerate(gaps) if math.isclose(j, delta, rel_tol=1e-5)]
    #print(max_indeces)
    changes = 0
    permutation = np.arange(n_pairs)
    assert len(permutation) == n_pairs
    # generate random permutation of n_pairs to sample
    np.random.shuffle(permutation)
    permutation = permutation.tolist()
    i = 0
    index = permutation[i]
    changed = False
    while not math.isclose(delta / (avg_plus + avg_minus), target_ratio, rel_tol=1e-5):
        cur_plus = w_plus[index]
        cur_minus = w_minus[index]
        if not math.isclose( abs(cur_plus - cur_minus), delta , rel_tol=1e-5): # avoid n_pairs where the maximum delta is obtained
            if delta / (avg_plus + avg_minus) > target_ratio:
                # increase avg plus or minus, by fixing delta. We increase only the smallest one since it allows for a maximum increase w.r.t. fixed delta
                if cur_minus <= cur_plus:
                    #print("Increase MINUS")
                    v = tot_pairs * ((delta / target_ratio) - avg_plus) - sum_minus + cur_minus
                    #print(v)
                    new_minus = min(ub, v, cur_plus + delta)
                    assert new_minus >= lb and new_minus <= ub
                    if not math.isclose(new_minus, cur_minus, rel_tol=1e-5):
                        changed = True
                    w_minus[index] = new_minus
                    sum_minus = sum_minus - cur_minus + new_minus
                    avg_minus = sum_minus / tot_pairs
                else:
                    #print("Increase PLUS")
                    v = tot_pairs * ((delta / target_ratio) - avg_minus) - sum_plus + cur_plus
                    #print(v)
                    new_plus = min(ub, v, cur_minus + delta)
                    #print(new_plus)
                    assert new_plus >= lb and new_plus <= ub
                    if not math.isclose(new_plus, cur_plus, rel_tol=1e-5):
                        changed = True
                    w_plus[index] = new_plus
                    sum_plus = sum_plus - cur_plus + new_plus
                    avg_plus = sum_plus / tot_pairs
            else:
                # decrease avg plus or minus, by fixing delta. For the same reason as before, decrease the biggest one
                if cur_minus <= cur_plus:
                    #print("Decrease PLUS")
                    v = tot_pairs * ((delta / target_ratio) - avg_minus) - sum_plus + cur_plus
                    #print(v)
                    new_plus = max(lb, v, cur_minus - delta)
                    #print(new_plus)
                    assert new_plus >= lb and new_plus <= ub
                    if not math.isclose(new_plus, cur_plus, rel_tol=1e-5):
                        changed = True
                    w_plus[index] = new_plus
                    sum_plus = sum_plus - cur_plus + new_plus
                    avg_plus = sum_plus / tot_pairs
                else:
                    #print("Decrease MINUS")
                    v = tot_pairs * ((delta / target_ratio) - avg_plus) - sum_minus + cur_minus
                    #print(v)
                    new_minus = max(lb, v, cur_plus - delta)
                    assert new_minus >= lb and new_minus <= ub
                    if not math.isclose(new_minus, cur_minus, rel_tol=1e-5):
                        changed = True
                    w_minus[index] = new_minus
                    sum_minus = sum_minus - cur_minus + new_minus
                    avg_minus = sum_minus / tot_pairs
            changes += 1
        i += 1
        if i == n_pairs:
            # generate another random permutation of n_pairs
            permutation = np.arange(n_pairs)
            np.random.shuffle(permutation)
            permutation = permutation.tolist()
            i = 0
            if changed == False:
                # local stuck, try to reduce the maximum gap delta by fixing avg plus and avg minus
                for ind in max_indeces:
                    k = random.random()
                    if delta / (avg_plus + avg_minus) > target_ratio: # decrease delta
                        if w_plus[ind] > w_minus[ind]:
                            var_plus = - delta * k
                            var_minus = delta * k
                        else:
                            var_plus = delta * k
                            var_minus = - delta * k
                    else: # increase delta
                        if w_plus[ind] > w_minus[ind]:
                            var_plus =  min(delta * k, ub - w_plus[ind])
                            var_minus = - min(delta * k, w_minus[ind] - lb)
                        else:
                            var_plus = - min(delta * k, w_plus[ind] - lb)
                            var_minus = min(delta * k, ub - w_minus[ind])
                    sum_plus = sum_plus + var_plus
                    avg_plus = sum_plus / tot_pairs
                    sum_minus = sum_minus + var_minus
                    avg_minus = sum_minus / tot_pairs
                    w_plus[ind] = w_plus[ind] + var_plus
                    assert w_plus[ind] <= ub and w_plus[ind] >= lb
                    w_minus[ind] = w_minus[ind] + var_minus
                    assert w_minus[ind] <= ub and w_minus[ind] >= lb
                    changes += 2
                gaps = [ abs(x-y) for (x,y) in zip(w_plus, w_minus)]
                delta = max(gaps)
                max_indeces = [i for i, j in enumerate(gaps) if math.isclose(j, delta, rel_tol=1e-5)]
            changed = False
        index = permutation[i]

    # print("Number of changes = " + str(changes))
    assert(math.isclose(delta / (avg_plus + avg_minus), target_ratio, rel_tol=1e-5)) # check desired property
    sum_plus = sum(w_plus)
    sum_minus = sum(w_minus)
    avg_plus = sum_plus / tot_pairs
    avg_minus = sum_minus / tot_pairs
    gaps = [ abs(x-y) for (x,y) in zip(w_plus, w_minus)]
    delta = max(gaps)
    assert(math.isclose(delta / (avg_plus + avg_minus), target_ratio, rel_tol=1e-5)) # check desired property again, by recomputing from scratch the involved measures
    cont = 0
    for p, m in zip(w_plus, w_minus):
        if p >= m:
            cont += 1
    #print("% n_pairs w+ >= w- = " + str(cont/len(w_plus)))
    return w_plus, w_minus

if __name__ == '__main__':
    n = 100
    n_pairs = int(n * (n - 1) / 2)
    target_ratio = 20
    lb=0
    ub=1
    fraction_zero_pairs = 0.9
    zero_pairs = int(n_pairs * fraction_zero_pairs)
    generate_weight_pairs_m(n_pairs - zero_pairs, target_ratio, lb, ub, zero_pairs)