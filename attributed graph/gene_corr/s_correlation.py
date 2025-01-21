import numpy as np
import scipy.sparse as sp

def cal_s_correlation(X, Y):
    n = len(X)
    if len(Y) != n:
        raise ValueError("X and Y must have the same length.")
    if n < 2:
        raise ValueError("X and Y must have at least two elements.")

    # Step 1: Randomly shuffle X and Y to break ties uniformly at random
    random_indices = np.random.permutation(n)
    X = np.array(X)[random_indices]
    Y = np.array(Y)[random_indices]

    # Step 2: Sort X and rearrange Y accordingly
    sorted_indices = np.argsort(X)  # Get the indices that would sort X
    sorted_Y = Y[sorted_indices]  # Rearrange Y based on sorted X

    # Step 3: Define r_i as the ranks of sorted_Y
    r = np.argsort(np.argsort(sorted_Y)) + 1  # Ranking of sorted_Y (1-based indexing)

    # Step 4: Compute the numerator: sum of |r_{i+1} - r_i| for i in 1 to n-1
    rank_diffs = np.abs(np.diff(r))
    numerator = n * np.sum(rank_diffs)

    # Step 5: Compute the denominator

    l = np.array([np.sum(sorted_Y >= sorted_Y[i]) for i in range(n)])  # Compute l_i
    denominator = 2 * np.sum(l * (n - l))
    
    # Step 6: Compute xi correlation
    xi = 1 - numerator / denominator
    return xi

if __name__ == "__main__":
    X = [3, 1, 4, 2]
    Y = [10, 20, 30, 40]
    correlation = cal_s_correlation(X, Y)
    print("S_correlation:", correlation)