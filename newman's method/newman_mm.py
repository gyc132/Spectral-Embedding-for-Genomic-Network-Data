import numpy as np
from utils import power_method

class NewmanMM:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.splits_log = []
        self.B, self.k, self.m = self.calculate_modularity_matrix(adj_matrix)
        self.labels = np.zeros(adj_matrix.shape[0], dtype=int)

    def calculate_modularity_matrix(self, adj_matrix):
        k = np.sum(adj_matrix, axis=1)
        m = np.sum(adj_matrix) / 2
        B = adj_matrix - np.outer(k, k) / (2 * m)
        return B, k, m

    def calculate_delta_modularity(self, s, group):
        n_g = len(group)
        B_g = self.B[np.ix_(group, group)]
        for i in range(n_g):
            delta_ij = np.zeros(n_g)
            delta_ij[i] = 1  # Kronecker delta for i == j
            B_g[i, :] -= delta_ij * np.sum(self.B[group[i], :])
        return (1 / (4 * self.m)) * np.dot(s.T, np.dot(B_g, s))

    def calculate_modularity(self, s, group = None):
        B, _, m = self.calculate_modularity_matrix(self.adj_matrix)
        return (1 / (4 * m)) * np.dot(s.T, np.dot(B, s))

    def vertex_moving_ft(self, s, max_func, group = None):
        n = len(s)
        best_overall_s = s.copy()  # Store the overall best split
        best_overall_modularity = max_func(s=best_overall_s, group=group)  # Best overall modularity

        improvement = True  # To track if modularity improves across rounds
        while improvement:
            improvement = False  # Reset improvement flag for each round
            moved = np.zeros(n, dtype=bool)  # Track which nodes have been moved
            # Track the intermediate states and their modularity within this round
            intermediate_states = []
            intermediate_modularities = []

            # Loop until all vertices have been moved once in this round
            for _ in range(n):
                best_move_modularity = None
                best_move_idx = None
                
                # Check all un-moved vertices for the best move
                for i in range(n):
                    if not moved[i]:
                        s[i] = -s[i]  # Move vertex i to the other group
                        modularity = max_func(s=s, group=group)  # Calculate modularity
                        # Track the best move (maximizing modularity or minimizing decrease)
                        if best_move_modularity is None or modularity > best_move_modularity:
                            best_move_modularity = modularity
                            best_move_idx = i
                        s[i] = -s[i]  # Undo the move
                # If no move improves modularity, finish this round
                if best_move_modularity is None: break
                
                # Apply the best move
                s[best_move_idx] = -s[best_move_idx]
                moved[best_move_idx] = True  # Mark this vertex as moved
                # Track intermediate state and modularity
                intermediate_states.append(s.copy())
                intermediate_modularities.append(best_move_modularity)

            # After all nodes have been moved, find the best intermediate state for this round
            best_state_idx = np.argmax(intermediate_modularities)
            best_s = intermediate_states[best_state_idx]
            best_modularity = intermediate_modularities[best_state_idx]
            # Compare with the overall best state across all rounds
            if best_modularity > best_overall_modularity:
                best_overall_s = best_s.copy()
                best_overall_modularity = best_modularity
                improvement = True  # If there was improvement, continue to another round
        
        return best_overall_s, best_overall_modularity

    def additional_modularity_division(self, group):
        # Calculate B_g
        n_g = len(group)
        B_g = self.B[np.ix_(group, group)]
        for i in range(n_g):
            delta_ij = np.zeros(n_g)
            delta_ij[i] = 1  # Kronecker delta for i == j
            B_g[i, :] -= delta_ij * np.sum(self.B[group[i], :])
        
        max_evalue, eigenvector = power_method(B_g)
        s = np.where(eigenvector > 0, 1, -1)
        delta_Q = (1 / (4 * self.m)) * np.dot(s.T, np.dot(B_g, s))

        if delta_Q <= 0: return None, None, delta_Q
        return s, B_g, delta_Q


    def further_dividing(self, adj_matrix, s_initial, label_base):
        group_1 = np.where(s_initial == 1)[0]
        group_2 = np.where(s_initial == -1)[0]
        if len(group_1) == 0 or len(group_2) == 0: return "indivisible", s_initial

        self.labels[group_1] = label_base
        self.labels[group_2] = label_base + 1
        next_label = label_base + 2

        # Step 2: Check for further division on the left group (group_1)
        s_l, B_l, delta_Q = self.additional_modularity_division(group_1)
        if s_l is not None:
            s_l_finetuned, best_delta_Q = self.vertex_moving_ft(s_l, self.calculate_delta_modularity, group_1)
            self.splits_log.append({'split': s_l_finetuned, 'delta_Q': best_delta_Q, 'group': group_1})
            # Recursively divide the left group
            self.further_dividing(adj_matrix[np.ix_(group_1, group_1)], s_l_finetuned, next_label)

        # Step 3: Check for further division on the right group (group_2)
        s_r, B_r, delta_Q = self.additional_modularity_division(group_2)
        if s_r is not None:
            s_r_finetuned, best_delta_Q = self.vertex_moving_ft(s_r, self.calculate_delta_modularity, group_2)
            self.splits_log.append({'split': s_r_finetuned, 'delta_Q': best_delta_Q, 'group': group_2})
            # Recursively divide the right group
            self.further_dividing(adj_matrix[np.ix_(group_2, group_2)], s_r_finetuned, next_label)


    def main(self, adj_matrix):
        # Calculate modularity matrix B
        max_evalue, eigenvector = power_method(self.B)

        if max_evalue <= 0: return "indivisible"
        s = np.where(eigenvector > 0, 1, -1)
        s_finetuned, best_modularity = self.vertex_moving_ft(s, self.calculate_modularity)
        self.splits_log.append({'split': s_finetuned, 'delta_Q': best_modularity})

        self.further_dividing(adj_matrix, s_finetuned, 0)

        return self.labels, self.splits_log
