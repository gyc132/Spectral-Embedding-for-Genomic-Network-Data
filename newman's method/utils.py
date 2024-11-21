import numpy as np

def power_method(B, num_iterations=1000, tol=1e-6):
    # Power method to compute the largest eigenvalue and corresponding eigenvector
    n = B.shape[0]
    u = np.random.rand(n)
    u /= np.linalg.norm(u)
    lambda_old = 0
    for _ in range(num_iterations):
        # Matrix-vector multiplication
        u_new = np.dot(B, u)
        u_new /= np.linalg.norm(u_new)
        # Eigenvalue corresponding to u
        lambda_new = np.dot(u_new, np.dot(B, u_new))
        # Check for convergence
        if np.abs(lambda_new - lambda_old) < tol:
            break
        u = u_new
        lambda_old = lambda_new 
    return lambda_new, u