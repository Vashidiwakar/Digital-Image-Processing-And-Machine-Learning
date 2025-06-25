import numpy as np

def gradient_descent(f, grad_f, project, w0, learning_rate=0.01, max_iter=1000, tol=1e-6):
    """
    Perform Projected Gradient Descent.

    Args:
        f (function): Objective function to minimize.
        grad_f (function): Gradient (or subgradient) of the objective function.
        project (function): Projection function to ensure feasibility.
        w0 (np.array): Initial point.
        learning_rate (float): Step size.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        w (np.array): Optimal point found.
        history (list): History of objective values.
    """
    w = w0
    history = []

    for t in range(max_iter):
        g = grad_f(w)  # Compute gradient
        w_new = w - learning_rate * g  # Gradient step
        w_new = project(w_new)  # Projection step

        # Check for convergence
        if np.linalg.norm(w_new - w) < tol:
            break

        w = w_new
        history.append(f(w))

    return w, history

# Example usage:
if __name__ == "__main__":
    # Define an example quadratic function and its gradient
    def f(w):
        return np.sum(w**2)

    def grad_f(w):
        return 2 * w

    # Define a projection function for the feasible set C (e.g., projection onto a ball of radius 1)
    def project(w):
        return np.clip(w, -1, 1)

    # Initial point
    w0 = np.array([3.0, 4.0])

    # Run Projected Gradient Descent
    optimal_w, history = gradient_descent(f, grad_f, project, w0)

    print("Optimal w:", optimal_w)
    print("History of objective values:", history)
