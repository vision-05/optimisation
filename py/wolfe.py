import matplotlib.pyplot as plt
import numpy as np

def f_1(x):
    return x**4 + 20 * x **3

def f_1_grad(x):
    return 4*x**3 * 60 * x**2

def f_1_hess(x):
    x_arr = np.atleast_1d(x)
    return 12 * np.diag(x_arr**2) + 120*np.diag(x_arr)

def linesearch_newton(f, f_grad, f_hess, x0, eps, iters, logger):
    x_prev = 0
    x = x0
    p = 0
    c_1 = 0.0001
    c_2 = 0.9
    rho = 0.5
    rho2 = 1.1
    for i in range(iters):
        g = f_grad(x)
        H = f_hess(x)
        p = np.linalg.solve(H, -g)
        alpha = 1.0

        while f(x + alpha*p) > f(x) + c_1 * alpha * np.dot(g ,p):
            alpha *= rho #decrease alpha until we meet the Armijo condition

        while np.dot(f_grad(x + alpha*p), p) < c_2 * np.dot(g, p):
            alpha *= rho2 #increase alpha until we meet the curvature condition

        x = x + alpha * p
        logger(f"Iteration {i} x_est {x}")

        if np.abs(x - x_prev) < eps:
            logger(f"Terminated due to convergence at iteration {i}")
            return x
        
    return x

min = linesearch_newton(f_1, f_1_grad, f_1_hess, np.array([100]), 1e-4, 10000, print)
        

t = np.linspace(-50,50,100)
xs = f_1(t)
plt.figure()
plt.plot(t, xs)
plt.show()
print(min)
