import numpy as np

def sqn(x0, H1, m, alpha, g, dist):
    """
    Stochastic Quasi Newton's Method
    x0:    the starting point
    H1:    the initial approximation of the gradient
    m:     a sequence of batch sizes
    alpha: a sequence of step sizes
    g:     R^n x R^d -> R, stochastic gradient of f taking location and random input
    dist:  the distribution of the random variable

    """

    # iteration parameters
    MAX_ITER = 100
    tol = 1e-8
    p = 4  # memory size


    # iterate data, which is computed and eventually to be returned     
    x = np.empty((MAX_ITER+1, *x0.shape))
    s = np.empty((MAX_ITER+1, *x0.shape))
    y = np.empty((MAX_ITER+1, *x0.shape))
    err = np.empty(MAX_ITER+1)

    x[0] = x0
    err[0] = np.linalg.norm(x0)

    k = 0

    # main loop
    while err[k] >= tol and k < MAX_ITER:
        # general SQN step
        x[k+1] = x[k] - alpha[k] * sdlbfgs_step(k, dist, m, g, x, s, y, p)
        err[k+1] = np.linalg.norm(x[k+1] - x[k])

    return x[:k], err[:k]


def sdlbfgs_step(k, dist, m, g, x, s, y, p):
    """
    Step computation using SdLBFGS
    k:    current iteration
    dist: the distribution of the random variable
    m:    sequence of batch sizes
    g:    R^n x R^d -> R, stochastic gradient of f taking location and random input
    x:    sequence of points in algorithm [TODO: this needs rewriting]
    s:    blah 
    y:    blah
    """

    mu = np.empty((np.min(p,k-1), 1))
    u = np.empty((np.min(p,k-1), *y[0].shape))
    u[0] = g[k]

    # draw the random sample
    xi_k = dist(m[k])
    
    
    s[k-1] = x[k] - x[k-1]
    y[k-1] = np.sum(g(x[k], xi_k[i]) - g(x[k-1], xi_k[i-1]) for i in range(len(xi_k))) / m[k]

    gamma = np.max(
            delta,
            np.dot(y[k-1], y[k-1]) / np.dot(s[k-1], y[k-1])
            )

    thetakm1 = 1.
    if skm1.dot(ykm1) < 0.25 * skm1.dot(ihessk).dot(skm1):
        thetakm1 =0.75 * skm1.dot(ihessk).dot(skm1) / \
                (skm1.dot(ihessk).dot(skm1) - skm1.dot(ykm1))

    ybarkm1 = thetakm1 * ykm1 + (1-thetakm1) * ihessk.dot(skm1)
    rhokm1 = 1./skm1.dot(ybarkm1)

    for i in range(min(p, k-1)):
        mu[i] = rho[k-i-1] * u[i].dot(s[k-i-1])
        u[i+1] = u[i] - mu[i] * ybar[k-i-1]

    v0 = u[p] / gammak
    for i in range(min(p,k-1)):
        nu[i] = rho[k-p+i] * v[i].dot(ybar[k-p+i])
        v[i+1] = v[i] + (mu[p-i-1] - nu[i]) * s[k-p+i]

    return v[p]


