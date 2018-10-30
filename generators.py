from sklearn.utils import check_random_state
from scipy.integrate import odeint
import numpy as np


def sphere(n_samples, noise=0, uniform=False):

    if uniform:
        p = np.linspace(0,60*np.pi, n_samples)
        t = np.linspace(0,14*np.pi, n_samples)
    else:
        random_state = check_random_state(0)
        p = random_state.rand(n_samples) * (2 * np.pi)
        t = random_state.rand(n_samples) * np.pi
    color = np.cos(t)
    x, y, z = np.sin(t) * np.cos(p), \
        np.sin(t) * np.sin(p), \
        np.cos(t)
    X = np.array([x, y, z]).T

    if noise != 0:
        X = np.random.normal(X, noise)

    return X, color


def globe(n_samples, noise=0, uniform=False):

    if uniform:
        p = np.linspace(0,60*np.pi, n_samples)
        t = np.linspace(0,14*np.pi, n_samples)
    else:
        random_state = check_random_state(0)
        p = random_state.rand(n_samples) * (2 * np.pi)
        t = random_state.rand(n_samples) * np.pi

    # Sever the poles from the sphere.
    indices = ((np.cos(t)<0.92) & (np.cos(t) > -0.92))
    color = p[indices] % (np.pi)
    x, y, z = np.sin(t[indices]) * np.cos(p[indices]), \
        np.sin(t[indices]) * np.sin(p[indices]), \
        np.cos(t[indices])
    X = np.array([x, y, z]).T

    if noise != 0:
        X = np.random.normal(X, noise)

    return X, color


def torus(n_samples, noise=0, uniform=False):

    if uniform:
        theta = np.linspace(0, 60 * np.pi, n_samples)
        phi = np.linspace(0, 26 * np.pi, n_samples)
    else:
        random_state = check_random_state(0)
        theta = random_state.rand(n_samples) * (2 * np.pi)
        phi = random_state.rand(n_samples) * (2 * np.pi)
    r1 = 2
    r2 = 0.5
    x = np.ravel((r1 + r2 * np.cos(phi)) * np.cos(theta))
    y = np.ravel((r1 + r2 * np.cos(phi)) * np.sin(theta))
    z = np.ravel(r2 * np.sin(phi))
    color = theta % (2*np.pi)
    X = np.array([x, y, z]).T

    if noise != 0:
        X = np.random.normal(X, noise)

    return X, color


def mobius_strip(n_samples, noise=0, uniform=False):

    if uniform:
        theta = np.linspace(0, 60 * np.pi, n_samples)
        w = np.linspace(-0.25, 0.25, n_samples)
    else:
        random_state = check_random_state(0)
        theta = random_state.rand(n_samples) * (2 * np.pi)
        w = random_state.rand(n_samples) * 0.5 - 0.25
    phi = 0.5 * theta
    # radius in x-y plane
    r = 1 + w * np.cos(phi)

    x = np.ravel(r * np.cos(theta))
    y = np.ravel(r * np.sin(theta))
    z = np.ravel(w * np.sin(phi))
    color = theta % (2*np.pi)
    X = np.array([x, y, z]).T

    if noise != 0:
        X = np.random.normal(X, noise)

    return X, color


def trefoil_knot(n_samples, noise=0, uniform=True):

    if uniform:
        phi = np.linspace(0,2*np.pi,n_samples)
    else:
        random_state = check_random_state(0)
        phi = random_state.rand(n_samples) * (2 * np.pi)
    x = np.sin(phi)+2*np.sin(2*phi)
    y = np.cos(phi)-2*np.cos(2*phi)
    z = -np.sin(3*phi)
    color = phi
    X = np.array([x, y, z]).T

    if noise != 0:
        X = np.random.normal(X, noise)

    return X, color


def lorenz_attractor(n_samples, noise=0, uniform=True):
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot

    dt = 0.01
    stepCnt = 10000

    # Need one more for the initial values
    xs = np.empty((stepCnt + 1,))
    ys = np.empty((stepCnt + 1,))
    zs = np.empty((stepCnt + 1,))

    # Setting initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)

    # Stepping through "time".
    for i in range(stepCnt):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    if uniform:
        indices = range(0, 10000, int(10000/n_samples))
    else:
        indices = np.random.choice(10000, n_samples, replace=False)

    color = indices
    X = np.array([xs, ys, zs]).T[indices,:]

    if noise != 0:
        X = np.random.normal(X, noise)

    return X, color


def rossler_attractor(n_samples, noise=0, uniform=True):
    def rossler(x, y, z, a=0.1, b=0.1, c=14):
        x_dot = - y - z
        y_dot = x + a*y
        z_dot = b + z*(x - c)
        return x_dot, y_dot, z_dot

    # basic parameters
    dt = 0.01 
    steps = 100000

    # initialize solutions arrays (+1 for initial conditions)
    xs = np.empty((steps + 1))
    ys = np.empty((steps + 1))
    zs = np.empty((steps + 1))

    # fill in initial conditions
    xs[0], ys[0], zs[0] = (0.1, 0., 0.1)

    # solve equation system
    for i in range(steps):
        # Calculate derivatives
        x_dot, y_dot, z_dot = rossler(xs[i], ys[i], zs[i])

        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    if uniform:
        indices = range(0, 10000, int(10000/n_samples))
    else:
        indices = np.random.choice(10000, n_samples, replace=False)

    color = indices
    X = np.array([xs, ys, zs]).T[indices,:]

    if noise != 0:
        X = np.random.normal(X, noise)

    return X, color


def thomas_attractor(n_samples, noise=0, uniform=True):
    def thomas(x, y, z, b=0.1):
        x_dot = np.sin(y) - b*x
        y_dot = np.sin(z) - b*y
        z_dot = np.sin(x) - b*z
        return x_dot, y_dot, z_dot

    # basic parameters
    dt = 0.05 
    steps = 100000

    # initialize solutions arrays (+1 for initial conditions)
    xs = np.empty((steps + 1))
    ys = np.empty((steps + 1))
    zs = np.empty((steps + 1))

    # fill in initial conditions
    xs[0], ys[0], zs[0] = (0.1, 0., 0.1)

    # solve equation system
    for i in range(steps):
        # Calculate derivatives
        x_dot, y_dot, z_dot = thomas(xs[i], ys[i], zs[i])

        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    if uniform:
        indices = range(0, 10000, int(10000/n_samples))
    else:
        indices = np.random.choice(10000, n_samples, replace=False)

    color = indices
    X = np.array([xs, ys, zs]).T[indices,:]

    if noise != 0:
        X = np.random.normal(X, noise)

    return X, color


def folded_klein_bottle(n_samples, noise=0, uniform=False):

    if uniform:
        theta = np.linspace(0, 60 * np.pi, n_samples)
        phi = np.linspace(0, 26 * np.pi, n_samples)
    else:
        random_state = check_random_state(0)
        theta = random_state.rand(n_samples) * (2 * np.pi)
        phi = random_state.rand(n_samples) * (2 * np.pi)
    r1 = 2
    r2 = 0.6
    e  = 0.2
    w = np.ravel((r1 * (np.cos(theta/2)*np.cos(phi) - np.sin(theta/2)*np.sin(2*phi))))
    x = np.ravel((r1 * (np.sin(theta/2)*np.cos(phi) + np.cos(theta/2)*np.sin(2*phi))))
    y = np.ravel(r2 * np.cos(theta) * (1 + e * np.sin(phi)))
    z = np.ravel(r2 * np.sin(theta) * (1 + e * np.sin(phi)))
    color = theta % (2*np.pi)
    X = np.array([w, x, y, z]).T

    if noise != 0:
        X = np.random.normal(X, noise)

    return X, color


def klein_bottle(n_samples, noise=0, uniform=False):

    if uniform:
        theta = np.linspace(0, 60 * np.pi, n_samples)
        phi = np.linspace(0, 26 * np.pi, n_samples)
    else:
        random_state = check_random_state(0)
        theta = random_state.rand(n_samples) * (2 * np.pi)
        phi = random_state.rand(n_samples) * (2 * np.pi)

    w = np.ravel(np.cos(theta) * (1 + np.cos(theta/2)*np.sin(phi) - np.sin(theta/2)*np.sin(2*phi)/2))
    x = np.ravel(np.sin(theta) * (1 + np.cos(theta/2)*np.sin(phi) - np.sin(theta/2)*np.sin(2*phi)/2))
    y = np.ravel(np.sin(theta/2)*np.cos(phi/2) + np.cos(theta/2)*np.cos(phi)/2)
    z = np.ravel(np.sin(theta/2)*np.cos(phi) + np.sin(theta/2)*np.cos(2*phi)/2)
    color = theta % (2*np.pi)
    X = np.array([w, x, y, z]).T

    if noise != 0:
        X = np.random.normal(X, noise)

    return X, color


def S3_klein_bottle(n_samples, noise=0, uniform=False):

    if uniform:
        theta = np.linspace(0, 60 * np.pi, n_samples)
        phi = np.linspace(0, 26 * np.pi, n_samples)
    else:
        random_state = check_random_state(0)
        theta = random_state.rand(n_samples) * (np.pi)
        phi = random_state.rand(n_samples) * (2 * np.pi)
    w = np.ravel(np.cos(theta) * np.sin(phi))
    x = np.ravel(np.sin(theta) * np.sin(phi))
    y = np.ravel(np.cos(2*theta)*np.cos(phi))
    z = np.ravel(np.sin(2*theta)*np.cos(phi))
    color = theta % (2*np.pi)
    X = np.array([w, x, y, z]).T

    if noise != 0:
        X = np.random.normal(X, noise)

    return X, color


def S3_torus(n_samples, noise=0, uniform=False):

    if uniform:
        theta = np.linspace(0, 60 * np.pi, n_samples)
        phi = np.linspace(0, 26 * np.pi, n_samples)
    else:
        random_state = check_random_state(0)
        theta = random_state.rand(n_samples) * (2 * np.pi)
        phi = random_state.rand(n_samples) * (2 * np.pi)

    w = np.ravel(np.cos(theta) * np.cos(phi))
    x = np.ravel(np.cos(theta) * np.sin(phi))
    y = np.ravel(np.sin(theta) * np.cos(phi))
    z = np.ravel(np.sin(theta) * np.sin(phi))
    color = theta % (2*np.pi)
    X = np.array([w, x, y, z]).T

    if noise != 0:
        X = np.random.normal(X, noise)

    return X, color


def Lorenz96_attractor(n_samples, noise=0, uniform=True, dim = 36, F = 8):

    def Lorenz96(x,t):
        # compute state derivatives
        d = np.zeros(dim)
        # first the 3 edge cases: i=1,2,N
        d[0] = (x[1] - x[dim-2]) * x[dim-1] - x[0]
        d[1] = (x[2] - x[dim-1]) * x[0]- x[1]
        d[dim-1] = (x[0] - x[dim-3]) * x[dim-2] - x[dim-1]
        # then the general case
        for i in range(2, dim-1):
            d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
        # add the forcing term
        d = d + F

        # return the state derivatives
        return d

    x0 = F*np.ones(dim) # initial state (equilibrium)
    x0[19] += 0.01 # add small perturbation to 20th variable
    t = np.arange(0.0, 30.0, 0.003)

    if uniform:
        indices = range(0, 10000, int(10000/n_samples))
    else:
        indices = np.random.choice(10000, n_samples, replace=False)

    color = t[indices]/30
    X = odeint(Lorenz96, x0, t)[indices]

    if noise != 0:
        X = np.random.normal(X, noise)

    return X, color

