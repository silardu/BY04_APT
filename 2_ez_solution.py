"""
Epstein-Zin fixed-point solution and CRRA benchmark.
Requires tauchen_data.pkl produced by 1_tauchen.py.
"""
import sys
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

# Load 0_utils.py once per session (input() only fires on first run)
if '_by04_utils' not in sys.modules:
    _spec = importlib.util.spec_from_file_location('_by04_utils', '0_utils.py')
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules['_by04_utils'] = _mod
    _spec.loader.exec_module(_mod)
_u = sys.modules['_by04_utils']
save_pkl, load_pkl, save_fig = _u.save_pkl, _u.load_pkl, _u.save_fig

# Load Tauchen grid
data = load_pkl('tauchen_data.pkl')
x, P, pi, K = data['x'], data['P'], data['pi'], data['K']

# Parameters
delta = 0.998; gamma = 10.0; psi_ez = 1.5
mu = 0.0015; sigma = 0.0078; phi = 3.0; phi_d = 4.5

# g_c(j) = exp(mu + x_j)  (gross consumption growth in state j)
gc = np.exp(mu + x)



print("CRRA BENCHMARK")


# Psi^w_{ij} = P_{ij} * delta * gc(j)^{1-gamma}
Psi_w = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        Psi_w[i, j] = P[i, j] * delta * gc[j]**(1.0 - gamma)

# Spectral radius
sr = np.max(np.abs(np.linalg.eigvals(Psi_w)))
print(f"Spectral radius of Psi^w = {sr:.6f} (< 1? {sr < 1})")

# Solve (I - Psi^w) w = Psi^w * 1
ones = np.ones(K)
w_crra = np.linalg.solve(np.eye(K) - Psi_w, Psi_w @ ones)
print(f"w^CRRA: min = {w_crra.min():.4f}, max = {w_crra.max():.4f}")
print(f"E[w^CRRA] = {np.dot(pi, w_crra):.4f}")



print("EPSTEIN-ZIN SOLUTION")

theta = (1.0 - gamma) / (1.0 - 1.0/psi_ez)
print(f"theta = {theta:.2f}")

def ez_fixed_point(x, P, gamma, psi, delta, mu, K, damping=0.3, tol=1e-12, maxiter=100000):
    theta = (1.0 - gamma) / (1.0 - 1.0/psi)
    gc = np.exp(mu + x)

    # Initial guess: CRRA solution
    Psi_w_init = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            Psi_w_init[i, j] = P[i, j] * delta * gc[j]**(1.0 - gamma)
    w = np.linalg.solve(np.eye(K) - Psi_w_init, Psi_w_init @ np.ones(K))
    log_w = np.log(np.maximum(w, 1e-20))

    for it in range(maxiter):
        w_old = np.exp(log_w)

        # T(w)_i = [ sum_j P_{ij} (delta * gc(j)^{1-1/psi})^theta * (1+w_j)^theta ]^{1/theta}
        rhs = np.zeros(K)
        for i in range(K):
            s = 0.0
            for j in range(K):
                base = delta * gc[j]**(1.0 - 1.0/psi)
                s += P[i, j] * base**theta * (1.0 + w_old[j])**theta
            rhs[i] = s

        # w_new = rhs^{1/theta}
        w_new = rhs**(1.0/theta)
        log_w_new = np.log(w_new)

        # Damped update on log scale
        log_w_updated = (1.0 - damping) * log_w + damping * log_w_new

        err = np.max(np.abs(log_w_updated - log_w))
        log_w = log_w_updated

        if err < tol:
            print(f"Converged in {it+1} iterations, max|Delta log(w)| = {err:.2e}")
            return np.exp(log_w), it+1

    print(f"WARNING: Did not converge after {maxiter} iters, err = {err:.2e}")
    return np.exp(log_w), maxiter

w_ez, n_iter = ez_fixed_point(x, P, gamma, psi_ez, delta, mu, K)
print(f"w^EZ: min = {w_ez.min():.4f}, max = {w_ez.max():.4f}")
print(f"E[w^EZ] = {np.dot(pi, w_ez):.4f}")

# Verify Euler equation: sum_j P_{ij} Lambda_{ij} R^w_{ij} = 1 for all i
print("\nEuler equation verification:")
max_euler_err = 0.0
for i in range(K):
    euler_sum = 0.0
    for j in range(K):
        Rw_ij = gc[j] * (1.0 + w_ez[j]) / w_ez[i]
        Lambda_ij = delta**theta * gc[j]**(-theta/psi_ez) * Rw_ij**(theta - 1.0)
        euler_sum += P[i, j] * Lambda_ij * Rw_ij
    err = abs(euler_sum - 1.0)
    if err > max_euler_err:
        max_euler_err = err

print(f"Max Euler equation error: {max_euler_err:.2e}")

# === Plot w^EZ vs w^CRRA ===
_u = sys.modules['_by04_utils']
fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
ax.plot(x * 100, w_ez,   '-',  color=_u.BLUE,      linewidth=1.8, label='Epstein–Zin ($\\psi=1.5$, $\\gamma=10$)')
ax.plot(x * 100, w_crra, '--', color=_u.DARKGREEN,  linewidth=1.8, label='CRRA ($\\psi=1/\\gamma=0.1$, $\\gamma=10$)')
ax.set_xlabel('$x_i$ (% per month)')
ax.set_ylabel('Wealth–consumption ratio $w_i$')
ax.legend(frameon=False)
fig.tight_layout()
save_fig(fig, 'fig_wc_ratio')

# Save for subsequent scripts
save_pkl({'w_ez': w_ez, 'w_crra': w_crra, 'gc': gc, 'theta': theta}, 'ez_solution.pkl')
