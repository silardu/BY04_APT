"""
Tauchen (1986) discretization of the expected growth process
    x_{t+1} = rho * x_t + phi_e * sigma * e_{t+1}
onto K=50 grid points spanning ±4 unconditional standard deviations.
"""
import sys
import importlib.util
import numpy as np
from scipy.stats import norm
from scipy.linalg import eig

# Load 0_utils.py once per session (input() only fires on first run)
if '_by04_utils' not in sys.modules:
    _spec = importlib.util.spec_from_file_location('_by04_utils', '0_utils.py')
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules['_by04_utils'] = _mod
    _spec.loader.exec_module(_mod)
save_pkl = sys.modules['_by04_utils'].save_pkl

# === Parameters ===
rho = 0.979
phi_e = 0.044
sigma = 0.0078
sigma_e = phi_e * sigma  # innovation SD for x process

K = 50
m = 4  # number of unconditional SDs

# === Tauchen method ===
sigma_x = sigma_e / np.sqrt(1.0 - rho**2)  # unconditional SD of x_t
x = np.linspace(-m * sigma_x, m * sigma_x, K)
dx = x[1] - x[0]

P = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        if j == 0:
            P[i, j] = norm.cdf((x[0] + dx/2 - rho * x[i]) / sigma_e)
        elif j == K - 1:
            P[i, j] = 1.0 - norm.cdf((x[K-1] - dx/2 - rho * x[i]) / sigma_e)
        else:
            P[i, j] = (norm.cdf((x[j] + dx/2 - rho * x[i]) / sigma_e)
                      - norm.cdf((x[j] - dx/2 - rho * x[i]) / sigma_e))

# Verify rows sum to 1
row_sums = P.sum(axis=1)
print(f"Grid: K = {K}, x in [{x[0]:.6f}, {x[-1]:.6f}]")
print(f"Unconditional SD of x: sigma_x = {sigma_x:.6f}")
print(f"Grid spacing dx = {dx:.6f}")
print(f"Max |row sum - 1| = {np.max(np.abs(row_sums - 1.0)):.2e}")

# Stationary distribution: left eigenvector for eigenvalue 1
eigenvalues, eigenvectors = eig(P.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi = np.real(eigenvectors[:, idx])
pi = pi / pi.sum()

print(f"Stationary dist: sum = {pi.sum():.10f}, min = {pi.min():.2e}, max = {pi.max():.4f}")
print(f"E[x] under pi = {np.dot(pi, x):.2e} (should be ~0)")
print(f"Std[x] under pi = {np.sqrt(np.dot(pi, x**2) - np.dot(pi, x)**2):.6f} (analytical: {sigma_x:.6f})")

# Save for use in subsequent scripts
save_pkl({'x': x, 'P': P, 'pi': pi, 'K': K}, 'tauchen_data.pkl')
