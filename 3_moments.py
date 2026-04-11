"""
Asset pricing moments, approximation quality, and risk-neutral probabilities.
Requires tauchen_data.pkl and ez_solution.pkl from earlier scripts.
"""
import sys
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# Load 0_utils.py once per session (input() only fires on first run)
if '_by04_utils' not in sys.modules:
    _spec = importlib.util.spec_from_file_location('_by04_utils', '0_utils.py')
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules['_by04_utils'] = _mod
    _spec.loader.exec_module(_mod)
_u = sys.modules['_by04_utils']
save_pkl, load_pkl, save_fig = _u.save_pkl, _u.load_pkl, _u.save_fig

# Load data
td = load_pkl('tauchen_data.pkl')
x, P, pi, K = td['x'], td['P'], td['pi'], td['K']

ez = load_pkl('ez_solution.pkl')
w_ez, w_crra, gc, theta = ez['w_ez'], ez['w_crra'], ez['gc'], ez['theta']

# Parameters
delta = 0.998; gamma = 10.0; psi = 1.5
mu = 0.0015; sigma = 0.0078; phi = 3.0; phi_d = 4.5
kappa1 = 0.997; rho = 0.979; phi_e = 0.044

# ══════════════════════════════════════════════════════════
# ASSET PRICING MOMENTS
# ══════════════════════════════════════════════════════════

print("ASSET PRICING MOMENTS")


# (a) State-price matrix Psi_{ij} = P_{ij} * Lambda_{ij}
# Lambda_{ij} = delta^theta * gc(j)^{-theta/psi} * R^w_{ij}^{theta-1}
# R^w_{ij} = gc(j) * (1 + w_j) / w_i

Psi = np.zeros((K, K))
Lambda = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        Rw_ij = gc[j] * (1.0 + w_ez[j]) / w_ez[i]
        Lambda[i, j] = delta**theta * gc[j]**(-theta/psi) * Rw_ij**(theta - 1.0)
        Psi[i, j] = P[i, j] * Lambda[i, j]

# Risk-free rate: R^f_i = 1 / sum_j Psi_{ij}
Rf = 1.0 / Psi.sum(axis=1)
rf = np.log(Rf)  # log risk-free rate

# Unconditional mean risk-free rate
E_rf = np.dot(pi, rf)
print(f"E[r_f] monthly = {E_rf:.6f}")
print(f"E[r_f] annual  = {E_rf*12*100:.2f}%")
print(f"E[R_f] annual  = {(np.dot(pi, Rf)**12 - 1)*100:.2f}%")

# (b) Levered equity claim
mu_d = mu  # mu_d = mu = 0.0015
gd_i = np.exp(mu_d + phi * x + 0.5 * phi_d**2 * sigma**2)  # E[D_{t+1}/D_t | x_t = x_i]

Psi_d = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        Psi_d[i, j] = Psi[i, j] * gd_i[i]

# Check spectral radius
sr_d = np.max(np.abs(np.linalg.eigvals(Psi_d)))
print(f"\nSpectral radius of Psi^d = {sr_d:.6f} (< 1? {sr_d < 1})")

# Solve (I - Psi^d) w^d = Psi^d * 1
w_d = np.linalg.solve(np.eye(K) - Psi_d, Psi_d @ np.ones(K))
print(f"w^d (P/D ratio): min = {w_d.min():.2f}, max = {w_d.max():.2f}")
print(f"E[w^d] = {np.dot(pi, w_d):.2f}")

# (c) Equity return and equity premium
E_rm = np.zeros(K)
for i in range(K):
    E_log_gd = mu_d + phi * x[i]  # E[log(D_{t+1}/D_t) | x_i]
    E_log_pd = np.dot(P[i, :], np.log(1.0 + w_d)) - np.log(w_d[i])
    E_rm[i] = E_log_gd + E_log_pd

# Equity premium in each state
ep_i = E_rm - rf

# Unconditional
E_ep = np.dot(pi, ep_i)
E_rf_val = np.dot(pi, rf)

print(f"\n--- Unconditional moments (annualized) ---")
print(f"E[r_m - r_f] = {E_ep*12*100:.2f}%")
print(f"E[r_f]       = {E_rf_val*12*100:.2f}%")
print(f"\nAnalytical: E[r_m - r_f] = 3.93%, E[r_f] = 2.64%")

# Volatility of market return
log_1pw = np.log(1.0 + w_d)
cond_var_rm = np.zeros(K)
for i in range(K):
    E_log1pw = np.dot(P[i, :], log_1pw)
    E_log1pw2 = np.dot(P[i, :], log_1pw**2)
    cond_var_rm[i] = phi_d**2 * sigma**2 + (E_log1pw2 - E_log1pw**2)

E_cond_var = np.dot(pi, cond_var_rm)
var_E_rm = np.dot(pi, E_rm**2) - np.dot(pi, E_rm)**2
var_rm_total = E_cond_var + var_E_rm
sigma_rm = np.sqrt(var_rm_total * 12) * 100
print(f"sigma(R_m)   = {sigma_rm:.2f}%")

# Volatility of risk-free rate
var_rf = np.dot(pi, rf**2) - np.dot(pi, rf)**2
sigma_rf = np.sqrt(var_rf * 12) * 100
print(f"sigma(R_f)   = {sigma_rf:.2f}%")

# Volatility of log price-dividend ratio
log_pd = np.log(w_d)
var_pd = np.dot(pi, log_pd**2) - np.dot(pi, log_pd)**2
sigma_pd = np.sqrt(var_pd)
print(f"sigma(p-d)   = {sigma_pd:.4f}")



print("APPROXIMATION QUALITY")


ep_analytical = 3.93  # percent annual
rf_analytical = 2.64  # percent annual

ep_numerical = E_ep * 12 * 100
rf_numerical = E_rf_val * 12 * 100

print(f"Equity premium: analytical = {ep_analytical:.2f}%, numerical = {ep_numerical:.2f}%")
print(f"  Discrepancy = {(ep_analytical - ep_numerical)/ep_numerical*100:.1f}%")
print(f"Risk-free rate: analytical = {rf_analytical:.2f}%, numerical = {rf_numerical:.2f}%")
print(f"  Discrepancy = {(rf_analytical - rf_numerical)/rf_numerical*100:.1f}%")

# Plot log P/C: numerical vs analytical
A0_numerical = np.dot(pi, np.log(w_ez))
A1 = (1.0 - 1.0/psi) / (1.0 - kappa1 * rho)
log_pc_analytical = A0_numerical + A1 * x
log_pc_numerical = np.log(w_ez)

fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
ax.plot(x * 100, log_pc_numerical,  '-',  color=_u.BLUE,     linewidth=1.8, label='Numerical (exact)')
ax.plot(x * 100, log_pc_analytical, '--', color=_u.DARKGREEN, linewidth=1.8, label='Analytical ($A_0 + A_1 x_i$)')
ax.set_xlabel('$x_i$ (% per month)')
ax.set_ylabel('$\\log(P_i/C_i)$')
ax.legend(frameon=False)
fig.tight_layout()
save_fig(fig, 'fig_pc_approx')

# ══════════════════════════════════════════════════════════
# RISK-NEUTRAL PROBABILITIES
# ══════════════════════════════════════════════════════════

print("RISK-NEUTRAL PROBABILITIES")


# Q_{ij} = Psi_{ij} * R^f_i
Q = np.zeros((K, K))
for i in range(K):
    Q[i, :] = Psi[i, :] * Rf[i]

print(f"Max |Q row sum - 1| = {np.max(np.abs(Q.sum(axis=1) - 1.0)):.2e}")

# Stationary distribution of Q
eigenvalues_Q, eigenvectors_Q = eig(Q.T)
idx_Q = np.argmin(np.abs(eigenvalues_Q - 1.0))
pi_Q = np.real(eigenvectors_Q[:, idx_Q])
pi_Q = pi_Q / pi_Q.sum()

print(f"pi^Q: sum = {pi_Q.sum():.10f}, min = {pi_Q.min():.2e}")
print(f"E^Q[x] = {np.dot(pi_Q, x)*100:.4f}% (vs E^P[x] = {np.dot(pi, x)*100:.4f}%)")

# Plot pi vs pi^Q
fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
ax.plot(x * 100, pi,   '-',  color=_u.BLUE,     linewidth=1.8, label='Physical ($\\pi$)')
ax.plot(x * 100, pi_Q, '--', color=_u.DARKGREEN, linewidth=1.8, label='Risk-neutral ($\\pi^Q$)')
ax.set_xlabel('$x_i$ (% per month)')
ax.set_ylabel('Stationary probability')
ax.legend(frameon=False)
fig.tight_layout()
save_fig(fig, 'fig_pi_vs_piQ')

# Save all results
save_pkl({
    'E_ep': E_ep, 'E_rf': E_rf_val, 'sigma_rm': sigma_rm,
    'sigma_rf': sigma_rf, 'sigma_pd': sigma_pd,
    'w_d': w_d, 'Rf': Rf, 'rf': rf, 'Psi': Psi, 'Q': Q,
    'pi_Q': pi_Q, 'Lambda': Lambda
}, 'numerical_results.pkl')
