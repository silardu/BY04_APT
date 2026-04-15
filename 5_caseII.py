"""
Stochastic Volatility (Case II) — Q17-Q19
"""
import sys
import importlib.util
import numpy as np
from scipy.stats import norm as norm_dist
import matplotlib.pyplot as plt

# Load 0_utils.py once per session (input() only fires on first run)
if '_by04_utils' not in sys.modules:
    _spec = importlib.util.spec_from_file_location('_by04_utils', '0_utils.py')
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules['_by04_utils'] = _mod
    _spec.loader.exec_module(_mod)
_u = sys.modules['_by04_utils']
save_fig = _u.save_fig

# ═══════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════
delta = 0.998; gamma = 10.0; psi = 1.5
mu = 0.0015; mu_d = 0.0015; sigma = 0.0078; rho = 0.979
phi_e = 0.044; phi = 3.0; phi_d = 4.5
kappa1 = 0.997; kappa1m = kappa1
nu1 = 0.987; sigma_w = 0.23e-5

sigma2 = sigma**2
theta = (1 - gamma) / (1 - 1/psi)

# ═══════════════════════════════════════════════════════════
# Case II coefficients and Table IV replication
# ═══════════════════════════════════════════════════════════
print("CASE II COEFFICIENTS AND TABLE IV")

# A1 (unchanged from Case I)
A1 = (1 - 1/psi) / (1 - kappa1 * rho)

# A2
num_A2 = 0.5 * ((theta - theta/psi)**2 + (theta * A1 * kappa1 * phi_e)**2)
A2 = num_A2 / (theta * (1 - kappa1 * nu1))

# Market prices of risk
lam_eta = -gamma
B = kappa1 * A1 * phi_e
lam_e = (1 - theta) * B
lam_w = (1 - theta) * A2 * kappa1

print(f"theta = {theta:.2f}")
print(f"A1 = {A1:.4f}")
print(f"A2 = {A2:.4f}")
print(f"lambda_{{m,eta}} = {lam_eta:.4f}")
print(f"lambda_{{m,e}} = {lam_e:.4f}")
print(f"lambda_{{m,w}} = {lam_w:.4f}")

# Dividend claim coefficients
A1m = (phi - 1/psi) / (1 - kappa1m * rho)
beta_me = kappa1m * A1m * phi_e

# A2m from eq (A20)
Hm = lam_eta**2 + (-lam_e + beta_me)**2 + phi_d**2
A2m = ((1 - theta) * A2 * (1 - kappa1 * nu1) + 0.5 * Hm) / (1 - kappa1m * nu1)
beta_mw = kappa1m * A2m

print(f"\nA1m = {A1m:.4f}")
print(f"A2m = {A2m:.4f}")
print(f"beta_{{m,e}} = {beta_me:.6f}")
print(f"beta_{{m,w}} = {beta_mw:.6f}")

# Conditional EP at sigma^2_t = sigma^2
var_rm = (beta_me**2 + phi_d**2) * sigma2 + beta_mw**2 * sigma_w**2
risk_e = beta_me * lam_e * sigma2
risk_w = beta_mw * lam_w * sigma_w**2
jensen = -0.5 * var_rm
ep_monthly = risk_e + risk_w + jensen
ep_annual = ep_monthly * 12 * 100

print(f"\n--- Conditional EP at sigma^2_t = sigma^2 ---")
print(f"Risk comp (e): {risk_e*12*100:.4f}%")
print(f"Risk comp (w): {risk_w*12*100:.4f}%")
print(f"Jensen:        {jensen*12*100:.4f}%")
print(f"EP annual:     {ep_annual:.2f}%")

# Risk-free rate
var_ra = (1 + B**2) * sigma2 + (A2 * kappa1)**2 * sigma_w**2
epra = (gamma * sigma2 + (1 - theta) * B**2 * sigma2
        + kappa1 * A2 * lam_w * sigma_w**2 - 0.5 * var_ra)
var_m = (lam_eta**2 + lam_e**2) * sigma2 + lam_w**2 * sigma_w**2
rf_monthly = (-np.log(delta) + (1/psi) * mu
              + ((1 - theta)/theta) * epra - (1/(2*theta)) * var_m)
rf_annual = rf_monthly * 12 * 100

# Unconditional sigma(Rm)
sigma2_x = (phi_e * sigma)**2 / (1 - rho**2)
var_sigma2 = sigma_w**2 / (1 - nu1**2)
var_rm_uncond = (sigma2_x / psi**2
                + (beta_me**2 + phi_d**2) * sigma2
                + (A2m * (nu1 * kappa1m - 1))**2 * var_sigma2
                + beta_mw**2 * sigma_w**2)
sigma_rm = np.sqrt(var_rm_uncond * 12) * 100

# sigma(Rf)
Q2_val = lam_eta**2 + lam_e**2
Q1_val = -lam_eta + (1 - theta) * B**2 - 0.5 * (1 + B**2)
coeff_sigma2_rf = ((1 - theta)/theta * Q1_val - Q2_val/(2*theta))
var_rf = (1/psi)**2 * sigma2_x + coeff_sigma2_rf**2 * var_sigma2
sigma_rf = np.sqrt(var_rf * 12) * 100

# sigma(p-d)
var_zm = A1m**2 * sigma2_x + A2m**2 * var_sigma2
sigma_pd = np.sqrt(var_zm)

print(f"\n--- Table IV (gamma=10) ---")
print(f"E[rm - rf]   = {ep_annual:.2f}%   (Paper: 6.84%)")
print(f"E[rf]        = {rf_annual:.2f}%   (Paper: 0.93%)")
print(f"sigma(Rm)    = {sigma_rm:.2f}%   (Paper: 18.65%)")
print(f"sigma(Rf)    = {sigma_rf:.2f}%   (Paper: 0.57%)")
print(f"sigma(p-d)   = {sigma_pd:.4f}    (Paper: 0.21)")


# ═══════════════════════════════════════════════════════════
# Pricing kernel decomposition
# ═══════════════════════════════════════════════════════════
print("\nQ18: PRICING KERNEL DECOMPOSITION")

var_eta = lam_eta**2 * sigma2
var_e = lam_e**2 * sigma2
var_w_comp = lam_w**2 * sigma_w**2
var_total = var_eta + var_e + var_w_comp

share_eta = var_eta / var_total * 100
share_e = var_e / var_total * 100
share_w = var_w_comp / var_total * 100

max_sr = np.sqrt(12) * np.sqrt(var_total)

print(f"Variance shares:")
print(f"  eta (consumption):     {share_eta:.1f}%  (Paper: 14%)")
print(f"  e   (expected growth): {share_e:.1f}%  (Paper: 47%)")
print(f"  w   (volatility):      {share_w:.1f}%  (Paper: 39%)")
print(f"Annualized max Sharpe ratio: {max_sr:.2f}  (Paper: 0.73)")

# Sensitivity to rho = 0.95
print(f"\n--- Sensitivity: rho = 0.95 ---")
rho95 = 0.95
A1_95 = (1 - 1/psi) / (1 - kappa1 * rho95)
B_95 = kappa1 * A1_95 * phi_e
lam_e_95 = (1 - theta) * B_95
num_A2_95 = 0.5 * ((theta - theta/psi)**2 + (theta * A1_95 * kappa1 * phi_e)**2)
A2_95 = num_A2_95 / (theta * (1 - kappa1 * nu1))
lam_w_95 = (1 - theta) * A2_95 * kappa1

var_eta_95 = lam_eta**2 * sigma2
var_e_95 = lam_e_95**2 * sigma2
var_w_95 = lam_w_95**2 * sigma_w**2
var_tot_95 = var_eta_95 + var_e_95 + var_w_95

print(f"  eta: {var_eta_95/var_tot_95*100:.1f}%")
print(f"  e:   {var_e_95/var_tot_95*100:.1f}%")
print(f"  w:   {var_w_95/var_tot_95*100:.1f}%")
print(f"  Max SR: {np.sqrt(12)*np.sqrt(var_tot_95):.2f}")


# ═══════════════════════════════════════════════════════════
# Time-varying risk premia
# ═══════════════════════════════════════════════════════════
print("\nTIME-VARYING RISK PREMIA")

# (a) Conditional EP vs sigma^2_t
sigma2_range = np.linspace(0.5 * sigma2, 1.5 * sigma2, 100)
ep_cond = np.zeros(len(sigma2_range))
for k, s2 in enumerate(sigma2_range):
    var_rm_k = (beta_me**2 + phi_d**2) * s2 + beta_mw**2 * sigma_w**2
    ep_cond[k] = (beta_me * lam_e * s2 + beta_mw * lam_w * sigma_w**2
                  - 0.5 * var_rm_k) * 12 * 100

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(sigma2_range / sigma2, ep_cond, '-', color=_u.BLUE, linewidth=1.8)
ax.axvline(x=1.0, color='#aaaaaa', linestyle=':', linewidth=1.2)
ax.set_xlabel('$\\sigma_t^2 / \\bar{\\sigma}^2$')
ax.set_ylabel('Conditional equity premium (\\% annual)')
fig.tight_layout()
save_fig(fig, 'fig_conditional_ep')

# (b) Volatility feedback
cov_feedback = beta_mw * (beta_me**2 + phi_d**2) * sigma_w**2
print(f"\n(b) Volatility feedback:")
print(f"Cov = beta_mw * (beta_me^2 + phi_d^2) * sigma_w^2 = {cov_feedback:.2e}")
print(f"Sign: {'negative' if cov_feedback < 0 else 'positive'}")
print(f"A2m = {A2m:.4f} < 0, so beta_mw < 0 => Cov < 0")

# (c) Pr(sigma^2_{t+1} < 0)
arg_at_mean = -((1 - nu1) * sigma2 + nu1 * sigma2) / sigma_w
prob_at_mean = norm_dist.cdf(arg_at_mean)

arg_at_zero = -((1 - nu1) * sigma2) / sigma_w
prob_at_zero = norm_dist.cdf(arg_at_zero)

print(f"\n(c) Pr(sigma^2_{{t+1}} < 0):")
print(f"At sigma^2_t = sigma^2: Pr = {prob_at_mean:.2e}")
print(f"At sigma^2_t = 0:       Pr = {prob_at_zero:.2%}")
print(f"Paper footnote 13: ~5% of simulated realizations are negative.")
print(f"Reconciliation: with nu1={nu1} near 1, sigma^2_t can drift well")
print(f"below sigma^2 along a sample path, making negative draws possible.")
