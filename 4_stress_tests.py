
import sys
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Load 0_utils.py once per session (input() only fires on first run)
if '_by04_utils' not in sys.modules:
    _spec = importlib.util.spec_from_file_location('_by04_utils', '0_utils.py')
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules['_by04_utils'] = _mod
    _spec.loader.exec_module(_mod)
_u = sys.modules['_by04_utils']
save_fig = _u.save_fig

# ═══════════════════════════════════════════════════════════
# Baseline parameters
# ═══════════════════════════════════════════════════════════
delta = 0.998; gamma_base = 10.0; psi_base = 1.5
mu = 0.0015; sigma = 0.0078; rho_base = 0.979
phi_e_base = 0.044; phi_base = 3.0; phi_d = 4.5
kappa1 = 0.997
sigma2 = sigma**2

def compute_moments(gamma, psi, rho, phi_e, phi, delta=0.998, mu=0.0015,
                     sigma=0.0078, phi_d=4.5, kappa1=0.997):
    """Compute annualized EP, Rf, sigma(Rm) from analytical formulas."""
    sigma2 = sigma**2
    theta = (1 - gamma) / (1 - 1/psi)
    A1 = (1 - 1/psi) / (1 - kappa1 * rho)
    A1m = (phi - 1/psi) / (1 - kappa1 * rho)

    B = kappa1 * A1 * phi_e
    lam_eta = -gamma
    lam_e = (1 - theta) * B
    beta_me = kappa1 * A1m * phi_e

    # Conditional variance of rm
    var_rm = (beta_me**2 + phi_d**2) * sigma2

    # Log equity premium
    risk_comp = beta_me * lam_e * sigma2
    jensen = -0.5 * var_rm
    ep_monthly = risk_comp + jensen
    ep_annual = ep_monthly * 12 * 100

    # Risk-free rate
    epra = gamma * sigma2 + (1 - theta) * B**2 * sigma2 - 0.5 * (1 + B**2) * sigma2
    var_m = (lam_eta**2 + lam_e**2) * sigma2
    rf_monthly = -np.log(delta) + (1/psi) * mu + ((1 - theta)/theta) * epra - (1/(2*theta)) * var_m
    rf_annual = rf_monthly * 12 * 100

    # Unconditional sigma(Rm)
    sigma2_x = (phi_e * sigma)**2 / (1 - rho**2) if abs(rho) < 1 else 0
    var_rm_uncond = sigma2_x / psi**2 + (beta_me**2 + phi_d**2) * sigma2
    sigma_rm = np.sqrt(var_rm_uncond * 12) * 100

    return {'ep': ep_annual, 'rf': rf_annual, 'sigma_rm': sigma_rm,
            'A1': A1, 'A1m': A1m, 'theta': theta, 'risk_comp': risk_comp * 12 * 100}


# ═══════════════════════════════════════════════════════════
# Q11: Persistence and the equity premium
# ═══════════════════════════════════════════════════════════
print("PERSISTENCE AND THE EQUITY PREMIUM")

rho_vals = [0.50, 0.80, 0.90, 0.95, 0.979]
print(f"{'rho':>6s} {'EP (%)':>8s} {'Rf (%)':>8s} {'sigma_x':>10s}")
print("-" * 35)
ep_rho = []
rf_rho = []
for r in rho_vals:
    res = compute_moments(gamma_base, psi_base, r, phi_e_base, phi_base)
    ep_rho.append(res['ep'])
    rf_rho.append(res['rf'])
    sigma_x = phi_e_base * sigma / np.sqrt(1 - r**2) if abs(r) < 1 else float('inf')
    print(f"{r:6.3f} {res['ep']:8.2f} {res['rf']:8.2f} {sigma_x:10.6f}")

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax2 = ax1.twinx()
ax1.plot(rho_vals, ep_rho, 'o-',  color='#E07B39',   linewidth=1.8, markersize=5, label='$E[r_m - r_f]$')
ax2.plot(rho_vals, rf_rho, 's--', color=_u.DARKGREEN, linewidth=1.8, markersize=5, label='$E[r_f]$')
ax1.set_xlabel('$\\rho$')
ax1.set_ylabel('Equity premium (\\%)')
ax2.set_ylabel('Risk-free rate (\\%)')
ax2.spines['right'].set_visible(True)
ax1.grid(False)
ax2.grid(True)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=True, fancybox=False, edgecolor='#cccccc')
fig.tight_layout()
save_fig(fig, 'fig_stress_persistence')


# ═══════════════════════════════════════════════════════════
# Identification problem (i.i.d. case)
# ═══════════════════════════════════════════════════════════
print("IDENTIFICATION PROBLEM (i.i.d. CASE)")

res_baseline = compute_moments(gamma_base, psi_base, rho_base, phi_e_base, phi_base)
res_iid = compute_moments(gamma_base, psi_base, rho_base, 0.0, phi_base)

print(f"Baseline EP = {res_baseline['ep']:.2f}%")
print(f"i.i.d. EP (phi_e=0) = {res_iid['ep']:.2f}%")
print(f"Difference = {res_baseline['ep'] - res_iid['ep']:.2f}%")
print(f"As fraction of baseline = {(res_baseline['ep'] - res_iid['ep'])/res_baseline['ep']*100:.1f}%")
print(f"\ni.i.d. Rf = {res_iid['rf']:.2f}%")


# ═══════════════════════════════════════════════════════════
# IES sensitivity
# ═══════════════════════════════════════════════════════════
print("IES SENSITIVITY")

psi_vals = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
print(f"{'psi':>6s} {'EP (%)':>8s} {'Rf (%)':>8s} {'A1':>8s}")
print("-" * 35)
ep_psi = []
rf_psi = []
for p in psi_vals:
    if abs(p - 1.0) < 1e-10:
        print(f"{p:6.2f}     ---      ---      ---  (singularity)")
        ep_psi.append(np.nan)
        rf_psi.append(np.nan)
        continue
    res = compute_moments(gamma_base, p, rho_base, phi_e_base, phi_base)
    print(f"{p:6.2f} {res['ep']:8.2f} {res['rf']:8.2f} {res['A1']:8.2f}")
    ep_psi.append(res['ep'])
    rf_psi.append(res['rf'])

psi_plot = [p for p, e in zip(psi_vals, ep_psi) if not np.isnan(e)]
ep_plot  = [e for e in ep_psi if not np.isnan(e)]
rf_plot  = [r for r in rf_psi if not np.isnan(r)]

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax2 = ax1.twinx()
ax1.plot(psi_plot, ep_plot, 'o-',  color='#E07B39',   linewidth=1.8, markersize=5, label='$E[r_m - r_f]$')
ax2.plot(psi_plot, rf_plot, 's--', color=_u.DARKGREEN, linewidth=1.8, markersize=5, label='$E[r_f]$')
ax1.axvline(x=1.0, color='#aaaaaa', linestyle=':', linewidth=1.2, label='$\\psi = 1$')
ax1.set_xlabel('$\\psi$ (IES)')
ax1.set_ylabel('Equity premium (\\%)')
ax2.set_ylabel('Risk-free rate (\\%)')
ax2.spines['right'].set_visible(True)
ax1.grid(False)
ax2.grid(True)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=True, fancybox=False, edgecolor='#cccccc')
fig.tight_layout()
save_fig(fig, 'fig_stress_ies')


# ═══════════════════════════════════════════════════════════
# Leverage parameter
# ═══════════════════════════════════════════════════════════
print("LEVERAGE PARAMETER")

phi_vals = [1.0, 2.0, 3.0, 3.5, 4.0]
print(f"{'phi':>6s} {'EP (%)':>8s}")
print("-" * 18)
ep_phi = []
for ph in phi_vals:
    res = compute_moments(gamma_base, psi_base, rho_base, phi_e_base, ph)
    ep_phi.append(res['ep'])
    print(f"{ph:6.1f} {res['ep']:8.2f}")

ep_phi3 = ep_phi[phi_vals.index(3.0)]
ep_phi1 = ep_phi[phi_vals.index(1.0)]
print(f"\nEP at phi=3: {ep_phi3:.2f}%")
print(f"EP at phi=1: {ep_phi1:.2f}%")
print(f"Fraction lost: {(ep_phi3 - ep_phi1)/ep_phi3*100:.1f}%")


# ═══════════════════════════════════════════════════════════
# Risk aversion
# ═══════════════════════════════════════════════════════════
print("\RISK AVERSION")

gamma_vals = [5, 7.5, 10, 15, 20]
print(f"{'gamma':>6s} {'EP (%)':>8s} {'Rf (%)':>8s}")
print("-" * 25)
for g in gamma_vals:
    res = compute_moments(g, psi_base, rho_base, phi_e_base, phi_base)
    print(f"{g:6.1f} {res['ep']:8.2f} {res['rf']:8.2f}")

def ep_minus_target(g):
    return compute_moments(g, psi_base, rho_base, phi_e_base, phi_base)['ep'] - 6.33
gamma_star = brentq(ep_minus_target, 5, 30)
print(f"\nGamma that matches 6.33% EP: {gamma_star:.2f}")
print(f"Within Mehra-Prescott bound (gamma <= 10)? {gamma_star <= 10}")

res_ez   = compute_moments(gamma_base, psi_base,         rho_base, phi_e_base, phi_base)
res_crra = compute_moments(gamma_base, 1.0/gamma_base,   rho_base, phi_e_base, phi_base)
print(f"\nEZ (psi=1.5):       EP = {res_ez['ep']:.2f}%")
print(f"CRRA (psi=1/gamma): EP = {res_crra['ep']:.2f}%")
print(f"EP attributable to separation: {res_ez['ep'] - res_crra['ep']:.2f}%")
print(f"As fraction of EZ EP: {(res_ez['ep'] - res_crra['ep'])/res_ez['ep']*100:.1f}%")


# ═══════════════════════════════════════════════════════════
# Joint fragility
# ═══════════════════════════════════════════════════════════
print("JOINT FRAGILITY")

configs = [
    ("Baseline",                        gamma_base, psi_base, rho_base, phi_base),
    ("$(\\rho,\\psi) = (0.95, 1.0)$",  gamma_base, 1.001,   0.95,     phi_base),
    ("$(\\rho,\\phi) = (0.95, 2.0)$",  gamma_base, psi_base, 0.95,    2.0),
    ("$(\\gamma,\\psi) = (7.5, 0.75)$", 7.5,       0.75,    rho_base, phi_base),
]

print(f"{'Config':>35s} {'EP (%)':>8s} {'Rf (%)':>8s} {'sigma(Rm)':>10s}")
print("-" * 68)
for name, g, p, r, ph in configs:
    res = compute_moments(g, p, r, phi_e_base, ph)
    print(f"{name:>35s} {res['ep']:8.2f} {res['rf']:8.2f} {res['sigma_rm']:10.2f}")

res_below = compute_moments(gamma_base, 0.999, 0.95, phi_e_base, phi_base)
print(f"\n  Note: psi=1.0 is a singularity. Using psi=1.001 as proxy.")
print(f"  With psi=0.999: EP = {res_below['ep']:.2f}%, Rf = {res_below['rf']:.2f}%")

print("\nDone. All stress test figures saved.")
