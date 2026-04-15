# BY04 Asset Pricing Replication

Numerical replication of **Bansal & Yaron (2004), "Risks for the Long Run: A Potential Resolution of Asset Pricing Puzzles"**, *Journal of Finance* 59(4).

The project solves the Epstein–Zin long-run risks model numerically (via Tauchen discretization and fixed-point iteration) and analytically (log-linearisation), replicating the key tables and figures from the paper. All scripts are designed to be run sequentially in the provided Jupyter notebook.

---

## Repository structure

| File | Description |
|---|---|
| `0_utils.py` | Path configuration, plot style (Helvetica Neue, blue/green palette), and I/O helpers |
| `1_tauchen.py` | Tauchen (1986) discretization of the expected growth process onto a 50-point grid |
| `2_ez_solution.py` | Epstein–Zin fixed-point solution and CRRA benchmark (Q6) |
| `3_moments.py` | Asset pricing moments, approximation quality, and risk-neutral probabilities (Q8–Q10) |
| `4_stress_tests.py` | Analytical sensitivity analysis over persistence, IES, leverage, and risk aversion (Q11–Q16) |
| `5_caseII.py` | Stochastic volatility extension (Case II), Table IV replication, pricing kernel decomposition, time-varying risk premia (Q17–Q19) |
| `BY04_replication.ipynb` | Jupyter notebook — runs all scripts in order with explanatory text |

---

## Setup

### Requirements

```
numpy
scipy
matplotlib
jupyter
```

Install with:

```bash
pip install numpy scipy matplotlib jupyter
```

### Running the notebook

1. Open `BY04_replication.ipynb` in Jupyter.
2. Run **Step 1** first — you will be prompted to enter your project directory path (e.g. `/Users/you/BY04_APT`). An `output/` subfolder is created automatically for all saved files.
3. Run the remaining steps in order. Each script loads `0_utils.py` automatically on first use, so the directory prompt only appears once per session.

---

## Output

All pickle files and figures are saved to `<your_project_dir>/output/`:

| File | Contents |
|---|---|
| `tauchen_data.pkl` | Grid points, transition matrix, stationary distribution |
| `ez_solution.pkl` | Wealth–consumption ratios for EZ and CRRA, consumption growth |
| `numerical_results.pkl` | State-price matrix, equity moments, risk-neutral distribution |
| `fig_wc_ratio.png/pdf` | Wealth–consumption ratio: EZ vs CRRA |
| `fig_pc_approx.png/pdf` | Log P/C ratio: numerical vs analytical approximation |
| `fig_pi_vs_piQ.png/pdf` | Physical vs risk-neutral stationary distribution |
| `fig_stress_persistence.png/pdf` | Equity premium and risk-free rate vs persistence ρ |
| `fig_stress_ies.png/pdf` | Equity premium and risk-free rate vs IES ψ |
| `fig_conditional_ep.png/pdf` | Conditional equity premium vs volatility state σ²_t |

---

## Reference

Bansal, R. and Yaron, A. (2004). Risks for the Long Run: A Potential Resolution of Asset Pricing Puzzles. *Journal of Finance*, 59(4), 1481–1509.
