# PROJECT BRIEF: Provable Risk-Measure Stability for Deep Generative Models of Financial Returns
## A Wasserstein–CVaR Framework with Extreme Value Theory

---

## 1. WHAT THIS PROJECT IS (ONE PARAGRAPH)

This is a research paper targeting arXiv (categories: q-fin.RM, q-fin.CP, cross-listed stat.ML).
The central question: **if I train a generative AI model on historical financial return data and
use it to simulate fake-but-realistic future returns, how wrong can my risk estimates (specifically
CVaR / Expected Shortfall) be — and can I prove an explicit, computable bound on that error?**
The answer depends critically on how "wild" the asset's tail behavior is, which we measure using
Extreme Value Theory. Nobody has assembled these three specific pieces (generative model quality
→ Wasserstein distance → EVaR stability → EVT tail rate) into a single end-to-end bound and
empirically validated it. That's the paper.

---

## 2. THE RESEARCH CLAIM (ONE SENTENCE)

> For a trained normalizing flow model, the observable Wasserstein distance between generated
> and real return distributions, composed with a CVaR–Hölder stability bound, yields an
> explicit, computable upper bound on CVaR estimation error whose convergence rate degrades
> provably and measurably at a critical tail-index threshold (α = 2), confirmed empirically
> across assets with distinct tail behavior.

---

## 3. THE MATH FRAMEWORK (READ THIS CAREFULLY)

### 3.1 Notation

| Symbol | Meaning |
|--------|---------|
| L | Daily loss random variable (loss = negative return, so big positive = bad day) |
| P | True (unknown) distribution of L |
| Q_θ | Distribution implied by our trained normalizing flow |
| P̂_n | Empirical distribution from n real historical observations |
| Q̂_m | Empirical distribution from m flow-generated samples |
| q | CVaR confidence level (e.g. q=0.05 means 95% CVaR) |
| W_1(·,·) | Wasserstein-1 distance between two distributions |
| α | Tail index of the true return distribution (higher = thinner tail) |
| ξ | EVT shape parameter = 1/α |
| γ(α) | EVT-driven convergence rate exponent |
| ĥ | Hill estimator of α from data |

### 3.2 What VaR and CVaR Actually Are

**VaR_q(P):** The q-quantile of the loss distribution.
    VaR_q(P) = inf{ t : P(L > t) ≤ q }
    Plain English: "With probability (1-q), I lose less than this amount."

**CVaR_q(P):** The expected loss *given* that you're already past the VaR threshold.
    CVaR_q(P) = (1/q) * E_P[L · 1{L ≥ VaR_q(P)}]
    Equivalently (Rockafellar–Uryasev 2000):
    CVaR_q(P) = min_{t ∈ ℝ} { t + (1/q) * E_P[(L - t)^+] }
    Plain English: "Given I'm having one of my q-fraction worst days, this is the average loss."

CVaR is "coherent" (Artzner et al. 1999) - it respects diversification. VaR does not.

### 3.3 Wasserstein Distance (The "Gap Meter")

W_1(P, Q) = inf_{γ ∈ Γ(P,Q)} E_{(X,Y)~γ}[|X - Y|]

where Γ(P,Q) is the set of all joint distributions (couplings) with marginals P and Q.

In 1D (our case, since daily returns are scalars):
    W_1(P, Q) = ∫ |F_P(x) - F_Q(x)| dx
where F_P, F_Q are CDFs. This is the AREA between the two CDF curves.
Computationally: sort both samples, compute empirical CDFs, integrate the gap.

Key property: stays finite even when distributions don't fully overlap.
This makes it better than KL divergence for comparing a trained model to real data
(early in training, model may never generate values seen in real data).

### 3.4 Lemma 1 — CVaR Stability Under W_1 (CLASSICAL, IMPORTED)

SOURCE: Pflug (2000), Rockafellar–Uryasev (2000), Pichler robustness papers.
DO NOT CLAIM THIS AS YOUR OWN. Cite these explicitly.

|CVaR_q(P) - CVaR_q(Q_θ)| ≤ (1/q) · W_1(P, Q_θ)

PROOF SKETCH (import and cite, don't re-derive):
- The Rockafellar–Uryasev representation writes CVaR as min over t of an expectation.
- For fixed t, φ_t(L) = t + (1/q)(L-t)^+ is (1/q)-Lipschitz in L.
- By Kantorovich–Rubinstein duality: |E_P[φ_t] - E_Q[φ_t]| ≤ (1/q) · W_1(P,Q).
- Taking min_t on both sides (using |min f - min g| ≤ sup|f - g|) completes the proof.

WHAT THIS SAYS: model bias directly enters CVaR error with coefficient 1/q.
Better model (smaller W_1) → smaller CVaR error, linearly.

### 3.5 Lemma 2 — EVT-Driven Estimation Rate (YOUR KEY CONTRIBUTION)

The statistical noise floor when estimating W_1 from finite samples of size n
depends on the tail index α through the following convergence rate:

    γ(α) = min(1/2, (α-1)/α)

So the empirical error:
    E[W_1(P̂_n, P)] ~ C · n^{-γ(α)}

The CRITICAL THRESHOLD: at α = 2
    - α > 2 (finite variance): γ = 1/2 → standard root-n convergence
    - α ≤ 2 (infinite variance): γ = (α-1)/α < 1/2 → slower than root-n

INTUITION: When tails are wild enough that variance is infinite, you simply cannot
pin down the extreme tail precisely with finite data, regardless of model quality.
More data still helps, but proportionally less per additional observation.

SOURCE TO CITE: Brazauskas et al. on heavy-tailed CVaR estimation rates,
Fournier–Guillin (2015) on empirical Wasserstein convergence rates.
Hill (1975) for the estimator. Track these down in Month 1 reading.

### 3.6 THE COMPOSED THEOREM (YOUR ORIGINAL CONTRIBUTION)

Chaining Lemma 1 and Lemma 2:

    |CVaR_q(P̂_n) - CVaR_q(Q̂_m)|
        ≤  (1/q) · Ŵ_1(P̂_n, Q̂_m)          [model bias term — you measure this]
         + C_1 · n^{-γ(α̂)}                    [real-data finite-sample noise]
         + C_2 · m^{-γ(α̂)}                    [flow-sample finite-sample noise]

where α̂ is your Hill-estimated tail index.

ALL TERMS ARE COMPUTABLE FROM YOUR EXPERIMENT:
- Ŵ_1(P̂_n, Q̂_m): area between empirical CDFs of real vs. generated data
- n: number of real historical observations (you choose this)
- m: number of flow-generated samples (you choose this, typically m >> n)
- α̂: Hill estimator output from real data
- C_1, C_2: universal constants (or compute empirically via bootstrap)

THE FALSIFIABLE PREDICTION: for assets with α̂ comfortably above 2 (SPY),
observed CVaR errors should shrink close to root-n rate as n grows.
For assets with α̂ near or below 2 (BTC), observed errors should shrink
visibly slower, matching γ(α̂) < 1/2. This asymmetry IS YOUR FINDING.

---

## 4. THE GENERATIVE MODEL: NORMALIZING FLOW

### 4.1 Why Flows (Not Diffusion or GAN)

| Model | Exact density | Stable training | No extra theorem needed |
|-------|-------------|-----------------|------------------------|
| GAN | ❌ | ❌ | ❌ |
| Diffusion | ❌ | ✅ | ❌ (need SDE convergence theory) |
| **Flow** | **✅** | **✅** | **✅** |

Flows give exact p(x) via the change-of-variables formula. This means:
- Training loss IS the likelihood, directly measurable.
- No separate convergence theorem needed to link "training quality" to "distributional quality."
- Wasserstein distance can be computed directly from generated samples.

### 4.2 Architecture: RealNVP (Dinh et al. 2017)

Basic building block — an Affine Coupling Layer:
    Given input x = [x_A, x_B] (split into two halves):
        y_A = x_A                              (unchanged)
        y_B = x_B ⊙ exp(s(x_A)) + t(x_A)    (scale + shift, computed from x_A)
    where s(·) and t(·) are small MLPs.

Why this is clever:
    - Invertible: x_B = (y_B - t(y_A)) ⊙ exp(-s(y_A))  (trivially)
    - Jacobian determinant = product of diagonal terms = exp(sum(s(x_A)))  (cheap!)
    - Chain K of these layers (alternating which half transforms) = flexible density estimator

For 1D returns: use an autoregressive flow (MAF - Masked Autoregressive Flow)
since coupling layers need ≥2 dimensions. MAF generalizes the same idea to 1D.

### 4.3 Training Objective: Maximum Likelihood

loss = -E[log p_θ(x)]
     = -E[log p_z(f_θ^{-1}(x)) + log |det J_{f_θ^{-1}}(x)|]

where p_z is the base distribution (standard Gaussian), f_θ is the flow.
Standard Adam optimizer. No adversarial game. Stable.

### 4.4 What We Need From the Trained Model

After training:
1. GENERATE samples: z ~ N(0,1), x = f_θ(z). Produce m=50,000 fake return samples.
2. COMPUTE Ŵ_1: empirical CDFs of real and fake losses, integrate gap.
3. COMPUTE CVaR from fake samples: sort, take mean of worst q-fraction.
4. Compare to CVaR from real data. The gap is what the bound predicts.

---

## 5. EXPERIMENT DESIGN

### 5.1 Assets (Two Contrasting Series)

| Asset | Proxy | Expected α̂ | Expected regime |
|-------|-------|------------|----------------|
| Calm | SPY (S&P 500 ETF) | ~3-5 | α > 2, root-n rate |
| Wild | BTC-USD (Bitcoin) | ~1.5-2.5 | α near or below 2, slower rate |

These two are specifically chosen to straddle the α=2 threshold predicted by Lemma 2.
That contrast IS the empirical result. Everything is designed around demonstrating it.

### 5.2 Data Periods (Use Both)

| Period | Dates | Characterization |
|--------|-------|-----------------|
| Calm | 2023-01-01 to 2024-12-31 | Low vol, post-COVID recovery |
| Stress | 2020-02-01 to 2020-06-30 | COVID crash, extreme volatility |
| Combined | 2015-01-01 to 2025-12-31 | Full sample for Hill estimation |

Split: train flow on 2015–2022, test/evaluate on 2023–2025.

### 5.3 What to Measure and Plot

**Primary outputs (mandatory for paper):**
1. Hill plot for each asset: k vs. α̂(k), showing the plateau where we read off α̂.
2. Ŵ_1(P̂_n, Q̂_m) as a function of training epochs (should decrease and plateau).
3. Observed CVaR error: |CVaR_q(P̂_n) - CVaR_q(Q̂_m)| vs. the theoretical bound.
4. Convergence rate plot: CVaR error vs. n (subsample real data progressively),
   fit a line in log-log space, recover empirical γ, compare to theoretical γ(α̂).

**Secondary outputs (strengthen the paper):**
5. Empirical CDF overlay: real vs. generated returns, per asset.
6. QQ-plot: generated vs. real quantiles (visual diagnostic of tail fit quality).
7. Same plots for calm vs. stress windows separately.

### 5.4 The Main Result Table (What Paper's Table 1 Should Look Like)

| Asset | α̂ (Hill) | Theoretical γ(α̂) | Empirical γ (fitted) | Ŵ_1 | CVaR_q (real) | CVaR_q (model) | Bound |
|-------|-----------|------------------|----------------------|------|--------------|---------------|-------|
| SPY   | ~3.8      | 0.500            | ~0.48 ± 0.05         | ...  | ...          | ...           | ...   |
| BTC   | ~1.8      | ~0.44            | ~0.41 ± 0.07         | ...  | ...          | ...           | ...   |

The empirical γ being close to (but not above) the theoretical γ(α̂) is the validation.

---

## 6. CODEBASE STRUCTURE

```
cvar_flow_paper/
├── COPILOT_PROJECT_BRIEF.md     ← this file
├── data/
│   ├── calm_spy.csv             ← real SPY log-returns (fetch with src/fetch_data.py)
│   ├── wild_btc.csv             ← real BTC log-returns (fetch with src/fetch_data.py)
│   ├── synthetic_calm.csv       ← Student-t df=6 (known α=6, for testing)
│   └── synthetic_wild.csv       ← Student-t df=1.5 (known α=1.5, for testing)
├── src/
│   ├── fetch_data.py            ← download SPY + BTC from Yahoo Finance (run locally)
│   ├── make_synthetic_data.py   ← generate test data with known tail index
│   ├── hill_estimator.py        ← Hill estimator + hill_plot_data() + suggest_k()
│   ├── flow_model.py            ← [TO BUILD] Normalizing flow in PyTorch (MAF)
│   ├── train_flow.py            ← [TO BUILD] Training loop, saves checkpoints
│   ├── wasserstein.py           ← [TO BUILD] Empirical W_1 computation (1D, CDF method)
│   ├── cvar_utils.py            ← [TO BUILD] CVaR/VaR estimation from samples
│   ├── bound_calculator.py      ← [TO BUILD] Compose theorem, compute the full bound
│   └── run_experiment.py        ← [TO BUILD] Master script: load data → train → measure → plot
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_hill_plots.ipynb
│   ├── 03_flow_training_diagnostics.ipynb
│   └── 04_main_results.ipynb
├── results/
│   ├── figures/                 ← all paper figures saved here as PDFs/SVGs
│   └── tables/                  ← CSV/LaTeX tables of numerical results
├── paper/
│   ├── main.tex                 ← LaTeX paper
│   └── references.bib
└── requirements.txt
```

---

## 7. KEY FILES TO BUILD NEXT (IN ORDER)

### Priority 1: wasserstein.py

```python
# Compute W_1 between two 1D sample arrays
# Method: sort both, compute empirical CDFs, integrate the gap
# scipy.stats.wasserstein_distance also works and should match — use as sanity check
def wasserstein_1d(samples_P: np.ndarray, samples_Q: np.ndarray) -> float: ...
```

### Priority 2: cvar_utils.py

```python
# Historical simulation CVaR: sort losses, mean of worst q fraction
def cvar_historical(losses: np.ndarray, q: float = 0.05) -> float: ...
def var_historical(losses: np.ndarray, q: float = 0.05) -> float: ...
```

### Priority 3: flow_model.py

```python
# 1D Masked Autoregressive Flow using PyTorch
# Use nflows library (pip install nflows) OR build MAF from scratch
# Architecture: 8-10 MADE blocks, hidden size 64, tanh activations
# Base distribution: Normal(0,1)
# Input: standardized daily log-losses (subtract mean, divide by std — re-apply at inference)

import torch
import torch.nn as nn
# Recommended: use nflows.flows.MaskedAutoregressiveFlow
# OR build manually with MADE (Masked Autoencoder for Distribution Estimation)

class ReturnFlowModel(nn.Module):
    def __init__(self, n_layers=8, hidden_size=64): ...
    def log_prob(self, x: torch.Tensor) -> torch.Tensor: ...  # for training
    def sample(self, n: int) -> torch.Tensor: ...              # for generation
```

### Priority 4: train_flow.py

```python
# Training loop
# Optimizer: Adam, lr=1e-3, decay to 1e-4
# Batch size: 256
# Epochs: 200-500 (monitor NLL on validation set, early stopping)
# Save: checkpoint every 50 epochs, best model by val NLL
# Log: training NLL, val NLL, W_1 to real data (every 10 epochs)
```

### Priority 5: bound_calculator.py

```python
# Given: alpha_hat, n (real samples), m (generated samples), q (CVaR level), W1_hat
# Compute: full RHS of the composed theorem
# gamma = min(0.5, (alpha_hat - 1) / alpha_hat)
# bound = (1/q) * W1_hat + C1 * n**(-gamma) + C2 * m**(-gamma)
# C1, C2: estimate via bootstrap on real data (resample, compute CVaR spread)
def compute_bound(alpha_hat, W1_hat, n, m, q=0.05, C1=None, C2=None) -> float: ...
```

---

## 8. PYTHON DEPENDENCIES

```
numpy
pandas
scipy
matplotlib
torch
nflows           # normalizing flows library — pip install nflows
yfinance         # data fetching (run locally, not in sandboxed env)
statsmodels      # optional: additional statistical tests
jupyter          # for notebooks
```

Install all:
```bash
pip install numpy pandas scipy matplotlib torch nflows yfinance statsmodels jupyter
```

---

## 9. IMPORTANT IMPLEMENTATION NOTES FOR COPILOT

1. **ALL losses are expressed as positive numbers for losses, negative for gains.**
   Loss = -log_return. CVaR and VaR are computed on this sign convention.

2. **Standardize input to the flow.** Before training, subtract mean and divide by std
   of training losses. At inference, multiply samples back: x_real = x_std * std + mean.
   This prevents the flow from wasting capacity modeling the mean and scale.

3. **Hill estimator only uses POSITIVE loss values.** Filter losses > 0 before calling.

4. **W_1 computation for 1D is just `scipy.stats.wasserstein_distance(real, fake)`.**
   Always sanity-check the manual CDF implementation against this.

5. **For the convergence rate experiment:** subsample the real data at sizes
   n = [100, 200, 500, 1000, 2000, 5000, ...] (logarithmically spaced),
   fit the flow on each subsample, compute CVaR error, plot error vs. n in log-log space.
   The slope of the log-log line = the empirical γ. Compare to theoretical γ(α̂).

6. **Two q values to report:** q=0.05 (95% CVaR) and q=0.01 (99% CVaR).
   The bound coefficient (1/q) is 20× and 100× respectively — this shows how much
   more sensitive extreme CVaR is to model error.

7. **Hill plot reading:** look for a stable plateau in the α̂-vs-k curve.
   Don't just report a single point estimate. Report α̂ as a range
   (e.g. "α̂ ∈ [3.2, 4.1] over k ∈ [50, 150]"). This is standard practice.

---

## 10. PAPER STRUCTURE (LATEX SECTIONS)

```
Abstract (150 words max)
    - State the gap: generative models used for risk estimation, no reliability guarantees.
    - State the method: Wasserstein–CVaR stability + EVT tail classification.
    - State the finding: explicit threshold at α=2, confirmed empirically on SPY vs. BTC.

1. Introduction (1.5 pages)
    - Generative models in finance: stress testing, scenario generation, VaR/CVaR.
    - The reliability gap: nobody asks "how wrong can the risk estimate be?"
    - This paper: we do. We give an explicit, computable answer.
    - Related work paragraph (normalizing flows for finance, Wasserstein robustness,
      EVT in risk management — cite but distinguish).

2. Preliminaries (1 page)
    - CVaR definition and coherence (cite Artzner et al. 1999, Rockafellar–Uryasev 2000).
    - Wasserstein distance: definition, 1D simplification, Kantorovich–Rubinstein duality.
    - Normalizing flows: change-of-variables formula, MAF architecture.

3. Theoretical Framework (2 pages) ← YOUR CORE SECTION
    - Lemma 1: CVaR stability under W_1 (state, cite, brief proof sketch).
    - Lemma 2: EVT-driven convergence rates (state, cite Fournier-Guillin, Brazauskas et al.).
    - Theorem 1 (YOURS): Composed end-to-end bound. State formally. Prove by chaining.
    - Corollary (YOURS): at α = 2, convergence rate has a strict discontinuity.
      Assets below this threshold cannot achieve standard root-n CVaR reliability.

4. Empirical Validation (2 pages)
    - Data: SPY and BTC-USD, 2015–2025, training/test split.
    - Flow model: architecture, training details, convergence diagnostics.
    - Hill estimation: plots, α̂ values with ranges for both assets.
    - Main result: bound vs. observed CVaR error, calm and stress windows.
    - Convergence rate experiment: log-log plots, empirical γ vs. theoretical γ(α̂).

5. Discussion (0.5 page)
    - What this means for practitioners: a risk manager using a generative model
      on a crypto book has quantifiably less reliable CVaR than on an equity book,
      even with identical model quality — this is a theorem, not just intuition.
    - Limitations: Hill estimator bias for large α, finite sample sizes, 1D marginal
      (doesn't capture copula structure of multi-asset portfolios).

6. Conclusion (0.25 page)

Appendix
    - Extended proofs.
    - Additional assets (if time permits: GLD, VIX, EM FX).
    - Hill plot stability analysis.
```

---

## 11. KEY PAPERS TO READ (MONTH 1 PRIORITY LIST)

| Paper | Why You Need It | What to Extract |
|-------|----------------|-----------------|
| Rockafellar & Uryasev (2000) - "Optimization of CVaR" | Lemma 1 source | The variational representation of CVaR |
| Pflug (2000) - "Some remarks on the value-at-risk..." | Lemma 1 stability | Cite for Lipschitz/stability argument |
| Fournier & Guillin (2015) - "On the rate of convergence in Wasserstein distance" | Lemma 2 source | The n^{-1/d} rates, tail-index dependence |
| Hill (1975) - "A simple general approach to inference about the tail of a distribution" | Hill estimator | The formula, conditions for consistency |
| Dinh et al. (2017) - "Density estimation using Real-valued Non-Volume Preserving..." | RealNVP | The coupling layer architecture |
| Papamakarios et al. (2017) - "Masked Autoregressive Flow for Density Estimation" | MAF | The 1D flow you'll actually implement |
| Artzner et al. (1999) - "Coherent Measures of Risk" | CVaR coherence | Why CVaR > VaR, axioms |

All of these are freely available on Google Scholar or arXiv.

---

## 12. NOVELTY STATEMENT (FOR SOP AND ABSTRACT)

The novelty is **synthesis and empirical instantiation**, not new mathematics.

Each component (CVaR stability under W_1, EVT convergence rates, normalizing flows)
exists separately in the literature. What does not exist:
- The **explicit composition** of these three into a single end-to-end bound.
- The **α=2 threshold** identified as a practical reliability barrier for generative risk modeling.
- The **empirical validation** of that threshold on real assets with contrasting tail behavior.
- The **two-term decomposition** separating model bias (term 1) from estimation noise (term 2).

This is how 90% of strong research works. You're not inventing a new mathematical object;
you're assembling known tools into a framework nobody has built, and proving it works.
That's exactly the kind of intellectual move top MFE/MSQF committees want to see.

---

*End of project brief. Feed this file to GitHub Copilot as context before building any module.*
*Refer to the chat session where this was developed for concept explanations and code already written.*
