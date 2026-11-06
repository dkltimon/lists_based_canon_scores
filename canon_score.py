#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:32:49 2026

@author: keli
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize

# ============================================================
# 0. Settings
# ============================================================
TSV_PATH = "/Users/keli/Workstation/München/canon-shelf-main/lists/merged_all_books.csv"      
TITLE_COL = "Title"
YEAR_COL = "Publication year (EDTF)"

# A book is eligible for a list if it was published up to (STRICT = False) or before (STRICT = True) the list's year.
STRICT = False

ALPHA_MIN, BETA_MIN = 1e-4, 0.05   # lower bounds for the optimiser and the grid

LIST_ORDER = None               # e.g. ["2003_guardian", "2015_guardian", ...]; None = sort by year in the list name
SCORE_UNLISTED = False          # False: at stage t, score only books already on >= 1 of the first t lists

# ============================================================
# 1. TSV -> k, n  (one entry per listed book)
# ============================================================
df = pd.read_csv(TSV_PATH, sep="\t", dtype=str, encoding="utf-8-sig", keep_default_na=False)
df.columns = [c.strip() for c in df.columns]

list_cols = [c for c in df.columns if re.match(r"^\d{4}_", c)]
list_year = {c: int(c[:4]) for c in list_cols}
print("Lists found:", list_year)

def parse_year(s):
    m = re.search(r"-?\d{4}", str(s))
    return int(m.group()) if m else np.nan
df["pub_year"] = df[YEAR_COL].map(parse_year)
bad = df[df["pub_year"].isna()]
if len(bad):
    print(f"\n{len(bad)} book(s) dropped: publication year could not be read:")
    print(bad[[TITLE_COL, YEAR_COL]].to_string(index=False))
df = df.dropna(subset=["pub_year"]).copy()

NOT_IN = {"", "0", "0.0", "false", "no", "n", "na", "nan", "none", "-"}
is_in = lambda v: str(v).strip().lower() not in NOT_IN
included = np.column_stack([df[c].map(is_in).to_numpy() for c in list_cols])

eligible = np.zeros_like(included)
for j, c in enumerate(list_cols):
    before_list = df["pub_year"] < list_year[c] if STRICT else df["pub_year"] <= list_year[c]
    eligible[:, j] = before_list.to_numpy()

# A book on a list it was "not eligible" for means a wrong year in the data: flag it.
conflict = included & ~eligible
if conflict.any():
    print(f"\n{conflict.sum()} inclusion(s) fall outside the eligibility rule (counted as eligible anyway):")
    for r, j in zip(*np.where(conflict)):
        print(f"  {df.iloc[r][TITLE_COL]!r} ({int(df.iloc[r]['pub_year'])}) on {list_cols[j]}")
eligible = eligible | included

df["k"] = included.sum(axis=1)
df["n"] = eligible.sum(axis=1)
df = df[(df["k"] >= 1) & (df["n"] >= 1)].copy()
k = df["k"].astype(int).to_numpy()
n = df["n"].astype(int).to_numpy()
titles = df[TITLE_COL].tolist()

print(f"\n{len(k)} listed books.  Table of books by n (rows) and k (columns):")
print(pd.crosstab(pd.Series(n, name="n"), pd.Series(k, name="k")))

# ============================================================
# 2. Truncated Beta-Binomial log-likelihood (books with k >= 1 only)
# ============================================================
pairs, cnt = np.unique(np.c_[k, n], axis=0, return_counts=True)
ku, nu = pairs[:, 0], pairs[:, 1]

def loglik(a, b):
    ll = stats.betabinom.logpmf(ku, nu, a, b)
    lp0 = stats.betabinom.logpmf(0, nu, a, b)
    ll = ll - np.log(-np.expm1(np.minimum(lp0, -1e-12)))
    return (cnt * ll).sum()

# ============================================================
# 3. Grid of (alpha, beta)
# ============================================================
alphas = np.geomspace(ALPHA_MIN, 5, 120)
betas = np.geomspace(0.2, 40, 120)
LL = np.array([[loglik(a, b) for a in alphas] for b in betas])   # rows = beta, cols = alpha
A, B = np.meshgrid(alphas, betas)

# ============================================================
# 4. Optimiser (bounded) from several starting points
# ============================================================
nll = lambda th: -loglik(*np.exp(th))
bounds = [(np.log(ALPHA_MIN), np.log(50)), (np.log(BETA_MIN), np.log(200))]
fits = []
for a0, b0 in [(0.1, 1), (0.5, 3), (2, 2), (1, 10), (0.05, 0.5)]:
    r = optimize.minimize(nll, np.log([a0, b0]), method="L-BFGS-B", bounds=bounds)
    fits.append((*np.exp(r.x), -r.fun))
print("\nOptimiser from different starting points (alpha, beta, logL):")
for f in fits:
    print("  alpha=%.5f  beta=%.3f  logL=%.2f" % f)
a_hat, b_hat, ll_hat = max(fits, key=lambda f: f[2])
print(f"\nBest: alpha={a_hat:.5f}, beta={b_hat:.3f}, prior mean={a_hat/(a_hat+b_hat):.4f}, "
      f"strength alpha+beta={a_hat+b_hat:.2f}")
if a_hat < 10 * ALPHA_MIN:
    print(f"NOTE: alpha is at its lower bound ({ALPHA_MIN}); the likelihood is flat there, so alpha is not "
          f"identified. The score reduces to about k / (n + beta), beta={b_hat:.2f}.")

region = LL >= LL.max() - 3.0
print(f"Plausible region (score within 3.0 of best): alpha {A[region].min():.4f}-{A[region].max():.3f}, "
      f"beta {B[region].min():.2f}-{B[region].max():.2f}")
if region[:, 0].any():
    print("  (the region touches the lower alpha edge of the grid)")

# ============================================================
# 5. Plot
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
cs = ax1.contourf(A, B, np.clip(LL - LL.max(), -15, 0), levels=np.linspace(-15, 0, 16), cmap="viridis")
ax1.contour(A, B, LL - LL.max(), levels=[-3.0], colors="white", linewidths=1.5)
ax1.plot(max(a_hat, ALPHA_MIN), b_hat, "r*", ms=14, label="best (optimiser)")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("alpha"); ax1.set_ylabel("beta")
ax1.set_title("Score relative to the best (white line: plausible region)")
fig.colorbar(cs, ax=ax1, label="log-likelihood minus maximum")
ax1.legend(loc="upper right")

x = np.linspace(0.001, 0.999, 400)
idx = np.argwhere(region)
pick = idx[np.random.default_rng(0).choice(len(idx), size=min(8, len(idx)), replace=False)]
for r_, c_ in pick:
    ax2.plot(x, stats.beta.pdf(x, alphas[c_], betas[r_]), color="grey", alpha=0.5, lw=1)
ax2.plot(x, stats.beta.pdf(x, a_hat, b_hat), "r", lw=2.5,
         label="best" + (" (almost all mass at p=0)" if a_hat < 0.01 else ""))
ax2.set_yscale("log")
ax2.set_xlabel("a book's recognition probability p")
ax2.set_ylabel("density (log scale)")
ax2.set_title("Priors inside the plausible region (grey)")
ax2.legend()
fig.tight_layout()
plt.show()

# ============================================================
# 6. Scores with the fitted prior
# ============================================================
def scores(a, b):
    Ap, Bp = a + k, b + n - k
    return Ap / (Ap + Bp), stats.beta.ppf(0.025, Ap, Bp), stats.beta.ppf(0.975, Ap, Bp)

S, lo, hi = scores(a_hat, b_hat)
out = pd.DataFrame(dict(title=titles, k=k, n=n, raw=(k / n).round(3),
                        score=S.round(3), ci_low=lo.round(3), ci_high=hi.round(3)))
print("\nTop books by score:")
print(out.sort_values("score", ascending=False).head(15).to_string(index=False))

# ============================================================
# 7. Does the ranking depend on alpha? (alpha fixed, beta re-estimated)
# ============================================================
print("\nSensitivity of the ranking to alpha:")
print("  alpha    beta_hat   Spearman vs best")
for a_fix in [1e-3, 1e-2, 5e-2, 1e-1]:
    r = optimize.minimize_scalar(lambda lb: -loglik(a_fix, np.exp(lb)),
                                 bounds=(np.log(BETA_MIN), np.log(200)), method="bounded")
    b_fix = np.exp(r.x)
    rho = stats.spearmanr(scores(a_fix, b_fix)[0], S)[0] if len(k) > 2 else np.nan
    print(f"  {a_fix:<7g}  {b_fix:8.3f}   {rho:.4f}")
    
    
# ============================================================
# 8. Stage-wise scores: stage t uses only the first t lists
#    (the prior fitted on ALL lists is reused at every stage, so scores are comparable across stages)
# ============================================================
K_cum = np.cumsum(included, axis=1)       # books x stages: appearances so far
N_cum = np.cumsum(eligible, axis=1)       # books x stages: eligible lists so far

rows = []
for t, c in enumerate(list_cols):
    kt, nt = K_cum[:, t], N_cum[:, t]
    ok = (nt >= 1) if SCORE_UNLISTED else ((nt >= 1) & (kt >= 1))
    Ap, Bp = a_hat + kt[ok], b_hat + nt[ok] - kt[ok]
    mean = Ap / (Ap + Bp)
    low, high = stats.beta.ppf(0.025, Ap, Bp), stats.beta.ppf(0.975, Ap, Bp)
    for i, m_, l_, h_ in zip(np.where(ok)[0], mean, low, high):
        rows.append(dict(title=titles[i], stage=t + 1, last_list=c, k=int(kt[i]), n=int(nt[i]),
                         score=m_, ci_low=l_, ci_high=h_))
traj = pd.DataFrame(rows)

# wide table: one row per book, one column per stage, "score [low, high]"
traj["cell"] = traj.apply(lambda r: f"{r.score:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]  (k={r.k}, n={r.n})", axis=1)
wide = traj.pivot(index="title", columns="stage", values="cell").reindex(titles)
wide.columns = [f"stage {s_} ({list_cols[s_ - 1]})" for s_ in wide.columns]
pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_columns", 20)
print("\nScores by stage (cumulative lists):")
print(wide.fillna("-").to_string())
#traj.to_csv("/Users/keli/Workstation/München/canon-shelf-main/lists/calculated_scores.csv", sep='\t', index=False)     # uncomment to save the long table

# trajectories of the highest-scoring books
final = traj[traj["stage"] == len(list_cols)].sort_values("score", ascending=False)
top = final["title"].head(8).tolist()
fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True, squeeze=False)
for ax, title in zip(axes.ravel(), top):
    d = traj[traj["title"] == title].sort_values("stage")
    ax.plot(d["stage"], d["score"], "o-")
    ax.fill_between(d["stage"], d["ci_low"], d["ci_high"], alpha=0.25)
    ax.set_title(title[:28], fontsize=9)
    ax.set_xticks(range(1, len(list_cols) + 1))
    ax.set_ylim(0, 1)
for ax in axes[1]:
    ax.set_xlabel("stage (lists used so far)")
axes[0, 0].set_ylabel("score"); axes[1, 0].set_ylabel("score")
fig.tight_layout()
plt.show()    
    
# ------------------------------------------------------------
# Plot the score trajectories of chosen books
# ------------------------------------------------------------
BOOKS_TO_PLOT = [          # <- exact titles as they appear in your TSV (six books)
    "A Clockwork Orange",
    "Midnight’s Children",
    "Die Brücke über die Drina",
    "Tinker Tailor Soldier Spy",
    "Die Straße",
    "Ulysses",
]
MODE = "together"          # "together": one axes, six lines;  "panels": 2 x 3 small plots

import difflib
chosen = []
for t_ in BOOKS_TO_PLOT:
    if t_ in set(traj["title"]):
        chosen.append(t_)
    elif t_ in titles:
        print(f"'{t_}' is in the data but has no score at any stage (check its year/list columns).")
    else:
        print(f"'{t_}' not found. Did you mean: {difflib.get_close_matches(t_, titles, n=3)}")

stages = list(range(1, len(list_cols) + 1))
colors = plt.cm.tab10(np.arange(len(chosen)))

def draw(ax, title, color, dx=0.0, band=False):
    d = traj[traj["title"] == title].sort_values("stage")
    ax.plot(d["stage"] + dx, d["score"], "o-", color=color, label=title)
    if band:
        ax.fill_between(d["stage"], d["ci_low"], d["ci_high"], color=color, alpha=0.2)
    else:
        ax.vlines(d["stage"] + dx, d["ci_low"], d["ci_high"], color=color, alpha=0.6, lw=1.5)

if MODE == "together":
    fig, ax = plt.subplots(figsize=(9, 5))
    offsets = np.linspace(-0.12, 0.12, max(len(chosen), 1))      # small horizontal shifts so intervals do not overlap
    for title, col, dx in zip(chosen, colors, offsets):
        draw(ax, title, col, dx)
    ax.set_xticks(stages); ax.set_xticklabels(list_cols, rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("score (95% credible interval)")
    ax.set_xlabel("stage: lists used so far (cumulative)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
else:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True, squeeze=False)
    for ax, title, col in zip(axes.ravel(), chosen, colors):
        draw(ax, title, col, band=True)
        ax.set_title(title[:30], fontsize=9)
        ax.set_xticks(stages); ax.set_xticklabels(list_cols, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
    for ax in axes[:, 0]:
        ax.set_ylabel("score")
fig.tight_layout()
plt.show()


# ------------------------------------------------------------
# Grouped bar chart of chosen books with 95% credible intervals
# ------------------------------------------------------------
import difflib, textwrap

GROUP_BY = "book"          # "book": one group per book, one bar per stage;  "stage": one group per stage, one bar per book

chosen = []
for t_ in BOOKS_TO_PLOT:
    if t_ in set(traj["title"]):
        chosen.append(t_)
    elif t_ in titles:
        print(f"'{t_}' is in the data but has no score at any stage (check its year/list columns).")
    else:
        print(f"'{t_}' not found. Did you mean: {difflib.get_close_matches(t_, titles, n=3)}")

stages = list(range(1, len(list_cols) + 1))
piv = lambda col: traj.pivot(index="title", columns="stage", values=col).reindex(index=chosen, columns=stages)
M, LO, HI = piv("score").to_numpy(), piv("ci_low").to_numpy(), piv("ci_high").to_numpy()   # books x stages

if GROUP_BY == "book":
    groups = [textwrap.fill(t_, 16) for t_ in chosen]
    series = [1,2,3,4,5]
    colors = plt.cm.Blues(np.linspace(0.35, 0.95, len(stages)))      # light -> dark = earlier -> later lists
else:
    M, LO, HI = M.T, LO.T, HI.T
    groups = list_cols
    series = chosen
    colors = plt.cm.tab10(np.arange(len(chosen)))

n_groups, n_series = M.shape
width = 0.8 / n_series
fig, ax = plt.subplots(figsize=(max(9, 1.6 * n_groups), 5))
for j in range(n_series):
    x = np.arange(n_groups) + (j - (n_series - 1) / 2) * width
    ok = ~np.isnan(M[:, j])
    ax.bar(x[ok], M[ok, j], width * 0.95, color=colors[j], label=series[j],
           yerr=np.vstack([M[ok, j] - LO[ok, j], HI[ok, j] - M[ok, j]]),
           capsize=2, error_kw=dict(lw=1, ecolor="black", alpha=0.8))
    for xi in x[~ok]:                                   # stage with no score yet: mark with a dash
        ax.text(xi, 0.01, "-", ha="center", va="bottom", color="grey")
ax.set_xticks(np.arange(n_groups)); ax.set_xticklabels(groups, fontsize=9)
ax.set_ylim(0, 1.02)
ax.set_ylabel("score (95% credible interval)")
ax.legend(title="stage" if GROUP_BY == "book" else "book", fontsize=9)
fig.tight_layout()
plt.show()




    