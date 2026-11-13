"""
PC5203 Project 2, Problem 2, Q5 -- spectral flow of the Harper chain (Eq. 2) as k is varied
over one full period, and its relation to the Chern numbers of Q4.

Open Harper chain of L sites (open boundary conditions, OBC, in l):
    H_ll = 2 cos(2 pi b l - k),  H_{l,l+1} = H_{l+1,l} = 1,   l = 0..L-1.
This is the 2D lattice on a cylinder (open in x, periodic in y) at fixed k = k_y.
Changing k by 2 pi / L_y is equivalent to threading one flux quantum through the
cylinder (Laughlin's gauge argument), and k -> k + 2 pi returns H to itself.

Spectral flow count: for each bulk gap r (r bands below) we pick the reference energy
E_r = gap centre, sweep k over [0, 2 pi] on a fine grid and record every eigenvalue that
crosses E_r, with sign +1 (upward) / -1 (downward) and with the edge it lives on
(centre of mass < L/2: left, otherwise right).  N_L(r), N_R(r) = signed sums.

Checks performed (all printed):
  (a) N_L(r) = -N_R(r) for every gap  (charge pumped from one edge to the other);
  (b) N_L(r) = + sum_{j<=r} C_j  with C_j from p2_q4_chern.fhs (Q4), for several p/q
      (left-edge states flow upward through a gap of total Chern number C > 0);
  (c) the result does not depend on the chain length L (two lengths tested);
  (d) bulk bands at FIXED k (union over k_x only) obey 2cos(qk)-2 <= P(eps) <= 2cos(qk)+2,
      i.e. they are narrower than the full bands and slide up/down with period 2pi/q in k.
Figures: images/fig_q5_spectral_flow.png
"""
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

# ROOT = folder that holds the Q2 and Q4 scripts (this script's folder or its parent).
# Figures are saved to ROOT/images, e.g.
#   scripts: D:\PyCharm_Project\5203 project 2\
#   figures: D:\PyCharm_Project\5203 project 2\images\
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = next((d for d in (HERE, os.path.dirname(HERE))
             if os.path.isfile(os.path.join(d, "p2_q4_chern.py"))
             and os.path.isfile(os.path.join(d, "p2_q2_butterfly_square.py"))), None)
if ROOT is None:
    sys.exit("p2_q4_chern.py / p2_q2_butterfly_square.py not found in this script's "
             "folder or its parent folder.")
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True     # importing Q2/Q4 must not leave a __pycache__ folder behind
from p2_q4_chern import fhs, band_groups
from p2_q2_butterfly_square import band_edges

IMG = os.path.join(ROOT, "images")


def harper_obc(L, b, k):
    H = np.diag(2 * np.cos(2 * np.pi * b * np.arange(L) - k))
    H += np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1)
    return H


def sweep(L, b, nk):
    ks = np.linspace(0, 2 * np.pi, nk + 1)
    E = np.empty((nk + 1, L)); X = np.empty((nk + 1, L))
    l = np.arange(L)
    for i, k in enumerate(ks):
        e, v = np.linalg.eigh(harper_obc(L, b, k))
        E[i] = e
        X[i] = (np.abs(v) ** 2 * l[:, None]).sum(0) / (L - 1)   # centre of mass in [0,1]
    return ks, E, X


def flow(ks, E, X, Eref):
    """Signed crossings of energy Eref, split by edge. Requires <=1 crossing per k-step."""
    nL = nR = 0
    n = (E < Eref).sum(1)
    for i in range(len(ks) - 1):
        dn = n[i + 1] - n[i]
        if dn == 0:
            # a left-up and a right-down crossing in the same step would cancel here:
            # guard by checking that no eigenvalue sits within the step's sweep of Eref
            cross = ((E[i] - Eref) * (E[i + 1] - Eref) < 0).sum()
            assert cross == 0, "two crossings in one k-step: refine the k grid"
            continue
        assert abs(dn) == 1, "refine the k grid"
        if dn == -1:            # state n[i]-1 went up through Eref
            j, s = n[i] - 1, +1
        else:                   # state n[i] came down
            j, s = n[i], -1
        x = X[i, j]
        if x < 0.5:
            nL += s
        else:
            nR += s
    return nL, nR


def gap_centres(p, q):
    ed = band_edges(p, q)
    groups = band_groups(p, q)
    out = []
    filled = 0
    for g in groups[:-1]:
        filled += len(g)
        out.append((filled, 0.5 * (ed[filled - 1, 1] + ed[filled, 0])))
    return out


def checks():
    print("gap r | sum_{j<=r} C_j (Q4) | N_L, N_R (L1) | N_L, N_R (L2)")
    for (p, q) in [(1, 3), (2, 5), (1, 5), (2, 7), (3, 7), (3, 8)]:
        b = p / q
        C, groups, _, _ = fhs(p, q, 16, 8 * q)
        Ci = np.rint(C).astype(int)
        res = []
        for L in [10 * q + 1, 14 * q + 3]:
            ks, E, X = sweep(L, b, 4000)
            res.append([flow(ks, E, X, Er) for (_, Er) in gap_centres(p, q)])
        acc = 0
        for idx, (r, Er) in enumerate(gap_centres(p, q)):
            acc = Ci[:idx + 1].sum()
            (l1, r1), (l2, r2) = res[0][idx], res[1][idx]
            ok = (l1 == -r1 == acc) and (l2 == -r2 == acc)
            print(f"b={p}/{q} r={r:2d} | {acc:+d} | {l1:+d},{r1:+d} | {l2:+d},{r2:+d} | "
                  f"{'OK' if ok else 'MISMATCH'}")


def fixed_k_bands():
    from p2_q2_butterfly_square import H_mag
    worst = 0
    rng = np.random.default_rng(3)
    for (p, q) in [(1, 3), (2, 5), (3, 7)]:
        for k in rng.uniform(0, 2 * np.pi, 4):
            th = np.linspace(0, 2 * np.pi, 801)
            E = np.array([np.linalg.eigvalsh(H_mag(p, q, t / q, k)) for t in th])
            lo, hi = E.min(0), E.max(0)
            P = lambda e: np.linalg.det(e * np.eye(q) - H_mag(p, q, 0, 0)).real + 4
            for e in np.concatenate([lo, hi]):
                worst = max(worst, min(abs(P(e) - 2 * np.cos(q * k) - 2),
                                       abs(P(e) - 2 * np.cos(q * k) + 2)))
    print(f"(d) fixed-k band edges satisfy P(eps) = 2cos(qk) +- 2: max deviation {worst:.1e}")


def crossings(ks, E, X, Eref):
    """List of (k, sign, edge) for every eigenvalue crossing Eref (same rules as flow())."""
    out = []
    n = (E < Eref).sum(1)
    for i in range(len(ks) - 1):
        dn = n[i + 1] - n[i]
        if dn == 0:
            continue
        j, sgn = (n[i] - 1, +1) if dn == -1 else (n[i], -1)
        # interpolate the crossing momentum
        e0, e1 = E[i, j], E[i + 1, j if dn == -1 else n[i]]
        t = 0.5 if e1 == e0 else np.clip((Eref - e0) / (e1 - e0), 0, 1)
        out.append((ks[i] + t * (ks[i + 1] - ks[i]), sgn, "L" if X[i, j] < 0.5 else "R"))
    return out


def plot(fname):
    from matplotlib.lines import Line2D
    cases = [(1, 3, 31), (2, 5, 41)]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.9), constrained_layout=True)
    cL, cR = plt.get_cmap("coolwarm")(0.0), plt.get_cmap("coolwarm")(1.0)
    for ax, (p, q, L), tag in zip(axs, cases, "ab"):
        b = p / q
        ks, E, X = sweep(L, b, 600)
        K = np.repeat(ks[:, None], L, 1)
        sc = ax.scatter(K.ravel(), E.ravel(), c=X.ravel(), s=1.2, cmap="coolwarm",
                        vmin=0, vmax=1, rasterized=True)
        for lo, hi in band_edges(p, q):
            ax.axhspan(lo, hi, color="0.92", zorder=0)
        C, groups, _, _ = fhs(p, q, 16, 8 * q)
        Ci = np.rint(C).astype(int)
        ksf, Ef, Xf = sweep(L, b, 4000)
        for idx, (r, Er) in enumerate(gap_centres(p, q)):
            ax.axhline(Er, color="0.5", lw=0.6, ls=":", zorder=1)
            for (kc, sgn, edge) in crossings(ksf, Ef, Xf, Er):
                ax.plot(kc, Er, "^" if sgn > 0 else "v", ms=7, mec="k", mew=0.6,
                        color=cL if edge == "L" else cR, zorder=5)
            nL, nR = flow(ksf, Ef, Xf, Er)
            ax.text(2 * np.pi * 1.015, Er,
                    rf"$\sum C={Ci[:idx+1].sum():+d}$" + "\n" + rf"$N_L={nL:+d}$",
                    fontsize=8, va="center", linespacing=1.1)
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        ax.set_xlabel(r"$k$ (one full period)"); ax.set_ylabel(r"$\varepsilon$")
        ax.set_title(rf"({tag}) $b={p}/{q}$, open chain of $L={L}$ sites", fontsize=10)
    handles = [Line2D([], [], color="0.85", lw=6, label="bulk bands (grey strips)"),
               Line2D([], [], color=cL, lw=2.5, label="state on left edge"),
               Line2D([], [], color=cR, lw=2.5, label="state on right edge"),
               Line2D([], [], ls="", marker="^", mec="k", mfc="w", ms=7,
                      label=r"crosses gap centre upward ($+1$)"),
               Line2D([], [], ls="", marker="v", mec="k", mfc="w", ms=7,
                      label=r"crosses gap centre downward ($-1$)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8, frameon=False)
    cb = fig.colorbar(sc, ax=axs, shrink=0.8)
    cb.set_label(r"position of the state $\langle l\rangle/(L-1)$ (0: left, 1: right)")
    fig.suptitle(r"Spectral flow: in every gap the net number of left-edge crossings per period, "
                 r"$N_L$, equals $\sum_{j\leq r}C_j$", fontsize=10.5)
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    os.makedirs(IMG, exist_ok=True)
    checks()
    fixed_k_bands()
    plot(os.path.join(IMG, "fig_q5_spectral_flow.png"))
