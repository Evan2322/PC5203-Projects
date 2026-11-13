"""
PC5203 Project 2, Problem 2, Q4 -- Berry curvature and Chern numbers of the q Hofstadter
subbands at flux b = p/q, computed with the Fukui-Hatsugai-Suzuki (FHS) lattice method.

Magnetic Bloch Hamiltonian H(theta, k_y) (q x q) as in p2_q2_butterfly_square.py,
theta = q k_x.  The magnetic Brillouin zone of the q x 1 magnetic cell is
k_x in [0, 2pi/q), k_y in [0, 2pi)  <=>  theta in [0, 2pi), k_y in [0, 2pi).

FHS: on an N1 x N2 grid, link variables U = <u(k)|u(k+dk)>/|...|; the Berry flux through a
plaquette is F = arg(U1 U2(k+d1) U1(k+d2)^* U2(k)^*) in (-pi, pi]; C = sum F / 2pi is an
integer by construction and equals the Chern number once the grid resolves the curvature.
For degenerate band groups (even q, two central bands touch at eps = 0) the overlaps are
replaced by determinants of the overlap matrices (non-Abelian version).

Checks performed (all printed):
  (a) every C is an integer to 1e-10, and the Chern numbers of all q bands sum to zero;
  (b) grid convergence: C unchanged when the grid is doubled;
  (c) TKNN Diophantine equation: for each open gap r (r bands filled),
      r = q s_r + p t_r, |t_r| <= q/2, and the total Chern number below the gap
      sum_{j<=r} C_j = -t_r for all p/q with q <= QMAX (sign fixed by our conventions);
  (d) small flux b = 1/q: lowest band curvature becomes uniform, F -> -1/(2 pi b) = -l_B^2;
      uniformity measured by std(F)/|mean(F)| (the mean itself is fixed by C = -1).
Figures: images/fig_q4_berry_b1_3.png, images/fig_q4_berry_b2_5.png,
         images/fig_q4_colored_butterfly.png
"""
import numpy as np
from math import gcd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import os

# Figures go to ROOT/images, where ROOT is the project folder, e.g.
#   script : D:\PyCharm_Project\5203 project 2\p2_q4_chern.py
#   figures: D:\PyCharm_Project\5203 project 2\images\
# If the script itself sits in a sub-folder named "images" or "codes", ROOT is its parent.
# Nothing is created at import time (so importing from Q5 leaves no extra folders);
# the folder is made only when this script is run directly.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).lower() in ("images", "codes") else HERE
IMG = os.path.join(ROOT, "images")


def H_grid(p, q, th, ky):
    """Batched magnetic Bloch Hamiltonians on a grid: returns (N1, N2, q, q)."""
    TH, KY = np.meshgrid(th, ky, indexing="ij")
    l = np.arange(q)
    H = np.zeros(TH.shape + (q, q), complex)
    H[..., l, l] = 2 * np.cos(2 * np.pi * p * l / q - KY[..., None])
    for i in range(q):
        j = (i + 1) % q
        ph = np.exp(1j * TH) if j == 0 else np.ones_like(TH)
        H[..., j, i] += ph
        H[..., i, j] += np.conj(ph)
    return H


def band_groups(p, q):
    """Separate bands into groups that never touch. For even q the two central bands
    touch at eps = 0 (Dirac points) and are treated as one group."""
    if q % 2 == 0 and q > 1:
        m = q // 2
        return [[j] for j in range(m - 1)] + [[m - 1, m]] + [[j] for j in range(m + 1, q)]
    return [[j] for j in range(q)]


def fhs(p, q, N1, N2):
    """Return (list of Chern numbers per band group, groups, Berry flux per plaquette
    for each group (N1,N2), eigenvalues)."""
    th = 2 * np.pi * np.arange(N1) / N1
    ky = 2 * np.pi * np.arange(N2) / N2
    E, V = np.linalg.eigh(H_grid(p, q, th, ky))          # V[..., :, n]
    groups = band_groups(p, q)
    Cs, Fs = [], []
    for g in groups:
        U = V[..., :, g]                                 # (N1,N2,q,m)
        # x-links in the "physical" (position-including) Bloch convention:
        # u_I(k) = D(k_x) u(k), D = diag(e^{i k_x l}); the link then carries
        # diag(e^{i dk_x l}) with dk_x = dtheta / q.  This leaves C unchanged but makes
        # F(k) the physical, convention-independent Berry curvature.
        dkx = 2 * np.pi / N1 / q
        Dx = np.exp(1j * dkx * np.arange(q))[:, None]
        U1n = Dx * np.roll(U, -1, axis=0)                # k + d1 (H(theta) is 2pi-periodic)
        U2n = np.roll(U, -1, axis=1)

        def link(A, B):
            M = np.einsum("...ia,...ib->...ab", A.conj(), B)
            d = np.linalg.det(M)
            return d / np.abs(d)
        L1 = link(U, U1n)
        L2 = link(U, U2n)
        F = np.angle(L1 * np.roll(L2, -1, axis=0) * np.conj(np.roll(L1, -1, axis=1)) * np.conj(L2))
        Cs.append(F.sum() / (2 * np.pi))
        Fs.append(F)
    return np.array(Cs), groups, Fs, E


def diophantine_t(p, q, r):
    """Solve r = q s + p t with |t| <= q/2 (unique when the gap is open)."""
    sols = [t for t in range(-q, q + 1) if (r - p * t) % q == 0 and abs(t) <= q / 2]
    return sols


def checks(qmax=24):
    print("--- Chern numbers C_j (band index from the bottom) ---")
    table = {}
    for (p, q) in [(1, 3), (2, 5), (1, 4), (1, 5), (2, 7), (3, 7), (3, 8), (4, 9), (5, 11)]:
        C, groups, _, _ = fhs(p, q, 16, 8 * q)
        C2, _, _, _ = fhs(p, q, 32, 16 * q)
        Ci = np.rint(C).astype(int)
        assert np.abs(C - Ci).max() < 1e-10 and np.array_equal(Ci, np.rint(C2).astype(int))
        lab = ["+".join(str(j + 1) for j in g) for g in groups]
        table[(p, q)] = Ci
        print(f"b={p}/{q}: " + ", ".join(f"C[{a}]={c:+d}" for a, c in zip(lab, Ci))
              + f"   sum={Ci.sum():+d}")

    print("--- TKNN Diophantine check, all p/q with q <=", qmax, "---")
    n_gap, n_ok, bad = 0, 0, []
    for q in range(2, qmax + 1):
        for p in range(1, q):
            if gcd(p, q) != 1:
                continue
            C, groups, _, _ = fhs(p, q, 12, 6 * q)
            Ci = np.rint(C).astype(int)
            if np.abs(C - Ci).max() > 1e-8:
                bad.append((p, q, "non-integer"))
            filled, acc = 0, 0
            for g, c in zip(groups[:-1], Ci[:-1]):
                filled += len(g); acc += c
                t = diophantine_t(p, q, filled)
                n_gap += 1
                if len(t) == 1 and acc == -t[0]:
                    n_ok += 1
                else:
                    bad.append((p, q, filled, acc, t))
    print(f"open gaps checked: {n_gap}, agree with sigma_r = -t_r: {n_ok}, failures: {bad[:5]}")
    return table


def landau_uniformity():
    print("--- small flux: lowest band curvature vs -l_B^2 = -1/(2 pi b) ---")
    for q in [3, 6, 10, 20]:
        p = 1
        N1, N2 = 24, 8 * q
        C, groups, Fs, _ = fhs(p, q, N1, N2)
        dkx = 2 * np.pi / q / N1          # physical k_x step (theta = q k_x)
        dky = 2 * np.pi / N2
        F = Fs[0] / (dkx * dky)           # Berry curvature in physical units
        lB2 = 1 / (2 * np.pi * p / q)
        print(f"b=1/{q}: C1={np.rint(C[0]):+.0f}, <F>={F.mean():+.4f}, -l_B^2={-lB2:+.4f}, "
              f"std/|mean|={F.std() / abs(F.mean()):.2e}")


def plot_curvature(p, q, fname, N1=60, N2f=24):
    """Berry curvature Omega(k) of every subband (or touching group) over the magnetic BZ.
    Each panel has its own symmetric colour scale (white = 0). Dashed lines mark the
    k_y-period 2pi/q of the pattern (magnetic translations)."""
    N2 = N2f * q
    C, groups, Fs, E = fhs(p, q, N1, N2)
    dkx = 2 * np.pi / q / N1
    dky = 2 * np.pi / N2
    ng = len(groups)
    fig, axs = plt.subplots(1, ng, figsize=(2.75 * ng + 0.4, 3.5), constrained_layout=True)
    for a, (g, F, c) in enumerate(zip(groups, Fs, C)):
        ax = axs[a]
        Fp = F / (dkx * dky)
        vmax = np.abs(Fp).max()
        im = ax.imshow(Fp.T, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=[0, 2 * np.pi / q, 0, 2 * np.pi])
        for j in range(1, q):                                  # k_y-period 2pi/q
            ax.axhline(2 * np.pi * j / q, color="k", lw=0.6, ls="--", alpha=0.5)
        lab = f"band {g[0]+1}" + (f"+{g[1]+1}" if len(g) > 1 else "")
        ax.set_title(lab + rf",  $C={int(np.rint(c)):+d}$", fontsize=11)
        ax.text(0.03, 0.03, rf"$\Omega\in[{Fp.min():.2g},\,{Fp.max():.2g}]$",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
        ax.set_xlabel(r"$k_x$", fontsize=10)
        ax.set_xticks([0, 2 * np.pi / q]); ax.set_xticklabels(["0", rf"$2\pi/{q}$"])
        yt = [2 * np.pi * j / q for j in range(q + 1)]
        ax.set_yticks(yt)
        ax.set_yticklabels(["0"] + [rf"${2*j}\pi/{q}$" if (2*j) % q else rf"${2*j//q}\pi$".replace("$1\\pi", "$\\pi")
                                    for j in range(1, q + 1)], fontsize=8)
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.ax.tick_params(labelsize=7)
        if a == ng - 1:
            cb.set_label(r"Berry curvature $\Omega$ ($a^2$)", fontsize=9)
    axs[0].set_ylabel(r"$k_y$", fontsize=10)
    fig.suptitle(rf"Berry curvature $\Omega(\mathbf{{k}})$ of the {q} subbands at $b={p}/{q}$; "
                 rf"$C=\frac{{1}}{{2\pi}}\int\Omega\,d^2k$; dashed lines: period $2\pi/{q}$ in $k_y$",
                 fontsize=10.5)
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("saved", fname)


def colored_butterfly(qmax=24, fname=None):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    tmax = 6
    cols = plt.get_cmap("RdBu_r")(np.linspace(0.08, 0.92, 2 * tmax + 1))
    cols[tmax] = (0.75, 0.75, 0.75, 1)                    # sigma = 0 -> neutral gray
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(np.arange(-tmax - 0.5, tmax + 1.5), cmap.N)
    for q in range(1, qmax + 1):
        for p in range(0, q + 1):
            if gcd(p, q) != 1:
                continue
            b = p / q
            if q == 1:
                ax.plot([-4, 4], [b, b], color="k", lw=0.6)
                continue
            C, groups, _, E = fhs(p, q, 12, 6 * q)
            Ci = np.rint(C).astype(int)
            # exact band edges from the two special k points (see Q2, Chambers relation)
            e0 = np.linalg.eigvalsh(H_grid(p, q, np.array([0.]), np.array([0.]))[0, 0])
            e1 = np.linalg.eigvalsh(H_grid(p, q, np.array([np.pi]), np.array([np.pi / q]))[0, 0])
            lo, hi = np.minimum(e0, e1), np.maximum(e0, e1)
            for j in range(q):
                ax.plot([lo[j], hi[j]], [b, b], color="k", lw=0.6, solid_capstyle="butt")
            acc, filled = 0, 0
            for g, c in zip(groups[:-1], Ci[:-1]):
                filled += len(g); acc += c
                e1, e2 = hi[filled - 1], lo[filled]
                if e2 - e1 > 1e-6:
                    ax.plot([e1, e2], [b, b], color=cmap(norm(np.clip(acc, -tmax, tmax))),
                            lw=1.6 if q < 10 else 1.0, solid_capstyle="butt")
    # print the integer inside the widest gaps of a few representative fluxes
    for (p, q) in [(1, 10), (1, 6), (2, 9), (5, 13)]:
        for (pp, qq) in [(p, q), (q - p, q)]:
            C, groups, _, _ = fhs(pp, qq, 12, 6 * qq)
            Ci = np.rint(C).astype(int)
            e0 = np.linalg.eigvalsh(H_grid(pp, qq, np.array([0.]), np.array([0.]))[0, 0])
            e1 = np.linalg.eigvalsh(H_grid(pp, qq, np.array([np.pi]), np.array([np.pi / qq]))[0, 0])
            lo, hi = np.minimum(e0, e1), np.maximum(e0, e1)
            acc, filled = 0, 0
            for g, c in zip(groups[:-1], Ci[:-1]):
                filled += len(g); acc += c
                ga, gb = hi[filled - 1], lo[filled]
                if gb - ga > 0.45:
                    ax.text(0.5 * (ga + gb), pp / qq, f"{acc:+d}", ha="center", va="center",
                            fontsize=7.5, fontweight="bold",
                            bbox=dict(facecolor="white", edgecolor="0.4", lw=0.4, pad=0.8,
                                      boxstyle="round,pad=0.15"))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, ticks=range(-tmax, tmax + 1), shrink=0.9)
    cb.set_label(r"gap Chern number $\Sigma_{j\leq r}C_j=\sigma_{xy}/(e^2/h)$"
                 "\n" r"(values beyond $\pm6$ drawn as $\pm6$)", fontsize=9)
    ax.set_xlim(-4.1, 4.1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"energy $\varepsilon$ (units of $|t|$)")
    ax.set_ylabel(r"flux per plaquette $b=p/q$")
    ax.set_title(rf"Butterfly as a phase diagram: black = subbands, colour = gaps ($q\leq{qmax}$)"
                 "\n" r"each gap carries one integer $\Sigma_{j\leq r}C_j$ (boxed: examples)",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    os.makedirs(IMG, exist_ok=True)
    checks()
    landau_uniformity()
    plot_curvature(1, 3, os.path.join(IMG, "fig_q4_berry_b1_3.png"))
    plot_curvature(2, 5, os.path.join(IMG, "fig_q4_berry_b2_5.png"))
    colored_butterfly(fname=os.path.join(IMG, "fig_q4_colored_butterfly.png"))
