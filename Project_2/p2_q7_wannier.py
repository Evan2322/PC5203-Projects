"""
PC5203 Project 2, Problem 2, Q7 -- Wannier (hybrid Wannier-centre) polarization as a route to
the Chern number.

(A) k-space Wilson loops.  For each k_y, the Wilson loop along k_x over the magnetic BZ
    [0, 2pi/q) of a band (or of a group of bands) gives eigenphases gamma_i(k_y); the
    hybrid Wannier centres (localised in x, Bloch-like in y) are
        xbar_i(k_y) = (q / 2pi) * gamma_i(k_y)          (in lattice sites, modulo q).
    With the standard Berry curvature Omega = curl_k i<u|grad_k u> one has
        d xbar / d k_y = -(q / 2pi) * integral Omega dk_x ,   hence
        C = -[sum xbar(2pi) - sum xbar(0)] / q      (continuous unwrapping):
    the centres of a band with Chern number C move by -C magnetic cells per k_y period.
    Links use the position-including Bloch convention (as in p2_q4_chern.py).
(B) Real-space (Resta) version, for systems with no translation symmetry in x.
    Ring of L sites in x (b L integer), k_y still a good quantum number, plus an
    x-dependent random potential V_x in [-W/2, W/2].  With the lowest N_occ eigenvectors
    V(k_y) of H(k_y), the projected position operator  M = V^dag exp(i 2pi x/L) V  has
    eigenphases 2pi xbar_i / L.  The winding of arg det M over k_y in [0, 2pi] (in units
    of 2pi) equals the shift of sum xbar in units of L, i.e. -(Chern number of the
    occupied subspace).  No magnetic Bloch theorem in x
    (hence no FHS on a magnetic BZ) is needed.

Checks performed (all printed):
  (a) atomic-limit test: a band localised on site l0 of the cell has xbar = l0
      (asymmetric sites l0 = 1, 2 in a q = 5 cell, so the sign of xbar is tested);
  (e) independent sign check: Chern numbers from the Kubo formula
      Omega_n = -2 Im sum_m <n|d_x H|m><m|d_y H|n> / (E_m-E_n)^2 with the explicit
      position-including Bloch Hamiltonian agree with FHS (Q4) and the Wilson loops;
  (b) winding of per-band xbar(k_y) equals C_n of Q4 for every band, b = 1/3, 2/5, 1/5,
      2/7, 3/7, 3/8 (touching central bands of q = 8 treated as one group);
  (c) real-space winding for the clean ring equals -sum_{j<=r} C_j (several gaps);
  (d) disorder: winding of the lowest gap of b = 1/3 (clean value +1 = -C) vs W
      (20 realisations per W),
      together with the smallest gap above the occupied states.
Figures: images/fig_q7_wannier.png
"""
import numpy as np
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = next((d for d in (HERE, os.path.dirname(HERE))
             if os.path.isfile(os.path.join(d, "p2_q4_chern.py"))
             and os.path.isfile(os.path.join(d, "p2_q2_butterfly_square.py"))), None)
if ROOT is None:
    sys.exit("p2_q4_chern.py / p2_q2_butterfly_square.py not found in this script's "
             "folder or its parent folder.")
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True     # importing Q2/Q4 must not leave a __pycache__ folder behind
from p2_q4_chern import H_grid, fhs, band_groups
IMG = os.path.join(ROOT, "images")


# ---------------------------------------------------------------- (A) k-space Wilson loops
def wilson_centres(Hfun, q, groups, Nth, Nky):
    """Hybrid Wannier centres xbar[group][ky_index, i] (sites, in [0,q)) from Wilson loops
    along theta = q k_x for every k_y in [0, 2pi]. Hfun(th_array, ky_array) -> (Nth,Nky,q,q)."""
    th = 2 * np.pi * np.arange(Nth) / Nth
    ky = np.linspace(0, 2 * np.pi, Nky + 1)
    E, V = np.linalg.eigh(Hfun(th, ky))                 # (Nth, Nky+1, q, q)
    Dx = np.exp(1j * (2 * np.pi / Nth / q) * np.arange(q))[:, None]
    out = []
    for g in groups:
        U = V[..., :, g]                                # (Nth, Nky+1, q, m)
        W = np.broadcast_to(np.eye(len(g), dtype=complex), (Nky + 1, len(g), len(g))).copy()
        for i in range(Nth):
            A = U[i]; B = Dx * U[(i + 1) % Nth]
            M = np.einsum("kia,kib->kab", A.conj(), B)
            W = W @ M
        gam = np.angle(np.linalg.eigvals(W))            # (Nky+1, m)
        xb = np.mod(q * gam / (2 * np.pi), q)
        out.append(np.sort(xb, axis=1))
    return ky, out


def winding_from_det(phase_sum, strict=True):
    """phase_sum(k_y) sampled on [0,2pi] (radians, any branch); returns winding / 2pi.
    A step larger than 2.5 rad means the phase is not resolved (or the occupied
    subspace is ill-defined because the gap closed): error if strict, else NaN."""
    d = np.diff(phase_sum)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    if np.abs(d).max() >= 2.5:
        if strict:
            raise RuntimeError("refine the k_y grid")
        return np.nan
    return d.sum() / (2 * np.pi)


def wilson_winding(Hfun, q, groups, Nth, Nky):
    """Chern number of each group = -(1/2pi) winding of total Berry phase sum_i gamma_i."""
    th = 2 * np.pi * np.arange(Nth) / Nth
    ky = np.linspace(0, 2 * np.pi, Nky + 1)
    E, V = np.linalg.eigh(Hfun(th, ky))
    Dx = np.exp(1j * (2 * np.pi / Nth / q) * np.arange(q))[:, None]
    res = []
    for g in groups:
        U = V[..., :, g]
        detW = np.ones(Nky + 1, complex)
        for i in range(Nth):
            M = np.einsum("kia,kib->kab", U[i].conj(), Dx * U[(i + 1) % Nth])
            detW *= np.linalg.det(M)
        # xbar_total = -(q/2pi) arg detW ; winding in units of q  ->  C = -wind(arg detW)/2pi
        res.append(-winding_from_det(np.angle(detW)))
    return np.array(res)


# ---------------------------------------------------------------- (B) real-space (Resta)
def H_ring(L, b, ky, V=None):
    H = np.diag(2 * np.cos(2 * np.pi * b * np.arange(L) - ky)).astype(complex)
    if V is not None:
        H += np.diag(V)
    for x in range(L):
        H[(x + 1) % L, x] += 1.0
        H[x, (x + 1) % L] += 1.0
    return H


def resta_winding(L, b, Nocc, Nky=240, V=None):
    X = np.exp(1j * 2 * np.pi * np.arange(L) / L)[:, None]
    ky = np.linspace(0, 2 * np.pi, Nky + 1)
    ph, gap = [], np.inf
    for k in ky:
        e, v = np.linalg.eigh(H_ring(L, b, k, V))
        gap = min(gap, e[Nocc] - e[Nocc - 1])
        Vo = v[:, :Nocc]
        ph.append(np.angle(np.linalg.det(Vo.conj().T @ (X * Vo))))
    # det M ~ exp(i 2pi sum xbar / L): returns the shift of sum xbar in units of L (= -C)
    # a closed gap makes the winding ill-defined -> NaN (counted as "not quantised")
    return winding_from_det(np.array(ph), strict=(V is None)), gap


# ---------------------------------------------------------------- checks
def checks():
    # (a) atomic limit: deep well on site l0 of a q = 5 cell, weak hopping
    q = 5
    for l0 in (1, 2):
        def Hatom(th, ky, l0=l0):
            H = H_grid(1, q, th, ky) * 0.05
            H[..., l0, l0] += -10.0
            return H
        ky, xb = wilson_centres(Hatom, q, [[0]], 40, 8)
        print(f"(a) atomic limit: xbar = {xb[0].mean():.4f} (expected {l0})")

    # (b) per-band windings vs Q4
    allok = True
    for (p, q) in [(1, 3), (2, 5), (1, 5), (2, 7), (3, 7), (3, 8)]:
        groups = band_groups(p, q)
        Hfun = lambda th, ky, p=p, q=q: H_grid(p, q, th, ky)
        w = wilson_winding(Hfun, q, groups, 24, 12 * q)
        C, _, _, _ = fhs(p, q, 16, 8 * q)
        ok = np.allclose(w, np.rint(C), atol=1e-8)
        allok &= ok
        print(f"(b) b={p}/{q}: Wilson-loop windings {np.rint(w).astype(int)} | FHS {np.rint(C).astype(int)}"
              f" | {'OK' if ok else 'MISMATCH'}")

    # (c) clean real-space ring
    for (p, q, r) in [(1, 3, 1), (1, 3, 2), (2, 5, 1), (2, 5, 2), (3, 7, 2)]:
        M = 8
        L = M * q
        w, gap = resta_winding(L, p / q, r * M)
        C, _, _, _ = fhs(p, q, 16, 8 * q)
        target = -int(np.rint(C[:r]).sum())
        print(f"(c) ring L={L}, b={p}/{q}, r={r}: real-space winding {w:+.6f}, "
              f"-sum C = {target:+d} | {'OK' if abs(w - target) < 1e-8 else 'MISMATCH'}")
    # (e) Kubo-formula sign check (odd q: all gaps open)
    for (p, q) in [(1, 3), (2, 5)]:
        Ck = kubo_chern(p, q)
        C, _, _, _ = fhs(p, q, 16, 8 * q)
        print(f"(e) b={p}/{q}: Kubo C = {Ck.round(4)} | FHS {np.rint(C).astype(int)}")
    return allok


def kubo_chern(p, q, N1=40, N2f=40):
    """Chern numbers from the Kubo formula, explicit Bloch Hamiltonian with site positions
    x = l:  (H u)_l = e^{i k_x} u_{l+1} + e^{-i k_x} u_{l-1} + 2cos(2 pi p l/q - k_y) u_l."""
    def HI(kx, ky):
        l = np.arange(q)
        H = np.diag(2 * np.cos(2 * np.pi * p * l / q - ky)).astype(complex)
        for i in range(q):
            j = (i + 1) % q
            H[i, j] += np.exp(1j * kx); H[j, i] += np.exp(-1j * kx)
        return H
    dHx = lambda kx, ky: (HI(kx + 1e-6, ky) - HI(kx - 1e-6, ky)) / 2e-6
    dHy = lambda kx, ky: (HI(kx, ky + 1e-6) - HI(kx, ky - 1e-6)) / 2e-6
    N2 = N2f * q
    kxs = (np.arange(N1) + 0.5) * 2 * np.pi / q / N1
    kys = (np.arange(N2) + 0.5) * 2 * np.pi / N2
    C = np.zeros(q)
    for kx in kxs:
        for ky in kys:
            E, V = np.linalg.eigh(HI(kx, ky))
            X = V.conj().T @ dHx(kx, ky) @ V; Y = V.conj().T @ dHy(kx, ky) @ V
            dE = E[None, :] - E[:, None]; np.fill_diagonal(dE, np.inf)
            C += -2 * np.imag(np.sum(X * Y.T / dE ** 2, axis=1))
    return C * (2 * np.pi / q / N1) * (2 * np.pi / N2) / (2 * np.pi)


def disorder_scan(p=1, q=3, r=1, M=12, Ws=np.linspace(0, 4, 17), nreal=20, seed=1):
    rng = np.random.default_rng(seed)
    L = M * q
    rows = []
    for W in Ws:
        ws, gs = [], []
        for _ in range(nreal):
            V = W * (rng.random(L) - 0.5)
            w, g = resta_winding(L, p / q, r * M, Nky=360, V=V)
            ws.append(w); gs.append(g)
        ws = np.array(ws); gs = np.array(gs)
        rows.append((W, np.mean(np.isclose(ws, +1, atol=1e-6)), np.mean(np.isnan(ws)),
                     np.median(gs), gs.min()))
        other = np.sum(~np.isnan(ws) & ~np.isclose(ws, +1, atol=1e-6))
        print(f"(d) W={W:4.2f}: winding +1: {rows[-1][1]:.2f}, undefined (gap closed): "
              f"{rows[-1][2]:.2f}, other integers: {other}, median gap {np.median(gs):.3f}, "
              f"min gap {gs.min():.3f}")
    return np.array(rows)


def plot(rows, fname):
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)
    cols = ["#2a6fbb", "#d1495b", "#3a9e5c", "#e0a030", "#7a5195"]
    # (a) b = 1/3, all bands
    p, q = 1, 3
    ky, xb = wilson_centres(lambda th, k: H_grid(p, q, th, k), q, band_groups(p, q), 48, 300)
    C, _, _, _ = fhs(p, q, 16, 8 * q)
    for n in range(q):
        axs[0].plot(ky, xb[n][:, 0], ".", ms=1.6, color=cols[n],
                    label=rf"band {n+1}, $C={int(np.rint(C[n])):+d}$")
    axs[0].set_title(r"(a) hybrid Wannier centres, $b=1/3$", fontsize=10)
    axs[0].set_ylabel(r"$\bar x(k_y)$ (sites)"); axs[0].set_ylim(0, q)
    axs[0].legend(fontsize=7.5, loc="upper center", markerscale=5, frameon=False, ncol=3,
                  bbox_to_anchor=(0.5, -0.17))
    # (b) b = 2/5, lowest two bands
    p, q = 2, 5
    ky, xb = wilson_centres(lambda th, k: H_grid(p, q, th, k), q, [[0], [1]], 48, 400)
    C, _, _, _ = fhs(p, q, 16, 8 * q)
    for n in range(2):
        axs[1].plot(ky, xb[n][:, 0], ".", ms=1.4, color=cols[n],
                    label=rf"band {n+1}, $C={int(np.rint(C[n])):+d}$")
    axs[1].set_title(r"(b) hybrid Wannier centres, $b=2/5$", fontsize=10)
    axs[1].set_ylabel(r"$\bar x(k_y)$ (sites)"); axs[1].set_ylim(0, q)
    axs[1].legend(fontsize=7.5, loc="upper center", markerscale=5, frameon=False, ncol=2,
                  bbox_to_anchor=(0.5, -0.17))
    for ax in axs[:2]:
        ax.set_xlim(0, 2 * np.pi); ax.set_xlabel(r"$k_y$")
        ax.set_xticks([0, np.pi, 2 * np.pi]); ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    # (c) disorder
    W, frac, fnan, gmed, gmin = rows.T
    axs[2].plot(W, frac, "o-", color=cols[0], ms=4, label=r"fraction with winding $=+1$ ($=-C$)")
    ax2 = axs[2]
    ax2.plot(W, gmed / gmed[0], "s--", color=cols[1], ms=3.5, label="median gap / clean gap")
    ax2.set_xlabel(r"disorder strength $W$ (units of $|t|$)")
    ax2.set_ylim(-0.05, 1.08)
    ax2.set_title(r"(c) real-space winding, $b=1/3$, lowest gap, $L=36$", fontsize=10)
    ax2.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    os.makedirs(IMG, exist_ok=True)
    checks()
    rows = disorder_scan()
    plot(rows, os.path.join(IMG, "fig_q7_wannier.png"))
