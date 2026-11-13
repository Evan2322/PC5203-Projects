"""
PC5203 Project 2, Problem 2, Q8 -- magnetic field, Berry curvature and lattice structure.
Numerical support for the discussion: the SAME weak magnetic field produces two different
kinds of Landau levels, depending on the band geometry that the lattice provides.

(i)  Band bottom (eps -> -4, b -> 0): parabolic band, Berry phase 0 of the cyclotron orbit
     -> ordinary Landau levels eps_n = -4 + 4 pi b (n + 1/2), each a band with C = -1.
(ii) Band centre near half flux (b -> 1/2): the b = 1/2 lattice has two Dirac cones
     (eps = +-2 sqrt(cos^2 kx + cos^2 ky), velocity v = 2) carrying Berry phase pi.
     An extra flux delta = |b - 1/2| per plaquette (l_B^2 = 1/(2 pi delta)) gives
     relativistic Landau levels  eps_n = +- v sqrt(2 n / l_B^2) = +- 4 sqrt(pi n delta),
     including a zero mode n = 0; each level is a band with |C| = 2 (two cones).
The offset (n + 1/2 vs n) is the Berry phase in the Onsager-Roth rule
     S(eps_n) l_B^2 = 2 pi (n + 1/2 - gamma_B / 2 pi).

Checks performed (all printed):
  (a) bottom: centres of the lowest 3 subbands for b = 1/q vs eps_n, and their C;
  (b) centre: for b = M/(2M+1) (delta = 1/(2(2M+1))), the n = 0 band sits at eps = 0 and
      the ratio (n = 1, 2 band centre) / (4 sqrt(pi n delta)) -> 1 as delta -> 0; C of the
      central bands;
  (c) Berry phase of the lower b = 1/2 band around a small loop enclosing one Dirac point
      (= pi), and around a loop enclosing none (= 0).
Figure: images/fig_q8_landau_dirac.png
"""
import numpy as np
import os
import sys
from math import gcd
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
from p2_q4_chern import fhs
from p2_q2_butterfly_square import band_edges
IMG = os.path.join(ROOT, "images")


def checks():
    print("(a) band bottom, b = 1/q: subband centres vs -4 + 4 pi b (n+1/2); Chern numbers")
    for q in [10, 20, 40]:
        c = band_edges(1, q).mean(1)[:3]
        ll = -4 + 4 * np.pi / q * (np.arange(3) + 0.5)
        C, _, _, _ = fhs(1, q, 12, 6 * q)
        print(f"    b=1/{q}: centres {c.round(4)}  LL {ll.round(4)}  C = {np.rint(C[:3]).astype(int)}")

    print("(b) half flux, b = M/(2M+1): Dirac Landau levels +-4 sqrt(pi n delta)")
    for M in [5, 10, 20, 40]:
        p, q = M, 2 * M + 1
        d = 0.5 - p / q
        c = band_edges(p, q).mean(1)
        mid = q // 2                                   # index of the band at eps = 0
        r1 = c[mid + 1] / (4 * np.sqrt(np.pi * d))
        r2 = c[mid + 2] / (4 * np.sqrt(2 * np.pi * d))
        Cs = ""
        if q <= 41:
            C, _, _, _ = fhs(p, q, 12, 6 * q)
            Cs = f"  C(n=0,+-1,+-2) = {np.rint(C[mid-2:mid+3]).astype(int)}"
        print(f"    b={p}/{q}, delta={d:.4f}: eps(n=0)={c[mid]:+.1e}, "
              f"ratio n=1: {r1:.4f}, ratio n=2: {r2:.4f}{Cs}")

    print("(c) Berry phase of the lower b=1/2 band around small loops")
    # position-including (physical) Bloch Hamiltonian of the q = 2 magnetic cell, as in the
    # Kubo check of p2_q7_wannier.py; with the "all sites at x = 0" convention the loop
    # phase around a Dirac point is not exactly pi.
    def HI(kx, ky):
        # (H u)_l = e^{i k_x} u_{l+1} + e^{-i k_x} u_{l-1} + 2cos(pi l - k_y) u_l,  l = 0, 1
        H = np.diag(2 * np.cos(np.pi * np.arange(2) - ky)).astype(complex)
        H[0, 1] = 2 * np.cos(kx)      # u_1 enters row 0 via both l+1 and l-1 (= l+1-2)
        H[1, 0] = 2 * np.cos(kx)
        return H

    def berry_loop(k0, rad, N=400):
        ts = np.linspace(0, 2 * np.pi, N, endpoint=False)
        us = [np.linalg.eigh(HI(k0[0] + rad * np.cos(t), k0[1] + rad * np.sin(t)))[1][:, 0]
              for t in ts]
        prod = np.prod([np.vdot(us[i], us[(i + 1) % N]) for i in range(N)])
        return -np.angle(prod)
    for k0, lab in [((np.pi / 2, np.pi / 2), "around Dirac point (pi/2, pi/2)"),
                    ((-np.pi / 2, np.pi / 2), "around Dirac point (-pi/2, pi/2)"),
                    ((0.0, 0.0), "around (0,0), no Dirac point")]:
        g = berry_loop(k0, 0.05)
        print(f"    {lab}: gamma_B = {g:+.4f}  (|gamma_B|/pi = {abs(g)/np.pi:.4f})")


def subbands(bmin, bmax, emin, emax, qmax):
    """(b, lo, hi) of every subband of every coprime p/q (q <= qmax) inside the window."""
    out = []
    for q in range(1, qmax + 1):
        for p in range(0, q + 1):
            if gcd(p, q) != 1 or not (bmin <= p / q <= bmax):
                continue
            for lo, hi in band_edges(p, q):
                if hi >= emin and lo <= emax:
                    out.append((p / q, lo, hi))
    return np.array(out)


def draw(ax, sb):
    for b, lo, hi in sb:
        ax.plot([lo, hi], [b, b], color="k", lw=0.7, solid_capstyle="butt")
    ax.plot(0.5 * (sb[:, 1] + sb[:, 2]), sb[:, 0], "k.", ms=1.6)


def plot(fname):
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.9), constrained_layout=True)
    blue, red = "#2a6fbb", "#d1495b"
    # (a) band bottom, small b
    ax = axs[0]
    bmax, emin, emax = 0.05, -4.0, -3.2
    draw(ax, subbands(0.0, bmax, emin, emax, 160))
    b = np.linspace(0, bmax, 200)
    for n in range(4):
        e = -4 + 4 * np.pi * b * (n + 0.5)
        ax.plot(e, b, color=blue, lw=1.3, ls="--")
        bt = min(bmax, (emax + 0.02 - (-4)) / (4 * np.pi * (n + 0.5)))   # label where the line leaves
        ax.text(-4 + 4 * np.pi * bt * (n + 0.5) - 0.015, bt - 0.001, rf"$n={n}$",
                color=blue, fontsize=9, ha="right", va="top",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    ax.set_xlim(emin, emax); ax.set_ylim(0, bmax)
    ax.set_xlabel(r"energy $\varepsilon$"); ax.set_ylabel(r"flux per plaquette $b=p/q$")
    ax.set_title(r"(a) band bottom: ordinary Landau levels", fontsize=10.5)
    ax.text(0.97, 0.04,
            "parabolic band, Berry phase $\\gamma_B=0$\n"
            r"$\varepsilon_n=-4+4\pi b\,(n+\frac{1}{2})$: equal spacing, offset $\frac{1}{2}$" "\n"
            r"every level (subband) has $C=-1$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="0.7", pad=3))
    # (b) near half flux
    ax = axs[1]
    bmin, emax = 0.45, 1.3
    draw(ax, subbands(bmin, 0.5, -emax, emax, 160))
    b = np.linspace(bmin, 0.5, 400)
    d = 0.5 - b
    for n in range(4):
        e = 4 * np.sqrt(np.pi * n * d)
        ax.plot(e, b, color=red, lw=1.3, ls="--")
        ax.plot(-e, b, color=red, lw=1.3, ls="--")
        if n > 0:
            bt = max(bmin, 0.5 - (emax - 0.03) ** 2 / (16 * np.pi * n))
            et = 4 * np.sqrt(np.pi * n * (0.5 - bt))
            kw = dict(color=red, fontsize=9, va="bottom",
                      bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
            ax.text(et - 0.06, bt + 0.0015, rf"$n={n}$", ha="right", **kw)
            ax.text(-et + 0.06, bt + 0.0015, rf"$n=-{n}$", ha="left", **kw)
    ax.text(0.03, bmin + 0.002, r"$n=0$" "\n(zero mode)", color=red, fontsize=9, ha="left")
    ax.set_xlim(-emax, emax); ax.set_ylim(bmin, 0.5)
    ax.set_xlabel(r"energy $\varepsilon$"); ax.set_ylabel(r"flux per plaquette $b=p/q$")
    ax.set_title(r"(b) near half flux $b=1/2$: Dirac Landau levels", fontsize=10.5)
    ax.text(0.27, 0.03,
            "two Dirac cones at $b=1/2$,\nBerry phase $\\gamma_B=\\pi$\n"
            r"$\varepsilon_n=\pm4\sqrt{\pi n\,|b-\frac{1}{2}|}$" "\n"
            r"$\sqrt{n}$ spacing, zero mode" "\n"
            r"every level has $C=+2$" "\n(one per cone)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="0.7", pad=3))
    handles = [Line2D([], [], color="k", marker=".", lw=0.8,
                      label=r"numerical subbands, all $p/q$ with $q\leq160$ (segment + centre dot)"),
               Line2D([], [], color=blue, ls="--", lw=1.3, label=r"ordinary Landau levels, $n=0..3$"),
               Line2D([], [], color=red, ls="--", lw=1.3, label=r"Dirac Landau levels, $n=0,\pm1,\pm2,\pm3$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle("Same weak field, two kinds of Landau levels: the zero-field Berry phase of the "
                 "lattice band decides which", fontsize=10.5)
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    os.makedirs(IMG, exist_ok=True)
    checks()
    plot(os.path.join(IMG, "fig_q8_landau_dirac.png"))
