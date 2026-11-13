"""
PC5203 Project 2, Problem 2, Q2 -- rational flux b = p/q, magnetic Bloch bands and the
Hofstadter butterfly of the square lattice.

Magnetic Bloch Hamiltonian (q x q), from the Harper equation with psi_{l+q} = e^{i theta} psi_l,
theta = q k_x  (k_x in [0, 2pi/q)), and k = k_y:
    H_ll       = 2 cos(2 pi p l / q - k_y),  l = 0..q-1
    H_{l,l+1}  = 1 ;  corner H_{q-1,0} = e^{i theta}  (+ h.c.)

Checks performed (all printed):
  (a) b = 0     : eps = 2cos kx + 2cos ky ;
  (b) b = 1/2   : eps = +-2 sqrt(cos^2 kx + cos^2 ky) ;
  (c) Chambers relation  det(eps - H) = P_{p/q}(eps) - 2 [cos(q kx) + cos(q ky)],
      with P independent of k (tested at random eps, k);
  (d) band edges = eigenvalues at (q kx, q ky) = (0,0) and (pi,pi), compared with dense
      k-sampling of the magnetic Brillouin zone;
  (e) small flux b = 1/q: lowest subbands approach Landau levels
      eps_n = -4 + 4 pi b (n + 1/2)  (effective mass m* = 1/2 near the band bottom).
Figure: images/fig_q2_butterfly_square.png
"""
import numpy as np
from math import gcd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def H_mag(p, q, kx, ky):
    l = np.arange(q)
    H = np.diag(2 * np.cos(2 * np.pi * p * l / q - ky)).astype(complex)
    th = q * kx
    for i in range(q):              # bonds l -> l+1, the last one wraps with phase e^{i theta}
        j = (i + 1) % q
        ph = np.exp(1j * th) if j == 0 else 1.0
        H[j, i] += ph
        H[i, j] += np.conj(ph)
    return H


def band_edges(p, q):
    """Return (q,2) array of [lower, upper] edges of the q magnetic subbands."""
    e1 = np.linalg.eigvalsh(H_mag(p, q, 0.0, 0.0))
    e2 = np.linalg.eigvalsh(H_mag(p, q, np.pi / q, np.pi / q))
    return np.sort(np.stack([e1, e2], 1), 1)


def checks():
    rng = np.random.default_rng(0)
    # (a) b = 0
    kx, ky = rng.uniform(-np.pi, np.pi, 2)
    e = np.linalg.eigvalsh(H_mag(0, 1, kx, ky))[0]
    print(f"(a) b=0   : |eps - (2cos kx+2cos ky)| = {abs(e - 2*np.cos(kx) - 2*np.cos(ky)):.1e}")
    # (b) b = 1/2   (magnetic BZ: kx in [0,pi); our kx is the original crystal momentum)
    e = np.linalg.eigvalsh(H_mag(1, 2, kx, ky))
    ex = 2 * np.sqrt(np.cos(kx) ** 2 + np.cos(ky) ** 2)
    print(f"(b) b=1/2 : eigenvalues {e.round(6)}, analytic +-{ex:.6f}")
    # (c) Chambers relation
    worst = 0
    for (p, q) in [(1, 3), (2, 5), (3, 7), (5, 12)]:
        for _ in range(5):
            eps = rng.uniform(-4, 4)
            vals = []
            for _ in range(6):
                kx, ky = rng.uniform(-np.pi, np.pi, 2)
                d = np.linalg.det(eps * np.eye(q) - H_mag(p, q, kx, ky)).real
                vals.append(d + 2 * (np.cos(q * kx) + np.cos(q * ky)))
            worst = max(worst, np.ptp(vals))
    print(f"(c) Chambers: spread of det + 2[cos q kx + cos q ky] over k = {worst:.1e}")
    # (d) band edges vs dense sampling
    worst = 0
    for (p, q) in [(1, 3), (2, 5), (1, 4), (3, 8)]:
        n = 8 * q                                   # grid contains k = pi/q exactly
        ks = np.arange(n + 1) * 2 * np.pi / n
        E = np.array([np.linalg.eigvalsh(H_mag(p, q, kx / q, ky)) for kx in ks for ky in ks])
        dense = np.stack([E.min(0), E.max(0)], 1)
        worst = max(worst, np.abs(dense - band_edges(p, q)).max())
    print(f"(d) band edges (2 k-points) vs dense (8q+1)^2 sampling: max deviation = {worst:.1e}")


def landau_check():
    for q in [20, 50, 100]:
        b = 1 / q
        c = band_edges(1, q).mean(1)[:3]
        ll = -4 + 4 * np.pi * b * (np.arange(3) + 0.5)
        print(f"(e) b=1/{q}: subband centres {c.round(4)}, Landau levels {ll.round(4)}")


def butterfly(qmax=60, fname=None):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    for q in range(1, qmax + 1):
        for p in range(0, q + 1):
            if gcd(p, q) != 1:
                continue
            b = p / q
            for lo, hi in band_edges(p, q):
                ax.plot([lo, hi], [b, b], color="k", lw=0.5 if q < 20 else 0.35,
                        solid_capstyle="butt")
    # guide marks
    ax.axhline(1 / 3, color="#2a6fbb", lw=1.0, ls="--", alpha=0.9)
    ax.text(3.95, 1 / 3 + 0.012, r"$b=1/3$: 3 subbands, see (b)", color="#2a6fbb",
            fontsize=8, ha="right", bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax.annotate(r"$b=1/2$: two Dirac cones" "\n" r"(bands touch at $\varepsilon=0$)",
                xy=(0, 0.5), xytext=(1.3, 0.53), fontsize=8, color="#d1495b",
                arrowprops=dict(arrowstyle="->", color="#d1495b", lw=0.8),
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax.annotate("Landau-level fans\n(small $b$, band edges)", xy=(-3.28, 0.125),
                xytext=(-3.95, 0.25), fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8),
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax.text(-3.95, 0.975, r"symmetric under $\varepsilon\to-\varepsilon$" "\n" r"and $b\to1-b$",
            ha="left", va="top", fontsize=8, color="0.3")
    ax.set_xlabel(r"energy $\varepsilon$ (units of $|t|$)")
    ax.set_ylabel(r"flux per plaquette $b=p/q$ (units of $\Phi_0$)")
    ax.set_xlim(-4.1, 4.1); ax.set_ylim(0, 1)
    ax.set_title(r"(a) Hofstadter butterfly: $q$ subbands at every $b=p/q$, $q\leq%d$" % qmax,
                 fontsize=10)

    # (b) Chambers polynomial for b = 1/3
    p, q = 1, 3
    e = np.linspace(-4.2, 4.2, 2000)
    P = np.array([np.linalg.det(x * np.eye(q) - H_mag(p, q, 0, 0)).real
                  + 4 for x in e])
    ax2.axhspan(-4, 4, color="#2a6fbb", alpha=0.07, zorder=0)
    ax2.plot(e, P, color="#2a6fbb", lw=1.6)
    ax2.axhline(4, color="0.4", ls="--", lw=1); ax2.axhline(-4, color="0.4", ls="--", lw=1)
    for n, (lo, hi) in enumerate(band_edges(p, q)):
        ax2.axvspan(lo, hi, color="0.8", zorder=0)
        ax2.text(0.5 * (lo + hi), -11.3, f"band {n+1}", ha="center", fontsize=8)
    ax2.set_ylim(-12, 12); ax2.set_xlim(-4.2, 4.2)
    ax2.set_xlabel(r"energy $\varepsilon$"); ax2.set_ylabel(r"$P_{1/3}(\varepsilon)=\varepsilon^3-6\varepsilon$")
    ax2.text(-4.1, 4.5, r"$P=+4$", fontsize=8, color="0.3")
    ax2.text(-4.1, -5.4, r"$P=-4$", fontsize=8, color="0.3")
    ax2.set_title(r"(b) allowed energies at $b=1/3$: $|P_{1/3}(\varepsilon)|\leq4$", fontsize=10)
    ax2.legend(handles=[Line2D([], [], color="#2a6fbb", lw=1.6, label=r"Chambers polynomial $P_{1/3}$"),
                        Patch(color="0.8", label=r"subbands (where $|P|\leq4$)")],
               loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    if fname is None:
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here) if os.path.basename(here).lower() in ("images", "codes") else here
        os.makedirs(os.path.join(root, "images"), exist_ok=True)
        fname = os.path.join(root, "images", "fig_q2_butterfly_square.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    checks()
    landau_check()
    print("b=1/3 band edges:\n", band_edges(1, 3).round(4))
    butterfly()
