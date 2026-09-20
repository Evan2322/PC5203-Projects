"""
moire_bands.py -- band structure of the twisted bilayer square lattice (Problem 1, Q5).

Model (see Sec. V of the report)
  * two identical square lattices (a = 1), layer 2 rotated by theta = 2r, tan r = n/m
  * intralayer nearest-neighbour hopping  -t0
  * interlayer hopping  t_perp(d) = t_perp0 * exp(-(d - a)/lam) * Theta(d_c - d),
    d = in-plane distance between the two sites   (t_perp0 is the value at d = a;
    at the coincidence site d = 0 the amplitude is t_perp0 * exp(a/lam))
  * Bloch matrix  H_ab(k) = sum_L t_ab(L) exp[i k.(L + r_b - r_a)]   (2*Sigma x 2*Sigma)

Usage:  python moire_bands.py            -> moire_bands.png, band_stats.txt
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0, TP0, LAM, DC, A = 1.0, 0.3, 0.5, 1.6, 1.0     # parameters used in the report


# ------------------------------------------------------------------ geometry
def csl(m, n):
    """Moire Bravais vectors T1,T2 and CSL index Sigma for tan r = n/m (coprime)."""
    u = complex(m, n)
    w = u / (1 + 1j) if (m % 2 == 1 and n % 2 == 1) else u   # both odd: divide by (1+i)
    T1 = np.array([w.real, w.imag])
    T2 = np.array([-w.imag, w.real])
    return T1, T2, int(round(abs(w) ** 2))


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def cell_sites(R, T1, T2, sigma):
    """Sites of the lattice R*Z^2 inside the half-open moire cell spanned by T1,T2."""
    M = np.column_stack([T1, T2])
    Minv = np.linalg.inv(M)
    span = int(np.ceil(np.abs(M).sum())) + 2
    pts = []
    for p in range(-span, span + 1):
        for q in range(-span, span + 1):
            r = R @ np.array([p, q], float)
            f = Minv @ r
            f = np.round(f, 9)
            if np.all(f >= 0) and np.all(f < 1 - 1e-9):
                pts.append(r)
    pts = np.array(pts)
    assert len(pts) == sigma, (len(pts), sigma)
    return pts


class Bilayer:
    def __init__(self, m, n, tperp0=TP0):
        self.m, self.n = m, n
        self.T1, self.T2, self.sigma = csl(m, n)
        self.theta = 2 * np.arctan(n / m)
        R = rot(self.theta)
        s = self.sigma
        r1 = cell_sites(np.eye(2), self.T1, self.T2, s)
        r2 = cell_sites(R, self.T1, self.T2, s)
        self.pos = np.vstack([r1, r2])                # 2*Sigma sites, layer 1 then 2
        self.B = 2 * s
        M = np.column_stack([self.T1, self.T2])
        Minv = np.linalg.inv(M)
        def key(f):
            g = np.round(np.round(f, 6) % 1.0, 6) % 1.0     # wrap into [0,1), avoid 1.0
            return tuple(g)
        lookup = [{}, {}]                                   # one table per layer
        for idx, r in enumerate(self.pos):
            lookup[idx // s][key(Minv @ r)] = idx
        rows, cols, dvec, amp = [], [], [], []
        # intralayer nearest-neighbour bonds (-t0)
        for layer, Rl in enumerate([np.eye(2), R]):
            for i in range(s):
                a = layer * s + i
                for d in (Rl @ [1, 0], Rl @ [-1, 0], Rl @ [0, 1], Rl @ [0, -1]):
                    p = self.pos[a] + d
                    j = lookup[layer][key(Minv @ p)]
                    rows.append(a); cols.append(j); dvec.append(d); amp.append(-T0)
        # interlayer bonds: t_perp(d), all lattice translations L within the cutoff
        Ls = np.array([[p, q] for p in range(-2, 3) for q in range(-2, 3)], float) @ M.T
        for i in range(s):
            a = i
            for j in range(s):
                b = s + j
                dd = self.pos[b][None, :] + Ls - self.pos[a][None, :]
                dist = np.linalg.norm(dd, axis=1)
                for v, dist_v in zip(dd[dist < DC], dist[dist < DC]):
                    t = tperp0 * np.exp(-(dist_v - A) / LAM)
                    rows.append(a); cols.append(b); dvec.append(v); amp.append(t)
                    rows.append(b); cols.append(a); dvec.append(-v); amp.append(t)
        self.rows = np.array(rows); self.cols = np.array(cols)
        self.dvec = np.array(dvec); self.amp = np.array(amp)
        self.tperp_scale = np.ones(len(self.amp))
        self.inter = np.array([(r < s) != (c < s) for r, c in zip(rows, cols)])
        # moire BZ
        self.b1 = 2 * np.pi * self.T1 / (self.T1 @ self.T1)
        self.b2 = 2 * np.pi * self.T2 / (self.T2 @ self.T2)

    def H(self, k, coupled=True):
        h = np.zeros((self.B, self.B), complex)
        amp = self.amp if coupled else np.where(self.inter, 0.0, self.amp)
        np.add.at(h, (self.rows, self.cols), amp * np.exp(1j * (self.dvec @ k)))
        return h

    def bands(self, k, coupled=True):
        return np.linalg.eigvalsh(self.H(k, coupled))

    def path(self, npts=60):
        G = np.zeros(2); X = self.b1 / 2; M = (self.b1 + self.b2) / 2
        seg = [(G, X), (X, M), (M, G)]
        ks, xs, ticks, x0 = [], [], [0.0], 0.0
        for a, b in seg:
            t = np.linspace(0, 1, npts, endpoint=False)
            ks += [a + (b - a) * ti for ti in t]
            xs += list(x0 + t * np.linalg.norm(b - a))
            x0 += np.linalg.norm(b - a)
            ticks.append(x0)
        ks.append(G); xs.append(x0)
        return np.array(ks), np.array(xs), ticks


def path_bands(bl, coupled=True, npts=60):
    ks, xs, ticks = bl.path(npts)
    E = np.array([bl.bands(k, coupled) for k in ks])
    return xs, ticks, E


def grid_bands(bl, coupled=True, ng=12):
    """Bands on an ng x ng grid of the moire BZ (for statistics)."""
    fr = (np.arange(ng) + 0.5) / ng - 0.5
    E = []
    for u in fr:
        for v in fr:
            E.append(bl.bands(u * bl.b1 + v * bl.b2, coupled))
    return np.array(E)                                   # (nk, B)


def stats(E, delta):
    """fraction of states with |E|<delta; mean width of the bands whose centre lies there."""
    frac = np.mean(np.abs(E) < delta)
    centre = E.mean(axis=0)
    width = E.max(axis=0) - E.min(axis=0)
    sel = np.abs(centre) < delta
    return frac, (width[sel].mean() if sel.any() else np.nan), sel.sum()


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    pairs = [(2, 1), (4, 1), (6, 1), (8, 1), (12, 1), (16, 1)]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4), sharey=True)
    lines = ["m  Sigma    B   2r(deg)  B*theta^2 | frac(|E|<0.5) coupled/decoupled | "
             "frac(|E|<1.0) coupled/decoupled | mean width (bands centred in |E|<1) coupled/decoupled | "
             "width of highest band [its energy] coupled"]
    for ax, (m, n) in zip(axes.flat, pairs):
        bl = Bilayer(m, n)
        xs, ticks, E = path_bands(bl, True, npts=60 if bl.B < 200 else 40)
        ax.axhspan(-0.5, 0.5, color="tab:orange", alpha=0.18, lw=0)
        ax.axhline(0, color="tab:orange", lw=0.6, alpha=0.8)
        lw = 0.8 if bl.B < 100 else (0.55 if bl.B < 300 else 0.4)
        ax.plot(xs, E, color="tab:blue", lw=lw)
        for t in ticks:
            ax.axvline(t, color="gray", lw=0.4, ls="--")
        ax.set_xticks(ticks); ax.set_xticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$"])
        ax.set_xlim(0, ticks[-1]); ax.set_ylim(-7.5, 4.8)
        th = np.degrees(bl.theta)
        ax.set_title(f"$(m,n)=({m},{n})$,  $2r={th:.2f}^\\circ$,  $B={bl.B}$", fontsize=10)
        Eg_c = grid_bands(bl, True, ng=10)
        Eg_d = grid_bands(bl, False, ng=10)
        f5c, _, _ = stats(Eg_c, 0.5); f5d, _, _ = stats(Eg_d, 0.5)
        f1c, w1c, _ = stats(Eg_c, 1.0); f1d, w1d, _ = stats(Eg_d, 1.0)
        Wtop = Eg_c[:, -1].max() - Eg_c[:, -1].min(); Etop = Eg_c[:, -1].mean()
        lines.append(f"{m:2d} {bl.sigma:5d} {bl.B:4d}  {th:7.2f}  {bl.B*bl.theta**2:8.2f} | "
                     f"{f5c:.3f} / {f5d:.3f} | {f1c:.3f} / {f1d:.3f} | {w1c:.3f} / {w1d:.3f} | "
                     f"{Wtop:.2e} [{Etop:.2f}]")
        print(lines[-1], flush=True)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$E/t_0$")
    for ax, lab in zip(axes.flat, "abcdef"):
        # panel label above the axes, flush left (the bands would cover a label inside)
        ax.set_title(f"({lab})", loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig("moire_bands.png", dpi=220)
    open("band_stats.txt", "w").write("\n".join(lines) + "\n")
