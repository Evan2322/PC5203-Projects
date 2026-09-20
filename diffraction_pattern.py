"""
diffraction_pattern.py
-----------------------
Project 1, Problem 1, Sub-question 6: lattice Fourier transform
(~2D diffraction pattern) of the twisted bilayer square lattice of Fig. 1,
and its point-group symmetry.

Physics recap
-------------
Layer 1 (untwisted) sites:      r1_{l1,l2} = l1 * xhat + l2 * yhat,  l1,l2 in Z
Layer 2 (twisted by theta=2r):  r2_{l1,l2} = R(theta) @ r1_{l1,l2}
where tan(r) = n/m for coprime integers (m,n), as established in
sub-questions 1-3 (same (m,n) convention as the rest of the project).

The "lattice Fourier transform" (kinematic diffraction amplitude) is
    F(q) = sum_j exp(-i q . r_j)
summed over ALL atoms (both layers) in a finite real-space patch, and the
diffraction pattern is I(q) = |F(q)|^2.

Because each layer individually is an (undistorted) ideal square lattice,
F(q) = F1(q) + F2(q) where F1, F2 are each Dirac-comb-like sums peaked at
the layer's own reciprocal lattice points:
    G1_{hk} = 2*pi*(h,k)                       (layer 1)
    G2_{hk} = R(theta) @ G1_{hk}                (layer 2, rotated the same way)
i.e. the diffraction pattern is the superposition of two square Bragg-spot
lattices mutually rotated by theta = 2r.

This script:
  1. builds the atom coordinates for both layers (circular real-space patch),
  2. computes I(q) = |F(q)|^2 by direct (vectorized) summation on a q-grid,
  3. plots the resulting diffraction pattern and overlays the two
     theoretical square Bragg lattices as a check,
  4. numerically verifies the point-group symmetry of the FULL bilayer
     point set (exact real-space check, not grid-limited) for:
        - a generic commensurate angle (from Sec. 3's (m,n) table)
        - the theta = 45 deg (r = 22.5 deg) limit, which is NOT commensurate
          (tan(22.5 deg) is irrational) but recovers full 8-fold
          (octagonal) symmetry -- the Ammann-Beenker-type quasicrystal
          limit, connecting back to the quasicrystal/Penrose-tiling part
          of the same lecture.

Usage
-----
    python diffraction_pattern.py            # runs the default (2,1) example
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------
# 1. Atom coordinates
# ----------------------------------------------------------------------

def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def bilayer_atoms(theta, patch_radius=12):
    """Return (layer1, layer2, all_atoms) arrays of shape (N,2).

    layer1: ideal square lattice (a=1), truncated to a disk of radius
            `patch_radius` (a circular patch is used, rather than a square
            one, so that rotations about the origin map the patch exactly
            onto itself -- this matters for the symmetry checks below).
    layer2: layer1 rotated rigidly by `theta` = 2r about the origin.
    """
    l = np.arange(-patch_radius, patch_radius + 1)
    Lx, Ly = np.meshgrid(l, l)
    sq = np.stack([Lx.ravel(), Ly.ravel()], axis=1).astype(float)
    layer1 = sq[np.linalg.norm(sq, axis=1) <= patch_radius]
    layer2 = layer1 @ rot(theta).T
    return layer1, layer2, np.vstack([layer1, layer2])


# ----------------------------------------------------------------------
# 2. Lattice Fourier transform / diffraction intensity I(q)
# ----------------------------------------------------------------------

def structure_factor_intensity(atoms, qmax=13.0, qN=401, chunk=4000):
    """I(q) = |sum_j exp(-i q.r_j)|^2 on a qN x qN grid over [-qmax,qmax]^2."""
    qs = np.linspace(-qmax, qmax, qN)
    QX, QY = np.meshgrid(qs, qs)
    Q = np.stack([QX.ravel(), QY.ravel()], axis=1)

    F = np.zeros(Q.shape[0], dtype=complex)
    for i in range(0, Q.shape[0], chunk):
        qc = Q[i:i + chunk]
        phase = qc @ atoms.T          # (chunk, Natoms)
        F[i:i + chunk] = np.exp(-1j * phase).sum(axis=1)

    I = (np.abs(F) ** 2).reshape(qN, qN)
    return QX, QY, I


def theoretical_bragg_spots(theta, hkmax=6):
    """Return the two square Bragg lattices G1_{hk}=2*pi*(h,k) and
    G2_{hk}=R(theta) G1_{hk}, for |h|,|k| <= hkmax."""
    hk = np.arange(-hkmax, hkmax + 1)
    H, K = np.meshgrid(hk, hk)
    G1 = 2 * np.pi * np.stack([H.ravel(), K.ravel()], axis=1)
    G2 = G1 @ rot(theta).T
    return G1, G2


# ----------------------------------------------------------------------
# 3. Exact point-group symmetry check (real space, not grid-limited)
# ----------------------------------------------------------------------

def symmetry_report(theta, patch_radius=12, tol=1e-6):
    """Check, EXACTLY (via nearest-neighbour matching, not via the coarse
    q-grid), whether the full bilayer atom set is invariant under a set of
    candidate point-group operations about the origin."""
    _, _, atoms = bilayer_atoms(theta, patch_radius)
    tree = cKDTree(atoms)

    ops = {
        "R90  (4-fold rotation)": rot(np.pi / 2),
        "R45  (8-fold rotation)": rot(np.pi / 4),
        "mirror about x-axis":    np.array([[1, 0], [0, -1]]),
        "mirror about 45-line":   np.array([[0, 1], [1, 0]]),
    }

    print(f"theta = 2r = {np.rad2deg(theta):.3f} deg   "
          f"(tan r = {np.tan(theta/2):.6f})")
    for name, M in ops.items():
        moved = atoms @ M.T
        d, _ = tree.query(moved, k=1)
        frac_exact = np.mean(d < tol)
        verdict = "EXACT symmetry" if frac_exact > 1 - 1e-9 else "NOT a symmetry"
        print(f"  {name:26s}: fraction matched = {frac_exact:6.3f}  -> {verdict}")
    print()


# ----------------------------------------------------------------------
# 4. Plotting
# ----------------------------------------------------------------------

def plot_diffraction_pattern(theta, label, fname, patch_radius=12,
                              qmax=13.0, qN=401):
    _, _, atoms = bilayer_atoms(theta, patch_radius)
    QX, QY, I = structure_factor_intensity(atoms, qmax=qmax, qN=qN)
    G1, G2 = theoretical_bragg_spots(theta)

    fig, ax = plt.subplots(figsize=(6, 6))
    # log scale makes the (very sharp, high-dynamic-range) Bragg peaks visible
    ax.pcolormesh(QX, QY, np.log10(I + 1), shading="auto", cmap="inferno")
    ax.scatter(G1[:, 0], G1[:, 1], s=18, facecolors="none",
               edgecolors="cyan", linewidths=0.8, label="layer 1 (theory)")
    ax.scatter(G2[:, 0], G2[:, 1], s=18, facecolors="none",
               edgecolors="lime", linewidths=0.8, label="layer 2 (theory)")
    ax.set_xlim(-qmax, qmax)
    ax.set_ylim(-qmax, qmax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$q_x$")
    ax.set_ylabel(r"$q_y$")
    ax.set_title(f"Diffraction pattern, {label}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.3, labelcolor="white")
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"saved {fname}")


# ----------------------------------------------------------------------
# 5. Run the default examples
# ----------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        (2 * np.arctan(1 / 2), "(m,n)=(2,1), 2r=53.13deg", "diffraction_2r_53deg.png"),
        (2 * np.arctan(1 / 4), "(m,n)=(4,1), 2r=28.07deg", "diffraction_2r_28deg.png"),
        (np.deg2rad(45),       "2r=45deg (octagonal, non-commensurate)", "diffraction_2r_45deg.png"),
    ]

    for theta, label, fname in cases:
        print("=" * 70)
        print(label)
        symmetry_report(theta)
        plot_diffraction_pattern(theta, label, fname)
