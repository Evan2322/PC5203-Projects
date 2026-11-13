"""
PC5203 Project 2, Problem 2, Q1 -- numerical check of the Harper equation.

Model (hopping amplitude set to +1, energies in units of |t|):
    H = sum_{x,y} [ c^dag_{x+1,y} c_{x,y} + e^{+i 2 pi b x} c^dag_{x,y+1} c_{x,y} + h.c. ]
i.e. Landau gauge A = (0, B x, 0); the y-bond in column x carries the Peierls phase 2 pi b x.
With psi(x,y) = e^{i k y} psi_x this reduces to the Harper equation
    2 psi_x cos(2 pi b x - k) + psi_{x+1} + psi_{x-1} = eps psi_x .

Checks performed (all printed):
  (1) flux through every plaquette = 2 pi b (mod 2 pi), read off the hopping matrix;
  (2) full 2D spectrum on an Lx x Ly torus == union over k = 2 pi n / Ly of the
      Harper-chain spectra (length Lx, periodic);
  (3) gauge independence: Landau gauge A = (-B y, 0) gives the same 2D spectrum.
"""
import numpy as np


def H2d_landau_y(Lx, Ly, b):
    """Full 2D Hamiltonian, phase on y-bonds: e^{i 2 pi b x}. Requires b*Lx integer."""
    N = Lx * Ly
    idx = lambda x, y: (x % Lx) + Lx * (y % Ly)
    H = np.zeros((N, N), complex)
    for x in range(Lx):
        for y in range(Ly):
            i = idx(x, y)
            H[idx(x + 1, y), i] += 1.0                               # x-bond
            H[idx(x, y + 1), i] += np.exp(1j * 2 * np.pi * b * x)    # y-bond
    return H + H.conj().T


def H2d_landau_x(Lx, Ly, b):
    """Other Landau gauge A = (-B y, 0): phase on x-bonds e^{-i 2 pi b y}. Requires b*Ly integer."""
    N = Lx * Ly
    idx = lambda x, y: (x % Lx) + Lx * (y % Ly)
    H = np.zeros((N, N), complex)
    for x in range(Lx):
        for y in range(Ly):
            i = idx(x, y)
            H[idx(x + 1, y), i] += np.exp(-1j * 2 * np.pi * b * y)
            H[idx(x, y + 1), i] += 1.0
    return H + H.conj().T


def harper_chain(Lx, b, k):
    """Harper equation as an Lx x Lx matrix (periodic in x; valid when b*Lx is integer)."""
    H = np.diag(2 * np.cos(2 * np.pi * b * np.arange(Lx) - k)).astype(complex)
    for x in range(Lx):
        H[(x + 1) % Lx, x] += 1.0
        H[x, (x + 1) % Lx] += 1.0
    return H


def plaquette_flux(H, Lx, Ly):
    """Phase of the product of hoppings around each plaquette (counter-clockwise)."""
    idx = lambda x, y: (x % Lx) + Lx * (y % Ly)
    fl = []
    for x in range(Lx):
        for y in range(Ly):
            a, b_, c, d = idx(x, y), idx(x + 1, y), idx(x + 1, y + 1), idx(x, y + 1)
            prod = H[b_, a] * H[c, b_] * H[d, c] * H[a, d]
            fl.append(np.angle(prod))
    return np.array(fl)


if __name__ == "__main__":
    for (p, q) in [(1, 3), (2, 5), (3, 7)]:
        b = p / q
        Lx = Ly = 3 * q * (1 if q > 3 else 2)      # multiple of q so b*L is integer
        H = H2d_landau_y(Lx, Ly, b)
        assert np.allclose(H, H.conj().T)

        # (1) flux per plaquette
        fl = plaquette_flux(H, Lx, Ly)
        dev = np.abs(np.angle(np.exp(1j * (fl - 2 * np.pi * b)))).max()

        # (2) 2D spectrum vs Harper chains
        E2d = np.sort(np.linalg.eigvalsh(H))
        Eh = np.sort(np.concatenate([np.linalg.eigvalsh(harper_chain(Lx, b, 2 * np.pi * n / Ly))
                                     for n in range(Ly)]))
        # (3) other gauge
        Eg = np.sort(np.linalg.eigvalsh(H2d_landau_x(Lx, Ly, b)))

        print(f"b={p}/{q}, L={Lx}x{Ly}: max|flux-2pi b|={dev:.1e}, "
              f"max|E2d-E_Harper|={np.abs(E2d - Eh).max():.1e}, "
              f"max|E_gauge1-E_gauge2|={np.abs(E2d - Eg).max():.1e}")
