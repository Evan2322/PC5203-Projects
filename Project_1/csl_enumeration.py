import math
from math import gcd, atan, degrees

def enumerate_csl(sigma_max):
    results = []
    for m in range(1, 4*sigma_max):
        for n in range(1, m):
            if gcd(m, n) != 1:
                continue
            if m % 2 == 1 and n % 2 == 1:
                Sigma = (m*m + n*n) // 2      # both odd
            elif not (m % 2 == 0 and n % 2 == 0):
                Sigma = m*m + n*n             # opposite parity
            else:
                continue
            if Sigma < sigma_max:
                r = degrees(atan(n / m))
                results.append((Sigma, m, n, r, 2*r))
    return sorted(results)

for Sigma, m, n, r, two_r in enumerate_csl(50):
    print(f"Sigma={Sigma:3d}  N=2*Sigma={2*Sigma:3d}  "
          f"(m,n)=({m},{n})  r={r:6.2f} deg  2r={two_r:6.2f} deg")
