NIST Lennard-Jones liquid-vapor coexistence reference

Potential:
    u*(r*) = 4[(1/r*)^12 - (1/r*)^6]

Dataset:
    Grand-canonical transition-matrix Monte Carlo with histogram
    reweighting.

Truncation:
    Pair potential cut at 5 sigma, no long-range tail correction.

Reduced units:
    T*   = kBT / epsilon
    rho* = rho sigma^3
    P*   = P sigma^3 / epsilon

Reported range:
    T* = 0.60 to 1.25.

NIST critical-property estimate:
    Tc*   = 1.284
    rhoc* = 0.318
    Pc*   = 0.118

Purpose:
    Reference target for validation of the liquid perturbation and
    coexistence calculations. This is one specific NIST cutoff dataset;
    Huang Figure S3 includes NIST results using multiple cutoff choices.
