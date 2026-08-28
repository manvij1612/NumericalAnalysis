# Numerical Analysis

Notebooks and scripts on numerical methods for differential equations —
time-stepping schemes for ODEs, finite-difference methods for the heat and
advection/wave equations, and spectral methods with Fourier, Chebyshev and
Legendre bases.

Most files are [Pluto.jl](https://plutojl.org) notebooks. 

## Running the notebooks

```julia
import Pluto
Pluto.run()
```

## Contents

### ODEs — `ode_suff/`
| File | Topic |
| --- | --- |
| `basic_imp.jl` | Basic explicit / implicit time-stepping schemes and their stability |

### Parabolic & hyperbolic PDEs — `pde_stuff/`
| File | Topic |
| --- | --- |
| `heat_eqn/explicit.jl` | Explicit finite-difference scheme for the 1-D heat equation; CFL stability limit |
| `wave_eqn/advection.jl` | Linear advection equation; upwind scheme, numerical diffusion |
| `wave_eqn/more_schemes.jl` | Upwind, Lax–Friedrichs, Lax–Wendroff and leapfrog compared |
| `wave_eqn/heatvsadvection.py` | Side-by-side view of diffusive vs advective behaviour |

### Spectral methods — `spec_methods/`
| File | Topic |
| --- | --- |
| `fourier.jl` | Fourier spectral differentiation on periodic domains |
| `chebyshev.jl` | Chebyshev polynomials via recursion; coefficients through the DCT (FFTW `r2r`); the Runge phenomenon |
| `legendre.jl` | Legendre polynomials and Gauss–Legendre quadrature |
| `burger_eqn.jl` | Spectral solution of the viscous Burgers equation with dealiasing |

