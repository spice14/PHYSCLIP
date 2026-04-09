"""
Spectral solver for the 1D viscous Burgers equation.

Governing equation:
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

Domain: x ∈ [0, 2π] with periodic boundary conditions.

Method: Fourier pseudospectral with RK4 time integration.
- Spatial derivatives computed via FFT.
- 2/3 dealiasing to handle quadratic nonlinearity.
- Fixed-step RK4 time stepping.
- Deterministic and reproducible.
"""

import numpy as np


class BurgersSpectralSolver:
    """
    Spectral solver for 1D viscous Burgers equation with periodic BCs.
    
    Parameters
    ----------
    nu : float
        Viscosity coefficient (ν ≥ 0).
    nx_grid : int
        Number of spatial grid points (must be even for 2/3 dealiasing).
    dt : float
        Fixed time step size.
    """
    
    def __init__(self, nu, nx_grid=256, dt=0.01):
        """Initialize solver with grid and parameters."""
        self.nu = nu
        self.nx_grid = nx_grid
        self.dt = dt
        
        # Spatial grid: uniform on [0, 2π]
        self.x = np.linspace(0, 2*np.pi, nx_grid, endpoint=False)
        self.dx = 2*np.pi / nx_grid
        
        # Wavenumber grid: proper scaling for [0, 2π]
        # k_j = 2πj/L where L=2π, so k_j = j
        self.k = np.fft.fftfreq(nx_grid, d=self.dx) * 2 * np.pi
        
        # Sanity check: dt should be small relative to dx for stability
        if self.dt >= self.dx:
            raise ValueError(
                f"Time step dt={self.dt} is not small relative to dx={self.dx:.6f}. "
                "Consider reducing dt or increasing Nx."
            )

        # Diffusive CFL condition for explicit RK4: dt <= dx^2 / (2*nu)
        if self.nu > 0:
            dt_diff_limit = (self.dx ** 2) / (2.0 * self.nu)
            if self.dt > dt_diff_limit:
                raise ValueError(
                    f"Time step dt={self.dt} exceeds diffusion stability limit {dt_diff_limit:.6f} "
                    f"for nu={self.nu}. Reduce dt or use implicit treatment of viscosity."
                )
        
    def _spatial_derivatives(self, u_hat):
        """
        Compute spatial derivatives in spectral space.
        
        Parameters
        ----------
        u_hat : ndarray
            FFT of u at current time.
        
        Returns
        -------
        u : ndarray
            u in physical space.
        du_dx : ndarray
            ∂u/∂x in physical space.
        """
        # Transform to physical space
        u = np.fft.ifft(u_hat).real
        
        # First derivative: ∂u/∂x = IFFT(i·k·u_hat)
        du_dx_hat = 1j * self.k * u_hat
        du_dx = np.fft.ifft(du_dx_hat).real
        
        return u, du_dx
    
    def _rhs(self, u_hat):
        """
        Compute right-hand side: ∂u/∂t = -u·∂u/∂x + ν·∂²u/∂x²
        
        The nonlinear term (-u·∂u/∂x) is computed in physical space with 2/3 dealiasing.
        The viscous term (-ν·∂²u/∂x²) is computed directly in spectral space.
        
        Parameters
        ----------
        u_hat : ndarray
            FFT of u.
        
        Returns
        -------
        dudt_hat : ndarray
            FFT of ∂u/∂t.
        """
        u, du_dx = self._spatial_derivatives(u_hat)
        
        # Nonlinear term in physical space: -u·∂u/∂x
        nonlinear_phys = -u * du_dx
        
        # Transform to spectral space
        nonlinear_hat = np.fft.fft(nonlinear_phys)
        
        # Apply 2/3 dealiasing: zero modes beyond 2/3 of spectrum using index truncation
        nonlinear_hat[self.nx_grid//3 : -self.nx_grid//3] = 0
        
        # Viscous term directly in spectral space: -ν·∂²u/∂x² = -ν·k²·u_hat
        viscous_hat = -self.nu * (self.k**2) * u_hat
        
        # Combine: ∂u_hat/∂t = nonlinear + viscous
        rhs_hat = nonlinear_hat + viscous_hat
        
        return rhs_hat
    
    def step(self, u_hat):
        """
        Advance solution by one RK4 time step.
        
        Parameters
        ----------
        u_hat : ndarray
            FFT of u at current time.
        
        Returns
        -------
        u_hat_new : ndarray
            FFT of u at next time.
        """
        # RK4 coefficients: k_i = ∂u/∂t evaluated at intermediate states
        k1 = self._rhs(u_hat)
        k2 = self._rhs(u_hat + 0.5*self.dt*k1)
        k3 = self._rhs(u_hat + 0.5*self.dt*k2)
        k4 = self._rhs(u_hat + self.dt*k3)
        
        # RK4 update: u_{n+1} = u_n + (dt/6)(k1 + 2k2 + 2k3 + k4)
        u_hat_new = u_hat + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return u_hat_new
    
    def solve(self, u0, t_final, return_trajectory=False):
        """
        Integrate from initial condition to time t_final.
        
        Parameters
        ----------
        u0 : ndarray
            Initial condition u(x, 0) in physical space.
        t_final : float
            Final time.
        return_trajectory : bool
            If True, return solution at all time steps (expensive).
            If False, return only final state.
        
        Returns
        -------
        u_final : ndarray
            Solution at time t_final.
        trajectory : list of ndarray (optional)
            All intermediate states if return_trajectory=True.
        """
        # Initial condition in spectral space
        u_hat = np.fft.fft(u0)
        
        # Determine number of steps
        n_steps = int(np.round(t_final / self.dt))
        
        trajectory = []
        if return_trajectory:
            trajectory.append(np.fft.ifft(u_hat).real.copy())
        
        # Time integration
        for _ in range(n_steps):
            u_hat = self.step(u_hat)
            if return_trajectory:
                trajectory.append(np.fft.ifft(u_hat).real.copy())
        
        u_final = np.fft.ifft(u_hat).real
        
        if return_trajectory:
            return u_final, trajectory
        else:
            return u_final
