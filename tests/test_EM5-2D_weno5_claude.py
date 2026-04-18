import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================================================
# 2D Finite Volume Maxwell Solver - WENO5 Reconstruction
# TM Mode: Ez, Hx, Hy with Jz Dipole Source
# Periodic Boundary Conditions
# ==============================================================


def weno5_reconstruct_left(Q, axis):
    """
    Fifth-order WENO reconstruction of the left state at
    each interface along the specified axis.

    For axis=0, reconstructs Q^L at interface i+1/2 using
    cells {i-2, i-1, i, i+1, i+2}.

    Q has shape (3, Nx, Ny).

    Returns Q_L of the same shape, where Q_L[:, i, j] is the
    left-biased reconstruction at the interface between cell
    (i,j) and cell (i+1,j) for axis=0.
    """
    ax = axis + 1  # shift for the field component dimension

    # Stencil values using periodic rolls
    qm2 = np.roll(Q, 2, axis=ax)
    qm1 = np.roll(Q, 1, axis=ax)
    q0  = Q
    qp1 = np.roll(Q, -1, axis=ax)
    qp2 = np.roll(Q, -2, axis=ax)

    # Three candidate third-order reconstructions at i+1/2
    # Stencil 0: {i-2, i-1, i}
    q0_hat = (2.0 * qm2 - 7.0 * qm1 + 11.0 * q0) / 6.0
    # Stencil 1: {i-1, i, i+1}
    q1_hat = (-qm1 + 5.0 * q0 + 2.0 * qp1) / 6.0
    # Stencil 2: {i, i+1, i+2}
    q2_hat = (2.0 * q0 + 5.0 * qp1 - qp2) / 6.0

    # Smoothness indicators
    beta0 = (13.0 / 12.0) * (qm2 - 2.0 * qm1 + q0)**2 + \
            (1.0 / 4.0) * (qm2 - 4.0 * qm1 + 3.0 * q0)**2
    beta1 = (13.0 / 12.0) * (qm1 - 2.0 * q0 + qp1)**2 + \
            (1.0 / 4.0) * (qm1 - qp1)**2
    beta2 = (13.0 / 12.0) * (q0 - 2.0 * qp1 + qp2)**2 + \
            (1.0 / 4.0) * (3.0 * q0 - 4.0 * qp1 + qp2)**2

    # Optimal linear weights for fifth-order accuracy
    d0 = 1.0 / 10.0
    d1 = 6.0 / 10.0
    d2 = 3.0 / 10.0

    # Nonlinear weights with small parameter to avoid division by zero
    eps_w = 1.0e-6

    alpha0 = d0 / (eps_w + beta0)**2
    alpha1 = d1 / (eps_w + beta1)**2
    alpha2 = d2 / (eps_w + beta2)**2

    alpha_sum = alpha0 + alpha1 + alpha2

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum

    # Final WENO reconstruction
    Q_L = w0 * q0_hat + w1 * q1_hat + w2 * q2_hat

    return Q_L


def weno5_reconstruct_right(Q, axis):
    """
    Fifth-order WENO reconstruction of the right state at
    each interface along the specified axis.

    For axis=0, reconstructs Q^R at interface i+1/2 using
    cells {i-1, i, i+1, i+2, i+3} (i.e. the left-biased
    reconstruction from cell i+1 looking backward).

    This is equivalent to mirroring the left reconstruction.
    """
    ax = axis + 1

    # Stencil values: reconstruct from the right side of i+1/2,
    # meaning we center on cell i+1 and look leftward
    qm1 = np.roll(Q, -3, axis=ax)
    q0  = np.roll(Q, -2, axis=ax)
    qp1 = np.roll(Q, -1, axis=ax)
    qp2 = Q
    qp3 = np.roll(Q, 1, axis=ax)

    # Three candidate reconstructions at i+1/2 from the right
    # Stencil 0: {i+3, i+2, i+1}
    q0_hat = (2.0 * qm1 - 7.0 * q0 + 11.0 * qp1) / 6.0
    # Stencil 1: {i+2, i+1, i}
    q1_hat = (-q0 + 5.0 * qp1 + 2.0 * qp2) / 6.0
    # Stencil 2: {i+1, i, i-1}
    q2_hat = (2.0 * qp1 + 5.0 * qp2 - qp3) / 6.0

    # Smoothness indicators
    beta0 = (13.0 / 12.0) * (qm1 - 2.0 * q0 + qp1)**2 + \
            (1.0 / 4.0) * (qm1 - 4.0 * q0 + 3.0 * qp1)**2
    beta1 = (13.0 / 12.0) * (q0 - 2.0 * qp1 + qp2)**2 + \
            (1.0 / 4.0) * (q0 - qp2)**2
    beta2 = (13.0 / 12.0) * (qp1 - 2.0 * qp2 + qp3)**2 + \
            (1.0 / 4.0) * (3.0 * qp1 - 4.0 * qp2 + qp3)**2

    # Optimal linear weights (mirrored)
    d0 = 1.0 / 10.0
    d1 = 6.0 / 10.0
    d2 = 3.0 / 10.0

    eps_w = 1.0e-6

    alpha0 = d0 / (eps_w + beta0)**2
    alpha1 = d1 / (eps_w + beta1)**2
    alpha2 = d2 / (eps_w + beta2)**2

    alpha_sum = alpha0 + alpha1 + alpha2

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum

    Q_R = w0 * q0_hat + w1 * q1_hat + w2 * q2_hat

    return Q_R


def compute_flux_x(Q_L, Q_R, Z, c, epsilon, mu):
    """
    Upwind numerical flux in the x-direction using the
    characteristic Riemann solver.
    """
    Ez_L, Hx_L, Hy_L = Q_L[0], Q_L[1], Q_L[2]
    Ez_R, Hx_R, Hy_R = Q_R[0], Q_R[1], Q_R[2]

    wp = Ez_L - Z * Hy_L
    wm = Ez_R + Z * Hy_R

    Ez_star = 0.5 * (wp + wm)
    Hy_star = 0.5 * (wm - wp) / Z

    F = np.zeros_like(Q_L)
    F[0] = -Hy_star / epsilon
    F[1] = 0.0
    F[2] = -Ez_star / mu

    return F


def compute_flux_y(Q_L, Q_R, Z, c, epsilon, mu):
    """
    Upwind numerical flux in the y-direction using the
    characteristic Riemann solver.
    """
    Ez_L, Hx_L, Hy_L = Q_L[0], Q_L[1], Q_L[2]
    Ez_R, Hx_R, Hy_R = Q_R[0], Q_R[1], Q_R[2]

    wp = Ez_L + Z * Hx_L
    wm = Ez_R - Z * Hx_R

    Ez_star = 0.5 * (wp + wm)
    Hx_star = 0.5 * (wp - wm) / Z

    G = np.zeros_like(Q_L)
    G[0] = Hx_star / epsilon
    G[1] = Ez_star / mu
    G[2] = 0.0

    return G


def compute_rhs(Q, dx, dy, Z, c, epsilon, mu):
    """
    Compute the spatial operator using WENO5 reconstruction
    and upwind fluxes.
    """
    # x-direction
    Q_L_x = weno5_reconstruct_left(Q, axis=0)
    Q_R_x = weno5_reconstruct_right(Q, axis=0)
    F = compute_flux_x(Q_L_x, Q_R_x, Z, c, epsilon, mu)

    # y-direction
    Q_L_y = weno5_reconstruct_left(Q, axis=1)
    Q_R_y = weno5_reconstruct_right(Q, axis=1)
    G = compute_flux_y(Q_L_y, Q_R_y, Z, c, epsilon, mu)

    dFdx = (F - np.roll(F, 1, axis=1)) / dx
    dGdy = (G - np.roll(G, 1, axis=2)) / dy

    return -dFdx - dGdy


def dipole_source(t, Nx, Ny, dx, dy, x0, y0, freq, sigma_src, J0):
    """Jz current source for a z-directed dipole."""
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')

    ## delta function
    delta_fct = np.zeros((npts_x, npts_y))
    delta_fct[Nx // 2, Ny // 2] = np.sin(2 * np.pi * freq * t)
    Jz = J0 * delta_fct

    # x = (np.arange(Nx) + 0.5) * dx
    # y = (np.arange(Ny) + 0.5) * dy
    # X, Y = np.meshgrid(x, y, indexing='ij')
    #
    # r_sq = (X - x0)**2 + (Y - y0)**2
    # spatial_profile = np.exp(-r_sq / (2.0 * sigma_src**2))
    # spatial_profile /= (2.0 * np.pi * sigma_src**2)
    #
    # omega = 2.0 * np.pi * freq
    # n_ramp = 3.0
    # t_ramp = n_ramp / freq
    # if t < t_ramp:
    #     envelope = 0.5 * (1.0 - np.cos(np.pi * t / t_ramp))
    # else:
    #     envelope = 1.0
    #
    # temporal = envelope * np.sin(omega * t)
    # Jz = J0 * spatial_profile * temporal

    S = np.zeros((3, Nx, Ny))
    S[0] = -Jz

    return S


def ssp_rk3_step(Q, dt, dx, dy, Z, c, epsilon, mu,
                  t, Nx, Ny, x0, y0, freq, sigma_src, J0):
    """
    Third-order strong stability preserving Runge-Kutta
    (SSP-RK3) time step to match the spatial accuracy of
    the WENO5 reconstruction more closely.

    Three-stage Shu-Osher form:
        Q1   = Qn + dt * L(Qn)
        Q2   = 3/4 Qn + 1/4 (Q1 + dt * L(Q1))
        Qn+1 = 1/3 Qn + 2/3 (Q2 + dt * L(Q2))
    """
    def full_rhs(Q_in, t_eval):
        rhs = compute_rhs(Q_in, dx, dy, Z, c, epsilon, mu)
        S = dipole_source(t_eval, Nx, Ny, dx, dy, x0, y0,
                          freq, sigma_src, J0)
        S[0] /= epsilon
        rhs += S
        return rhs

    # Stage 1
    rhs1 = full_rhs(Q, t)
    Q1 = Q + dt * rhs1

    # Stage 2
    rhs2 = full_rhs(Q1, t + dt)
    Q2 = 0.75 * Q + 0.25 * (Q1 + dt * rhs2)

    # Stage 3
    rhs3 = full_rhs(Q2, t + 0.5 * dt)
    Q_new = (1.0 / 3.0) * Q + (2.0 / 3.0) * (Q2 + dt * rhs3)

    return Q_new


def solve_maxwell_weno5(
    Lx=4.0, Ly=4.0,
    Nx=200, Ny=200,
    T_final=6.0,
    CFL=0.4,
    epsilon=1.0, mu=1.0,
    freq=2.0,
    J0=1.0
):
    """
    Main solver routine using WENO5 reconstruction with
    upwind fluxes and SSP-RK3 time integration.
    """
    dx = Lx / Nx
    dy = Ly / Ny

    c = 1.0 / np.sqrt(epsilon * mu)
    Z = np.sqrt(mu / epsilon)

    dt = CFL / (c * (1.0 / dx + 1.0 / dy))
    Nt = int(np.ceil(T_final / dt))
    dt = T_final / Nt

    x0, y0 = Lx / 2.0, Ly / 2.0
    sigma_src = 2.0 * dx

    print(f"Grid: {Nx} x {Ny}")
    print(f"dx = {dx:.6f}, dy = {dy:.6f}")
    print(f"Wave speed c = {c:.4f}, Impedance Z = {Z:.4f}")
    print(f"dt = {dt:.6e}, Nt = {Nt}")
    print(f"Source frequency = {freq:.2f} Hz, wavelength = {c/freq:.4f}")
    print(f"Points per wavelength = {c / freq / dx:.1f}")

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy

    Q = np.zeros((3, Nx, Ny))

    snapshots = []
    snapshot_times = []
    n_snapshots = 100
    snapshot_interval = max(1, Nt // n_snapshots)

    for n in range(Nt):
        t = n * dt

        if n % snapshot_interval == 0:
            snapshots.append(Q[0].copy())
            snapshot_times.append(t)
            print(f"  Step {n:6d}/{Nt}, t = {t:.4f}, "
                  f"max|Ez| = {np.max(np.abs(Q[0])):.6e}")

        Q = ssp_rk3_step(Q, dt, dx, dy, Z, c, epsilon, mu,
                         t, Nx, Ny, x0, y0, freq, sigma_src, J0)

    snapshots.append(Q[0].copy())
    snapshot_times.append(T_final)

    print(f"\nSimulation complete at t = {T_final:.4f}")
    print(f"Final max|Ez| = {np.max(np.abs(Q[0])):.6e}")

    return Q, x, y, snapshots, snapshot_times


def plot_final_state(Q, x, y):
    """Plot the final state of all three field components."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = [r'$E_z$', r'$H_x$', r'$H_y$']

    for idx, ax in enumerate(axes):
        vmax = np.max(np.abs(Q[idx])) * 0.8
        if vmax == 0:
            vmax = 1.0
        im = ax.pcolormesh(
            x, y, Q[idx].T,
            shading='auto', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(labels[idx])
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('WENO5 Dipole Radiation — Final State', fontsize=14)
    plt.tight_layout()
    plt.savefig('weno5_dipole_final.png', dpi=150, bbox_inches='tight')
    plt.show()


def animate_field(snapshots, snapshot_times, x, y):
    """Create an animation of the Ez field evolution."""
    fig, ax = plt.subplots(figsize=(7, 6))

    mid = len(snapshots) // 2
    vmax = np.max(np.abs(snapshots[mid])) * 0.9
    if vmax == 0:
        vmax = 1.0

    im = ax.pcolormesh(
        x, y, snapshots[0].T,
        shading='auto', cmap='RdBu_r',
        vmin=-vmax, vmax=vmax
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    title = ax.set_title(f'$E_z$ at t = {snapshot_times[0]:.4f}')
    fig.colorbar(im, ax=ax, shrink=0.8)

    def update(frame):
        im.set_array(snapshots[frame].T.ravel())
        title.set_text(f'$E_z$ at t = {snapshot_times[frame]:.4f}')
        return im, title

    ani = FuncAnimation(
        fig, update, frames=len(snapshots),
        interval=80, blit=False
    )
    ani.save('weno5_dipole_radiation.gif', writer='pillow', fps=15)
    plt.show()

    return ani

def plot_field_along_axes(Q, x, y, x0, y0):
    """
    Plot cross-sections of Ez along the x and y axes
    passing through the dipole location.
    """
    ix0 = np.argmin(np.abs(x - x0))
    iy0 = np.argmin(np.abs(y - y0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x, Q[0][:, iy0], 'b-', linewidth=1.2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel(r'$E_z$')
    axes[0].set_title(f'$E_z$ along y = {y[iy0]:.2f}')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x0, color='r', linestyle='--', alpha=0.5,
                    label='Source')
    axes[0].legend()

    axes[1].plot(y, Q[0][ix0, :], 'b-', linewidth=1.2)
    axes[1].set_xlabel('y')
    axes[1].set_ylabel(r'$E_z$')
    axes[1].set_title(f'$E_z$ along x = {x[ix0]:.2f}')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(y0, color='r', linestyle='--', alpha=0.5,
                    label='Source')
    axes[1].legend()

    plt.suptitle('Field Cross-Sections Through Dipole', fontsize=14)
    plt.tight_layout()
    plt.savefig('dipole_cross_sections.png', dpi=150, bbox_inches='tight')
    plt.show()



# ==============================================================
# Run the solver
# ==============================================================
if __name__ == '__main__':
    # Domain
    L = 8
    npts_x = npts_y = 2 ** L

    permittivity = 1  # 1.0    # permittivity of free space
    permeability = 1.0  # permeability of free space
    eV = 1.0  # elementary charge or Joule-eV conversion factor

    ## dipole properties
    c = 1 / np.sqrt(permittivity * permeability)
    dp = 0.2  # dipole length
    I0 = 1.0  # current strength
    omega = 2 * np.pi * 2
    lam = 2 * np.pi / (omega / c)

    ## box size
    Lx = 10.0 * lam
    Ly = Lx

    x0_vals = np.linspace(-Lx, Lx, npts_x, endpoint=True)
    y0_vals = np.linspace(-Ly, Ly, npts_y, endpoint=True)
    dx = (x0_vals[1] - x0_vals[0])
    dy = (y0_vals[1] - y0_vals[0])
    # x1_vals = x0_vals + dx/2
    # y1_vals = y0_vals + dy/2

    x0_mesh, y0_mesh = np.meshgrid(x0_vals, y0_vals)
    # x1_mesh, y1_mesh = np.meshgrid(x1_vals, y1_vals)

    ## time stepping
    T = 5.0
    dt = None  # 0.05  # 0.04  # None
    cfl = 0.9

    Q, x, y, snapshots, snapshot_times = solve_maxwell_weno5(
        Lx=Lx * 2, Ly=Ly * 2,
        Nx=npts_x, Ny=npts_y,
        T_final=T,
        CFL=cfl,
        epsilon=1.0, mu=1.0,
        freq=omega / 2 / np.pi,
        J0=1.0,
    )

    plot_final_state(Q, x, y)
    # ani = animate_field(snapshots, snapshot_times, x, y)
    plot_field_along_axes(Q, x, y, x0=Lx, y0=Ly)