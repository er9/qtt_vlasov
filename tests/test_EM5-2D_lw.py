import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================================================
# 2D Finite Volume Maxwell Solver - Lax-Wendroff Method
# TM Mode: Ez, Hx, Hy with Jz Dipole Source
# Periodic Boundary Conditions
# ==============================================================
#
# The TM mode Maxwell equations in conservation form:
#
#   dQ/dt + dF/dx + dG/dy = S
#
# where:
#   Q = [Ez, Hx, Hy]^T
#
#   F = [-Hy/eps, 0, -Ez/mu]^T       (x-flux)
#   G = [Hx/eps, Ez/mu, 0]^T         (y-flux)
#   S = [-Jz/eps, 0, 0]^T            (source)
#
# The flux Jacobians are:
#
#   A = dF/dQ = [[0,       0,    -1/eps],
#                [0,       0,     0    ],
#                [-1/mu,   0,     0    ]]
#
#   B = dG/dQ = [[0,     1/eps,   0],
#                [1/mu,  0,       0],
#                [0,     0,       0]]
#
# The Lax-Wendroff flux in the x-direction at interface i+1/2:
#
#   F_{i+1/2} = 0.5 * (F(Q_i) + F(Q_{i+1}))
#             - 0.5 * (dt/dx) * A^2 * (Q_{i+1} - Q_i)
#
# where A^2 = A*A is computed analytically:
#
#   A^2 = [[1/(eps*mu),  0,  0],
#          [0,           0,  0],
#          [0,           0,  1/(eps*mu)]]
#
#      = c^2 * diag(1, 0, 1)
#
# Similarly, B^2 = B*B:
#
#   B^2 = [[1/(eps*mu),  0,  0],
#          [0,  1/(eps*mu), 0],
#          [0,           0,  0]]
#
#      = c^2 * diag(1, 1, 0)
#
# The cross-derivative term for the fully 2D Lax-Wendroff
# method requires the mixed Jacobian products A*B and B*A:
#
#   A*B = [[0,        0,        0 ],
#          [0,        0,        0 ],
#          [0,  -1/(eps*mu),    0 ]]
#
#   B*A = [[0,     0,        0 ],
#          [0,     0,  -1/(eps*mu)],
#          [0,     0,        0 ]]
#


def dipole_source(t, Nx, Ny, dx, dy, x0, y0, freq, sigma_src, J0):
    """
    Compute the Jz current source for a z-directed dipole.
    Returns the full source vector S = [-Jz/eps, 0, 0]
    (the 1/eps factor is applied in the main loop).
    """
    # x = (np.arange(Nx) + 0.5) * dx
    # y = (np.arange(Ny) + 0.5) * dy
    # X, Y = np.meshgrid(x, y, indexing='ij')

    ## delta function
    delta_fct = np.zeros((npts_x, npts_y))
    delta_fct[Nx // 2, Ny // 2] = np.sin(2 * np.pi * freq * t)
    print('Jz', J0 * delta_fct[Nx //2, Ny//2])
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


def dipole_source_dt(t, Nx, Ny, dx, dy, x0, y0, freq, sigma_src, J0):
    """
    Time derivative of the Jz source, needed for the
    Lax-Wendroff correction term involving dS/dt.

    For second-order accuracy, the source contribution to the
    update is:
        Q^{n+1} = ... + dt*S^n + 0.5*dt^2 * dS/dt|^n
    """
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')

    ## delta function
    omega = 2 * np.pi * freq
    delta_fct = np.zeros((npts_x, npts_y))
    delta_fct[Nx // 2, Ny // 2] = 1.0  #np.sin(omega * t)

    # d/dt [envelope * sin(omega*t)]
    dJz_dt = J0 * delta_fct * omega * np.cos(omega * t)

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
    #
    # if t < t_ramp:
    #     envelope = 0.5 * (1.0 - np.cos(np.pi * t / t_ramp))
    #     d_envelope = 0.5 * np.pi / t_ramp * np.sin(np.pi * t / t_ramp)
    # else:
    #     envelope = 1.0
    #     d_envelope = 0.0
    #
    # sin_val = np.sin(omega * t)
    # cos_val = np.cos(omega * t)
    #
    # # d/dt [envelope * sin(omega*t)]
    # d_temporal = d_envelope * sin_val + envelope * omega * cos_val

    # dJz_dt = J0 * spatial_profile * d_temporal

    dS_dt = np.zeros((3, Nx, Ny))
    dS_dt[0] = -dJz_dt
    return dS_dt


def compute_x_flux(Q, dt, dx, epsilon, mu, c2):
    """
    Compute the Lax-Wendroff numerical flux in the x-direction
    at all interfaces i+1/2.

    F_{i+1/2} = 0.5*(F_i + F_{i+1}) - 0.5*(dt/dx)*A^2*(Q_{i+1} - Q_i)

    Since A^2 = c^2 * diag(1, 0, 1), the correction term only
    affects the Ez and Hy components.
    """
    # Neighbor to the right (periodic)
    Q_p1 = np.roll(Q, -1, axis=1)

    Ez   = Q[0];    Ez_p1 = Q_p1[0]
    Hx   = Q[1];    Hx_p1 = Q_p1[1]
    Hy   = Q[2];    Hy_p1 = Q_p1[2]

    # Average flux: 0.5*(F_i + F_{i+1})
    # F = [-Hy/eps, 0, -Ez/mu]
    F_avg = np.zeros_like(Q)
    F_avg[0] = -0.5 * (Hy + Hy_p1) / epsilon
    F_avg[1] = 0.0
    F_avg[2] = -0.5 * (Ez + Ez_p1) / mu

    # Lax-Wendroff correction: -0.5*(dt/dx)*A^2*(Q_{i+1} - Q_i)
    # A^2 = c^2 * diag(1, 0, 1)
    dQ = Q_p1 - Q
    coeff = 0.5 * dt / dx * c2

    F_lw = np.zeros_like(Q)
    F_lw[0] = F_avg[0] - coeff * dQ[0]
    F_lw[1] = 0.0
    F_lw[2] = F_avg[2] - coeff * dQ[2]

    return F_lw


def compute_y_flux(Q, dt, dy, epsilon, mu, c2):
    """
    Compute the Lax-Wendroff numerical flux in the y-direction
    at all interfaces j+1/2.

    G_{j+1/2} = 0.5*(G_j + G_{j+1}) - 0.5*(dt/dy)*B^2*(Q_{j+1} - Q_j)

    Since B^2 = c^2 * diag(1, 1, 0), the correction term only
    affects the Ez and Hx components.
    """
    Q_p1 = np.roll(Q, -1, axis=2)

    Ez   = Q[0];    Ez_p1 = Q_p1[0]
    Hx   = Q[1];    Hx_p1 = Q_p1[1]
    Hy   = Q[2];    Hy_p1 = Q_p1[2]

    # Average flux: 0.5*(G_j + G_{j+1})
    # G = [Hx/eps, Ez/mu, 0]
    G_avg = np.zeros_like(Q)
    G_avg[0] = 0.5 * (Hx + Hx_p1) / epsilon
    G_avg[1] = 0.5 * (Ez + Ez_p1) / mu
    G_avg[2] = 0.0

    # Lax-Wendroff correction: -0.5*(dt/dy)*B^2*(Q_{j+1} - Q_j)
    # B^2 = c^2 * diag(1, 1, 0)
    dQ = Q_p1 - Q
    coeff = 0.5 * dt / dy * c2

    G_lw = np.zeros_like(Q)
    G_lw[0] = G_avg[0] - coeff * dQ[0]
    G_lw[1] = G_avg[1] - coeff * dQ[1]
    G_lw[2] = 0.0

    return G_lw


def compute_cross_term(Q, dt, dx, dy, epsilon, mu, c2):
    """
    Compute the cross-derivative correction term for the
    fully 2D Lax-Wendroff method.

    The Taylor expansion of Q to second order in 2D gives a
    mixed derivative term:

        0.5 * dt^2 * (A * dG/dx + B * dF/dy)

    which can be written as:

        0.5 * dt^2 / (dx*dy) * [A*B*(Q_{i,j+1} - Q_{i,j-1}
                                      - Q_{i+1,j+1} + Q_{i+1,j})_{x-diff}
                                + B*A*(...)]

    For Maxwell's equations, we compute this using the
    analytical forms of A*B and B*A:

        A*B = c^2 * [[0, 0, 0], [0, 0, 0], [0, -1, 0]]
        B*A = c^2 * [[0, 0, 0], [0, 0, -1], [0, 0, 0]]

    The cross term simplifies to corrections on Hx and Hy
    involving mixed spatial differences of Ez, Hx, Hy.
    """
    # We need the mixed second derivative d^2Q/(dx dy)
    # Approximated as:
    # d^2Q/(dx dy) ≈ (1/(4*dx*dy)) * (Q_{i+1,j+1} - Q_{i+1,j-1}
    #                                   - Q_{i-1,j+1} + Q_{i-1,j-1})

    Q_pp = np.roll(np.roll(Q, -1, axis=1), -1, axis=2)  # (i+1, j+1)
    Q_pm = np.roll(np.roll(Q, -1, axis=1),  1, axis=2)  # (i+1, j-1)
    Q_mp = np.roll(np.roll(Q,  1, axis=1), -1, axis=2)  # (i-1, j+1)
    Q_mm = np.roll(np.roll(Q,  1, axis=1),  1, axis=2)  # (i-1, j-1)

    d2Q_dxdy = (Q_pp - Q_pm - Q_mp + Q_mm) / (4.0 * dx * dy)

    # Cross correction to the update:
    # dt^2/2 * (A*B + B*A) * d^2Q/(dx dy)
    #
    # (A*B + B*A) applied to [Ez, Hx, Hy]:
    #   component 0 (Ez):  0
    #   component 1 (Hx):  -c^2 * Hy component of d^2Q/(dx dy)
    #   component 2 (Hy):  -c^2 * Hx component of d^2Q/(dx dy)

    cross = np.zeros_like(Q)
    coeff = 0.5 * dt**2 * c2
    cross[0] = 0.0
    cross[1] = -coeff * d2Q_dxdy[2]  # -c^2 * d^2(Hy)/(dx dy)
    cross[2] = -coeff * d2Q_dxdy[1]  # -c^2 * d^2(Hx)/(dx dy)
    # print('cross Bx', np.linalg.norm(cross[1]))
    # print('cross By', np.linalg.norm(cross[2]))
    # print('coeff', coeff)
    #
    # if np.linalg.norm(cross[1]) > 0:
    #     plt.figure()
    #     plt.imshow(Q[1])
    #     plt.title('Bx')
    #     plt.colorbar()
    #
    #     plt.figure()
    #     plt.imshow(Q[2])
    #     plt.title('By')
    #     plt.colorbar()
    #
    #     plt.figure()
    #     plt.imshow(d2Q_dxdy[1])
    #     plt.title('d2/dxdy Bx')
    #     plt.colorbar()
    #
    #     plt.figure()
    #     plt.imshow(d2Q_dxdy[2])
    #     plt.title('d2/dxdy By')
    #     plt.colorbar()
    #     plt.show()


    return cross


def solve_maxwell_lax_wendroff(
    Lx=4.0, Ly=4.0,
    Nx=400, Ny=400,
    T_final=6.0,
    CFL=0.5,
    epsilon=1.0, mu=1.0,
    freq=2.0,
    J0=1.0
):
    """
    Main solver for 2D TM Maxwell equations using the
    Lax-Wendroff method with periodic boundary conditions
    and a Jz dipole source.

    The complete single-step update is:

        Q^{n+1} = Q^n
            - (dt/dx) * (F_{i+1/2} - F_{i-1/2})
            - (dt/dy) * (G_{j+1/2} - G_{j-1/2})
            + cross derivative correction
            + dt * S^n + 0.5 * dt^2 * dS/dt|^n
    """
    dx = Lx / Nx
    dy = Ly / Ny

    c = 1.0 / np.sqrt(epsilon * mu)
    c2 = c**2
    Z = np.sqrt(mu / epsilon)

    # CFL condition for Lax-Wendroff in 2D
    dt_max = 1.0 / (c * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))
    dt = CFL * dt_max

    Nt = int(np.ceil(T_final / dt))
    dt = T_final / Nt

    # Verify CFL
    nu_x = c * dt / dx
    nu_y = c * dt / dy
    print(f"Grid: {Nx} x {Ny}")
    print(f"dx = {dx:.6f}, dy = {dy:.6f}")
    print(f"Wave speed c = {c:.4f}")
    print(f"dt = {dt:.6e}, Nt = {Nt}")
    print(f"Courant numbers: nu_x = {nu_x:.4f}, nu_y = {nu_y:.4f}")
    print(f"nu_x^2 + nu_y^2 = {nu_x**2 + nu_y**2:.4f} (must be <= 1)")
    print(f"Source frequency = {freq:.2f} Hz, wavelength = {c/freq:.4f}")
    print(f"Points per wavelength = {c / freq / dx:.1f}")

    # Source parameters
    x0, y0 = Lx / 2.0, Ly / 2.0
    sigma_src = 2.0 * dx

    # Cell center coordinates
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy

    # Initialize fields to zero
    Q = np.zeros((3, Nx, Ny))

    # Snapshots
    snapshots = []
    snapshot_times = []
    n_snapshots = 100
    snapshot_interval = max(1, Nt // n_snapshots)

    # ==========================================================
    # Main Lax-Wendroff time-stepping loop
    # ==========================================================
    for n in range(Nt):
        t = n * dt

        if n % snapshot_interval == 0:
            snapshots.append(Q[0].copy())
            snapshot_times.append(t)
            print(f"  Step {n:6d}/{Nt}, t = {t:.4f}, "
                  f"max|Ez| = {np.max(np.abs(Q[0])):.6e}")

        # Lax-Wendroff x-flux at all i+1/2 interfaces
        F = compute_x_flux(Q, dt, dx, epsilon, mu, c2)

        # Lax-Wendroff y-flux at all j+1/2 interfaces
        G = compute_y_flux(Q, dt, dy, epsilon, mu, c2)

        # Flux divergence
        dFdx = (F - np.roll(F, 1, axis=1)) / dx
        dGdy = (G - np.roll(G, 1, axis=2)) / dy

        # Cross-derivative correction
        cross = compute_cross_term(Q, dt, dx, dy, epsilon, mu, c2)

        # Source terms (second-order accurate in time)
        S = dipole_source(t, Nx, Ny, dx, dy, x0, y0,
                          freq, sigma_src, J0)
        S[0] /= epsilon

        dS_dt = dipole_source_dt(t, Nx, Ny, dx, dy, x0, y0,
                                  freq, sigma_src, J0)
        dS_dt[0] /= epsilon

        # Full Lax-Wendroff update
        Q = Q - dt * (dFdx + dGdy) + cross + dt * S # + 0.5 * dt**2 * dS_dt

    # Final snapshot
    snapshots.append(Q[0].copy())
    snapshot_times.append(T_final)

    print(f"\nSimulation complete at t = {T_final:.4f}")
    print(f"Final max|Ez| = {np.max(np.abs(Q[0])):.6e}")

    results = {
        'Q': Q, 'x': x, 'y': y,
        'snapshots': snapshots,
        'snapshot_times': snapshot_times,
        'dx': dx, 'dy': dy,
        'x0': x0, 'y0': y0
    }

    return results


def plot_final_state(results):
    """Plot the final state of all three field components."""
    Q = results['Q']
    x, y = results['x'], results['y']

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

    plt.suptitle('Lax-Wendroff Dipole Radiation — Final State',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('lw_dipole_final.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_snapshot_sequence(results, n_panels=6):
    """Plot a sequence of Ez snapshots."""
    snapshots = results['snapshots']
    times = results['snapshot_times']
    x, y = results['x'], results['y']

    indices = np.linspace(0, len(snapshots) - 1, n_panels, dtype=int)

    nrows = 2
    ncols = n_panels // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 10))
    axes = axes.flatten()

    vmax = np.max(np.abs(snapshots[-1])) * 0.8
    if vmax == 0:
        vmax = np.max(np.abs(snapshots[len(snapshots) // 2])) * 0.8
    if vmax == 0:
        vmax = 1.0

    for k, idx in enumerate(indices):
        ax = axes[k]
        im = ax.pcolormesh(
            x, y, snapshots[idx].T,
            shading='auto', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax
        )
        ax.set_title(f't = {times[idx]:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle(r'$E_z$ — Lax-Wendroff Dipole Radiation', fontsize=14)
    plt.tight_layout()
    plt.savefig('lw_dipole_sequence.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_field_along_axes(results):
    """Plot cross-sections of Ez through the dipole."""
    Q = results['Q']
    x, y = results['x'], results['y']
    x0, y0 = results['x0'], results['y0']

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

    plt.suptitle('Lax-Wendroff Cross-Sections Through Dipole',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('lw_dipole_cross_sections.png', dpi=150,
                bbox_inches='tight')
    plt.show()


def animate_field(results):
    """Create an animation of the Ez field evolution."""
    snapshots = results['snapshots']
    times = results['snapshot_times']
    x, y = results['x'], results['y']

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
    title = ax.set_title(f'$E_z$ at t = {times[0]:.4f}')
    fig.colorbar(im, ax=ax, shrink=0.8)

    def update(frame):
        im.set_array(snapshots[frame].T.ravel())
        title.set_text(f'$E_z$ at t = {times[frame]:.4f}')
        return im, title

    ani = FuncAnimation(
        fig, update, frames=len(snapshots),
        interval=80, blit=False
    )
    ani.save('lw_dipole_radiation.gif', writer='pillow', fps=15)
    plt.show()

    return ani


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

    T = 5.0
    dt = None  # 0.05  # 0.04  # None
    cfl = 0.9

    results = solve_maxwell_lax_wendroff(
        Lx=Lx * 2, Ly=Ly * 2,
        Nx=npts_x, Ny=npts_y,
        T_final=T,
        CFL=cfl,
        epsilon=1.0, mu=1.0,
        freq=omega / 2 / np.pi,
        J0=1.0,
    )

    plot_final_state(results)
    # plot_snapshot_sequence(results, n_panels=6)
    plot_field_along_axes(results)
    # ani = animate_field(results)