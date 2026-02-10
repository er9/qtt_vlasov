import numpy as np
import matplotlib.pyplot as plt

"""
updated current source to be more accurate...
i think the main difference is probably an updated treatment of the charge?
for the wave-propagation version
"""

# =============================================================================
# Physical Constants
# =============================================================================
c = 299792458.0
mu_0 = 4 * np.pi * 1e-7
epsilon_0 = 1.0 / (mu_0 * c ** 2)

# =============================================================================
# Simulation Parameters
# =============================================================================
Nx, Ny, Nz = 80, 80, 80
use_wave = True  # False

dx = 1e-2
dy = 1e-2
dz = 1e-2

cfl_limit = 1.0 / (c * np.sqrt(1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2))
dt = 0.95 * cfl_limit

n_steps = 400

src_x, src_y, src_z = Nx // 2, Ny // 2, Nz // 2

frequency = 2.0e9
omega = 2 * np.pi * frequency
wavelength = c / frequency

# Dipole moment amplitude (current * length, in A*m)
p0 = 1e-6

turn_on_time = 0.0 # 3.0 / frequency

print("=" * 70)
print("3D Maxwell Solver - Corrected Dipole Source")
print("=" * 70)
print(f"Grid size: {Nx} x {Ny} x {Nz} cells")
print(f"Cell size: dx = {dx * 1e3:.2f} mm")
print(f"Time step: dt = {dt * 1e12:.4f} ps")
print(f"Frequency: {frequency / 1e9:.2f} GHz")
print(f"Wavelength: {wavelength * 1e2:.2f} cm")
print(f"Grid extent: {Nx * dx * 1e2:.1f} cm")
print("=" * 70)

# =============================================================================
# Initialize Field Arrays
# =============================================================================
Ax_prev = np.zeros((Nx, Ny, Nz))
Ay_prev = np.zeros((Nx, Ny, Nz))
Az_prev = np.zeros((Nx, Ny, Nz))

Ax_curr = np.zeros((Nx, Ny, Nz))
Ay_curr = np.zeros((Nx, Ny, Nz))
Az_curr = np.zeros((Nx, Ny, Nz))

Ax_next = np.zeros((Nx, Ny, Nz))
Ay_next = np.zeros((Nx, Ny, Nz))
Az_next = np.zeros((Nx, Ny, Nz))

if use_wave:
    V_prev = np.zeros((Nx, Ny, Nz))
    V_curr = np.zeros((Nx, Ny, Nz))
    V_next = np.zeros((Nx, Ny, Nz))
else:
    ## if use gauge condition
    # For V with the Lorenz gauge approach, we use a leapfrog scheme
    # V is staggered in time: V is stored at half-integer time steps
    # V_half stores V at time (n - 1/2), V_half_new will be V at (n + 1/2)
    V_half = np.zeros((Nx, Ny, Nz))  # V at time (n - 1/2)
    V_curr = np.zeros((Nx, Ny, Nz))  # V at time n (interpolated)

c2dt2 = (c * dt) ** 2
c2 = c ** 2


# =============================================================================
# Helper Functions
# =============================================================================
def laplacian(f, dx, dy, dz):
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1, 1:-1] = (
            (f[2:, 1:-1, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / dx ** 2 +
            (f[1:-1, 2:, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / dy ** 2 +
            (f[1:-1, 1:-1, 2:] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / dz ** 2
    )
    return lap

def divergence(Ax, Ay, Az, dx, dy, dz):
    """
    Computes the discrete Laplacian of a 3D scalar field f using
    second-order central differences.

    ∇²f ≈ (Ax[i+1] - Ax[i-1])/2/dx + (same for y and z)

    Returns an array of the same shape with zeros at boundaries.
    """
    div = np.zeros_like(Ax)

    # Interior points only (boundaries handled separately)
    div[1:-1, 1:-1, 1:-1] = (
            (Ax[2:, 1:-1, 1:-1] - Ax[:-2, 1:-1, 1:-1]) / 2 / dx +
            (Ay[1:-1, 2:, 1:-1] - Ay[1:-1, :-2, 1:-1]) / 2 / dy +
            (Az[1:-1, 1:-1, 2:] - Az[1:-1, 1:-1, :-2]) / 2 / dz
    )
    return div

def apply_mur_abc(f_next, f_curr, f_prev, c, dt, dx, dy, dz):
    coef_x = (c * dt - dx) / (c * dt + dx)
    coef_y = (c * dt - dy) / (c * dt + dy)
    coef_z = (c * dt - dz) / (c * dt + dz)

    f_next[0, :, :] = f_curr[1, :, :] + coef_x * (f_next[1, :, :] - f_curr[0, :, :])
    f_next[-1, :, :] = f_curr[-2, :, :] + coef_x * (f_next[-2, :, :] - f_curr[-1, :, :])
    f_next[:, 0, :] = f_curr[:, 1, :] + coef_y * (f_next[:, 1, :] - f_curr[:, 0, :])
    f_next[:, -1, :] = f_curr[:, -2, :] + coef_y * (f_next[:, -2, :] - f_curr[:, -1, :])
    f_next[:, :, 0] = f_curr[:, :, 1] + coef_z * (f_next[:, :, 1] - f_curr[:, :, 0])
    f_next[:, :, -1] = f_curr[:, :, -2] + coef_z * (f_next[:, :, -2] - f_curr[:, :, -1])

    return f_next


# =============================================================================
# Precompute distance arrays for source injection
# =============================================================================
# Create coordinate arrays centered on the source
x = (np.arange(Nx) - src_x) * dx
y = (np.arange(Ny) - src_y) * dy
z = (np.arange(Nz) - src_z) * dz

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
R[R < dx / 2] = dx / 2  # Avoid division by zero at origin

# For the dipole radiation pattern
cos_theta = Z / R
sin_theta = np.sqrt(X ** 2 + Y ** 2) / R

# Unit vectors in spherical coordinates (expressed in Cartesian)
# r_hat components
r_hat_x = X / R
r_hat_y = Y / R
r_hat_z = Z / R

# theta_hat components (pointing toward increasing theta)
# theta_hat = cos(theta)*cos(phi)*x_hat + cos(theta)*sin(phi)*y_hat - sin(theta)*z_hat
rho_cyl = np.sqrt(X ** 2 + Y ** 2)
rho_cyl[rho_cyl < dx / 2] = dx / 2

theta_hat_x = cos_theta * X / rho_cyl
theta_hat_y = cos_theta * Y / rho_cyl
theta_hat_z = -sin_theta

# Handle the z-axis where rho_cyl is zero
on_axis = rho_cyl < dx
theta_hat_x[on_axis] = 0
theta_hat_y[on_axis] = 0
theta_hat_z[on_axis] = 0

# =============================================================================
# Storage
# =============================================================================
snapshot_interval = 10
Ez_snapshots_xy = []
Ez_snapshots_xz = []
snapshot_times = []

# =============================================================================
# Main Time Loop - Soft Source Injection
# =============================================================================
print("\nStarting time-stepping with soft source injection...")

for n in range(n_steps):
    t = n * dt

    # -------------------------------------------------------------------------
    # Step 1: Compute source terms
    # -------------------------------------------------------------------------
    # For a Hertzian dipole with moment p(t) = p0 * sin(omega*t) * envelope
    # oscillating along z, the current density and charge density are related by
    # the continuity equation.
    #
    # We use a "soft source" approach: inject the known analytical solution
    # for Az in a small region around the source.
    #
    # The retarded potential for Az in the near zone is approximately:
    # Az(r,t) ≈ (mu_0 / 4*pi) * (1/r) * d p(t_ret)/dt
    # where t_ret = t - r/c
    #
    # For the scalar potential in Lorenz gauge:
    # V(r,t) ≈ (1 / 4*pi*epsilon_0) * (cos_theta / r^2) * p(t_ret)
    #        + radiation term

    envelope = 1.0 - np.exp(-t / turn_on_time) if turn_on_time > 0 else 1.0
    d_envelope_dt = (1.0 / turn_on_time) * np.exp(-t / turn_on_time) if turn_on_time > 0 else 0.0

    # p(t) = p0 * sin(omega*t) * envelope
    # dp/dt = p0 * [omega * cos(omega*t) * envelope + sin(omega*t) * d_envelope_dt]
    # d²p/dt² = p0 * [-omega² * sin(omega*t) * envelope + 2*omega*cos(omega*t)*d_envelope_dt + ...]

    p_t = p0 * np.sin(omega * t) * envelope
    dp_dt = p0 * (omega * np.cos(omega * t) * envelope + np.sin(omega * t) * d_envelope_dt)

    # -------------------------------------------------------------------------
    # Step 2: Update A using wave equation with current source
    # -------------------------------------------------------------------------
    # Current density for a point dipole: J = dp/dt * delta(r) * z_hat
    # In discrete form, we spread this over a small region

    # Create current density array
    Jz = np.zeros((Nx, Ny, Nz))

    # Spread the current over a small Gaussian region for stability
    sigma = dx  #  1.5 * dx
    source_region = np.exp(-(R ** 2) / (2 * sigma ** 2))
    source_region = source_region / (np.sum(source_region) * dx * dy * dz)  # Normalize

    Jz = dp_dt * source_region

    # Charge density from continuity: ∂ρ/∂t + ∇·J = 0
    # For a z-directed current: ∂ρ/∂t = -∂Jz/∂z
    # Since Jz = dp/dt * f(r), we have ρ = -p(t) * ∂f/∂z
    # This gives us point charges at the ends of the dipole

    # Compute ∂(source_region)/∂z
    d_source_dz = np.zeros_like(source_region)
    d_source_dz[:, :, 1:-1] = (source_region[:, :, 2:] - source_region[:, :, :-2]) / (2 * dz)

    rho = -p_t * d_source_dz

    # Compute Laplacians
    lap_Ax = laplacian(Ax_curr, dx, dy, dz)
    lap_Ay = laplacian(Ay_curr, dx, dy, dz)
    lap_Az = laplacian(Az_curr, dx, dy, dz)
    lap_V = laplacian(V_curr, dx, dy, dz)

    # Update A components
    Ax_next[1:-1, 1:-1, 1:-1] = (
            2 * Ax_curr[1:-1, 1:-1, 1:-1]
            - Ax_prev[1:-1, 1:-1, 1:-1]
            + c2dt2 * lap_Ax[1:-1, 1:-1, 1:-1]
    )

    Ay_next[1:-1, 1:-1, 1:-1] = (
            2 * Ay_curr[1:-1, 1:-1, 1:-1]
            - Ay_prev[1:-1, 1:-1, 1:-1]
            + c2dt2 * lap_Ay[1:-1, 1:-1, 1:-1]
    )

    Az_next[1:-1, 1:-1, 1:-1] = (
            2 * Az_curr[1:-1, 1:-1, 1:-1]
            - Az_prev[1:-1, 1:-1, 1:-1]
            + c2dt2 * lap_Az[1:-1, 1:-1, 1:-1]
            + c2dt2 * mu_0 * Jz[1:-1, 1:-1, 1:-1]
    )

    if use_wave:
        # Update V with charge source
        V_next[1:-1, 1:-1, 1:-1] = (
                2 * V_curr[1:-1, 1:-1, 1:-1]
                - V_prev[1:-1, 1:-1, 1:-1]
                + c2dt2 * lap_V[1:-1, 1:-1, 1:-1]
                + c2dt2 * rho[1:-1, 1:-1, 1:-1] / epsilon_0
        )
        V_next = apply_mur_abc(V_next, V_curr, V_prev, c, dt, dx, dy, dz)

    else:
        div_A_curr = divergence(Ax_curr, Ay_curr, Az_curr, dx, dy, dz)

        # Update V at the half time step
        V_half_new = np.zeros_like(V_half)
        V_half_new[1:-1, 1:-1, 1:-1] = (
                V_half[1:-1, 1:-1, 1:-1]
                - c ** 2 * dt * div_A_curr[1:-1, 1:-1, 1:-1]
        )

        # Apply boundary conditions to V
        V_half_new = apply_mur_abc(V_half_new, V_half, None, c, dt, dx, dy, dz)

        # Interpolate to get V at integer time step for E field calculation
        # V^n ≈ (V^(n-1/2) + V^(n+1/2)) / 2
        V_curr = 0.5 * (V_half + V_half_new)

        # #### dV/dt   ## i guess this method is only first-order accurate...
        # dVdt = -c**2 * divergence(Ax_curr, Ay_curr, Az_curr, dx, dy, dz)
        # V_next = V_curr + dVdt * dt

    # Apply ABCs
    Ax_next = apply_mur_abc(Ax_next, Ax_curr, Ax_prev, c, dt, dx, dy, dz)
    Ay_next = apply_mur_abc(Ay_next, Ay_curr, Ay_prev, c, dt, dx, dy, dz)
    Az_next = apply_mur_abc(Az_next, Az_curr, Az_prev, c, dt, dx, dy, dz)


    # -------------------------------------------------------------------------
    # Step 3: Store snapshots
    # -------------------------------------------------------------------------
    if n % snapshot_interval == 0:
        # Compute E = -∇V - ∂A/∂t
        dV_dx = np.zeros((Nx, Ny, Nz))
        dV_dy = np.zeros((Nx, Ny, Nz))
        dV_dz = np.zeros((Nx, Ny, Nz))

        dV_dx[1:-1, :, :] = (V_curr[2:, :, :] - V_curr[:-2, :, :]) / (2 * dx)
        dV_dy[:, 1:-1, :] = (V_curr[:, 2:, :] - V_curr[:, :-2, :]) / (2 * dy)
        dV_dz[:, :, 1:-1] = (V_curr[:, :, 2:] - V_curr[:, :, :-2]) / (2 * dz)

        dAx_dt = (Az_next - Az_prev) / (2 * dt)
        dAy_dt = (Ay_next - Ay_prev) / (2 * dt)
        dAz_dt = (Az_next - Az_prev) / (2 * dt)

        Ex = -dV_dx - (Ax_next - Ax_prev) / (2 * dt)
        Ey = -dV_dy - (Ay_next - Ay_prev) / (2 * dt)
        Ez = -dV_dz - (Az_next - Az_prev) / (2 * dt)

        Ez_snapshots_xy.append(Ez[:, :, Nz // 2].copy())
        Ez_snapshots_xz.append(Ez[:, Ny // 2, :].copy())
        snapshot_times.append(t)

        if n % 50 == 0:
            print(f"  Step {n:4d}/{n_steps}, t = {t * 1e9:.3f} ns, "
                  f"max|Ez| = {np.max(np.abs(Ez)):.2e} V/m")

    # -------------------------------------------------------------------------
    # Step 4: Advance time
    # -------------------------------------------------------------------------
    Ax_prev, Ax_curr = Ax_curr.copy(), Ax_next.copy()
    Ay_prev, Ay_curr = Ay_curr.copy(), Ay_next.copy()
    Az_prev, Az_curr = Az_curr.copy(), Az_next.copy()
    if use_wave:
        V_prev, V_curr = V_curr.copy(), V_next.copy()
    else:
        V_half = V_half_new.copy()

print("\nTime-stepping complete!")

# =============================================================================
# Compute Final Fields
# =============================================================================
print("\nComputing final fields...")

dV_dx = np.zeros((Nx, Ny, Nz))
dV_dy = np.zeros((Nx, Ny, Nz))
dV_dz = np.zeros((Nx, Ny, Nz))

dV_dx[1:-1, :, :] = (V_curr[2:, :, :] - V_curr[:-2, :, :]) / (2 * dx)
dV_dy[:, 1:-1, :] = (V_curr[:, 2:, :] - V_curr[:, :-2, :]) / (2 * dy)
dV_dz[:, :, 1:-1] = (V_curr[:, :, 2:] - V_curr[:, :, :-2]) / (2 * dz)

Ex = -dV_dx - (Ax_curr - Ax_prev) / dt
Ey = -dV_dy - (Ay_curr - Ay_prev) / dt
Ez = -dV_dz - (Az_curr - Az_prev) / dt

Bx = np.zeros((Nx, Ny, Nz))
By = np.zeros((Nx, Ny, Nz))
Bz = np.zeros((Nx, Ny, Nz))

Bx[:, 1:-1, 1:-1] = (
        (Az_curr[:, 2:, 1:-1] - Az_curr[:, :-2, 1:-1]) / (2 * dy) -
        (Ay_curr[:, 1:-1, 2:] - Ay_curr[:, 1:-1, :-2]) / (2 * dz)
)

By[1:-1, :, 1:-1] = (
        (Ax_curr[1:-1, :, 2:] - Ax_curr[1:-1, :, :-2]) / (2 * dz) -
        (Az_curr[2:, :, 1:-1] - Az_curr[:-2, :, 1:-1]) / (2 * dx)
)

Bz[1:-1, 1:-1, :] = (
        (Ay_curr[2:, 1:-1, :] - Ay_curr[:-2, 1:-1, :]) / (2 * dx) -
        (Ax_curr[1:-1, 2:, :] - Ax_curr[1:-1, :-2, :]) / (2 * dy)
)

# =============================================================================
# Visualization
# =============================================================================
print("\nGenerating visualizations...")

fig = plt.figure(figsize=(16, 12))

# 1. Ez in xy-plane (z = Nz//2) - should show circular wavefronts
ax1 = fig.add_subplot(2, 2, 1)
z_slice = Nz // 2
Ez_xy = Ez[:, :, z_slice]
vmax1 = np.percentile(np.abs(Ez_xy), 98)
if vmax1 == 0:
    vmax1 = 1e-10
im1 = ax1.imshow(Ez_xy.T, origin='lower', cmap='RdBu_r',
                 extent=[-Nx * dx / 2 * 100, Nx * dx / 2 * 100, -Ny * dy / 2 * 100, Ny * dy / 2 * 100],
                 vmin=-vmax1, vmax=vmax1)
ax1.plot(0, 0, 'k*', markersize=15, label='Dipole')
ax1.set_xlabel('x (cm)')
ax1.set_ylabel('y (cm)')
ax1.set_title(f'Ez in xy-plane (z=0, equatorial plane)')
ax1.legend()
ax1.set_aspect('equal')
plt.colorbar(im1, ax=ax1, label='Ez (V/m)')

# 2. Ez in xz-plane (y = Ny//2) - should show butterfly pattern with null on z-axis
ax2 = fig.add_subplot(2, 2, 2)
y_slice = Ny // 2
Ez_xz = Ez[:, y_slice, :]
vmax2 = np.percentile(np.abs(Ez_xz), 98)
if vmax2 == 0:
    vmax2 = 1e-10
im2 = ax2.imshow(Ez_xz.T, origin='lower', cmap='RdBu_r',
                 extent=[-Nx * dx / 2 * 100, Nx * dx / 2 * 100, -Nz * dz / 2 * 100, Nz * dz / 2 * 100],
                 vmin=-vmax2, vmax=vmax2)
ax2.plot(0, 0, 'k*', markersize=15, label='Dipole')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Equatorial plane')
ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Dipole axis')
ax2.set_xlabel('x (cm)')
ax2.set_ylabel('z (cm)')
ax2.set_title(f'Ez in xz-plane (y=0) - Should show null along z-axis')
ax2.legend(loc='upper right')
ax2.set_aspect('equal')
plt.colorbar(im2, ax=ax2, label='Ez (V/m)')

# 3. Ez along z-axis (should be ~0 in far field for radiation)
ax3 = fig.add_subplot(2, 2, 3)
Ez_along_z = Ez[src_x, src_y, :]
z_coords = (np.arange(Nz) - src_z) * dz * 100  # in cm

# Also plot Ez along x-axis for comparison
Ez_along_x = Ez[:, src_y, src_z]
x_coords = (np.arange(Nx) - src_x) * dx * 100  # in cm

ax3.plot(z_coords, Ez_along_z, 'b-', linewidth=2, label='Along z-axis (should → 0)')
ax3.plot(x_coords, Ez_along_x, 'r--', linewidth=2, label='Along x-axis (max radiation)')
ax3.set_xlabel('Distance from dipole (cm)')
ax3.set_ylabel('Ez (V/m)')
ax3.set_title('Ez along principal axes')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# 4. |B| field in xy-plane (should show circular pattern)
ax4 = fig.add_subplot(2, 2, 4)
B_mag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
B_xy = B_mag[:, :, z_slice]
im4 = ax4.imshow(B_xy.T, origin='lower', cmap='hot',
                 extent=[-Nx * dx / 2 * 100, Nx * dx / 2 * 100, -Ny * dy / 2 * 100, Ny * dy / 2 * 100])
ax4.plot(0, 0, 'w*', markersize=15, label='Dipole')
ax4.set_xlabel('x (cm)')
ax4.set_ylabel('y (cm)')
ax4.set_title('|B| in xy-plane')
ax4.legend()
ax4.set_aspect('equal')
plt.colorbar(im4, ax=ax4, label='|B| (T)')

plt.tight_layout()
plt.savefig('dipole_corrected_fields.png', dpi=150, bbox_inches='tight')
print("Saved: dipole_corrected_fields.png")
plt.show()

# =============================================================================
# Wave propagation animation frames for xz-plane
# =============================================================================
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

n_frames = min(6, len(Ez_snapshots_xz))
frame_indices = np.linspace(len(Ez_snapshots_xz) // 3, len(Ez_snapshots_xz) - 1, n_frames, dtype=int)

all_max = np.percentile([np.abs(snap).max() for snap in Ez_snapshots_xz], 90)
if all_max == 0:
    all_max = 1e-10

for idx, frame_idx in enumerate(frame_indices):
    ax = axes[idx]
    snap = Ez_snapshots_xz[frame_idx]
    im = ax.imshow(snap.T, origin='lower', cmap='RdBu_r',
                   extent=[-Nx * dx / 2 * 100, Nx * dx / 2 * 100, -Nz * dz / 2 * 100, Nz * dz / 2 * 100],
                   vmin=-all_max, vmax=all_max)
    ax.plot(0, 0, 'k*', markersize=8)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('z (cm)')
    ax.set_title(f't = {snapshot_times[frame_idx] * 1e9:.2f} ns')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Ez (V/m)')

fig2.suptitle('Ez in xz-plane: Wave Propagation from z-oriented Dipole\n(Should show null along z-axis)', fontsize=12)
plt.tight_layout()
plt.savefig('dipole_wave_propagation_xz.png', dpi=150, bbox_inches='tight')
print("Saved: dipole_wave_propagation_xz.png")
plt.show()

# =============================================================================
# Quantitative check: Ez ratio between x-axis and z-axis
# =============================================================================
print("\n" + "=" * 70)
print("Quantitative Check: Ez radiation pattern")
print("=" * 70)

# Sample at a fixed distance from the source
check_distance = 10  # cells from center

Ez_at_x = Ez[src_x + check_distance, src_y, src_z]
Ez_at_z = Ez[src_x, src_y, src_z + check_distance]

print(f"At distance {check_distance * dx * 100:.1f} cm from dipole:")
print(f"  Ez along x-axis (equatorial): {Ez_at_x:.4e} V/m")
print(f"  Ez along z-axis (polar):      {Ez_at_z:.4e} V/m")
print(f"  Ratio |Ez(x)| / |Ez(z)|:      {abs(Ez_at_x) / (abs(Ez_at_z) + 1e-20):.2f}")
print(f"  (For ideal far-field dipole radiation, Ez along z should be ~0)")
print("=" * 70)