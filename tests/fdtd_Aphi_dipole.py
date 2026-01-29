import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# Physical Constants
# =============================================================================
c = 299792458.0  # Speed of light (m/s)
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
epsilon_0 = 1.0 / (mu_0 * c ** 2)  # Permittivity of free space (F/m)

# =============================================================================
# Simulation Parameters
# =============================================================================
# Grid dimensions (number of cells)
Nx, Ny, Nz = 60, 60, 60
use_wave = False  # True  # False

# Spatial resolution (cell size in meters)
dx = 1e-2
dy = 1e-2
dz = 1e-2

# Time step (must satisfy CFL condition: c * dt * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2) <= 1)
cfl_limit = 1.0 / (c * np.sqrt(1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2))
dt = 0.5 * cfl_limit # 0.99 * cfl_limit

# Number of time steps
n_steps = 300 # 300

# Dipole source parameters
# Location: center of the grid
src_x, src_y, src_z = Nx // 2, Ny // 2, Nz // 2

# Oscillation frequency (Hz) - chosen so wavelength is comparable to grid size
frequency = 1.5e9
omega = 2 * np.pi * frequency
wavelength = c / frequency

# Dipole current amplitude (Amperes)
I0 = 1.0

# Dipole length (meters) - should be small compared to wavelength for a Hertzian dipole
dipole_length = dx  # Approximate as one cell

# Smooth turn-on time constant (to avoid sharp transients)
turn_on_time = 2.0 / frequency

print("=" * 60)
print("3D Maxwell Solver using Vector Potentials (A-V Formulation)")
print("=" * 60)
print(f"Grid size: {Nx} x {Ny} x {Nz} cells")
print(f"Cell size: dx={dx * 1e3:.2f} mm, dy={dy * 1e3:.2f} mm, dz={dz * 1e3:.2f} mm")
print(f"Time step: dt = {dt * 1e12:.4f} ps")
print(f"Frequency: {frequency / 1e9:.2f} GHz")
print(f"Wavelength: {wavelength * 1e2:.2f} cm")
print(f"Grid extent: {Nx * dx * 1e2:.1f} x {Ny * dy * 1e2:.1f} x {Nz * dz * 1e2:.1f} cm")
print(f"Simulation time: {n_steps * dt * 1e9:.2f} ns")
print(f"CFL factor: {dt / cfl_limit:.2f}")
print("=" * 60)

# =============================================================================
# Initialize Field Arrays
# =============================================================================
# Magnetic vector potential A (three components)
# We store values at three time levels: previous (n-1), current (n), next (n+1)
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
    # Electric scalar potential V
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


# Current density J (only z-component for a z-oriented dipole)
Jz = np.zeros((Nx, Ny, Nz))

# Charge density rho
rho = np.zeros((Nx, Ny, Nz))

# Convenience factor
c2dt2 = (c * dt) ** 2


# =============================================================================
# Helper Function: Discrete Laplacian
# =============================================================================
def laplacian(f, dx, dy, dz):
    """
    Computes the discrete Laplacian of a 3D scalar field f using
    second-order central differences.

    ∇²f ≈ (f[i+1] - 2f[i] + f[i-1])/dx² + (same for y and z)

    Returns an array of the same shape with zeros at boundaries.
    """
    lap = np.zeros_like(f)

    # Interior points only (boundaries handled separately)
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



# =============================================================================
# Helper Function: Absorbing Boundary Conditions (Simple Mur ABC)
# =============================================================================
def apply_mur_abc(f_next, f_curr, f_prev, c, dt, dx, dy, dz):
    """
    Applies first-order Mur absorbing boundary conditions to all six faces
    of the computational domain. This allows outgoing waves to exit the
    domain with minimal reflection.

    The Mur condition approximates an outgoing wave: ∂f/∂t + c * ∂f/∂n = 0
    where n is the outward normal direction.
    """
    # Mur coefficient
    coef_x = (c * dt - dx) / (c * dt + dx)
    coef_y = (c * dt - dy) / (c * dt + dy)
    coef_z = (c * dt - dz) / (c * dt + dz)

    # x = 0 face (left)
    f_next[0, :, :] = f_curr[1, :, :] + coef_x * (f_next[1, :, :] - f_curr[0, :, :])

    # x = Nx-1 face (right)
    f_next[-1, :, :] = f_curr[-2, :, :] + coef_x * (f_next[-2, :, :] - f_curr[-1, :, :])

    # y = 0 face (front)
    f_next[:, 0, :] = f_curr[:, 1, :] + coef_y * (f_next[:, 1, :] - f_curr[:, 0, :])

    # y = Ny-1 face (back)
    f_next[:, -1, :] = f_curr[:, -2, :] + coef_y * (f_next[:, -2, :] - f_curr[:, -1, :])

    # z = 0 face (bottom)
    f_next[:, :, 0] = f_curr[:, :, 1] + coef_z * (f_next[:, :, 1] - f_curr[:, :, 0])

    # z = Nz-1 face (top)
    f_next[:, :, -1] = f_curr[:, :, -2] + coef_z * (f_next[:, :, -2] - f_curr[:, :, -1])

    return f_next


# =============================================================================
# Storage for Visualization
# =============================================================================
# Store field snapshots at the central z-plane
snapshot_interval = 10
Ez_snapshots = []
snapshot_times = []

# =============================================================================
# Main Time-Stepping Loop
# =============================================================================
print("\nStarting time-stepping loop...")

for n in range(n_steps):
    t = n * dt

    # -------------------------------------------------------------------------
    # Step 1: Update Source Terms (Hertzian Dipole)
    # -------------------------------------------------------------------------
    # The dipole is a small current element oscillating in the z-direction
    # Current: I(t) = I0 * sin(omega * t) * smooth_envelope(t)
    # Current density: Jz = I(t) / (cell_volume) at the source location

    # Smooth turn-on envelope to avoid sharp transients
    envelope = 1.0 - np.exp(-t / turn_on_time) if turn_on_time > 0 else 1.0

    # Oscillating current
    I_t = I0 * np.sin(omega * t) * envelope

    # Current density (Amperes per square meter)
    # For a Hertzian dipole, we approximate the current as distributed over one cell
    cell_volume = dx * dy * dz
    Jz[src_x, src_y, src_z] = I_t * dipole_length / cell_volume

    # Charge density from continuity equation: ∂ρ/∂t = -∇·J
    # For a z-directed current changing in time:
    # dρ/dt ≈ -dJz/dz, but for a point dipole we use: ρ = ±q at the dipole ends
    # Here we approximate: d(I)/dt = I0 * omega * cos(omega*t) * envelope
    dI_dt = I0 * omega * np.cos(omega * t) * envelope

    # Charge accumulates at the ends of the dipole (±dz/2 from center)
    # This is a simplified model; for a true point dipole both charges are at the same point
    # We place positive charge just above and negative just below the center
    if src_z > 0 and src_z < Nz - 1:
        # Rate of charge accumulation: dq/dt = I(t), so q accumulates
        # We model the charge density contribution
        charge_factor = dI_dt * dipole_length / (cell_volume * c ** 2)
        rho[src_x, src_y, src_z + 1] = charge_factor
        rho[src_x, src_y, src_z - 1] = -charge_factor

    # -------------------------------------------------------------------------
    # Step 2: Compute Laplacians of Current Potentials
    # -------------------------------------------------------------------------
    lap_Ax = laplacian(Ax_curr, dx, dy, dz)
    lap_Ay = laplacian(Ay_curr, dx, dy, dz)
    lap_Az = laplacian(Az_curr, dx, dy, dz)
    lap_V = laplacian(V_curr, dx, dy, dz)

    # -------------------------------------------------------------------------
    # Step 3: Update Potentials Using Wave Equation
    # -------------------------------------------------------------------------
    # Wave equation: ∇²A - (1/c²)∂²A/∂t² = -μ₀J
    # Discretized: A_next = 2*A_curr - A_prev + (c*dt)² * (∇²A + μ₀J)
    # Note: The source term sign depends on convention; here we use +μ₀J
    # because the wave equation is ∇²A - (1/c²)∂²A/∂t² = -μ₀J
    # Rearranging: A_next = 2*A_curr - A_prev + c²dt²*∇²A + c²dt²*μ₀J
    #            = 2*A_curr - A_prev + c²dt²*∇²A + dt²*J/ε₀

    # Interior update for Ax (no source in x-direction)
    Ax_next[1:-1, 1:-1, 1:-1] = (
            2 * Ax_curr[1:-1, 1:-1, 1:-1]
            - Ax_prev[1:-1, 1:-1, 1:-1]
            + c2dt2 * lap_Ax[1:-1, 1:-1, 1:-1]
    )

    # Interior update for Ay (no source in y-direction)
    Ay_next[1:-1, 1:-1, 1:-1] = (
            2 * Ay_curr[1:-1, 1:-1, 1:-1]
            - Ay_prev[1:-1, 1:-1, 1:-1]
            + c2dt2 * lap_Ay[1:-1, 1:-1, 1:-1]
    )

    # Interior update for Az (has source Jz)
    Az_next[1:-1, 1:-1, 1:-1] = (
            2 * Az_curr[1:-1, 1:-1, 1:-1]
            - Az_prev[1:-1, 1:-1, 1:-1]
            + c2dt2 * lap_Az[1:-1, 1:-1, 1:-1]
            + c2dt2 * mu_0 * Jz[1:-1, 1:-1, 1:-1]
    )

    if use_wave:
        # Interior update for V (has source rho)
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
                - c**2 * dt * div_A_curr[1:-1, 1:-1, 1:-1]
        )

        # Apply boundary conditions to V
        V_half_new = apply_mur_abc(V_half_new, V_half, None, c, dt, dx, dy, dz)

        # Interpolate to get V at integer time step for E field calculation
        # V^n ≈ (V^(n-1/2) + V^(n+1/2)) / 2
        V_curr = 0.5 * (V_half + V_half_new)

        # #### dV/dt   ## i guess this method is only first-order accurate...
        # dVdt = -c**2 * divergence(Ax_curr, Ay_curr, Az_curr, dx, dy, dz)
        # V_next = V_curr + dVdt * dt

    # -------------------------------------------------------------------------
    # Step 4: Apply Absorbing Boundary Conditions
    # -------------------------------------------------------------------------
    Ax_next = apply_mur_abc(Ax_next, Ax_curr, Ax_prev, c, dt, dx, dy, dz)
    Ay_next = apply_mur_abc(Ay_next, Ay_curr, Ay_prev, c, dt, dx, dy, dz)
    Az_next = apply_mur_abc(Az_next, Az_curr, Az_prev, c, dt, dx, dy, dz)

    # -------------------------------------------------------------------------
    # Step 5: Store Snapshots for Visualization
    # -------------------------------------------------------------------------
    if n % snapshot_interval == 0:
        # Compute Ez from potentials: E = -∇V - ∂A/∂t
        # Ez = -∂V/∂z - ∂Az/∂t

        # ∂V/∂z using central differences
        dV_dz = np.zeros((Nx, Ny, Nz))
        dV_dz[:, :, 1:-1] = (V_curr[:, :, 2:] - V_curr[:, :, :-2]) / (2 * dz)

        # ∂Az/∂t using central differences in time
        dAz_dt = (Az_next - Az_prev) / (2 * dt)

        # Ez field
        Ez = -dV_dz - dAz_dt

        # Store the central xy-plane (z = Nz//2)
        Ez_snapshots.append(Ez[:, :, Nz // 2].copy())
        snapshot_times.append(t)

        if n % 50 == 0:
            print(f"  Step {n}/{n_steps}, t = {t * 1e9:.3f} ns, max|Ez| = {np.max(np.abs(Ez)):.2e} V/m")

    # -------------------------------------------------------------------------
    # Step 6: Advance Time Levels
    # -------------------------------------------------------------------------
    Ax_prev, Ax_curr = Ax_curr.copy(), Ax_next.copy()
    Ay_prev, Ay_curr = Ay_curr.copy(), Ay_next.copy()
    Az_prev, Az_curr = Az_curr.copy(), Az_next.copy()
    if use_wave:
        V_prev, V_curr = V_curr.copy(), V_next.copy()
    else:
        V_half = V_half_new.copy()
        # V_prev, V_curr = V_curr.copy(), V_next.copy()

print("Time-stepping complete!")

# =============================================================================
# Compute Final Fields from Potentials
# =============================================================================
print("\nComputing final E and B fields from potentials...")

# Electric field: E = -∇V - ∂A/∂t
# Using central differences for spatial derivatives
# Using the last two time steps for time derivative

# Spatial derivatives of V
dV_dx = np.zeros((Nx, Ny, Nz))
dV_dy = np.zeros((Nx, Ny, Nz))
dV_dz = np.zeros((Nx, Ny, Nz))

dV_dx[1:-1, :, :] = (V_curr[2:, :, :] - V_curr[:-2, :, :]) / (2 * dx)
dV_dy[:, 1:-1, :] = (V_curr[:, 2:, :] - V_curr[:, :-2, :]) / (2 * dy)
dV_dz[:, :, 1:-1] = (V_curr[:, :, 2:] - V_curr[:, :, :-2]) / (2 * dz)

# Time derivatives of A
dAx_dt = (Ax_curr - Ax_prev) / dt
dAy_dt = (Ay_curr - Ay_prev) / dt
dAz_dt = (Az_curr - Az_prev) / dt

# Electric field components
Ex = -dV_dx - dAx_dt
Ey = -dV_dy - dAy_dt
Ez = -dV_dz - dAz_dt

# Magnetic field: B = ∇ × A
# Bx = ∂Az/∂y - ∂Ay/∂z
# By = ∂Ax/∂z - ∂Az/∂x
# Bz = ∂Ay/∂x - ∂Ax/∂y

Bx = np.zeros((Nx, Ny, Nz))
By = np.zeros((Nx, Ny, Nz))
Bz = np.zeros((Nx, Ny, Nz))

# Compute curl components using central differences
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

print(f"Final field magnitudes:")
print(f"  max|Ex| = {np.max(np.abs(Ex)):.2e} V/m")
print(f"  max|Ey| = {np.max(np.abs(Ey)):.2e} V/m")
print(f"  max|Ez| = {np.max(np.abs(Ez)):.2e} V/m")
print(f"  max|Bx| = {np.max(np.abs(Bx)):.2e} T")
print(f"  max|By| = {np.max(np.abs(By)):.2e} T")
print(f"  max|Bz| = {np.max(np.abs(Bz)):.2e} T")

# =============================================================================
# Visualization
# =============================================================================
print("\nGenerating visualizations...")

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Ez field in central xy-plane (z = Nz//2)
ax1 = fig.add_subplot(2, 2, 1)
z_slice = Nz // 2
Ez_slice = np.log10(np.abs(Ez[:, :, z_slice]))
vmax = np.max(np.abs(Ez_slice)) * 0.8
if vmax == 0:
    vmax = 1e-10
im1 = ax1.imshow(Ez_slice.T, origin='lower', cmap='RdBu_r',
                 extent=[0, Nx * dx * 100, 0, Ny * dy * 100],
                 vmin=-vmax, vmax=vmax)
ax1.plot(src_x * dx * 100, src_y * dy * 100, 'k*', markersize=15, label='Dipole')
ax1.set_xlabel('x (cm)')
ax1.set_ylabel('y (cm)')
ax1.set_title(f'Ez field in xy-plane (z = {z_slice * dz * 100:.1f} cm)')
ax1.legend()
plt.colorbar(im1, ax=ax1, label='Ez (V/m)')

# 2. Ez field in central xz-plane (y = Ny//2)
ax2 = fig.add_subplot(2, 2, 2)
y_slice = Ny // 2
Ez_xz = np.log10(np.abs(Ez[:, y_slice, :]))
vmax2 = np.max(np.abs(Ez_xz)) * 0.8
if vmax2 == 0:
    vmax2 = 1e-10
im2 = ax2.imshow(Ez_xz.T, origin='lower', cmap='RdBu_r',
                 extent=[0, Nx * dx * 100, 0, Nz * dz * 100],
                 vmin=-vmax2, vmax=vmax2)
ax2.plot(src_x * dx * 100, src_z * dz * 100, 'k*', markersize=15, label='Dipole')
ax2.set_xlabel('x (cm)')
ax2.set_ylabel('z (cm)')
ax2.set_title(f'Ez field in xz-plane (y = {y_slice * dy * 100:.1f} cm)')
ax2.legend()
plt.colorbar(im2, ax=ax2, label='Ez (V/m)')

# 3. Magnetic field magnitude in xy-plane
ax3 = fig.add_subplot(2, 2, 3)
# B_mag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
B_mag = np.abs(Bx)
B_slice = np.log10(B_mag[:, :, z_slice])
im3 = ax3.imshow(B_slice.T, origin='lower', cmap='hot',
                 extent=[0, Nx * dx * 100, 0, Ny * dy * 100])
ax3.plot(src_x * dx * 100, src_y * dy * 100, 'w*', markersize=15, label='Dipole')
ax3.set_xlabel('x (cm)')
ax3.set_ylabel('y (cm)')
ax3.set_title(f'|B| field in xy-plane (z = {z_slice * dz * 100:.1f} cm)')
ax3.legend()
plt.colorbar(im3, ax=ax3, label='|B| (T)')

# 4. Time evolution of Ez at a monitoring point
ax4 = fig.add_subplot(2, 2, 4)
if len(Ez_snapshots) > 0:
    # Extract Ez at a point offset from the source
    monitor_x = src_x + 10
    monitor_y = src_y
    Ez_time_series = [snap[monitor_x, monitor_y] for snap in Ez_snapshots]
    times_ns = [t * 1e9 for t in snapshot_times]
    ax4.plot(times_ns, Ez_time_series, 'b-', linewidth=1.5)
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Ez (V/m)')
    ax4.set_title(
        f'Ez vs time at ({monitor_x * dx * 100:.1f}, {monitor_y * dy * 100:.1f}, {z_slice * dz * 100:.1f}) cm')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('maxwell_potential_solver_results.png', dpi=150, bbox_inches='tight')
print("Saved: maxwell_potential_solver_results.png")
plt.show()

# Create animation frames showing wave propagation
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

n_frames = min(6, len(Ez_snapshots))
frame_indices = np.linspace(0, len(Ez_snapshots) - 1, n_frames, dtype=int)

# Find global color scale
all_max = max(np.max(np.abs(snap)) for snap in Ez_snapshots)
if all_max == 0:
    all_max = 1e-10

for idx, frame_idx in enumerate(frame_indices):
    ax = axes[idx]
    snap = np.log10(np.abs(Ez_snapshots[frame_idx]))
    im = ax.imshow(snap.T, origin='lower', cmap='RdBu_r',
                   extent=[0, Nx * dx * 100, 0, Ny * dy * 100],
                   # vmin=-all_max * 0.8, vmax=all_max * 0.8
                   )
    ax.plot(src_x * dx * 100, src_y * dy * 100, 'k*', markersize=10)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title(f't = {snapshot_times[frame_idx] * 1e9:.2f} ns')
    plt.colorbar(im, ax=ax, label='Ez (V/m)')

fig2.suptitle('Wave Propagation from Radiating Dipole (Ez in xy-plane)', fontsize=14)
plt.tight_layout()
plt.savefig('maxwell_wave_propagation.png', dpi=150, bbox_inches='tight')
print("Saved: maxwell_wave_propagation.png")
plt.show()

# =============================================================================
# Verify Divergence of B (should be ~0)
# =============================================================================
print("\nVerifying ∇·B = 0 (Gauss's law for magnetism)...")
div_B = np.zeros((Nx, Ny, Nz))
div_B[1:-1, 1:-1, 1:-1] = (
        (Bx[2:, 1:-1, 1:-1] - Bx[:-2, 1:-1, 1:-1]) / (2 * dx) +
        (By[1:-1, 2:, 1:-1] - By[1:-1, :-2, 1:-1]) / (2 * dy) +
        (Bz[1:-1, 1:-1, 2:] - Bz[1:-1, 1:-1, :-2]) / (2 * dz)
)
print(f"  max|∇·B| = {np.max(np.abs(div_B)):.2e} T/m")
print(f"  This should be very small (numerical noise level)")

print("\nSimulation complete!")