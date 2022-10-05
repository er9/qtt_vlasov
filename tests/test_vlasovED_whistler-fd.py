import os, sys, pickle, time, glob

import numpy as np

sys.path.append('../')

from setup.configs import *
import setup.helper as helper_test
import helper_quimb as helper
import setup_test as test_setup

from axis import Axis
import axis_map
from basis.basis_k import FourierBasis
from basis.basis_spatial import SpatialBasis
from basis.basis_hermite import AWHermiteGaussianBasis
from basis.basis_hermite import HermiteGaussianBasis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from grid_comb import GridsComb
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_vlasovEM import VlasovMaxwell, Maxwell
from pde_vlasovED import VlasovDarwin, Darwin
from vlasov_tests import Whistler

"""
Landau damping (k-space in space, real space in velocity)
"""

save_figs = False
save_data = False
load_data = False
restart   = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_pde_v2/data2/VM13-whistler-fd/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data2/VM13-whistler-fd/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)

#######################
# standard parameters #
#######################

plasma_config = UnitsConfiguration()  # eps0=1.0e-3, mu0=1.0e-3)
charge = plasma_config.e  # unit of charge
eV = plasma_config.eV     # unit of kB*T
mu0 = plasma_config.mu0   # permeability
eps0 = plasma_config.eps0 # permittivity

mass_e = 1.0            # mass of electron
mass_i = 1836 * mass_e  # mass of ion
n0_e = 1.0              # electron number density
n0_i = n0_e             # ion number density
T_e = 1.0e-6            # electron temperature [eV], so that vth_e/c = 1.0e-3
T_i = T_e               # ion temperature [eV]

ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config.vth    # electron thermal speed
wp_e  = elc_config.wp     # plasma e' frequency
lamD  = elc_config.lamD   # Debye length
wp_i  = ion_config.wp     # plasma i+ frequency
vth_i = ion_config.vth    # ion thermal speed
d_e   = elc_config.skin_depth

print('vth_e', vth_e/plasma_config.c)
print('wp_e', wp_e, 'lamD', lamD, 'elc skin depth', d_e)


def perturb_E_amplitude(k, vy_e, vz_e):
    """
    k: wavevector
    vy_i, vz_i:  amplitudes of velocity displacement
    solves for Ey, Ez
        i omega m vy_i = - e (Ey + vz_i Bx)
        eps0 (omega^2 + c^2 k^2) Ey = i omega n0 e vy_i
        --> Ey = -i omega m / e * vy_i - vz_i * Bx
        --> (-i omega) = ( Ey + vz_i * Bx ) * e / m / vy_i
        --> eps0 (-(Ey + vz_i * Bx)**2 * (e/m/vy_i)^2 + c^2/k^2) * Ey = - (Ey + vz_i * Bx ) * e/m * n0 * e
        --> eps0 (Ey^2 + 2 * vz_i * Bx * Ey + vz_i ^2 Bx^2) * (e/m/vy_i)^2 - c^2/k^2) Ey = ...
        ---> Ey^3 * (eps0 * (e/m/vy_i)^2) + Ey^2 * (eps0 * vz_i*Bx*(e/m/vy_i)^2) +
                Ey * (eps0 * vz_i^2 Bx^2 * (e/m/vy_i)^2 - c^2/k^2 - e/m * n0 * e) - vz_i * Bx * e/m * n0 * e = 0
    by symmetry, Ey = Ez
    THIS SHOULD BE CORRECT
    """
    e = elc_config.e * elc_config.Z
    m = elc_config.mass
    Bx0 = B0
    c_speed = plasma_config.c
    eps0 = plasma_config.eps0

    c3 = - eps0 * (e / m / vy_e) ** 2
    c2 = - 2 * eps0 * vz_e * Bx0 * (e / m / vy_e) ** 2
    c1 = + e ** 2 * n0_e / m - c_speed ** 2 * eps0 * k ** 2 - eps0 * vz_e ** 2 * Bx0 ** 2 * (e / m / vy_e) ** 2
    c0 = + vz_e * Bx0 * e ** 2 * n0_e / m
    Ey_vals = np.roots([c3, c2, c1, c0])
    print('Ey values?', Ey_vals)

    c3 = - eps0 * (e / m / vz_e) ** 2
    c2 = + 2 * eps0 * vy_e * Bx0 * (e / m / vz_e) ** 2
    c1 = + e ** 2 * n0_e / m - c_speed ** 2 * eps0 * k ** 2 - eps0 * vy_e ** 2 * Bx0 ** 2 * (e / m / vz_e) ** 2
    c0 = - vy_e * Bx0 * e ** 2 * n0_e / m
    Ez_vals = np.roots([c3, c2, c1, c0])
    print('Ez values?', Ez_vals)


    omega = 1.j * (Ey_vals + vz_e * Bx0) * e / m / vy_e
    omega_z = 1.j * (Ez_vals - vy_e * Bx0) * e / m / vz_e
    # Ey_check = 1.j * vy_i * omega * m / e - vz_i * Bx0
    Ey_check = 1.j * omega * n0_e * e * vy_e / eps0 / (omega ** 2 - c_speed ** 2 * k ** 2)
    print('E value check', Ey_check)    #
    print('omegas', omega)
    print('omega z', omega_z)

    select_ind = np.argmin(np.abs(np.imag(omega)))
    Ey = Ey_vals[select_ind]
    Ez = Ez_vals[select_ind]
    omega = omega[select_ind]
    # if np.real(omega) < 0:
    #     omega *= -1
    #     Ey_mag *= -1
    print('Ey', Ey, 'Ez', Ez, 'omega', omega)
    return Ey, Ez, omega


def initial_state_real(x_vals, vx_vals, vy_vals, vz_vals, A, k,
                       vth=1.0, density=1.0, flow=0.0):
    """ initial state for w
    """
    x_mesh, vx_mesh, vy_mesh, vz_mesh = np.meshgrid(x_vals, vx_vals, vy_vals, vz_vals, indexing='ij')

    dv_y = A*np.sin(k*x_mesh)
    dv_z = A*np.cos(k*x_mesh)
    vy_mesh = vy_mesh - dv_y
    vz_mesh = vz_mesh - dv_z

    out = helper_test.maxwellian_ndim( vx_mesh, vy_mesh, vz_mesh,
                                        vth2=vth**2, density=density, flow=flow )

    plt.figure()
    plt.imshow(out[:,len(vx_vals)//2,:,len(vz_vals)//2], aspect='auto')
    plt.title('f(x,vy)')
    plt.colorbar()
    plt.show()

    return out


def initial_state_k(kx_vals, vx_vals, vy_vals, vz_vals, A, k, vth=1.0, density=1.0, flow=0.0):
    """ initial state for w
    """
    dk = kx_vals[1] - kx_vals[0]
    x_vals = np.linspace(-np.pi/dk, np.pi/dk, len(kx_vals), endpoint=False)
    f_real = initial_state_real(x_vals, vx_vals, vy_vals, vz_vals, A, k, vth=vth, density=density, flow=flow)
    f_k = np.fft.fftshift( np.fft.fft(f_real, axis=0), axes=[0])

    plt.figure()
    plt.imshow(np.real(f_k[:, len(vx_vals) // 2, :, len(vz_vals) // 2]))
    plt.figure()
    plt.imshow(np.real(f_k[:, :, len(vy_vals) // 2, len(vz_vals) // 2]))
    plt.show()
    return f_k



#####################
# define parameters #
#####################

### system parameters ###
A = 1.e-3
# dk = 0.10 / lamD
dk = 0.10 / d_e
B0 = 1.0

## runtime parameter
q = 2
KX = 1
KV = 3
Lx, Lve, Lvi = 7, 6, 5

TN_layout = LayoutType.SEQUENTIAL
x_map_key = 'F'
v_map_key = 'B'

# TN_layout = LayoutType.COMB
# x_map_key = 'B'
# v_map_key = 'B'

order = 1   ## derivative order
T = 2000 # 50  # 2000 / wp_e
print('T', T)
do_adapt_dt = True
dt_frac = 0.9
te_order = 313
dt = None  # 0.05 # None # 0.005

save_every_nt = 250

DMAX = 32
DMAX_F = DMAX

VP_test = Whistler(ks=(dk,), A=A,
                   Lxs=(Lx,), Lves=(Lve,Lve,Lve), Lvis=(Lvi,Lvi,Lvi), KX=KX, KV=KV, q=q,
                   x_map_key = x_map_key, ve_map_key = v_map_key,
                   basis_xs = SpatialBasis(),
                   #
                   ion_config = ion_config, elc_config = elc_config,
                   coll_type=None, do_adapt_coll = True, coll_coeff_e = 0.005, coll_rate_e = 0.01,
                   DMAX = DMAX, DMAX_F = DMAX_F, compress_style = 2, compress_style_mod = 0,
                   do_adapt_dt=do_adapt_dt, dt_frac=dt_frac, dt=dt, T=T, te_order=te_order,
                   grid_layout=TN_layout
                   )


te_compress_i = VP_test.get_compression_config(DMAX=DMAX)
te_compress_e = VP_test.get_compression_config(DMAX=DMAX)
te_compress_E = VP_test.get_compression_config(DMAX=DMAX)
te_compress_B = VP_test.get_compression_config(DMAX=DMAX)

print('te compress fe', te_compress_e)
print('te compress fi', te_compress_i)
print('te compress E', te_compress_E)
print('te compress B', te_compress_B)

comp_levels = VP_test.get_compress_levels()

print('compress', DMAX, VP_test.compress_style, VP_test.compress_style_mod)

#############################
## start building PDE system
#############################

## initial conditions

## define axis objects; position -> k-space,
ax_x, = VP_test.pos_axes
ax_vx_i, ax_vy_i, ax_vz_i = VP_test.vi_axes
ax_vx_e, ax_vy_e, ax_vz_e = VP_test.ve_axes
print('ax_x zero ind', ax_x.zero_ind)

x_min, x_max = VP_test.x_lims[0]
print('x range',ax_x.dx, x_min, x_max)
ve_x_min, ve_x_max = VP_test.ve_lims[0]
ve_y_min, ve_y_max = VP_test.ve_lims[1]
ve_z_min, ve_z_max = VP_test.ve_lims[2]
vi_x_min, vi_x_max = VP_test.vi_lims[0]
vi_y_min, vi_y_max = VP_test.vi_lims[1]
vi_z_min, vi_z_max = VP_test.vi_lims[2]
tot_x = x_max - x_min

coords_x  = VP_test.pos_coordsys
coords_vi = VP_test.vi_coordsys
coords_ve = VP_test.ve_coordsys

X, Y, Z = coords_x.coords
VX, VY, VZ = coords_ve.coords

npts_x = ax_x.npts
npts_vxe = ax_vx_e.npts
npts_vye = ax_vy_e.npts
npts_vze = ax_vz_e.npts
npts_vxi = ax_vx_i.npts
npts_vyi = ax_vy_i.npts
npts_vzi = ax_vz_i.npts

# ## define MPS grids
grid_i = VP_test.grid_i
grid_e = VP_test.grid_e
grid_X = VP_test.grid_X

x_vals = ax_x.xpts
xB_vals = x_vals + ax_x.dx/2    # for staggered Yee cell

vx_e_vals = ax_vx_e.xpts
vy_e_vals = ax_vy_e.xpts
vz_e_vals = ax_vz_e.xpts
vx_i_vals = ax_vx_i.xpts
vy_i_vals = ax_vy_i.xpts
vz_i_vals = ax_vz_i.xpts


## electron density:  1 + A * np.cos(k * x)
fe_data = initial_state_real(x_vals, vx_e_vals, vy_e_vals, vz_e_vals, A, dk, vth=vth_e, flow=0.0, density=n0_e)

## ion density
fi_x = ax_x.get_iden_mps()
helper.scalar_multiply(fi_x, n0_i, inplace=True)
fi_vx = helper_test.maxwellian(vx_i_vals, vth2=vth_i**2, density=n0_i, flow=0.0)
fi_vy = helper_test.maxwellian(vy_i_vals, vth2=vth_i**2, density=n0_i, flow=0.0)
fi_vz = helper_test.maxwellian(vz_i_vals, vth2=vth_i**2, density=n0_i, flow=0.0)

## E field:
Ey_mag, Ez_mag, omega = perturb_E_amplitude(dk, A, -1.j*A)
# Ey = E_mag * np.cos(dk * x_vals)
# Ez = E_mag * np.sin(dk * x_vals)

## Bz field: - d/dx Bz = mu_0 (Jz + eps_0 dEz/dt) = mu_0 * (q * n0 * A + 1.j * eps_0 * omega * E1) * sin(kx)
# Bz = - mu0 * (charge * n0_e * A + 1.j * eps0 * omega * E_mag) / dk * np.cos(dk * xB_vals)
Bz = - mu0 * (charge * n0_e * A) / dk * np.cos(dk * xB_vals)

## By field:   d/dx By = mu_0 (Jy + eps_0 dEy/dt) = mu_0 * (q * n0 * A + 1.j * eps_0 * omega * E1) * cos(kx)
# By = - mu0 * (charge * n0_e * A + 1.j * eps0 * omega * E_mag) / dk * np.sin(dk * xB_vals)
By = - mu0 * (charge * n0_e * A) / dk * np.sin(dk * xB_vals)

### B0
Bx0 = ax_x.get_iden_mps()
helper.scalar_multiply(Bx0, B0, inplace=True)



## define initial grid1DTNs
init_fe_gtn = grid_e.map_state_to_mps(fe_data)

# fi_x = ax_x.map_state_to_mps(fi_x)
# fi_vx = ax_vx_i.map_state_to_mps(fi_vx)
# fi_vy = ax_vy_i.map_state_to_mps(fi_vy)
# fi_vz = ax_vz_i.map_state_to_mps(fi_vz)
init_fi_gtn = grid_i.make_mps_ndim({ax_x: fi_x, ax_vx_i: fi_vx, ax_vy_i: fi_vy, ax_vz_i: fi_vz})
init_Ex_gtn  = grid_X.make_empty_gridTN()
init_Ey_gtn  = grid_X.make_empty_gridTN()
init_Ez_gtn  = grid_X.make_empty_gridTN()
# init_Ey_gtn  = grid_X.make_mps_ndim({ax_x: Ey})
# init_Ez_gtn  = grid_X.make_mps_ndim({ax_x: Ez})
init_Bx0_gtn = grid_X.make_mps_ndim({ax_x: Bx0})
init_Bx_gtn  = grid_X.make_empty_gridTN()
init_By_gtn  = grid_X.make_mps_ndim({ax_x: By})
init_Bz_gtn  = grid_X.make_mps_ndim({ax_x: Bz})
init_phi_gtn = grid_X.make_empty_gridTN()
init_psi_gtn = grid_X.make_empty_gridTN()

deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
deriv_v = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)
init_fe_gtn.ax_deriv_configs.update({ax_x: deriv_x, ax_vx_e: deriv_v, ax_vy_e: deriv_v, ax_vz_e: deriv_v})
init_fi_gtn.ax_deriv_configs.update({ax_x: deriv_x, ax_vx_i: deriv_v, ax_vy_i: deriv_v, ax_vz_i: deriv_v})
init_Ex_gtn.ax_deriv_configs.update({ax_x: deriv_x})
init_Ey_gtn.ax_deriv_configs.update({ax_x: deriv_x})
init_Ez_gtn.ax_deriv_configs.update({ax_x: deriv_x})
init_Bx_gtn.ax_deriv_configs.update({ax_x: deriv_x})
init_By_gtn.ax_deriv_configs.update({ax_x: deriv_x})
init_Bz_gtn.ax_deriv_configs.update({ax_x: deriv_x})


### time step info
if dt is None:
    Fmax_e = B0 * np.array([0, ve_z_max, ve_y_max]) * charge / mass_e
    Fmax_i = B0 * np.array([0, vi_z_max, vi_y_max]) * charge / mass_i
    dt = VP_test.get_dt(Fmax_e, Fmax_i)
num_tsteps = int(T / dt)
print('dt', dt, num_tsteps)

### get file naming string
fstr = VP_test.get_fstr()
print('filename', fstr)

########################
#### run simulation ####
########################

### define v2 mpo for measurement ###
# vx2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vx_e], x_power=2)
# vy2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vy_e], x_power=2)
# vz2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vz_e], x_power=2)
# v2e_mpo = vx2e_mpo.add(vy2e_mpo)
# v2e_mpo = v2e_mpo.add(vz2e_mpo)
v2e_mpo = test_setup.get_v2_mpo(grid_e, VP_test.ve_axes)

# vx2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vx_i], x_power=2)
# vy2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vy_i], x_power=2)
# vz2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vz_i], x_power=2)
# v2i_mpo = vx2i_mpo.add(vy2i_mpo)
# v2i_mpo = v2i_mpo.add(vz2i_mpo)
v2i_mpo = test_setup.get_v2_mpo(grid_i, VP_test.vi_axes)


#### define fields
init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, compress_config=te_compress_e)
init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, compress_config=te_compress_i)
init_ET_field = Field('E', grid_X, data={}, is_sqrt=True, compress_config=te_compress_E)
init_EL_field = Field('E', grid_X, data={}, is_sqrt=True, compress_config=te_compress_E)
init_B_field = Field('B', grid_X, data={X: init_Bx_gtn, Y: init_By_gtn, Z: init_Bz_gtn},
                     is_sqrt=True, compress_config=te_compress_B)
B0_field = Field('B0',grid_X, data={X: init_Bx0_gtn})
curlB0_field = Field('B0', grid_X, data={})
init_phi_field = ScalarField('phi', grid_X, data=init_phi_gtn, is_sqrt=True, compress_config=te_compress_E)
init_psi_field = ScalarField('psi', grid_X, data=init_psi_gtn, is_sqrt=True, compress_config=te_compress_B)


### load data if restarting ###
try:
    out = test_setup.load_data_ED(VP_test,
                            init_fe_field, init_fi_field, init_ET_field, init_EL_field, init_B_field,
                            fdir=fdir, sdir=sdir, save_every_nt=save_every_nt,
                            load_data=load_data, restart=restart, restart_from_T=restart_from_T)

    (init_fe_field, init_fi_field, init_ET_field, init_EL_field, init_B_field,
     ts, nrg_fe_ts, nrg_fi_ts, nrg_E_ts, nrg_B_ts,
     max_bond_fe, max_bond_fi, max_bond_ET, max_bond_EL, max_bond_B) = out

    nt = len(ts) - 1
    restarted = True

except(IOError, OSError, NameError, ValueError, FileNotFoundError):

    ts = [0]
    nt = 0
    restarted = False

    nrg_fe = init_fe_gtn.meas_expec(v2e_mpo) * mass_e / 2 / tot_x
    nrg_fi = init_fi_gtn.meas_expec(v2i_mpo) * mass_i / 2 / tot_x
    # nrg_E  = init_E_field.norm() ** 2 * 1. / 2 / tot_x
    # nrg_B  = init_B_field.norm() ** 2 * 1. / 2 / tot_x
    nrg_E  = np.array([np.nan]*3)
    # nrg_B  = [np.nan] * 3 # A / dk
    nrg_B = np.array(init_B_field.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_x
    print('tot_x', dk, 2 * np.pi / dk, tot_x)

    ## maybe should initialize system, run EM_sys for half time step to start

    nrg_fe_ts = [nrg_fe]
    nrg_fi_ts = [nrg_fi]
    nrg_E_ts  = [nrg_E]
    nrg_B_ts  = [nrg_B]
    # nrg_By_ts = [nrg_B]

    print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
    max_bond_fe = [init_fe_gtn.max_bond()]
    max_bond_fi = [init_fi_gtn.max_bond()]
    max_bond_ET = [init_ET_field.max_bonds(compIDs=coords_x.coords) if init_ET_field is not None else np.nan]
    max_bond_EL = [init_EL_field.max_bonds(compIDs=coords_x.coords) if init_EL_field is not None else np.nan]
    max_bond_B = [init_B_field.max_bonds(compIDs=coords_x.coords) if init_B_field is not None else np.nan]
    max_bond_phi = [init_phi_field.max_bond() if init_phi_field is not None else np.nan]
    max_bond_psi = [init_psi_field.max_bond() if init_psi_field is not None else np.nan]

print('current nrgs', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1], nrg_B_ts[-1])

norm_fe = init_fe_gtn.norm()
norm_fi = init_fi_gtn.norm()
print('field norm', norm_fe, norm_fi)


### BUILD PDE
em_sys = Darwin(init_ET_field, init_EL_field, init_B_field,
                coords_x=coords_x, matl_params=plasma_config, te_order=te_order,
                background_B0=B0_field, normalize=False,)
em_sys.curlB0 = curlB0_field

vm_sys = VlasovDarwin(init_fe_field, init_fi_field, em_sys,
                       coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                       elc_params=elc_config, ion_params=ion_config, normalize=True, zipup=True,
                       te_order=te_order, compress_levels=comp_levels, )
print('initialized vm_sys')

vm_sys.collision = VP_test.collision_config
print('colls', vm_sys.collision.coll_params)

vm_sys.compress_F = VP_test.compress_F
vm_sys.compress_F_opts = VP_test.compress_F_opts
print('compress E?',  vm_sys.compress_F, vm_sys.compress_F_opts)

print('E nrg', vm_sys.E.norm()**2 * 1./ 2 / tot_x)
print('B nrg', vm_sys.B.norm()**2 * 1./ 2 / tot_x)


if nt == 0:
    fig1, ax1 = plt.subplots(2, 1)
    time_ = nt * dt
    fi_data = init_fi_gtn.get_data(ax_select={ax_vx_i: ax_vx_i.npts // 2, ax_vy_i: ax_vy_i.npts // 2})
    fi_data = ax_x.basis.get_realspace_1D(fi_data, 0)
    fi_data = ax_vz_i.basis.get_realspace_1D(fi_data, 1)
    im1 = ax1[0].imshow(np.real(fi_data).T,
                        extent=(-np.pi / dk, np.pi / dk, vi_z_min, vi_z_max),
                        aspect='auto', origin='lower')

    fe_data = init_fe_gtn.get_data(ax_select={ax_vx_e: ax_vx_e.npts // 2, ax_vy_e: ax_vy_e.npts // 2})
    fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
    fe_data = ax_vz_e.basis.get_realspace_1D(fe_data, 1)
    im2 = ax1[1].imshow(np.real(fe_data).T,
                        extent=(-np.pi / dk, np.pi / dk, ve_z_min, ve_z_max),
                        aspect='auto', origin='lower')

    fig1.colorbar(im1, ax=ax1[0])
    fig1.colorbar(im2, ax=ax1[1])
    ax1[0].set_title(f'time {time_:3.2f}')
    ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
    ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
    # if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')
    # plt.close(fig1)

    plt.show()


if restarted:
    if do_adapt_dt:
        em_term = vm_sys.compute_force_term(is_ion=False)
        max_em = []
        for coord in [X,Y,Z]:
            try:
                em_data = em_term.get_comp_data(coord, ax_select={ax_vx_e: 0, ax_vy_e: 0, ax_vz_e: 0})
                em_max = np.max(np.abs(em_data))
            except KeyError:
                em_max = 0.
            max_em += [em_max]

        Fe_maxs = np.array(max_em)
        # Fi_maxs = np.array(max_em) * charge / mass_i
        VP_test.get_dt(Fe_maxs)
        print('restarted dt', dt)

time1 = time.time()

print('starting time evolution')
while ts[-1] < T:

    walltime = time.time()
    vm_sys = vm_sys.next_time_step(dt, compress_level=1, err_tol=1.0e-7, verbose_plot=False,
                                   is_first_time_step=(nt==0))
    print('wall time', time.time() - walltime)

    fe_field = vm_sys.fe
    max_bond_fe += [fe_field.component.max_bond()]
    nrg_fe = fe_field.component.meas_expec(v2e_mpo) * mass_e / 2 / tot_x

    fi_field = vm_sys.fi
    max_bond_fi += [fi_field.component.max_bond()]
    nrg_fi = fi_field.component.meas_expec(v2i_mpo) * mass_i / 2 / tot_x

    E_field = vm_sys.E
    max_bond_ET += [vm_sys.ET.max_bonds(compIDs=coords_x.coords)]
    max_bond_EL += [vm_sys.EL.max_bonds(compIDs=coords_x.coords)]
    energy_E_t = np.array(E_field.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_x  # E field energy

    B_field = vm_sys.B
    max_bond_B += [B_field.max_bonds(compIDs=coords_x.coords)]
    energy_B_t = np.array(B_field.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_x  # B field energy
    # norm_BY = B_field.norms([Y])[0]
    # energy_By_t = norm_BY ** 2 * 1. / 2 / tot_x if norm_BY is not None else np.nan

    nrg_fe_ts += [nrg_fe]
    nrg_fi_ts += [nrg_fi]
    nrg_E_ts += [energy_E_t]
    nrg_B_ts += [energy_B_t]
    # nrg_By_ts += [energy_By_t]

    ts += [ts[-1] + dt]
    nt += 1

    if do_adapt_dt:
        em_term = vm_sys.compute_force_term(is_ion=False)
        max_em = []
        for coord in [X, Y, Z]:
            try:
                em_data = em_term.get_comp_data(coord, ax_select={ax_vx_e: 0, ax_vy_e: 0, ax_vz_e: 0})
                em_max = np.max(np.abs(em_data))
            except KeyError:
                em_max = 0.
            max_em += [em_max]

        Fe_maxs = np.array(max_em)
        dt = VP_test.get_dt(Fe_maxs)
        # print('restarted dt', dt)

    norm = vm_sys.fe.norm() - vm_sys.fi.norm()
    print('result norm', nt, ts[-1], dt, norm, # nrg_fe, nrg_fi,
          energy_E_t, energy_B_t,
          E_field.max_bond(X), E_field.max_bond(Y), E_field.max_bond(Z),
          B_field.max_bond(X), B_field.max_bond(Y), B_field.max_bond(Z),
          fi_field.max_bond(), fe_field.max_bond(), Lx, Lvi, Lve)

    if nt % (save_every_nt//5) == 0:

        if save_data:
            if not os.path.exists(sdir + 'restart/'):
                os.makedirs(sdir + 'restart/')

            test_setup.save_data_ED( VP_test, vm_sys, sdir+'restart/', add_fstr=f'-nt{nt}',
                                     ts=ts, nrg_fe_ts=nrg_fe_ts, nrg_fi_ts=nrg_fi_ts, nrg_E_ts=nrg_E_ts,
                                     nrg_B_ts=nrg_B_ts, max_bond_fe=max_bond_fe, max_bond_fi=max_bond_fi,
                                     max_bond_ET=max_bond_ET, max_bond_EL=max_bond_EL, max_bond_B=max_bond_B,)

        fig1, ax1 = plt.subplots(2, 1)
        time_ = ts[-1]
        # fi_data = fi_field.get_comp_data(ax_select={ax_vy_i: npts_vyi//2, ax_vz_i: npts_vzi//2})
        fi_data = fi_field.get_comp_data(ax_select={ax_vy_i: npts_vyi // 2, ax_vx_i: npts_vxi // 2})
        fi_data = ax_x.basis.get_realspace_1D(fi_data, 0)
        fi_data = ax_vz_i.basis.get_realspace_1D(fi_data, 1)
        im1 = ax1[0].imshow(np.real(fi_data).T, extent=(-np.pi/dk, np.pi/dk, vi_z_min, vi_z_max),
                            aspect='auto', origin='lower')

        # fe_data = fe_field.get_comp_data(ax_select={ax_vy_e: npts_vye//2, ax_vz_e: npts_vze//2})
        fe_data = init_fe_gtn.get_data(ax_select={ax_vx_e: ax_vx_e.npts // 2, ax_vy_e: ax_vy_e.npts // 2})
        fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
        fe_data = ax_vz_e.basis.get_realspace_1D(fe_data, 1)
        im2 = ax1[1].imshow(np.real(fe_data).T, extent=(-np.pi/dk, np.pi/dk, ve_z_min, ve_z_max),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].xaxis.set_visible(False)
        ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
        ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
        if save_figs:
            fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')
        else:
            plt.show()
        plt.close(fig1)

        real_xs = np.linspace(-np.pi / dk, np.pi / dk, npts_x, endpoint=False)

        plt.figure()
        E_data = vm_sys.E.get_comp_data(Y)
        E_data = ax_x.basis.get_realspace_1D(E_data, 0)
        plt.plot(real_xs, np.real(E_data), label='Ey field real')
        plt.plot(real_xs, np.imag(E_data), '--', label='Ey field imag')
        E_data = vm_sys.E.get_comp_data(Z)
        E_data = ax_x.basis.get_realspace_1D(E_data, 0)
        plt.plot(real_xs, np.real(E_data), label='Ez field real')
        plt.plot(real_xs, np.imag(E_data), '--', label='Ez field imag')
        plt.plot((real_xs[0],real_xs[-1]), [Ez_mag, Ez_mag],':')
        plt.legend()

        plt.figure()
        By_data = vm_sys.B.get_comp_data(Y)
        By_data = ax_x.basis.get_realspace_1D(By_data, 0)
        plt.plot(real_xs, np.real(By_data), label='By field real')
        plt.plot(real_xs, np.imag(By_data), '--', label='By field imag')
        Bz_data = vm_sys.B.get_comp_data(Z)
        Bz_data = ax_x.basis.get_realspace_1D(Bz_data, 0)
        plt.plot(real_xs, np.real(Bz_data), label='Bz field real')
        plt.plot(real_xs, np.imag(Bz_data), '--', label='Bz field imag')
        B_mag = mu0 * charge * n0_e * A / dk
        plt.plot((real_xs[0], real_xs[-1]), [B_mag, B_mag], ':')
        plt.legend()

        if not save_figs:
            plt.show()
        plt.close()
        plt.close()

        fig7, ax7 = plt.subplots()
        ax7.plot(ts, [nrg[1] for nrg in nrg_B_ts], label='By')
        ax7.legend()
        ax7.set_xlabel('t')
        ax7.set_ylabel('energy')
        ax7.legend()
        if save_figs:
            fig7.savefig(fdir + 'plot_nrg_By_' + fstr + '.png')
        else:
            plt.show()
        plt.close()

if save_data:
    test_setup.save_data_ED(VP_test, vm_sys, fdir,
                            ts=ts, nrg_fe_ts=nrg_fe_ts, nrg_fi_ts=nrg_fi_ts, nrg_E_ts=nrg_E_ts,
                            nrg_B_ts=nrg_B_ts, max_bond_fe=max_bond_fe, max_bond_fi=max_bond_fi,
                            max_bond_ET=max_bond_ET, max_bond_EL=max_bond_EL, max_bond_B=max_bond_B,)

    # vm_sys.fe.save_data(fdir + 'fe_' + fstr)
    # vm_sys.fi.save_data(fdir + 'fi_' + fstr)
    # vm_sys.E.save_data(fdir + 'E_' + fstr)
    # vm_sys.B.save_data(fdir + 'B_' + fstr)
    # vm_sys.phi.save_data(fdir + 'phi_' + fstr)
    # vm_sys.psi.save_data(fdir + 'psi_' + fstr)
    # pickle.dump(max_bond_fe, open(fdir + 'maxD_fe_' + fstr + '.pkl', 'wb'))
    # pickle.dump(max_bond_fi, open(fdir + 'maxD_fi_' + fstr + '.pkl', 'wb'))
    # pickle.dump(max_bond_E, open(fdir + 'maxD_E_' + fstr + '.pkl', 'wb'))
    # # pickle.dump(max_bond_B, open(fdir + 'maxD_B_' + fstr + '.pkl', 'wb'))
    # # pickle.dump(max_bond_phi, open(fdir + 'maxD_phi_' + fstr + '.pkl', 'wb'))
    # # pickle.dump(max_bond_psi, open(fdir + 'maxD_psi_' + fstr + '.pkl', 'wb'))
    # pickle.dump(nrg_fe_ts, open(fdir + 'nrg_e_' + fstr + '.pkl', 'wb'))
    # pickle.dump(nrg_fi_ts, open(fdir + 'nrg_i_' + fstr + '.pkl', 'wb'))
    # pickle.dump(nrg_E_ts, open(fdir + 'nrg_E_' + fstr + '.pkl', 'wb'))
    # pickle.dump(nrg_B_ts, open(fdir + 'nrg_B_' + fstr + '.pkl', 'wb'))
    # pickle.dump(ts, open(fdir + 'ts_' + fstr + '.pkl', 'wb'))

print('wall time', time.time() - time1)

fig1, ax1 = plt.subplots(2, 1)
time_ = ts[-1]
fi_data = vm_sys.fi.get_comp_data(ax_select={ax_vy_i: npts_vyi//2, ax_vz_i: npts_vzi//2})
fi_real = ax_x.basis.get_realspace_1D(fi_data, 0)
fi_real = ax_vz_i.basis.get_realspace_1D(fi_real, 1)
im1 = ax1[0].imshow(np.real(fi_real).T, extent=(-np.pi / dk, np.pi / dk, vi_x_min, vi_x_max),
                    aspect='auto', origin='lower')

fe_data = vm_sys.fe.get_comp_data(ax_select={ax_vy_e: npts_vye//2, ax_vz_e: npts_vze//2})
fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
fe_real = ax_vx_e.basis.get_realspace_1D(fe_real, 1)
im2 = ax1[1].imshow(np.real(fe_real).T, extent=(-np.pi / dk, np.pi / dk, ve_x_min, ve_x_max),
                    aspect='auto', origin='lower')

fig1.colorbar(im1, ax=ax1[0])
fig1.colorbar(im2, ax=ax1[1])
ax1[0].set_title(f'time {time_:3.2f}')
ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_T{time_:2.1f}.png')

fig2, ax2 = plt.subplots()
ax2.plot(np.log(max_bond_fe) / np.log(q), label=r'$f_e$')
ax2.plot(np.log(max_bond_fi) / np.log(q), label=r'$f_i$')
max_bond_Ey = [mb[1] if mb[1] is not None else np.nan for mb in max_bond_E]
ax2.plot(np.log(max_bond_Ey) / np.log(q), label='Ey')
max_bond_By = [mb[1] if mb[1] is not None else np.nan for mb in max_bond_B]
ax2.plot(np.log(max_bond_By) / np.log(q), label='By')
ax2.plot([0, num_tsteps], [(Lx+KV*Lve) // 2] * 2, ':', label='max fe')
ax2.plot([0, num_tsteps], [(Lx) // 2] * 2, ':', label='max V')
ax2.legend()
ax2.set_xlabel('t')
ax2.set_ylabel('log_{q^K}(max bond dimension)')
if save_figs:   fig2.savefig(fdir + 'plot_maxbond_' + fstr + '.png')

# energy_E_t = np.sum(Ex_t*np.conj(Ex_t),axis=1)*dx*1./2   # electric field energy
energy_E_t = np.array(nrg_E_ts)
energy_B_t = np.array(nrg_B_ts)
energy_fi_t = np.array(nrg_fi_ts)
energy_fe_t = np.array(nrg_fe_ts)
total_nrg_t = energy_fe_t + energy_fi_t + np.sum(energy_E_t,1) + np.sum(energy_B_t,1)
# delta_nrg_fe_t = energy_fe_t - energy_fe_t[0]
# delta_nrg_fi_t = energy_fi_t - energy_fi_t[0]
# delta_nrg_E_t  = energy_E_t  - energy_E_t[0]
# total_E_t = delta_nrg_fe_t + delta_nrg_fi_t + delta_nrg_E_t


##############

fig7, ax7 = plt.subplots()
# ts = np.arange(num_tsteps+1)*dt
ax7.semilogy(ts, np.abs(energy_fe_t), label='elec')
ax7.semilogy(ts, np.abs(energy_fi_t), label='ion')
ax7.semilogy(ts, np.abs(energy_E_t), label='E field')
ax7.semilogy(ts, np.abs(energy_B_t), label='B field')
ax7.semilogy(ts, np.abs(total_nrg_t), ls='--',label='total')

try:
    pass
    ref_fstr1 = f'testU1t_TN{TN_layout.value}_cut16_k{dk:1.2f}_A{Astr}'
    ref_fstr = ref_fstr1 + \
               f'_L{L:02d}_cfl{cfl}_T{T}_te{te_order}_m{mass_i}'
    nrg_E_ts = pickle.load(open(fdir + 'nrg_E_' + ref_fstr + '.pkl', 'rb'))
    ts = pickle.load(open(fdir + 'ts_' + ref_fstr + '.pkl', 'rb'))
    ax7.semilogy(ts, nrg_E_ts, 'k--', label='ref (cutoff 1.0e-8)')
except(NameError, IOError):
    pass

ax7.legend()
ax7.set_xlabel('t')
ax7.set_ylabel('energy')
# ax7.set_ylim(top=10**4)
if save_figs:
    fig7.savefig(fdir + 'plot_nrg_' + fstr + '.png')
    print(fdir + 'plot_nrg_' + fstr + '.png')

plt.show()


######## plot EE vs bond #####
fig7, ax7 = plt.subplots()
ax7.semilogy( vm_sys.fe.component.entanglement_entropy_all(), '-x', label='elec')
ax7.semilogy( vm_sys.fi.component.entanglement_entropy_all(), '-o', label='ion')

ax7.legend()
ax7.set_xlabel('bond number')
ax7.set_ylabel('EE')
if save_figs:
    fig7.savefig(fdir + 'plot_EE_' + fstr + '.png')
    print(fdir + 'plot_EE_' + fstr + '.png')

plt.show()


print('done')

