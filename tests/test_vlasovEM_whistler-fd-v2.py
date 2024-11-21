import os, sys, pickle, time, glob

import numpy as np
import matplotlib
import scipy.optimize

sys.path.append('../')

from setup_.configs import *
import setup_.helper as helper_test
import helper_quimb as helper
import setup_test as test_setup

from axis import Axis
import axis_map
from basis.basis_spatial import SpatialBasis
from basis.basis_hermite import AWHermiteGaussianBasis
from basis.basis_hermite import HermiteGaussianBasis
from field import Field, ScalarField
from pde_vlasovEM import VlasovMaxwell, Maxwell
from vlasov_tests import Whistler

"""
Whistler wave with only real part
(previous verions is complex f)
"""

save_figs = False
save_data = False  # True
load_data = False  # True
restart   = False  # True
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

is_sqrt = False # True
if restart or save_data or save_figs:
    if is_sqrt:
        sdir = '/pool001/erikaye/tns_pde_v2/data15/VM13-whistler-fd-sq-v2/'
        fdir = 'data15/VM13-whistler-fd-sq-v2/'
    else:
        sdir = '/pool001/erikaye/tns_pde_v2/data15/VM13-whistler-fd/'
        fdir = 'data15/VM13-whistler-fd/'

    if not os.path.exists(sdir):
        os.makedirs(sdir)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)

extra_str = ''

#######################
# standard parameters #
#######################

plasma_config = UnitsConfiguration()  # eps0=1.0e-3, mu0=1.0e-3)
charge = plasma_config.e  # unit of charge
eV = plasma_config.eV  # unit of kB*T
mu0 = plasma_config.mu0  # permeability
eps0 = plasma_config.eps0  # permittivity

mass_e = 1.0  # mass of electron
mass_i = 1836 * mass_e  # mass of ion
n0_e = 1.0  # electron number density
n0_i = n0_e  # ion number density
T_e = 1.0e-6  # electron temperature [eV], so that vth_e/c = 1.0e-3
T_i = T_e  # ion temperature [eV]

ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config.vth  # electron thermal speed
wp_e = elc_config.wp  # plasma e' frequency
lamD = elc_config.lamD  # Debye length
wp_i = ion_config.wp  # plasma i+ frequency
vth_i = ion_config.vth  # ion thermal speed
d_e = elc_config.skin_depth

print('vth_e', vth_e, 'vth_i', vth_i, 'c', plasma_config.c)
print('wp_e', wp_e, 'lamD', lamD)
print('d_e', elc_config.skin_depth, 'd_p', ion_config.skin_depth)


# def perturb_E_amplitude(k, vy_e, vz_e):
#     """
#     k: wavevector
#     vy_i, vz_i:  amplitudes of velocity displacement
#     solves for Ey, Ez
#         i omega m vy_i = - e (Ey + vz_i Bx)
#         eps0 (omega^2 + c^2 k^2) Ey = i omega n0 e vy_i
#         --> Ey = -i omega m / e * vy_i - vz_i * Bx
#         --> (-i omega) = ( Ey + vz_i * Bx ) * e / m / vy_i
#         --> eps0 (-(Ey + vz_i * Bx)**2 * (e/m/vy_i)^2 + c^2/k^2) * Ey = - (Ey + vz_i * Bx ) * e/m * n0 * e
#         --> eps0 (Ey^2 + 2 * vz_i * Bx * Ey + vz_i ^2 Bx^2) * (e/m/vy_i)^2 - c^2/k^2) Ey = ...
#         ---> Ey^3 * (eps0 * (e/m/vy_i)^2) + Ey^2 * (eps0 * vz_i*Bx*(e/m/vy_i)^2) +
#                 Ey * (eps0 * vz_i^2 Bx^2 * (e/m/vy_i)^2 - c^2/k^2 - e/m * n0 * e) - vz_i * Bx * e/m * n0 * e = 0
#     by symmetry, Ey = Ez
#     THIS SHOULD BE CORRECT
#     """
#     e = elc_config.e * elc_config.Z
#     m = elc_config.mass
#     Bx0 = B0
#     c_speed = plasma_config.c
#     eps0 = plasma_config.eps0
#
#     c3 = - eps0 * (e / m / vy_e) ** 2
#     c2 = - 2 * eps0 * vz_e * Bx0 * (e / m / vy_e) ** 2
#     c1 = + e ** 2 * n0_e / m - c_speed ** 2 * eps0 * k ** 2 - eps0 * vz_e ** 2 * Bx0 ** 2 * (e / m / vy_e) ** 2
#     c0 = + vz_e * Bx0 * e ** 2 * n0_e / m
#     Ey_vals = np.roots([c3, c2, c1, c0])
#     print('Ey values?', Ey_vals)
#
#     c3 = - eps0 * (e / m / vz_e) ** 2
#     c2 = + 2 * eps0 * vy_e * Bx0 * (e / m / vz_e) ** 2
#     c1 = + e ** 2 * n0_e / m - c_speed ** 2 * eps0 * k ** 2 - eps0 * vy_e ** 2 * Bx0 ** 2 * (e / m / vz_e) ** 2
#     c0 = - vy_e * Bx0 * e ** 2 * n0_e / m
#     Ez_vals = np.roots([c3, c2, c1, c0])
#     print('Ez values?', Ez_vals)
#
#     omega = 1.j * (Ey_vals + vz_e * Bx0) * e / m / vy_e
#     omega_z = 1.j * (Ez_vals - vy_e * Bx0) * e / m / vz_e
#     # Ey_check = 1.j * vy_i * omega * m / e - vz_i * Bx0
#     Ey_check = 1.j * omega * n0_e * e * vy_e / eps0 / (omega ** 2 - c_speed ** 2 * k ** 2)
#     print('E value check', Ey_check)  #
#     print('omegas', omega)
#     print('omega z', omega_z)
#
#     select_ind = np.argmin(np.abs(np.imag(omega)))
#     Ey = Ey_vals[select_ind]
#     Ez = Ez_vals[select_ind]
#     omega = omega[select_ind]
#     # if np.real(omega) < 0:
#     #     omega *= -1
#     #     Ey_mag *= -1
#     print('Ey', Ey, 'Ez', Ez, 'omega', omega)
#     return Ey, Ez, omega


# def initial_state_real(x_vals, vx_vals, vy_vals, vz_vals, A, k,
#                        vth=1.0, density=1.0, flow=0.0):
#     """ initial state for w
#     """
#     x_mesh, vx_mesh, vy_mesh, vz_mesh = np.meshgrid(x_vals, vx_vals, vy_vals, vz_vals, indexing='ij')
#
#     dv_y = A * np.sin(k * x_mesh)
#     dv_z = A * np.cos(k * x_mesh)
#     vy_mesh = vy_mesh - dv_y
#     vz_mesh = vz_mesh - dv_z
#
#     out = helper_test.maxwellian_ndim(vx_mesh, vy_mesh, vz_mesh,
#                                       vth2=vth ** 2, density=density, flow=flow)
#
#     plt.figure()
#     plt.imshow(out[:, len(vx_vals) // 2, :, len(vz_vals) // 2], aspect='auto')
#     plt.title('f(x,vy)')
#     plt.colorbar()
#     plt.show()
#
#     return out


def initial_state_k(kx_vals, vy_vals, vz_vals, vth=1.0, density=1.0, flow=0.0):
    """ initial state for w
    """
    dk = kx_vals[1] - kx_vals[0]
    x_vals = np.linspace(-np.pi / dk, np.pi / dk, len(kx_vals), endpoint=False)
    f_real = initial_state_real_vy_vz(x_vals, vy_vals, vz_vals, vth=vth, density=density, flow=flow)
    f_k = np.fft.fftshift(np.fft.fft(f_real, axis=0), axes=[0])

    plt.figure()
    plt.imshow(np.real(f_k[:, len(vx_vals) // 2, :, len(vz_vals) // 2]))
    plt.figure()
    plt.imshow(np.real(f_k[:, :, len(vy_vals) // 2, len(vz_vals) // 2]))
    plt.show()
    return f_k


def initial_state_real_vy_vz(x_vals, vy_vals, vz_vals, vth=1.0, density=1.0, flow=0.0):
    """ initial state for w
    """
    x_mesh, vy_mesh, vz_mesh = np.meshgrid(x_vals, vy_vals, vz_vals, indexing='ij')

    dv_y = dvey_func(x_mesh)
    dv_z = dvez_func(x_mesh)
    vy_mesh = vy_mesh - dv_y
    vz_mesh = vz_mesh - dv_z

    out = helper_test.maxwellian_ndim(vy_mesh, vz_mesh,
                                      vth2=vth ** 2, density=density, flow=flow)


    npts = len(vy_vals)

    plt.figure()
    plt.imshow(np.real(out[:, npts//2, :]))
    plt.title('real fe')
    plt.figure()
    plt.imshow(np.imag(out[:, npts // 2, :]))
    plt.title('imag fe')
    plt.colorbar()
    plt.show()

    return out


def calc_omega(k):
    ## n^2 = (k c / omega)^2 = 1 - wp_e ** 2 / (w + wC_e) / (w + wC_i)
    guess = B0 * k ** 2 / (1 + k ** 2)
    n2 = lambda w: (dk * plasma_config.c / w)**2
    if lr_sign > 0:
        disp = lambda w:  1 - wp_e ** 2 / (w + wC_e) / (w + wC_i)
    else:
        disp = lambda w:  1 - wp_e ** 2 / (w - wC_e) / (w - wC_i)

    out = scipy.optimize.fsolve( lambda w: n2(w) - disp(w),  guess)
    return out.item()



#####################
# define parameters #
#####################

### system parameters ###
A = 1.e-3  # * (vth_e/plasma_config.c)
dk = 10.0 / d_e  # 0.10 / d_e
B0 = 1.0
lr_sign = 1

dvey_coeff = A
dvez_coeff = -A

dvey_func = lambda x: dvey_coeff * np.cos(dk * x)
dvez_func = lambda x: dvez_coeff * np.sin(dk * x)


wC_i = ion_config.cyclotron_frequency(B0)
wC_e = elc_config.cyclotron_frequency(B0)

print('dk / de', dk, d_e)
print('OmegaC_i', ion_config.cyclotron_frequency(B0), 'OmegaC_e', elc_config.cyclotron_frequency(B0))
print('omega', calc_omega(dk), 2 * np.pi / calc_omega(dk))
print('dt max?', 2. * np.pi / wC_e)
print('dt max?', 2. * np.pi / wp_e, wp_e)
# exit()

## runtime parameter
q = 2
KX = 1
KV = 3
# Lx, Lve, Lvi = 6, 7, 6  ## [413] dt = 0.212 - 0.168  [433] similar; EM dt = 0.98
Lx, Lve, Lvi = 5, 5, 5
# Lx, Lve, Lvi = 7, 7, 7  ## [413] dt = 0.105 - 0.08   [433] similar; EM dt = 0.49
# Lx, Lve, Lvi = 8, 8, 8

order = 2   ## derivative order
if order != 1:
    extra_str = f'_o{order}' + extra_str

TN_layout = LayoutType.SEQUENTIAL
x_map_key = 'F'
v_map_key = 'F'  # 'B'

# # TN_layout = LayoutType.PARALLEL_GROUP
# TN_layout = LayoutType.PARALLEL
# x_map_key = 'F'
# v_map_key = 'F'

# TN_layout = LayoutType.COMB
# x_map_key = 'F' # 'B'
# v_map_key = 'F' # 'B'

grid_idx = 2

T = np.round(min(3000, 50 * 2 * np.pi / calc_omega(dk)))  # 50  # 2000 / wp_e
print('T', T)

# do_adapt_dt = True
# dt_frac = 0.9    # 0.9
# te_order = 413
# dt = None

do_adapt_dt = True  # True  # False
dt_frac = 0.9  # 2.0  # 2.0  # 0.9  # 0.9
te_order = 313   # 31 # 313  # 3184   # 4164  # 4164
dt_max = 1.0   # 1.0 for k=0.10, limited by EM oscillations
dt = 0.01      # min( round(0.01 / omega(dk), 4)

te_order_EM = 31  # 4, 31, 223
is_yee = True if te_order_EM == 31 else False

save_every_nt = 250  # 125 # 250  # 250

DMAX = 8  # None  #  64   # 16
DMAX_F = DMAX
comp_style = 3

DMAX_EM = DMAX  # 64  # DMAX
if DMAX_EM != DMAX:
   extra_str = f'_DE{DMAX_EM}' + extra_str

VP_test = Whistler(ks=(dk,), A=A,
                   Lxs=(Lx,), Lves=(Lve, Lve, Lve), Lvis=(Lvi, Lvi, Lvi), KX=KX, KV=KV, q=q,
                   x_map_key=x_map_key, ve_map_key=v_map_key, grid_idx=grid_idx,
                   basis_xs=SpatialBasis(),
                   #
                   ion_config=ion_config, elc_config=elc_config,
                   coll_type=None, do_adapt_coll=True, coll_coeff_e=0.005, coll_rate_e=0.01,
                   DMAX=DMAX, DMAX_F=DMAX_F, compress_style=comp_style, compress_style_mod=0,
                   do_adapt_dt=do_adapt_dt, dt_frac=dt_frac, dt=dt, T=T, te_order=te_order, te_order_EM=te_order_EM,
                   grid_layout=TN_layout
                   )

# CUTOFF = 1.0e-16
te_compress_i = VP_test.get_compression_config(DMAX=DMAX, cutoff=CUTOFF)
te_compress_e = VP_test.get_compression_config(DMAX=DMAX, cutoff=CUTOFF)
te_compress_E = VP_test.get_compression_config(DMAX=DMAX_EM)
te_compress_B = VP_test.get_compression_config(DMAX=DMAX_EM)

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
print('x range', ax_x.dx, x_min, x_max)
ve_x_min, ve_x_max = VP_test.ve_lims[0]
ve_y_min, ve_y_max = VP_test.ve_lims[1]
ve_z_min, ve_z_max = VP_test.ve_lims[2]
vi_x_min, vi_x_max = VP_test.vi_lims[0]
vi_y_min, vi_y_max = VP_test.vi_lims[1]
vi_z_min, vi_z_max = VP_test.vi_lims[2]
tot_x = x_max - x_min

coords_x = VP_test.pos_coordsys
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
xB_vals = x_vals + ax_x.dx / 2 if is_yee else x_vals  # for staggered Yee cell

vx_e_vals = ax_vx_e.xpts
vy_e_vals = ax_vy_e.xpts
vz_e_vals = ax_vz_e.xpts
vx_i_vals = ax_vx_i.xpts
vy_i_vals = ax_vy_i.xpts
vz_i_vals = ax_vz_i.xpts

## EM stability condition:  dE / dt = - c curl B + 4 pi J
##                          dB / dt = -c curl E
##       --> dt < dx / c * dB / dE ~ dt < dx / c
##       --> FDTD ref says dt < dx / vp for 1D, * 1/sqrt(2) for 2D
if str(te_order_EM)[:2] not in ['21', '22']:
    dt_max = min(dt_max, 0.1 * ax_x.dx / plasma_config.c)
print('EM dt max', 0.1 * ax_x.dx / plasma_config.c)
print('EM dt max', dt_max, ax_x.dx, plasma_config.c)

## electron density:  1 + A * np.cos(k * x)
# fe_data = initial_state_real(x_vals, vx_e_vals, vy_e_vals, vz_e_vals, A, dk, vth=vth_e, flow=0.0, density=n0_e)
# if is_sqrt:
#     fe_data = np.sqrt(fe_data)

fe_x_vy_vz = initial_state_real_vy_vz(x_vals, vy_e_vals, vz_e_vals, vth=vth_e, flow=0.0, density=n0_e)
fe_vx = helper_test.maxwellian(vx_e_vals, vth2=vth_e ** 2, density=1.0, flow=0.0)
if is_sqrt:
    fe_x_vy_vz = np.sqrt(fe_x_vy_vz)
    fe_vx = np.sqrt(fe_vx)

ax_inds_list = [(0, 2, 3), (1,)]
gtn_data_list = [fe_x_vy_vz, fe_vx]
init_fe_gtn = grid_e.build_mps_from_subgtns(list(zip(ax_inds_list, gtn_data_list)))
print('init fe max bond', init_fe_gtn.max_bond())
init_fe_gtn.compress(compress_opts={'max_bond': DMAX})

## ion density
fi_x = ax_x.get_iden_mps()
if is_sqrt:
    helper.scalar_multiply(fi_x, np.sqrt(n0_i), inplace=True)
else:
    helper.scalar_multiply(fi_x, n0_i, inplace=True)
fi_vx = helper_test.maxwellian(vx_i_vals, vth2=vth_i ** 2, density=1.0, flow=0.0)
fi_vy = helper_test.maxwellian(vy_i_vals, vth2=vth_i ** 2, density=1.0, flow=0.0)
fi_vz = helper_test.maxwellian(vz_i_vals, vth2=vth_i ** 2, density=1.0, flow=0.0)
if is_sqrt:
    fi_vx = np.sqrt(fi_vx)
    fi_vy = np.sqrt(fi_vy)
    fi_vz = np.sqrt(fi_vz)

init_fi_gtn = grid_i.make_mps_ndim({ax_x: fi_x, ax_vx_i: fi_vx, ax_vy_i: fi_vy, ax_vz_i: fi_vz})
print('init fi max bond', init_fi_gtn.max_bond())
print('init fi gtn norm', init_fi_gtn.frobenius_norm())
print('init fi', init_fi_gtn.norm(is_sqrt=is_sqrt), is_sqrt)
init_fi_gtn.compress(compress_opts={'max_bond': DMAX})

plt.figure()
plt.imshow(init_fi_gtn.get_data(ax_select={ax_vx_i: ax_vx_i.npts//2, ax_vy_i: ax_vy_i.npts//2}))
plt.title('fi')
plt.figure()
plt.imshow(init_fe_gtn.get_data(ax_select={ax_vx_e: ax_vx_e.npts//2, ax_vy_e: ax_vy_e.npts//2}))
plt.title('fe')
plt.show()


## E field:
#### get Ey, Ez (new) ####
omega = calc_omega(dk)
c = plasma_config.c
Jy_coeff = -1. * n0_e * charge * dvey_coeff
Jz_coeff = -1. * n0_e * charge * dvez_coeff
Jy = Jy_coeff * np.cos(dk * x_vals)
Jz = Jz_coeff * np.sin(dk * x_vals)
Ey_coeff = 1. / (1 - (c * dk / omega) ** 2) / (eps0 * omega) * Jy_coeff
Ez_coeff = 1. / (1 - (c * dk / omega) ** 2) / (eps0 * omega) * Jz_coeff * -1
Ey = Ey_coeff * np.sin(dk * x_vals)
Ez = Ez_coeff * np.cos(dk * x_vals)

By_coeff = - dk / omega * Ez_coeff
Bz_coeff = + dk / omega * Ey_coeff
By = By_coeff * np.cos(dk * xB_vals)
Bz = Bz_coeff * np.sin(dk * xB_vals)

Ey_mag = Ez_mag = np.max(np.abs(Ey))
print('predicted omega', omega)


print('Ey', -omega * Ey_coeff, -c**2 * dk * Bz_coeff - 1/eps0 * Jy_coeff)
print('Ez', omega * Ez_coeff, -c**2 * dk * By_coeff - 1/eps0 * Jz_coeff)
print('By', omega * By_coeff, -dk * Ez_coeff)
print('Bz', -omega * Bz_coeff, -dk * Ey_coeff)
# exit()

# plt.figure()
# plt.plot(Ey, '--', label='Ey')
# plt.plot(Ez, '--', label='Ez')
# plt.plot(By, label='By')
# plt.plot(Bz, label='Bz')
# plt.legend()
# plt.show()

### B0
Bx0 = ax_x.get_iden_mps()
helper.scalar_multiply(Bx0, B0, inplace=True)

## define initial grid1DTNs
# init_fe_gtn = grid_e.map_state_to_mps(fe_data)

# fi_x = ax_x.map_state_to_mps(fi_x)
# fi_vx = ax_vx_i.map_state_to_mps(fi_vx)
# fi_vy = ax_vy_i.map_state_to_mps(fi_vy)
# fi_vz = ax_vz_i.map_state_to_mps(fi_vz)
# init_fi_gtn = grid_i.make_mps_ndim({ax_x: fi_x, ax_vx_i: fi_vx, ax_vy_i: fi_vy, ax_vz_i: fi_vz})

init_Ex_gtn = grid_X.make_empty_gridTN()
# init_Ey_gtn  = grid_X.make_empty_gridTN()
# init_Ez_gtn  = grid_X.make_empty_gridTN()
init_Ey_gtn = grid_X.make_mps_ndim({ax_x: Ey})
init_Ez_gtn = grid_X.make_mps_ndim({ax_x: Ez})
init_Bx0_gtn = grid_X.make_mps_ndim({ax_x: Bx0})
init_Bx0_gtn.is_constant = True
init_Bx_gtn = grid_X.make_empty_gridTN()
# init_By_gtn  = grid_X.make_empty_gridTN()
# init_Bz_gtn  = grid_X.make_empty_gridTN()
init_By_gtn = grid_X.make_mps_ndim({ax_x: By})
init_Bz_gtn = grid_X.make_mps_ndim({ax_x: Bz})
init_phi_gtn = grid_X.make_empty_gridTN()
init_psi_gtn = grid_X.make_empty_gridTN()

deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
if te_order in [3154, 3164, 3174, 3184, 3150, 3160, 4154, 4164, 4174, 4184, 4150, 4160, 64, 60]:
    deriv_v = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
else:
    deriv_v = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)
init_fe_gtn.ax_deriv_configs.update({ax_x: deriv_x, ax_vx_e: deriv_v, ax_vy_e: deriv_v, ax_vz_e: deriv_v})
init_fi_gtn.ax_deriv_configs.update({ax_x: deriv_x, ax_vx_i: deriv_v, ax_vy_i: deriv_v, ax_vz_i: deriv_v})
deriv_x_EM = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
init_Ex_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})
init_Ey_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})
init_Ez_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})
init_Bx_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})
init_By_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})
init_Bz_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})

### time step info
if te_order in [412, 413, 433, 414, 416, 4154, 4164, 4174, 4184, 4160, 4150]:
    dt_cfl = VP_test.get_dt(np.array([0, np.abs(Ey_mag), np.abs(Ez_mag)]) * charge / mass_e, do_advec_x_cfl=False)
else:
    Fmax_e = (B0 * np.array([0, ve_z_max, ve_y_max]) + np.abs(Ey_mag)) * charge / mass_e
    Fmax_i = (B0 * np.array([0, vi_z_max, vi_y_max]) + np.abs(Ez_mag)) * charge / mass_i
    dt_cfl = VP_test.get_dt(Fmax_e, Fmax_i)

print('dt', dt, dt_cfl, dt_max)
dt = dt_cfl * dt_frac if dt is None or do_adapt_dt else dt
dt = min(dt_max, dt)
num_tsteps = int(T / dt)
print('dt', dt, num_tsteps)

### get file naming string
if is_sqrt:
    extra_str += '_sq'
fstr = VP_test.get_fstr(extra_str=extra_str)
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
init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, is_sqrt=is_sqrt, compress_config=te_compress_e)
init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, is_sqrt=is_sqrt, compress_config=te_compress_i)
init_E_field = Field('E', grid_X, data={X: init_Ex_gtn, Y: init_Ey_gtn, Z: init_Ez_gtn},
                     is_sqrt=True, compress_config=te_compress_E)
init_B_field = Field('B', grid_X, data={X: init_Bx_gtn, Y: init_By_gtn, Z: init_Bz_gtn},
                     is_sqrt=True, compress_config=te_compress_B)
B0_field = Field('B0', grid_X, data={X: init_Bx0_gtn})
curlB0_field = Field('B0', grid_X, data={})
init_phi_field = ScalarField('phi', grid_X, data=init_phi_gtn, is_sqrt=True, compress_config=te_compress_E)
init_psi_field = ScalarField('psi', grid_X, data=init_psi_gtn, is_sqrt=True, compress_config=te_compress_B)

### load data if restarting ###
try:
    out = test_setup.load_data_EM(VP_test,
                                  init_fe_field, init_fi_field, init_E_field, init_B_field, init_phi_field,
                                  init_psi_field,
                                  fdir=fdir, sdir=sdir, extra_str=extra_str, save_every_nt=save_every_nt,
                                  load_data=load_data, restart=restart, restart_from_T=restart_from_T)

    (init_fe_field, init_fi_field, init_E_field, init_B_field, init_phi_field, init_psi_field,
     ts, nrg_fe_ts, nrg_fi_ts, nrg_E_ts, nrg_B_ts,
     max_bond_fe, max_bond_fi, max_bond_E, max_bond_B, max_bond_phi, max_bond_psi) = out

    nt = len(ts) - 1
    restarted = True

except(IOError, OSError, NameError, ValueError, FileNotFoundError):

    ts = [0]
    nt = 0
    restarted = False

    nrg_fe = init_fe_gtn.meas_expec(v2e_mpo, is_sqrt=is_sqrt) * mass_e / 2 / tot_x
    nrg_fi = init_fi_gtn.meas_expec(v2i_mpo, is_sqrt=is_sqrt) * mass_i / 2 / tot_x
    # nrg_E  = init_E_field.norm() ** 2 * 1. / 2 / tot_x
    # nrg_B  = init_B_field.norm() ** 2 * 1. / 2 / tot_x
    nrg_E = np.array([np.nan] * 3)
    # nrg_B  = [np.nan] * 3 # A / dk
    nrg_B = np.array(init_B_field.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_x
    print('tot_x', dk, 2 * np.pi / dk, tot_x)

    ## maybe should initialize system, run EM_sys for half time step to start

    nrg_fe_ts = [nrg_fe]
    nrg_fi_ts = [nrg_fi]
    nrg_E_ts = [nrg_E]
    nrg_B_ts = [nrg_B]
    # nrg_By_ts = [nrg_B]

    Ex_mags = [0]
    Ey_mags = [Ey[npts_x // 2]]  # [(Ey_mag[npts_x//2])]
    Ez_mags = [Ez[npts_x // 2]]  # [(Ez_mag[npts_x//2])]

    print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
    max_bond_fe = [init_fe_gtn.max_bond()]
    max_bond_fi = [init_fi_gtn.max_bond()]
    max_bond_E = [init_E_field.max_bonds(compIDs=coords_x.coords) if init_E_field is not None else np.nan]
    max_bond_B = [init_B_field.max_bonds(compIDs=coords_x.coords) if init_B_field is not None else np.nan]
    max_bond_phi = [init_phi_field.max_bond() if init_phi_field is not None else np.nan]
    max_bond_psi = [init_psi_field.max_bond() if init_psi_field is not None else np.nan]

print('current nrgs', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1], nrg_B_ts[-1])

norm_fe = init_fe_gtn.norm(is_sqrt=is_sqrt)
norm_fi = init_fi_gtn.norm(is_sqrt=is_sqrt)
if is_sqrt:
    print('dist norm', is_sqrt, norm_fe ** 2, norm_fi ** 2)
else:
    print('dist norm', is_sqrt, norm_fe, norm_fi)

### BUILD PDE
em_sys = Maxwell(init_E_field, init_B_field, phi=init_phi_field, psi=init_psi_field,
                 coords_x=coords_x, matl_params=plasma_config, te_order=te_order_EM,
                 background_B0=B0_field, normalize=False,
                 conservative=False,
                 is_electrostatic=False, clean=False, is_yee=is_yee)
em_sys.curlB0 = curlB0_field

vm_sys = VlasovMaxwell(init_fe_field, init_fi_field, em_sys,
                       evolve_ion=False,
                       coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                       elc_params=elc_config, ion_params=ion_config, normalize=True, zipup=True,
                       conservative=False,
                       te_order=te_order, compress_levels=comp_levels, )
print('initialized vm_sys', vm_sys.te_order)

### initial current / velocity displacement
print('n0', n0_e, n0_i)
jze = vm_sys.sys_fe.compute_flow() # * (1./ n0_i)       ## no charge
jzi = vm_sys.sys_fi.compute_flow() # * (1./ n0_e)
jz = vm_sys.compute_current()

plt.figure()
plt.plot(ax_x.xpts, np.real(jze.get_comp_data(Y)), label='e (y) re')
plt.plot(ax_x.xpts, np.imag(jze.get_comp_data(Y)), '--', label='e (y) im')
# plt.plot(ax_x.xpts, jzi.get_comp_data(Y), '--', label='i (y)')
plt.plot(ax_x.xpts, np.real(jze.get_comp_data(Z)), label='e (z) re')
plt.plot(ax_x.xpts, np.imag(jze.get_comp_data(Z)), '--', label='e (z) im')
# plt.plot(ax_x.xpts, jzi.get_comp_data(Z), '--', label='i (z)')
# plt.plot(ax_x.xpts, jz.get_comp_data(Y), label='J (y)')
# plt.plot(ax_x.xpts, jz.get_comp_data(Z), label='J (z)')
plt.title('velocity displacement and current')
plt.legend()

plt.figure()
plt.plot(ax_x.xpts, np.real(jze.get_comp_data(Y) - dvey_func(x_vals)), label='e (y) re')
plt.plot(ax_x.xpts, np.imag(jze.get_comp_data(Y) - dvey_func(x_vals)), '--', label='e (y) im')
plt.plot(ax_x.xpts, np.real(jze.get_comp_data(Z) - dvez_func(x_vals)), label='e (z) re')
plt.plot(ax_x.xpts, np.imag(jze.get_comp_data(Z) - dvez_func(x_vals)), '--', label='e (z) im')
plt.title('u error')
plt.legend()

plt.figure()
plt.plot(ax_x.xpts, np.real(Ey), label='Ey')
plt.plot(ax_x.xpts, np.imag(Ey), '--', label='Ey')
plt.plot(ax_x.xpts, np.real(Ez), label='Ez')
plt.plot(ax_x.xpts, np.imag(Ez), '--', label='Ez')
plt.title('E field')
plt.legend()

plt.figure()
plt.plot(ax_x.xpts, np.real(By), label='By')
plt.plot(ax_x.xpts, np.imag(By), '--', label='By')
plt.plot(ax_x.xpts, np.real(Bz), label='Bz')
plt.plot(ax_x.xpts, np.imag(Bz), '--', label='Bz')
plt.title('B field')
plt.legend()

plt.show()


vm_sys.collision = VP_test.collision_config
print('colls', vm_sys.collision.coll_params)

vm_sys.compress_F = VP_test.compress_F
vm_sys.compress_F_opts = VP_test.compress_F_opts
print('compress E?', vm_sys.compress_F, vm_sys.compress_F_opts)

print('E nrg', vm_sys.E.norm() ** 2 * 1. / 2 / tot_x)
print('B nrg', vm_sys.B.norm() ** 2 * 1. / 2 / tot_x)

if nt == 0:

    fig1, ax1 = plt.subplots(2, 1)
    time_ = nt * dt
    fi_data = init_fi_gtn.get_data(ax_select={ax_vx_i: ax_vx_i.npts // 2, ax_vy_i: ax_vy_i.npts // 2})
    fi_data = ax_x.basis.get_realspace_1D(fi_data, 0)
    fi_data = ax_vz_i.basis.get_realspace_1D(fi_data, 1)
    if is_sqrt:
        fi_data = np.abs(fi_data) ** 2
    im1 = ax1[0].imshow(np.real(fi_data).T,
                        extent=(-np.pi / dk, np.pi / dk, vi_z_min, vi_z_max),
                        aspect='auto', origin='lower')

    fe_data = init_fe_gtn.get_data(ax_select={ax_vx_e: ax_vx_e.npts // 2, ax_vy_e: ax_vy_e.npts // 2})
    fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
    fe_data = ax_vz_e.basis.get_realspace_1D(fe_data, 1)
    if is_sqrt:
        fe_data = np.abs(fe_data) ** 2
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
        include_bg = False if te_order in [412, 413, 433, 414, 416, 4164, 4154, 4174, 4184] else True
        do_advec_x_cfl = False if te_order in [412, 413, 414, 416, 4164, 4154, 4174, 4184,
                                               312, 313, 314, 316, 3164, 3154, 3174, 3184,] else True
        em_term = vm_sys.compute_force_term(is_ion=False, background_force=include_bg)
        max_em = []
        for coord in [X, Y, Z]:
            try:
                em_data = em_term.get_comp_data(coord, ax_select={ax_vx_e: 0, ax_vy_e: 0, ax_vz_e: 0})
                em_max = np.max(np.abs(em_data))
            except KeyError:
                em_max = 0.
            max_em += [em_max]

        Fe_maxs = np.array(max_em)
        # Fi_maxs = np.array(max_em) * charge / mass_i
        dt = min(dt_max, VP_test.get_dt(Fe_maxs, do_advec_x_cfl=do_advec_x_cfl))
        print('restarted dt', dt)

time1 = time.time()

print('starting time evolution')
while ts[-1] < T:

    vm_sys.EM_sys.verbose_plot = False # ((nt % save_every_nt) == 0)

    walltime = time.time()
    use_rk4 = (nt == 0)
    # use_rk4 = ((nt == 0) or (nt == 1))
    # use_rk4 = (nt < 11)
    # dt_ = 0.01 # / 4
    # dt_ = min(dt_cfl, dt_max) if use_rk4 and te_order in [3154, 3164, 3174, 3184,
    #                                                       4154, 4164, 4174, 4184] else dt
    dt_ = dt
    vm_sys = vm_sys.next_time_step(dt_, compress_level=1, err_tol=1.0e-7, verbose_plot=False,
                                   is_first_time_step=use_rk4)
    print('wall time', time.time() - walltime)

    fe_field = vm_sys.fe
    max_bond_fe += [fe_field.component.max_bond()]
    nrg_fe = fe_field.component.meas_expec(v2e_mpo, is_sqrt=is_sqrt) * mass_e / 2 / tot_x

    fi_field = vm_sys.fi
    max_bond_fi += [fi_field.component.max_bond()]
    nrg_fi = fi_field.component.meas_expec(v2i_mpo, is_sqrt=is_sqrt) * mass_i / 2 / tot_x

    E_field = vm_sys.E
    max_bond_E += [E_field.max_bonds(compIDs=coords_x.coords)]
    energy_E_t = np.array(E_field.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_x  # E field energy
    if not restarted:
        try:
            Ex_mags += [(E_field.get_comp_data(X)[npts_x // 2])]
        except:
            Ex_mags += [np.nan]
        Ey_mags += [(E_field.get_comp_data(Y)[npts_x // 2])]
        Ez_mags += [(E_field.get_comp_data(Z)[npts_x // 2])]

    # if nt % (save_every_nt // 10) == 0:
    #     plt.figure()
    #     if not np.isnan(Ex_mags[-1]):
    #         plt.plot(E_field.get_comp_data(X), label='EX')
    #     plt.plot(E_field.get_comp_data(Y), label='EY')
    #     plt.plot(E_field.get_comp_data(Z), label='EZ')
    #     plt.legend()
    #     plt.show()

    B_field = vm_sys.B
    max_bond_B += [B_field.max_bonds(compIDs=coords_x.coords)]
    energy_B_t = np.array(B_field.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_x  # B field energy
    # norm_BY = B_field.norms([Y])[0]
    # energy_By_t = norm_BY ** 2 * 1. / 2 / tot_x if norm_BY is not None else np.nan

    # gauss_err = vm_sys.EM_sys.check_poisson().component.norm()
    # gauss_err = gauss_err / np.sqrt(grid_X.npts) if gauss_err is not None else None
    # divB_err = vm_sys.EM_sys.check_divB().component.norm()
    # divB_err = gauss_err / np.sqrt(grid_X.npts) if divB_err is not None else None
    # print('gauss err', gauss_err, 'divB err', divB_err)

    nrg_fe_ts += [nrg_fe]
    nrg_fi_ts += [nrg_fi]
    nrg_E_ts += [energy_E_t]
    nrg_B_ts += [energy_B_t]
    # nrg_By_ts += [energy_By_t]

    ts += [ts[-1] + dt_]
    nt += 1

    if do_adapt_dt:
        include_bg = False if te_order in [412, 413, 433, 414, 416, 4164, 4174, 4154, 4184] else True
        do_advec_x_cfl = False if te_order in [412, 413, 414, 416, 4164, 4174, 4154, 4184,
                                               312, 313, 314, 316, 3164, 3174, 3154, 3184] else True
        em_term = vm_sys.compute_force_term(is_ion=False, background_force=include_bg)
        max_em = []
        for coord in [X, Y, Z]:
            try:
                em_data = em_term.get_comp_data(coord, ax_select={ax_vx_e: 0, ax_vy_e: 0, ax_vz_e: 0})
                em_max = np.max(np.abs(em_data))
            except (KeyError,TypeError):
                em_max = 0.
            max_em += [em_max]

        Fe_maxs = np.array(max_em)
        dt = min(dt_max, VP_test.get_dt(Fe_maxs, do_advec_x_cfl=do_advec_x_cfl))
        # print('restarted dt', dt)

    norm = vm_sys.fe.norm() - vm_sys.fi.norm()
    print('result norm', nt, ts[-1], dt_, norm,  # nrg_fe, nrg_fi,
          energy_E_t, energy_B_t,
          E_field.max_bond(X), E_field.max_bond(Y), E_field.max_bond(Z),
          B_field.max_bond(X), B_field.max_bond(Y), B_field.max_bond(Z),
          fi_field.max_bond(), fe_field.max_bond(), Lx, Lvi, Lve)

    if nt % (save_every_nt ) == 0:

        if save_data:
            if not os.path.exists(sdir + 'restart/'):
                os.makedirs(sdir + 'restart/')

            test_setup.save_data_EM(VP_test, vm_sys, sdir + 'restart/', extra_str=extra_str,
                                    add_fstr=f'-nt{nt}',
                                    ts=ts, nrg_fe_ts=nrg_fe_ts, nrg_fi_ts=nrg_fi_ts, nrg_E_ts=nrg_E_ts,
                                    nrg_B_ts=nrg_B_ts, max_bond_fe=max_bond_fe, max_bond_fi=max_bond_fi,
                                    max_bond_E=max_bond_E, max_bond_B=max_bond_B,
                                    max_bond_phi=max_bond_phi, max_bond_psi=max_bond_psi)

        if save_figs or matplotlib.get_backend() != 'agg':

            fig1, ax1 = plt.subplots(2, 1)
            time_ = ts[-1]
            # fi_data = fi_field.get_comp_data(ax_select={ax_vy_i: npts_vyi // 2, ax_vz_i: npts_vzi // 2})
            fi_data = fi_field.get_comp_data(ax_select={ax_vy_i: npts_vyi // 2, ax_vx_i: npts_vxi // 2})
            if fi_data is not None:
                fi_data = ax_x.basis.get_realspace_1D(fi_data, 0)
                fi_data = ax_vz_i.basis.get_realspace_1D(fi_data, 1)
                if is_sqrt:
                    fi_data = np.abs(fi_data) ** 2

                print('fi data norm', np.linalg.norm(fi_data))
                print('fi field norm', fi_field.component.frobenius_norm())
                im1 = ax1[0].imshow(np.real(fi_data).T, extent=(-np.pi / dk, np.pi / dk, vi_z_min, vi_z_max),
                                    aspect='auto', origin='lower')

            fe_data = fe_field.get_comp_data(ax_select={ax_vz_e: ax_vz_e.npts // 2, ax_vy_e: ax_vy_e.npts // 2})
            fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
            fe_data = ax_vx_e.basis.get_realspace_1D(fe_data, 1)
            if is_sqrt:
                fe_data = np.abs(fe_data) ** 2
            im2 = ax1[1].imshow(np.real(fe_data).T, extent=(-np.pi / dk, np.pi / dk, ve_x_min, ve_x_max),
                                aspect='auto', origin='lower')

            fig1.colorbar(im1, ax=ax1[0])
            fig1.colorbar(im2, ax=ax1[1])
            ax1[0].set_title(f'time {time_:3.2f}')
            ax1[0].xaxis.set_visible(False)
            ax1[0].set_xlabel('x'), ax1[0].set_ylabel('vz (i)')
            ax1[1].set_xlabel('x'), ax1[1].set_ylabel('vx (e)')
            if save_figs:
                fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')
            else:
                plt.show()
            plt.close(fig1)

            ## plot x, vy
            fig1, ax1 = plt.subplots(2, 1)
            time_ = ts[-1]
            fe_data = fe_field.get_comp_data(ax_select={ax_vz_e: npts_vze // 2, ax_vx_e: npts_vxe // 2})
            fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
            fe_data = ax_vy_e.basis.get_realspace_1D(fe_data, 1)
            if is_sqrt:
                fe_data = np.abs(fe_data) ** 2
            im1 = ax1[0].imshow(np.real(fe_data).T, extent=(-np.pi / dk, np.pi / dk, vi_y_min, ve_y_max),
                                aspect='auto', origin='lower')

            ## plot x, vz
            fe_data = fe_field.get_comp_data(ax_select={ax_vx_e: ax_vx_e.npts // 2, ax_vy_e: ax_vy_e.npts // 2})
            fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
            fe_data = ax_vz_e.basis.get_realspace_1D(fe_data, 1)
            if is_sqrt:
                fe_data = np.abs(fe_data) ** 2
            im2 = ax1[1].imshow(np.real(fe_data).T, extent=(-np.pi / dk, np.pi / dk, ve_z_min, ve_z_max),
                                aspect='auto', origin='lower')

            fig1.colorbar(im1, ax=ax1[0])
            fig1.colorbar(im2, ax=ax1[1])
            ax1[0].set_title(f'time {time_:3.2f}')
            ax1[0].xaxis.set_visible(False)
            ax1[0].set_xlabel('x'), ax1[0].set_ylabel('vy (e)')
            ax1[1].set_xlabel('x'), ax1[1].set_ylabel('vz (e)')
            plt.show()
            plt.close(fig1)

            real_xs = np.linspace(-np.pi / dk, np.pi / dk, npts_x, endpoint=False)

            plt.figure()
            E_data = vm_sys.E.get_comp_data(X)
            if E_data is not None and np.all(E_data != 0):
                E_data = ax_x.basis.get_realspace_1D(E_data, 0)
                plt.plot(real_xs, np.real(E_data), label='Ex field real')
                plt.plot(real_xs, np.imag(E_data), '--', label='Ex field imag')
            E_data = vm_sys.E.get_comp_data(Y)
            if np.all(E_data != 0):
                E_data = ax_x.basis.get_realspace_1D(E_data, 0)
                plt.plot(real_xs, np.real(E_data), label='Ey field real')
                plt.plot(real_xs, np.imag(E_data), '--', label='Ey field imag')
            E_data = vm_sys.E.get_comp_data(Z)
            if np.all(E_data != 0):
                E_data = ax_x.basis.get_realspace_1D(E_data, 0)
                plt.plot(real_xs, np.real(E_data), label='Ez field real')
                plt.plot(real_xs, np.imag(E_data), '--', label='Ez field imag')
            plt.plot((real_xs[0], real_xs[-1]), [Ez_mag, Ez_mag], ':')
            plt.title('E')
            plt.legend()

            plt.figure()
            Bx_data = vm_sys.B.get_comp_data(X)
            if Bx_data is not None and np.all(Bx_data != 0):
                Bx_data = ax_x.basis.get_realspace_1D(Bx_data, 0)
                plt.plot(real_xs, np.real(Bx_data), label='Bx field real')
                plt.plot(real_xs, np.imag(Bx_data), '--', label='Bx field imag')
            By_data = vm_sys.B.get_comp_data(Y)
            if np.all(By_data != 0):
                By_data = ax_x.basis.get_realspace_1D(By_data, 0)
                plt.plot(real_xs, np.real(By_data), label='By field real')
                plt.plot(real_xs, np.imag(By_data), '--', label='By field imag')
            Bz_data = vm_sys.B.get_comp_data(Z)
            if np.all(Bz_data != 0):
                Bz_data = ax_x.basis.get_realspace_1D(Bz_data, 0)
                plt.plot(real_xs, np.real(Bz_data), label='Bz field real')
                plt.plot(real_xs, np.imag(Bz_data), '--', label='Bz field imag')
            B_mag = mu0 * charge * n0_e * A / dk
            plt.plot((real_xs[0], real_xs[-1]), [B_mag, B_mag], ':')
            plt.title('B')
            plt.legend()

            if not save_figs:
                plt.show()
            plt.close()
            plt.close()

            fig7, ax7 = plt.subplots()
            ax7.semilogy(ts, [nrg[1] for nrg in nrg_B_ts], '-x', label='By')
            ax7.legend()
            ax7.set_xlabel('t')
            ax7.set_ylabel('energy')
            ax7.legend()

            fig8, ax8 = plt.subplots()
            ax8.semilogy(ts, [nrg[2] for nrg in nrg_E_ts], label='Ez')
            ax8.legend()
            ax8.set_xlabel('t')
            ax8.set_ylabel('energy')
            ax8.legend()
            if save_figs:
                fig7.savefig(fdir + 'plot_nrg_By_' + fstr + '.png')
            else:
                plt.show()
            plt.close()
            plt.close()

            if not restarted:
                fig8, ax8 = plt.subplots()
                ax8.plot(Ey_mags, Ez_mags, 'x-')
                ax8.set_xlabel('|Ey|')
                ax8.set_ylabel('|Ez|')
                plt.show()

if save_data:
    test_setup.save_data_EM(VP_test, vm_sys, fdir, extra_str=extra_str,
                            ts=ts, nrg_fe_ts=nrg_fe_ts, nrg_fi_ts=nrg_fi_ts, nrg_E_ts=nrg_E_ts,
                            nrg_B_ts=nrg_B_ts, max_bond_fe=max_bond_fe, max_bond_fi=max_bond_fi,
                            max_bond_E=max_bond_E, max_bond_B=max_bond_B,
                            max_bond_phi=max_bond_phi, max_bond_psi=max_bond_psi)

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
fi_data = vm_sys.fi.get_comp_data(ax_select={ax_vy_i: npts_vyi // 2, ax_vz_i: npts_vzi // 2})
fi_real = ax_x.basis.get_realspace_1D(fi_data, 0)
fi_real = ax_vz_i.basis.get_realspace_1D(fi_real, 1)
if is_sqrt:
    fi_real = np.abs(fi_real) ** 2
im1 = ax1[0].imshow(np.real(fi_real).T, extent=(-np.pi / dk, np.pi / dk, vi_x_min, vi_x_max),
                    aspect='auto', origin='lower')

fe_data = vm_sys.fe.get_comp_data(ax_select={ax_vy_e: npts_vye // 2, ax_vz_e: npts_vze // 2})
fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
fe_real = ax_vx_e.basis.get_realspace_1D(fe_real, 1)
if is_sqrt:
    fe_real = np.abs(fe_real) ** 2
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
ax2.plot([0, num_tsteps], [(Lx + KV * Lve) // 2] * 2, ':', label='max fe')
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
total_nrg_t = energy_fe_t + energy_fi_t + np.sum(energy_E_t, 1) + np.sum(energy_B_t, 1)
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
ax7.semilogy(ts, np.abs(total_nrg_t), ls='--', label='total')

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
ax7.semilogy(vm_sys.fe.component.entanglement_entropy_all(), '-x', label='elec')
ax7.semilogy(vm_sys.fi.component.entanglement_entropy_all(), '-o', label='ion')

ax7.legend()
ax7.set_xlabel('bond number')
ax7.set_ylabel('EE')
if save_figs:
    fig7.savefig(fdir + 'plot_EE_' + fstr + '.png')
    print(fdir + 'plot_EE_' + fstr + '.png')

plt.show()

print('done')