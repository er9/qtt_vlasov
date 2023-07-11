import os, sys, pickle, time, glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sp_opt

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
from pde_EM import Maxwell
from pde_vlasovEM import VlasovMaxwell
from vlasov_tests import Reconnection

"""
GEM reconnection problem (real space in x, real space in velocity)
https://aip-scitation-org.libproxy.mit.edu/doi/pdf/10.1063/1.2347101
updated speed of light to be greater than vth_e
"""

save_figs = False
save_data = True
load_data = True
restart   = True
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

is_sqrt = True

extra_str = ''
if is_sqrt:
    extra_str += '_sq'
    # raise NotImplementedError

if restart or save_data or save_figs:
    if is_sqrt:
        # sdir = '/pool001/erikaye/tns_pde_v2/data13/VM23-reconn-fd-sq/'
        # fdir = 'data13/VM23-reconn-fd-sq/'
        sdir = '/pool001/erikaye/tns_pde_v2/data13/VM23-reconn-fd-sq-v1/'
        fdir = 'data13/VM23-reconn-fd-sq-v1/'
    else:
        sdir = '/pool001/erikaye/tns_pde_v2/data13/VM23-reconn-fd/'
        fdir = 'data13/VM23-reconn-fd/'

    if not os.path.exists(sdir):
        os.makedirs(sdir)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)

### system parameters ###
lam = 0.5       # target Harris sheet thickness (to be scaled by lambda_i)
A = 0.5         # magnitude of initial vector potential perturbation ( to be scaled by B0 / lambda_i)

#######################
# plasma parameters   #
#######################

## normalization:
## ion inertial length = c / wp_i = 1
## ion cyclotron frequency wc_i = charge * B0 / np.pi / 2 / mass_i = 1
## (CGS: wc_i = charge * B / mass_i / c)

plasma_config = UnitsConfiguration(eps0=1./400, mu0=1.)  # eps0=1.0e-3, mu0=1.0e-3)
charge = plasma_config.e      # unit of charge
eV = plasma_config.eV         # unit of kB*T
mu0 = plasma_config.mu0       # permeability
eps0 = plasma_config.eps0     # permittivity
c_speed = plasma_config.c     # speed of light

mass_i = 1.0   # mass of ion; s.t ion plasma frequency c/wp_i = 1.0
mass_e = mass_i / 25    # mass of electron
Zi = 1
Ze = 1
TeTi_ratio = 0.2        # T_e / T_i ratio

lambda_i = 1.0        # ion inertial length = c / wp_i
wc_i = 1.0            # ion cyclotron frequency = 1
B0 = 1.0 * mass_i / charge        # chosen to satisfy wc_i
n0 = 1.0              # to satisfy wp_i = 1 so lambda_i = 1

print('B0', B0)
print('lambda_i', lambda_i)
print('ion cyclotron frequency', wc_i)

## A0 = - 2 * c * Ti / e / v0i = B0 * lam
## B0 = sqrt[ 2 * mu0 * n0 * (Te + Ti) ]
## lam = - B0 * Ti / e / mu0 / n0 / (Te + Ti) / v0i
## v0e = - Te/Ti * v0i
## n0 = 2 * c * Ti^2 / e^2 / mu0 / (Te + Ti) / v0i^2

def get_T_i(v0_i):
    """ u_i / T_i = - u_e / T_e
        ## CGS units (Biskamp)
            2 / B0 / lam = e / c * (u_e - u_i) / (T_i + T_e)
            2 / B0 / lam = e / c * (- u_i / T_i * T_e - u_i) / (T_i + T_e)
            2 / B0 / lam = e / c * (-2 u_i) * (T_e / T_i + 1) / (T_i + T_e)
            2 / B0 / lam = e / c * (-2 u_i) / T_i * (T_e + T_i) / (T_i + T_e)
            1 / B0 / lam = - e / c * u_i / T_i
            T_i = - e * B0 * lam * u_i / c
        ## SI units:  drop the c
    """
    return -B0 * lam * lambda_i * v0_i * charge / 2

def harris_sheet_magnitude(v0_i):
    """ calculate magnitude of Harris sheet (B0)
        B0 = np.sqrt(n0_i * (T_i + T_e) * 2 * mu0)
    """
    temp_i = get_T_i(v0_i)
    temp_e = TeTi_ratio * temp_i
    return np.sqrt(n0 * (temp_i + temp_e) * mu0 * 2)

# def harris_sheet_thickness(v0_i, temp_i):
#     """ calculate thickness of Harris sheet
#         v0_i:  drift velocity in z for ions
#     """
#     temp_e = TeTi_ratio * temp_i
#     v0_e = - TeTi_ratio * v0_i
#     return 2 * (temp_e + temp_i) / charge / B0 / (v0_i - v0_e)

def harris_sheet_thickness(v0_i):
    """ calculate thickness of Harris sheet (lam)
        v0_i:  drift velocity in z for ions
    """
    temp_i = get_T_i(v0_i)
    # temp_e = TeTi_ratio * temp_i
    # return -B0 * temp_i / charge / mu0 / n0 / (temp_e + temp_i) / v0_i
    return -2 / charge / B0 * temp_i / v0_i


def get_setup_params(input_params):
    """ satisfy Harris sheet magnitude, sheet thickness, density (for steady state)
    """
    v0_i, = input_params
    # temp_i = get_T_i(v0_i)
    B0_ = harris_sheet_magnitude(v0_i)         # Harris sheet magnitude
    lam_ = harris_sheet_thickness(v0_i)    # sheet thickness
    return np.array( abs(B0_ - B0), abs(lam_ - lam))


## electron, ion temperatures
## e drift velocities in z
res = sp_opt.minimize( get_setup_params, np.array([-0.1]) )
v0_i, = res.x                   # density, drift velocity in z for ions
T_i = get_T_i(v0_i)             # ion temperature [eV]
T_e = TeTi_ratio * T_i          # electron temperature [eV]
v0_e = -TeTi_ratio * v0_i       # drift velocity in z from elcs
n0_i = n0_e = n0                # elc, ion densities

B0_ = harris_sheet_magnitude(v0_i)
lam_ = harris_sheet_thickness(v0_i)

print('density', n0)
print('T', T_i, T_e)
print('harris sheet magnitude', B0_, B0)
print('lambda', lam * lambda_i, lam_)

print('v0_i', v0_i, -B0_ / lam_ / 1.2 / mu0)
print('v0_e', v0_e, +B0_ / lam_ / 1.2 * 0.2 / mu0)
# print('A0', B0_ * lam_, - 2 * c_speed * T_i / charge / v0_i )

ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, Z=Zi, units_config=plasma_config)
elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, Z=Ze, units_config=plasma_config)
vth_e = elc_config.vth    # electron thermal speed
wp_e  = elc_config.wp     # plasma e' frequency
lamD  = elc_config.lamD   # Debye length
vth_i = ion_config.vth    # electron thermal speed
wp_i  = ion_config.wp     # plasma i+ frequency,  SI: sqrt(n0 e^2/m/eps0) // CGS: sqrt(4*pi*n0*e^2/m)

print('vth_e', vth_e, 'vth_i', vth_i, 'c', plasma_config.c)
print('wp_e', wp_e, 'lamD', lamD)
print('wp_i', wp_i, 'lambda', ion_config.skin_depth, lambda_i)

print('vA_i', ion_config.alfven_speed(B0_), plasma_config.c)
print('vA_e', elc_config.alfven_speed(B0_), plasma_config.c)
print('beta_i', ion_config.beta(B0_))

print('wci', ion_config.cyclotron_frequency(B0_), wc_i)

print('dt max Wc_e?', 2. * np.pi / elc_config.cyclotron_frequency(B0_))
print('dt max wp_e?', 2. * np.pi / elc_config.plasma_frequency, elc_config.wp)



#####################
# define parameters #
#####################

grid_idx = 0    ## 0:  account for symmetries, 1: don't account for symmetries

## runtime parameters
q = 2
KX = 2
KV = 3


Lx, Ly = (8, 7) if grid_idx == 1 else (7, 6)
Lxe, Lye, Lze = 6, 6, 7
Lxi, Lyi, Lzi = 6, 6, 7


# TN_layout = LayoutType.SEQUENTIAL
# x_map_key = 'F'
# v_map_key = 'B'

TN_layout = LayoutType.COMB
x_map_key = 'B'
v_map_key = 'B'

order = 1   ## derivative order
T = 40

do_adapt_dt = False  # False  # True
dt_frac = 0.9
te_order = 3164  # 3164  # 3164  # 313
dt = 0.01  # 0.01  # None  # 0.01  # None  #  0.002  # None  # 0.001
dt_max = 0.25 * 2. * np.pi / max(np.abs(elc_config.cyclotron_frequency(B0_)), np.abs(elc_config.wp))

# do_adapt_dt = False  # True
# te_order = 3164  # 3164  # 313  # 333  # 4
# dt = 0.01  # None  #  0.002  # None  # 0.001
# dt_frac = None
# dt_max = 0.25 * 2. * np.pi / max(np.abs(elc_config.cyclotron_frequency(B0_)), np.abs(elc_config.wp))

te_order_EM = 22
is_yee = True if te_order_EM == 31 else False

save_every_nt = 100  # 250

DMAX = 32    # 128 # None
DMAX_F = 24  #  DMAX
comp_style = 3

VP_test = Reconnection(lam=lam, B0=B0, A=A, v0es=(0,0,v0_e), v0is=(0,0,v0_i), grid_idx=grid_idx,
                       Lxs=(Lx,Ly), Lves=(Lxe,Lye,Lze), Lvis=(Lxi,Lyi,Lzi), KX=KX, KV=KV, q=q,
                       x_map_key = x_map_key, ve_map_key = v_map_key,
                       basis_xs = SpatialBasis(),
                       #
                       ion_config = ion_config, elc_config = elc_config,
                       coll_type=None, do_adapt_coll = True, coll_coeff_e = 0.005, coll_rate_e = 0.01,
                       DMAX = DMAX, DMAX_F = DMAX_F, compress_style = comp_style, compress_style_mod = 0,
                       do_adapt_dt=do_adapt_dt, dt_frac=dt_frac, dt=dt, T=T, te_order=te_order,
                       te_order_EM=te_order_EM,
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
ax_x, ax_y, = VP_test.pos_axes
ax_vx_i, ax_vy_i, ax_vz_i = VP_test.vi_axes
ax_vx_e, ax_vy_e, ax_vz_e = VP_test.ve_axes
print('ax_x zero ind', ax_x.zero_ind)

x_min, x_max = VP_test.x_lims[0]
y_min, y_max = VP_test.x_lims[1]
print('x range', ax_x.dx, x_min, x_max, lambda_i)
print('x range', ax_y.dx, y_min, y_max, lambda_i)
ve_x_min, ve_x_max = VP_test.ve_lims[0]
vi_x_min, vi_x_max = VP_test.vi_lims[0]
ve_y_min, ve_y_max = VP_test.ve_lims[1]
vi_y_min, vi_y_max = VP_test.vi_lims[1]
ve_z_min, ve_z_max = VP_test.ve_lims[2]
vi_z_min, vi_z_max = VP_test.vi_lims[2]
tot_V = (x_max - x_min) * (y_max - y_min)

coords_x  = VP_test.pos_coordsys
coords_vi = VP_test.vi_coordsys
coords_ve = VP_test.ve_coordsys

X, Y, Z = coords_x.coords
VX, VY, VZ = coords_ve.coords

npts_x = ax_x.npts
npts_y = ax_y.npts
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
y_vals = ax_y.xpts
xB_vals = x_vals + ax_x.dx/2 if is_yee else x_vals   # for staggered Yee cell
yB_vals = y_vals + ax_y.dx/2 if is_yee else y_vals

vx_e_vals = ax_vx_e.xpts
vy_e_vals = ax_vy_e.xpts
vz_e_vals = ax_vz_e.xpts
vx_i_vals = ax_vx_i.xpts
vy_i_vals = ax_vy_i.xpts
vz_i_vals = ax_vz_i.xpts


## background field:
## Bx0 = B0_ * np.tanh(yB_vals/lam)
## curl(B0):
##    [z]: -d/dy Bx0 = - B0_ / lam / np.cosh(yB_vals/lam)**2

## perturbation in Az:  Az = A0 * cos(2*np.pi*x/Lx) * cos(np.pi * y/Ly) = A0 * cos(2 kx x) * cos(ky y)
## Lx = 2 * x_max, Ly = 2 * y_max
## B = curl(Az)
## Bx = d/dy Az - d/dz Ay = - A0 * ky / 2 * cos(kx * x) * sin(ky * y / 2)
## By = d/dz Ax - d/dx Az = + A0 * kx * sin(kx * x) * cos(ky * y / 2)
## Bz = d/dx Ay - d/dy Ax = 0
## div(B) = d/dx Bx + d/dy By + d/dz Bz =
##        = A0 * kx * ky / 2 * sin(kx * x) * sin(ky * y / 2) + (-A0 * ky / 2 * kx * sin(kx * x) * sin(ky * y / 2)
##        = 0

## j = 1/mu0 curl B = e * n0 (up - ue)
### perturbation:
## jx = 1/mu0 (d/dy Bz - d/dz By) = 0
## jy = 1/mu0 (d/dz Bx - d/dx Bz) = 0
## jz = 1/mu0 (d/dx By - d/dy Bx) = 1/mu0 (A0 * kx**2 * cos(kx * x) * cos(ky * y / 2)
##                                       + A0 * ky**2 / 4 * cos(kx * x) * cos(ky * y / 2)
##                                = 1/mu0 (A0 * 2 * kx**2 * cos(kx * x) * cos(ky * y / 2)
##                                (used ky = 2 kx)
### harris sheet:
## z: - d/dy Bx0
## jz = -1 / mu0 * B0_ * 1/lam * 1/cosh(y/lam)**2

## ue / up = -0.2
## j = e * n0 (up - ue) = e * n0 ( 1.2 up )
### perturbation
## up_z = 1 / mu0 / 1.2 * (A0 * 2 * kx**2 * cos(kx * x) * cos(ky * y / 2))
## ue_z = -0.2 / 1.2 / mu0 * (A0 * 2 * kx**2 * cos(kx * x) * cos(ky * y / 2))
### harris sheet
## up_z = B0_ / lam_ / mu0 * 1/cosh(y/lam)**2 / 1.2
## ue_z = B0_ / lam_ / mu0 * 1/cosh(y/lam)**2 / 1.2 * -0.2

## E = -u_e x B ??
## Ex = ue_y * Bx - ue_z * By = -ue_z * By
## Ey = ue_z * Bx - ue_x * Bz = +ue_z * Bx = ue_z * (B0_ * np.tanh(yB_vals/lam) + pert Bx)
## Ez - ue_x * By - ue_y * Bz = 0


def shifted_vz(x_vals, y_vals, vz_vals, dv_z_func, vth=1.0, density=1.0, flow=0.0, is_sqrt=False):
    """ initial state for w
    """
    x_mesh, y_mesh, vz_mesh = np.meshgrid(x_vals, y_vals, vz_vals, indexing='ij')

    dv_z = dv_z_func(x_mesh, y_mesh)
    vz_mesh = vz_mesh - dv_z

    out = helper_test.maxwellian( vz_mesh, vth2=vth**2, density=density, flow=flow, is_sqrt=is_sqrt )

    plt.figure()
    plt.imshow(out[:,:,len(vz_vals)//2], aspect='auto')
    plt.title('f(x,y;vz=0)')
    plt.colorbar()
    plt.show()

    return out

A0 = A * B0 / lambda_i
box_len_x, box_len_y = 25.6, 12.8
kx = 2 * np.pi / box_len_x
ky = 2 * np.pi / box_len_y
print('kx', kx, ky/2)

n0_bg = 0.2 * n0
ny = lambda y:  n0 * 1./np.cosh(y/lam) ** 2
pert_Bx_x = lambda x: np.cos(kx * x) * -A0 * ky / 2
pert_Bx_y = lambda y: np.sin(ky * y / 2)
pert_By_x = lambda x: np.sin(kx * x) * A0 * kx
pert_By_y = lambda y: np.cos(ky * y / 2)
back_Bx_y = lambda y: B0_ * np.tanh(y / lam)
pert_Jz = lambda x, y:  -A0 / mu0 * np.cos(kx * x) * np.cos( ky * y / 2) * (ky**2/4 + kx**2)
dvzi_func = lambda x, y: pert_Jz(x, y) / n0 / 1.2 + ny(y) / n0 * v0_i
dvze_func = lambda x, y: dvzi_func(x, y) * -0.2

plt.figure()
plt.plot(x_vals, pert_Bx_x(x_vals), label='Bx(x)')
plt.plot(x_vals, pert_By_x(x_vals), label='By(x)')
plt.legend()

plt.figure()
plt.plot(y_vals, pert_Bx_y(y_vals), label='Bx(y)')
plt.plot(y_vals, pert_By_y(y_vals), label='By(y)')
plt.legend()
plt.show()

## electron density:  n_0(y) = n_0 sech^2(y/lam) * Maxwellian(vx, vy, vz-v0e)
fy = ny(y_vals)  # n0 * 1./np.cosh(y_vals/lam) ** 2
if is_sqrt:
    fy = np.sqrt(fy)

fe_vx = helper_test.maxwellian(vx_e_vals, vth2=vth_e**2, density=1.0, flow=VP_test.v0es[0], is_sqrt=is_sqrt)
fe_vy = helper_test.maxwellian(vy_e_vals, vth2=vth_e**2, density=1.0, flow=VP_test.v0es[1], is_sqrt=is_sqrt)
fe_vz = helper_test.maxwellian(vz_e_vals, vth2=vth_e**2, density=1.0, flow=VP_test.v0es[2], is_sqrt=is_sqrt)
# fe_vz = shifted_vz(x_vals, y_vals, vz_e_vals, pert_dvze_func, vth=vth_e, density=1.0, flow=VP_test.v0es[2], is_sqrt=is_sqrt)
init_fe_gtn = grid_e.make_mps_ndim({ax_y: fy, ax_vx_e: fe_vx, ax_vy_e: fe_vy, ax_vz_e: fe_vz})

## ion density
fi_vx = helper_test.maxwellian(vx_i_vals, vth2=vth_i**2, density=1.0, flow=VP_test.v0is[0], is_sqrt=is_sqrt)
fi_vy = helper_test.maxwellian(vy_i_vals, vth2=vth_i**2, density=1.0, flow=VP_test.v0is[1], is_sqrt=is_sqrt)
fi_vz = helper_test.maxwellian(vz_i_vals, vth2=vth_i**2, density=1.0, flow=VP_test.v0is[2], is_sqrt=is_sqrt)
init_fi_gtn = grid_i.make_mps_ndim({ax_y: fy, ax_vx_i: fi_vx, ax_vy_i: fi_vy, ax_vz_i: fi_vz})

## background electron, ion density
fe_vx0 = helper_test.maxwellian(vx_e_vals, vth2=vth_e**2, density=1.0, flow=0.0, is_sqrt=is_sqrt)
fi_vx0 = helper_test.maxwellian(vx_i_vals, vth2=vth_i**2, density=1.0, flow=0.0, is_sqrt=is_sqrt)
fe_vy0 = helper_test.maxwellian(vy_e_vals, vth2=vth_e**2, density=1.0, flow=0.0, is_sqrt=is_sqrt)
fi_vy0 = helper_test.maxwellian(vy_i_vals, vth2=vth_i**2, density=1.0, flow=0.0, is_sqrt=is_sqrt)
fe_vz0 = helper_test.maxwellian(vz_e_vals, vth2=vth_e**2, density=1.0, flow=0.0, is_sqrt=is_sqrt)
# fe_vz0 = shifted_vz(x_vals, y_vals, vz_e_vals, pert_dvze_func, vth=vth_e, density=1.0, flow=0.0, is_sqrt=is_sqrt)
fi_vz0 = helper_test.maxwellian(vz_i_vals, vth2=vth_i**2, density=1.0, flow=0.0, is_sqrt=is_sqrt)
# fi_vz0 = shifted_vz(x_vals, y_vals, vz_i_vals, pert_dvzi_func, vth=vth_i, density=1.0, flow=0.0, is_sqrt=is_sqrt)

fe_vx0 = fe_vx0 * np.sqrt(n0_bg * n0_e) * 1.j if is_sqrt else fe_vx0 * 0.2 * n0_e
fi_vx0 = fi_vx0 * np.sqrt(n0_bg * n0_i) * 1.j if is_sqrt else fi_vx0 * 0.2 * n0_i

# ax_inds_list = [(2,), (3,), (0,1,4)]
# gtn_data_list = [fe_vx0, fe_vy0, fe_vz0]
# init_fe_gtn_0 = grid_e.build_mps_from_subgtns( list(zip(ax_inds_list, gtn_data_list)) )
init_fe_gtn_0 = grid_e.make_mps_ndim({ax_vx_e: fe_vx0, ax_vy_e: fe_vy0, ax_vz_e: fe_vz0})
init_fi_gtn_0 = grid_i.make_mps_ndim({ax_vx_i: fi_vx0, ax_vy_i: fi_vy0, ax_vz_i: fi_vz0})

print('v0es', VP_test.v0es)
print('v0is', VP_test.v0is)
# exit()

plt.figure()
if is_sqrt:
    plt.plot(fy**2)
else:
    plt.plot(fy)
plt.show()

fe = init_fe_gtn.add( init_fe_gtn_0 )
fi = init_fi_gtn.add( init_fi_gtn_0 )

## define initial grid1DTNs
init_fe_gtn = fe
init_fi_gtn = fi

#### B Fields
## background field
Bx0 = back_Bx_y(yB_vals)   # B0_ * np.tanh(yB_vals/lam)
init_B0_x_gtn = grid_X.make_mps_ndim({ax_y: Bx0})
# init_B0_x_gtn.constant_axes = [ax_x]
# print('is constant', init_B0_x_gtn.constant_axes)
curl_B0_z = - B0_ / lam / (np.cosh(y_vals/lam)**2)   # d/dx By - d/dy Bx
curl_B0_z_gtn = grid_X.make_mps_ndim({ax_y: curl_B0_z})

init_Bx_gtn  = grid_X.make_gtn_from_dicts([{ax_x: pert_Bx_x(xB_vals), ax_y: pert_Bx_y(yB_vals)}],
                                          data_type=DataType.MPS)
init_By_gtn  = grid_X.make_mps_ndim({ax_x: pert_By_x(xB_vals), ax_y: pert_By_y(yB_vals)})
                                    # {ax_x: np.sin(kx*xB_vals*2) * A0 * kx * 2,
                                    #  ax_y: np.cos(ky*yB_vals)})

# init_Bx_gtn = grid_X.make_empty_gridTN()
# init_By_gtn = grid_X.make_empty_gridTN()
init_Bz_gtn = grid_X.make_empty_gridTN()  # grid_X.make_mps_ndim({ax_x: Bz})

#### E fields
## force at equilbrium:  (E + v x B)  grad_v(f) = 0
## [x] Ex + vy Bz - vz By --> Ex = 0
## [y] Ey + vz Bx - vx Bz --> Ey = - vz Bx0
## [z] Ez + vx By - vy Bx --> Ez = - vy Bx0 = 0

## background E field: 0 ?   or -u_e x B

x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing='ij')

Ex_data = None
init_Ex_gtn = grid_X.make_empty_gridTN()
# Ex_data = - dvze_func(x_mesh, y_mesh) * (pert_By_x(x_mesh) * pert_By_y(y_mesh))
# init_Ex_gtn = grid_X.map_state_to_mps(Ex_data)
# init_Ex_gtn.elemental_multiply(init_By_gtn, inplace=True)

# Ey_data = None
Ey0_data = - ny(y_vals) / n0 * v0_e * back_Bx_y(y_vals)
# Ey0_data = - ny(y_vals) / n0 * v0_i * back_Bx_y(y_vals)
init_E0_y_gtn = grid_X.make_mps_ndim({ax_y: Ey0_data})
# init_Ey_gtn = init_E0_y_gtn.copy()
# Ey0_data = - dvze_func(x_mesh, y_mesh) * back_Bx_y(y_mesh)
# init_E0_y_gtn = grid_X.map_state_to_mps(Ey0_data)
# init_E0_y_gtn.constant_axes = [ax_x]
# print('is constant', init_E0_y_gtn.constant_axes)
init_Ey_gtn = grid_X.make_empty_gridTN()
# ## curl(E) = [z] d/dx Ey - [x] d/dz Ey = 0

init_Ez_gtn = grid_X.make_empty_gridTN()

# plt.figure()
# plt.imshow(init_Ex_gtn.get_data())
# plt.colorbar()
# plt.title('Ex')
#
# plt.figure()
# plt.imshow(init_Ey_gtn.get_data())
# plt.colorbar()
# plt.title('Ey')

# plt.figure()
# plt.imshow(init_Ez_gtn.get_data())
# plt.colorbar()
# plt.title('Ez')
# plt.figure()


### cleaning fields
init_phi_gtn = grid_X.make_empty_gridTN()
init_psi_gtn = grid_X.make_empty_gridTN()


##### boundary conditions ######
deriv_zg = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)

if grid_idx == 0:       # four-fold symmetry x=[0,Lx] and y=[0,Ly]
    deriv_sa = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, right_bc=BCType.ANTISYMMETRIC, order=order,
                                       fd_type=FDType.CENTER)
    deriv_as = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, right_bc=BCType.SYMMETRIC, order=order,
                                       fd_type=FDType.CENTER)
    deriv_aa = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, order=order, fd_type=FDType.CENTER)
    deriv_ss = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)

    init_fe_gtn.ax_deriv_configs.update({ax_x: deriv_ss, ax_y: deriv_ss,
                                         ax_vx_e: deriv_zg, ax_vy_e: deriv_zg, ax_vz_e: deriv_zg})
    init_fi_gtn.ax_deriv_configs.update({ax_x: deriv_ss, ax_y: deriv_ss,
                                         ax_vx_i: deriv_zg, ax_vy_i: deriv_zg, ax_vz_i: deriv_zg})
    # rho0.ax_deriv_configs.update({ax_x: deriv_aa, ax_y: deriv_sa})

    init_Ex_gtn.ax_deriv_configs.update({ax_x: deriv_aa, ax_y: deriv_sa})
    init_Ey_gtn.ax_deriv_configs.update({ax_x: deriv_ss, ax_y: deriv_as})
    # init_Ez_gtn.ax_deriv_configs.update({ax_x: deriv_ss, ax_y: deriv_aa})
    init_Ez_gtn.ax_deriv_configs.update({ax_x: deriv_ss, ax_y: deriv_sa})
    init_E0_y_gtn.ax_deriv_configs.update({ax_x: deriv_ss, ax_y: deriv_as})     ## not checked, but not used

    # jx0.ax_deriv_configs.update({ax_x: deriv_aa, ax_y: deriv_sa})
    # jy0.ax_deriv_configs.update({ax_x: deriv_ss, ax_y: deriv_as})
    # jz0.ax_deriv_configs.update({ax_x: deriv_ss, ax_y: deriv_sa})

    offset_B = 1 if is_yee else 0
    deriv_sa_B = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, right_bc=BCType.ANTISYMMETRIC, order=order,
                                         fd_type=FDType.CENTER, offset=offset_B)
    deriv_as_B = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, right_bc=BCType.SYMMETRIC, order=order,
                                         fd_type=FDType.CENTER, offset=offset_B)
    deriv_aa_B = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, order=order, fd_type=FDType.CENTER, offset=offset_B)
    deriv_ss_B = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER, offset=offset_B)

    init_Bx_gtn.ax_deriv_configs.update({ax_x: deriv_ss_B, ax_y: deriv_as_B})
    init_By_gtn.ax_deriv_configs.update({ax_x: deriv_aa_B, ax_y: deriv_sa_B})
    init_Bz_gtn.ax_deriv_configs.update({ax_x: deriv_aa_B, ax_y: deriv_as_B})
    init_B0_x_gtn.ax_deriv_configs.update({ax_x: deriv_ss_B, ax_y: deriv_as_B})

    # init_Bx0_gtn.ax_deriv_configs.update({ax_x: deriv_zg, ax_y: deriv_zg})
    # curl_B0_z_gtn.ax_deriv_configs.update({ax_x: deriv_zg, ax_y: deriv_zg})

elif grid_idx == 1:     # no symmetry x=[-Lx,Lx) and y=[-Ly,Ly)
    deriv_pp = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
    # deriv_ap = DerivativeConfiguration(left_bc=BCType.ANTIPERIODIC, order=order, fd_type=FDType.CENTER)

    deriv_ss = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)   # zero gradient
    deriv_aa = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, order=order, fd_type=FDType.CENTER)  # zero value
    deriv_op = DerivativeConfiguration(left_bc=BCType.OPEN, order=order, fd_type=FDType.CENTER)  # open bc

    init_fe_gtn.ax_deriv_configs.update({ax_x: deriv_pp, ax_y: deriv_zg,
                                         ax_vx_e: deriv_zg, ax_vy_e: deriv_zg, ax_vz_e: deriv_zg})
    init_fi_gtn.ax_deriv_configs.update({ax_x: deriv_pp, ax_y: deriv_zg,
                                         ax_vx_i: deriv_zg, ax_vy_i: deriv_zg, ax_vz_i: deriv_zg})
    # rho0.ax_deriv_configs.update({ax_x: deriv_pp, ax_y: deriv_zg})

    offset_B = 1 if is_yee else 0
    deriv_aa_B = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, order=order, fd_type=FDType.CENTER, offset=offset_B)
    deriv_ss_B = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER, offset=offset_B)

    init_Bx_gtn.ax_deriv_configs.update({ax_x: deriv_pp, ax_y: deriv_ss_B})
    init_By_gtn.ax_deriv_configs.update({ax_x: deriv_pp, ax_y: deriv_aa_B})
    init_Bz_gtn.ax_deriv_configs.update({ax_x: deriv_pp, ax_y: deriv_ss_B})

    init_B0_x_gtn.ax_deriv_configs.update({ax_x: deriv_pp, ax_y: deriv_ss_B})
    # curl_B0_z_gtn.ax_deriv_configs.update({ax_x: deriv_zg, ax_y: deriv_zg})

    ## Ex = ue_y * Bx - ue_z * By = -ue_z * By
    ## Ey = ue_z * Bx - ue_x * Bz = +ue_z * Bx = ue_z * (B0_ * np.tanh(yB_vals/lam) + pert Bx)
    ## Ez - ue_x * By - ue_y * Bz = 0
    init_Ex_gtn.ax_deriv_configs.update({ax_x: deriv_pp, ax_y: deriv_aa})
    init_Ey_gtn.ax_deriv_configs.update({ax_x: deriv_pp, ax_y: deriv_ss})
    init_Ez_gtn.ax_deriv_configs.update({ax_x: deriv_pp, ax_y: deriv_aa})

    init_E0_y_gtn.ax_deriv_configs.update({ax_x: deriv_pp, ax_y: deriv_ss})
else:
    raise ValueError(f'grid idx {grid_idx} not defined' )


### time step info
## EM stability
if te_order_EM != 22:
    dt_max = min(dt_max, 0.15 * ax_x.dx / plasma_config.c)
print('EM dt max', dt_max, ax_x.dx / plasma_config.c, ax_x.dx, plasma_config.c)

if te_order in [413, 414, 416, 4164, 4174]:
    Fmax_e = np.array([A * np.pi / 2 / y_max, A * np.pi / 2 / x_max]) * charge / mass_e
    Fmax_i = np.array([A * np.pi / 2 / y_max, A * np.pi / 2 / x_max]) * charge / mass_i
else:
    Fmax_e = np.array([B0 * ve_x_max + A * np.pi / 2 / y_max, A * np.pi / 2 / x_max]) * charge / mass_e
    Fmax_i = np.array([B0 * vi_x_max + A * np.pi / 2 / y_max, A * np.pi / 2 / x_max]) * charge / mass_i
do_advec_x_cfl = False if (te_order in [313, 314, 316, 3164, 3174, 413, 414, 416, 4174, 4164]) else True
dt_cfl = VP_test.get_dt(Fmax_e, Fi_maxs=Fmax_i, do_advec_x_cfl=do_advec_x_cfl)

print('dt', dt, dt_cfl, dt_max)
dt = dt_cfl if dt is None or do_adapt_dt else dt
dt = min(dt, dt_max)
num_tsteps = int(T / dt)
print('dt', dt, num_tsteps)


### get file naming string
fstr = VP_test.get_fstr(extra_str=extra_str)
print('filename', fstr)

########################
#### run simulation ####
########################

### define v2 mpo for measurement ###
v2e_mpo = test_setup.get_v2_mpo(grid_e, VP_test.ve_axes)
v2i_mpo = test_setup.get_v2_mpo(grid_i, VP_test.vi_axes)


#### define fields
init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, is_sqrt=is_sqrt, compress_config=te_compress_e)
init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, is_sqrt=is_sqrt, compress_config=te_compress_i)
init_E_field = Field('E', grid_X, data={X: init_Ex_gtn, Y: init_Ey_gtn, Z: init_Ez_gtn},
                     is_sqrt=True, compress_config=te_compress_E)
init_B_field = Field('B', grid_X, data={X: init_Bx_gtn, Y: init_By_gtn, Z: init_Bz_gtn},
                     is_sqrt=True, compress_config=te_compress_B)

init_phi_field = ScalarField('phi', grid_X, data=init_phi_gtn, is_sqrt=True, compress_config=te_compress_E)
init_psi_field = ScalarField('psi', grid_X, data=init_psi_gtn, is_sqrt=True, compress_config=te_compress_B)

E0_field = None
curl_E0_field = None
# E0_field = Field('E0', grid_X, data={Y: init_E0_y_gtn})
# curl_E0_field = Field('curl_E0', grid_X, {})
# print('background E0 bond dims', E0_field.max_bonds(), div_E0_field.max_bonds())

# B0_field = None
# curl_B0_field = None
B0_field = Field('B0', grid_X, data={X: init_B0_x_gtn})
curl_B0_field = Field('curl_B0', grid_X, {Z: curl_B0_z_gtn, })
# print('background B0 bond dims', B0_field.max_bonds(), curl_B0_field.max_bonds())

# charge0_field = ScalarField('charge_0', grid_X, rho0)
# current0_field = Field('current_0', grid_X, {X: jx0, Y: jy0, Z: jz0})
# print('charge0 bond dims', charge0_field.max_bonds(), current0_field.max_bonds())


### load data if restarting ###
try:
    out = test_setup.load_data_EM(VP_test,
                            init_fe_field, init_fi_field, init_E_field, init_B_field, init_phi_field, init_psi_field,
                            fdir=fdir, sdir=sdir, save_every_nt=save_every_nt, extra_str=extra_str,
                            load_data=load_data, restart=restart, restart_from_T=restart_from_T,
                            )

    (init_fe_field, init_fi_field, init_E_field, init_B_field, init_phi_field, init_psi_field,
     ts, nrg_fe_ts, nrg_fi_ts, nrg_E_ts, nrg_B_ts,
     max_bond_fe, max_bond_fi, max_bond_E, max_bond_B, max_bond_phi, max_bond_psi) = out

    nt = len(ts) - 1
    restarted = True

    flux_By_ts = pickle.load(open(fdir + 'flux_By_' + fstr + '.pkl', 'rb'))
    flux_By_ts = flux_By_ts[:nt + 1]

except(NameError, ValueError, FileNotFoundError):

    ts = [0]
    nt = 0
    restarted = False

    nrg_fe = init_fe_gtn.meas_expec(v2e_mpo, is_sqrt=is_sqrt) * mass_e / 2 / tot_V
    nrg_fi = init_fi_gtn.meas_expec(v2i_mpo, is_sqrt=is_sqrt) * mass_i / 2 / tot_V
    # nrg_E  = init_E_field.norm() ** 2 * 1. / 2 / tot_x
    # nrg_B  = init_B_field.norm() ** 2 * 1. / 2 / tot_x
    nrg_E  = [np.nan, np.nan, np.nan]
    nrg_B  = [np.nan, np.nan, np.nan]  # A / dk
    print('tot_x', x_max, y_max, tot_V)

    ## maybe should initialize system, run EM_sys for half timestep to start
    ## maybe should initialize B field on shifted lattice

    nrg_fe_ts = [nrg_fe]
    nrg_fi_ts = [nrg_fi]
    nrg_E_ts  = [nrg_E]
    nrg_B_ts  = [nrg_B]
    nrg_By_ts = [nrg_B]
    flux_By_ts = [pert_By_y(0) * A0 * kx * box_len_x / np.pi ]
    ## int_0^Lx/2 By(x,y=0) dx = By(y=0) * int_0^Lx/2 A0 * kx * sin(kx x) dx, where kx = 2*pi/Lx
    ##                         = By(y=0) * A0 * kx * Lx/pi

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
    print('field norm', norm_fe, norm_fi)
else:
    print('field norm', norm_fe**2, norm_fi**2)

### BUILD PDE
em_sys = Maxwell(init_E_field, init_B_field, # phi=init_phi_field, psi=init_psi_field,
                 coords_x=coords_x, matl_params=plasma_config, te_order=te_order_EM,
                 background_B0=B0_field, background_E0=E0_field, normalize=False,
                 is_electrostatic=False, clean=False, is_yee=is_yee)

vm_sys = VlasovMaxwell(init_fe_field, init_fi_field, em_sys,
                       coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                       elc_params=elc_config, ion_params=ion_config, zipup=True,
                       te_order=te_order, compress_levels=comp_levels, )

print('initialized vm_sys')
print('divB', em_sys.check_divB().norm(), 'divE', em_sys.check_poisson().norm())

em_sys.curlB0 = curl_B0_field
em_sys.curlE0 = curl_E0_field


charge = vm_sys.compute_charge_density()
plt.figure()
plt.imshow(np.real(charge.get_comp_data()))
plt.colorbar()
plt.title('charge density')
plt.show()

# calc_E = vm_sys.compute_E_from_charge_density(compress_opts=te_compress_E.get_compress_opts(1))
# for C in calc_E.componentIDs:
#     plt.figure()
#     plt.imshow(np.real(calc_E.get_comp_data(C)))
#     plt.colorbar()
#     plt.title(f'E{C}')
# plt.show()

# current = vm_sys.compute_current()
# current_Z = current.get_comp_data(Z)
# curl_B0_z_mesh = em_sys.curlB0[Z].get_data(pad_data=True)
# if current_Z is not None:
#     plt.figure()
#     plt.imshow(np.real(current_Z)/eps0)
#     plt.colorbar()
#     plt.title('Jz / eps0')
#
#     plt.figure()
#     plt.imshow(curl_B0_z_mesh * c_speed ** 2)
#     plt.colorbar()
#     plt.title('c^2 curl B0 [z]')
#
#     plt.figure()
#     plt.imshow(np.real(current_Z) / eps0 - curl_B0_z_mesh * c_speed ** 2)
#     plt.colorbar()
#     plt.title('Jz / eps0 - c^2 curl B0 [z]')
#     plt.show()

vm_sys.collision = VP_test.collision_config
print('colls', vm_sys.collision.coll_params)

vm_sys.compress_F = VP_test.compress_F
vm_sys.compress_F_opts = VP_test.compress_F_opts
print('compress E?',  vm_sys.compress_F, vm_sys.compress_F_opts)

print('E nrg', np.array(vm_sys.tot_E.norms(coords_x.coords)) ** 2 * 1. / 2 / tot_V)
print('B nrg', np.array(vm_sys.tot_B.norms(coords_x.coords)) ** 2 * 1. / 2 / tot_V)



if nt == 0:
    fig1, ax1 = plt.subplots(2, 1)
    time_ = nt * dt
    fi_data = init_fi_gtn.get_data(ax_select={ax_x: npts_x//2, ax_vx_i: npts_vxi // 2, ax_vy_i: npts_vyi // 2})
    fi_data = ax_x.basis.get_realspace_1D(fi_data, 0)
    fi_data = ax_vx_i.basis.get_realspace_1D(fi_data, 1)
    if is_sqrt:
        fi_data = np.abs(fi_data) ** 2
    im1 = ax1[0].imshow(np.real(fi_data).T,
                        extent=(y_min, y_max, vi_z_min, vi_z_max),
                        aspect='auto', origin='lower')

    fe_data = init_fe_gtn.get_data(ax_select={ax_x: npts_x//2, ax_vx_e: npts_vxe // 2, ax_vy_e: npts_vye // 2})
    fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
    fe_data = ax_vx_e.basis.get_realspace_1D(fe_data, 1)
    if is_sqrt:
        fe_data = np.abs(fe_data) ** 2
    im2 = ax1[1].imshow(np.real(fe_data).T,
                        extent=(y_min, y_max, ve_z_min, ve_z_max),
                        aspect='auto', origin='lower')

    fig1.colorbar(im1, ax=ax1[0])
    fig1.colorbar(im2, ax=ax1[1])
    ax1[0].set_title(f'time {time_:3.2f}')
    ax1[0].set_xlabel('y'), ax1[0].set_ylabel('vz')
    ax1[1].set_xlabel('y'), ax1[1].set_ylabel('vz')
    # if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')
    # plt.close(fig1)

    plt.show()


if restarted:
    if do_adapt_dt:
        # use_bg = False if (te_order in [413, 414, 416, 4164, 4174]) else True
        # do_advec_x_cfl = False if (te_order in [313, 314, 316, 3164, 3174, 413, 414, 416, 4164, 4174]) else True
        # # em_term = vm_sys.compute_force_term(is_ion=False, background_force=use_bg)
        # em_term = vm_sys.get_force_term(is_ion=False, background_force=use_bg)
        # max_em = []
        # for coord in [X,Y,Z]:
        #     try:
        #         em_data = em_term.get_comp_data(X, ax_select={ax_vx_e: 0, ax_vy_e: 0, ax_vz_e: 0})
        #         em_max = np.max(np.abs(em_data))
        #     except KeyError:
        #         em_max = 0.
        #     max_em += [em_max]

        # F_maxs = np.array(max_em)
        # dt = VP_test.get_dt(F_maxs, do_advec_x_cfl = do_advec_x_cfl)
        # dt = min(dt, dt_max)
        dt = ts[-1] - ts[-2]
        print('restarted dt', dt)

time1 = time.time()

print('starting time evolution')
while ts[-1] < T:

    walltime = time.time()
    dt_ = dt * 0.01 if nt == 0 and te_order in [3164, 3174, 316] else dt
    vm_sys = vm_sys.next_time_step(dt_, compress_level=1, err_tol=1.0e-7, verbose_plot=False,
                                   is_first_time_step=(nt==0))
    print('wall time', time.time() - walltime)

    fe_field = vm_sys.fe
    max_bond_fe += [fe_field.component.max_bond()]
    nrg_fe = fe_field.meas_expec(v2e_mpo).component * mass_e / 2 / tot_V

    fi_field = vm_sys.fi
    max_bond_fi += [fi_field.component.max_bond()]
    nrg_fi = fi_field.meas_expec(v2i_mpo).component * mass_i / 2 / tot_V

    E_field = vm_sys.tot_E
    max_bond_E += [E_field.max_bonds(compIDs=coords_x.coords)]
    energy_E_t = np.array(E_field.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_V   # E field energy

    B_field = vm_sys.tot_B
    max_bond_B += [B_field.max_bonds(compIDs=coords_x.coords)]
    energy_B_t = np.array(B_field.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_V   # magnetic field energy

    # print('ax_x zero?', ax_x.xpts[ax_x.npts // 2], ax_x.dx)
    # print('ax_y zero?', ax_y.xpts[ax_y.npts // 2], ax_y.dx)

    if grid_idx == 1:
        By_y0 = B_field[Y].get_data(ax_select={ax_y: ax_y.npts // 2})
        flux_By_ts += [np.sum(By_y0[ax_x.npts//2:]) * ax_x.dx]
    else:
        By_y0 = B_field[Y].get_data(ax_select={ax_y: 0})
        flux_By_ts += [np.sum(By_y0[:]) * ax_x.dx]
    print('flux By', flux_By_ts[-1])

    nrg_fe_ts += [nrg_fe]
    nrg_fi_ts += [nrg_fi]
    nrg_E_ts += [energy_E_t]
    nrg_B_ts += [energy_B_t]

    nrg_E_tot = np.sum([energy_E_t[i] if not np.isnan(energy_E_t[i]) else 0.0 for i in range(3)])
    nrg_B_tot = np.sum([energy_B_t[i] if not np.isnan(energy_B_t[i]) else 0.0 for i in range(3)])

    print('E nrg', np.array(vm_sys.E.norms(coords_x.coords)) ** 2 * 1. / 2 / tot_V)
    print('B nrg', np.array(vm_sys.B.norms(coords_x.coords)) ** 2 * 1. / 2 / tot_V)

    ts += [ts[-1] + dt_]
    nt += 1

    if do_adapt_dt:
        use_bg = False if (te_order in [413, 414, 416, 4164, 4174]) else True
        do_advec_x_cfl = False if (te_order in [313, 314, 316, 3164, 3174, 413, 414, 416, 4164, 4174]) else True
        em_term = vm_sys.compute_force_term(is_ion=False, background_force=use_bg)
        max_em = []
        for coord in [X, Y, Z]:
            try:
                em_data = em_term.get_comp_data(coord, ax_select={ax_vx_e: 0, ax_vy_e: 0, ax_vz_e: 0})
                ## select elements with max v on v axes (bc em_term = E + v x B)
                em_max = np.max(np.abs(em_data)) if em_data is not None else 0.0
            except KeyError:
                em_max = 0.
            max_em += [em_max]

        # print('max em', max_em, B0 * charge / mass_e * ax_vx_e.xpts[-1])
        F_maxs = np.array(max_em)
        dt = VP_test.get_dt(F_maxs, do_advec_x_cfl=do_advec_x_cfl)
        dt = min(dt, dt_max)
        print('restarted dt', dt)

    print('divB', em_sys.check_divB().norm(), 'divE', em_sys.check_poisson().norm())

    norm = vm_sys.fe.norm() - vm_sys.fi.norm()
    print('result norm', nt, ts[-1], dt, norm, nrg_fe, nrg_fi,
          nrg_E_tot, nrg_B_tot,
          E_field.max_bond(X), E_field.max_bond(Y), E_field.max_bond(Z),
          B_field.max_bond(X), B_field.max_bond(Y), B_field.max_bond(Z),
          fi_field.max_bond(), fe_field.max_bond(), Lx, Lxi, Lxe)

    if nt % (save_every_nt // 10) == 0:

        if save_data:
            if not os.path.exists(sdir + 'restart/'):
                os.makedirs(sdir + 'restart/')

            test_setup.save_data_EM( VP_test, vm_sys, sdir+'restart/', add_fstr=f'-nt{nt}',
                                     extra_str=extra_str,
                                     ts=ts, nrg_fe_ts=nrg_fe_ts, nrg_fi_ts=nrg_fi_ts, nrg_E_ts=nrg_E_ts,
                                     nrg_B_ts=nrg_B_ts, max_bond_fe=max_bond_fe, max_bond_fi=max_bond_fi,
                                     max_bond_E=max_bond_E, max_bond_B=max_bond_B,
                                     max_bond_phi=max_bond_phi, max_bond_psi=max_bond_psi)

            pickle.dump(flux_By_ts, open(fdir + 'flux_By_' + fstr + '.pkl', 'wb'))


        print('mpl backend', matplotlib.get_backend())
        fig1, ax1 = plt.subplots(2, 1)
        time_ = ts[-1]
        fi_data = fi_field.get_comp_data(ax_select={ax_x: npts_x // 2, ax_vx_i: npts_vxi // 2, ax_vy_i: npts_vyi // 2})
        if fi_data is not None:
            fi_data = np.abs(fi_data) ** 2 if is_sqrt else fi_data
            fi_data = ax_y.basis.get_realspace_1D(fi_data, 0)
            fi_data = ax_vz_i.basis.get_realspace_1D(fi_data, 1)
            im1 = ax1[0].imshow(np.real(fi_data).T, extent=(y_min, y_max, vi_z_min, vi_z_max),
                                aspect='auto', origin='lower')

        fe_data = fe_field.get_comp_data(ax_select={ax_x: npts_x // 2, ax_vx_e: npts_vxe // 2, ax_vy_e: npts_vye // 2})
        if fe_data is not None:
            fe_data = np.abs(fe_data) ** 2 if is_sqrt else fe_data
            fe_data = ax_y.basis.get_realspace_1D(fe_data, 0)
            fe_data = ax_vz_e.basis.get_realspace_1D(fe_data, 1)
            im2 = ax1[1].imshow(np.real(fe_data).T, extent=(y_min, y_max, ve_z_min, ve_z_max),
                                aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].xaxis.set_visible(False)
        ax1[0].set_xlabel('y'), ax1[0].set_ylabel('vz')
        ax1[1].set_xlabel('y'), ax1[1].set_ylabel('vz')
        if save_figs:
            fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')
        else:
            plt.show()
        plt.close(fig1)

        ###########
        # energy_E_t = np.sum(Ex_t*np.conj(Ex_t),axis=1)*dx*1./2   # electric field energy
        energy_E_t = np.array(nrg_E_ts)
        energy_B_t = np.array(nrg_B_ts)
        energy_fi_t = np.array(nrg_fi_ts)
        energy_fe_t = np.array(nrg_fe_ts)
        total_nrg_t = energy_fe_t + energy_fi_t + np.sum(energy_E_t, axis=1) + np.sum(energy_B_t, axis=1)

        fig7, ax7 = plt.subplots()
        # ts = np.arange(num_tsteps+1)*dt
        ax7.semilogy(ts, np.abs(energy_fe_t), label='elec')
        ax7.semilogy(ts, np.abs(energy_fi_t), label='ion')
        ax7.semilogy(ts, np.abs(energy_E_t), label='E field')
        ax7.semilogy(ts, np.abs(energy_B_t), label='B field')
        ax7.semilogy(ts, np.abs(total_nrg_t), ls='--', label='total')
        plt.xlabel('t')
        plt.ylabel('energy')
        plt.legend()
        plt.show()
        plt.close()

        if save_figs or matplotlib.get_backend() != 'agg':

            plt.figure()
            plt.plot(ts, flux_By_ts)
            plt.title('flux By(y=0)')
            plt.xlabel('t')
            plt.ylabel('flux By')
            plt.show()

            for compID in [X, Y, Z]:
                By_data = vm_sys.B.get_comp_data(compID)
                if By_data is not None:
                    plt.figure()
                    plt.imshow(np.real(By_data))
                    plt.colorbar()
                    plt.title(f'B{compID}')

            for compID in [X, Y, Z]:
                Ex_data = vm_sys.E.get_comp_data(compID)
                if Ex_data is not None:
                    plt.figure()
                    plt.imshow(np.real(Ex_data))
                    plt.colorbar()
                    plt.title(f'E{compID}')


            # je = vm_sys.sys_fe.compute_flow() * elc_config.charge
            # ji = vm_sys.sys_fi.compute_flow() * ion_config.charge

            # if je is not None:
            #     plt.figure()
            #     plt.imshow(np.real(je.get_comp_data(Z)))
            #     plt.colorbar()
            #     plt.title('jz,e')
            #
            # if ji is not None:
            #     plt.figure()
            #     plt.imshow(np.real(ji.get_comp_data(Z)))
            #     plt.colorbar()
            #     plt.title('jz,i')


            current: 'Field' = vm_sys.compute_current()
            print('current compIDs', current.componentIDs)
            hall_term = current.cross_product(vm_sys.B, coords_x, compress_level=1, zipup=True)

            current_Z = current.get_comp_data(Z)
            # curl_B0_z_mesh = em_sys.curlB0[Z].get_data(pad_data=True)
            if current_Z is not None:
                plt.figure()
                plt.imshow(np.real(current_Z)/eps0)
                plt.colorbar()
                plt.title('Jz / eps0')

                # plt.figure()
                # plt.imshow(curl_B0_z_mesh * c_speed ** 2)
                # plt.colorbar()
                # plt.title('c^2 curl B0 [z]')
                #
                # plt.figure()
                # plt.imshow(np.real(current_Z) / eps0 - curl_B0_z_mesh * c_speed ** 2)
                # plt.colorbar()
                # plt.title('Jz / eps0 - c^2 curl B0 [z]')
                plt.show()

            hall_Z = hall_term.get_comp_data(Z)
            if hall_Z is not None:
                plt.figure()
                plt.imshow(np.real(hall_Z))
                plt.colorbar()
                plt.title('(J x B)z')

            velocities: 'Field' = vm_sys.velocities_e
            print('velocities compIDs', velocities.componentIDs)
            lorentz = velocities.cross_product(vm_sys.B, coords_x)
            lorentz_Z = lorentz.get_comp_data(Z, ax_select={ax_vx_e: 0, ax_vy_e: 0, ax_vz_e: 0})    # -v values
            if lorentz_Z is not None:
                plt.figure()
                plt.imshow(np.real(lorentz_Z))
                plt.title('(v x B)z')

            if not save_figs:
                plt.show()
            plt.close()



if save_data:
    test_setup.save_data_EM(VP_test, vm_sys, fdir, extra_str=extra_str,
                            ts=ts, nrg_fe_ts=nrg_fe_ts, nrg_fi_ts=nrg_fi_ts, nrg_E_ts=nrg_E_ts,
                            nrg_B_ts=nrg_B_ts, max_bond_fe=max_bond_fe, max_bond_fi=max_bond_fi,
                            max_bond_E=max_bond_E, max_bond_B=max_bond_B,
                            max_bond_phi=max_bond_phi, max_bond_psi=max_bond_psi)

    pickle.dump(flux_By_ts, open(fdir + 'flux_By_' + fstr + '.pkl', 'wb'))

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
fi_data = vm_sys.fi.get_comp_data(ax_select={ax_x: npts_x//2, ax_vx_i: npts_vxi//2, ax_vz_i: npts_vzi//2})
fi_real = ax_x.basis.get_realspace_1D(fi_data, 0)
fi_real = ax_vz_i.basis.get_realspace_1D(fi_real, 1)
fi_real = np.abs(fi_real)**2 if is_sqrt else fi_real
im1 = ax1[0].imshow(np.real(fi_real).T, extent=(y_min, y_max, vi_y_min, vi_y_max),
                    aspect='auto', origin='lower')

fe_data = vm_sys.fe.get_comp_data(ax_select={ax_x: npts_x//2, ax_vx_e: npts_vxe//2, ax_vz_e: npts_vze//2})
fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
fe_real = ax_vx_e.basis.get_realspace_1D(fe_real, 1)
fe_real = np.abs(fe_real)**2 if is_sqrt else fe_real
im2 = ax1[1].imshow(np.real(fe_real).T, extent=(y_min, y_max, ve_y_min, ve_y_max),
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
max_bond_Ey = [mb[Y] if mb[Y] is not None else np.nan for mb in max_bond_E]
ax2.plot(np.log(max_bond_Ey) / np.log(q), label='Ey')
max_bond_By = [mb[Y] if mb[Y] is not None else np.nan for mb in max_bond_B]
ax2.plot(np.log(max_bond_By) / np.log(q), label='By')
ax2.plot([0, num_tsteps], [(Lx*KX+KV*Lxe) // 2] * 2, ':', label='max fe')
ax2.plot([0, num_tsteps], [Lx*KX // 2] * 2, ':', label='max V')
ax2.legend()
ax2.set_xlabel('t')
ax2.set_ylabel('log_{q^K}(max bond dimension)')
if save_figs:   fig2.savefig(fdir + 'plot_maxbond_' + fstr + '.png')

# energy_E_t = np.sum(Ex_t*np.conj(Ex_t),axis=1)*dx*1./2   # electric field energy
energy_E_t = np.array(nrg_E_ts)
energy_B_t = np.array(nrg_B_ts)
energy_fi_t = np.array(nrg_fi_ts)
energy_fe_t = np.array(nrg_fe_ts)
total_nrg_t = energy_fe_t + energy_fi_t + np.sum(energy_E_t,axis=1) + np.sum(energy_B_t,axis=1)
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
ax7.semilogy( helper.entanglement_entropy_all(vm_sys.fe.component.data), '-x', label='elec')
ax7.semilogy( helper.entanglement_entropy_all(vm_sys.fi.component.data), '-o', label='ion')

ax7.legend()
ax7.set_xlabel('bond number')
ax7.set_ylabel('EE')
if save_figs:
    fig7.savefig(fdir + 'plot_EE_' + fstr + '.png')
    print(fdir + 'plot_EE_' + fstr + '.png')

plt.show()


print('done')

