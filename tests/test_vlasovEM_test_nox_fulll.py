import os, sys, pickle, time, glob

import scipy.optimize

sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

# from defaults import *
from setup.configs import *
import setup.helper as helper_test
import helper_quimb as helper
import setup_test as test_setup

from axis import Axis
from basis.basis_spatial import SpatialBasis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_EM import Maxwell
from pde_vlasovEM import VlasovMaxwell
from pde_darwin import Darwin
from pde_vlasovED import VlasovDarwin
from vlasov_tests import VlasovTest

import helper_TE



"""
test distribution in the presence of time dependent electric field
http://ammar-hakim.org/sj/je/je32/je32-vlasov-test-ptcl.html
don't use low rank / TT
"""

save_figs = False
save_data = True
load_data = False
restart = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)
is_sqrt = False  # True
is_darwin = False

if restart or save_data or save_figs:
    # sdir = '/pool001/erikaye/tns_pde_v2/data15-2/VM12-test/'
    # if not os.path.exists(sdir):
    #     os.makedirs(sdir)

    # fdir = 'data22/VM12-test-full-F2/'
    fdir = 'data22/VM12-test-full-Ft/'
    # fdir = 'data15-3/VM12-test-no-x-adapt/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)

extra_str = ''
if is_sqrt:
    extra_str = '_sq'

#######################
# standard parameters #
#######################

plasma_config = UnitsConfiguration(eps0=1./8.0e-2, mu0=1./8.0e-2)
eps0 = plasma_config.eps0
mu0 = plasma_config.mu0
charge = plasma_config.e  # unit of charge
eV = plasma_config.eV  # unit of kB*T

mass_e = 1.0  # mass of electron
mass_i = 1836 * mass_e  # mass of ion
n0_e = 1.0  # electron number density
n0_i = n0_e  # ion number density
T_e = T_i = 1.0
ion_config1 = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
# elc_config1 = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
elc_config1 = IonConfiguration(n0=n0_i, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config1.vth  # electron thermal speed 1/2 m b^2 n0 = sqrt(5.0e-5)
wp_e = elc_config1.wp    # plasma e' frequency
lamD = elc_config1.lamD  # Debye length
wp_i = ion_config1.wp    # plasma i+ frequency
vth_i = ion_config1.vth  # ion thermal speed

charge_e = elc_config1.charge


#######################
# standard parameters #
#######################

### initial fields
Bz0 = 1.0
Ex0 = 0.9   # 1.0
omega = 0.4567    # 1.0
k = 0.0

def Ex(x,t):
    if k == 0.0:
        if isinstance(x, np.ndarray):
            return Ex0 * np.ones(x.shape) * np.cos(omega * t)
        else:
            return Ex0 * np.cos(omega * t)
    else:
        return Ex0 * np.cos( k * x - omega * t)


## runtime parameter
q = 2
KX = 0
KV = 2
Lx, Lve, Lvi = 4, 6, 6

TN_layout = LayoutType.SEQUENTIAL
x_map_key = 'F'
v_map_key = 'B'

# TN_layout = LayoutType.PARALLEL  # _GROUP
# x_map_key = 'F'  # 'F'
# v_map_key = 'M'  # 'M'

# TN_layout = LayoutType.COMB
# x_map_key = 'B'
# v_map_key = 'B'

order = 1     ## derivative order
if order > 1:
    extra_str += f'_o{order}'
T = 100
print('T', T)
do_adapt_dt = False  # False  # True
do_semiimplicit = False # True
te_order = 4  # 84  # 3184
dt_frac = 0.9
dt = 0.0125 / 2  #  0.1 for L=7
is_yee = False

use_E0_as_bg = False
use_B0_as_bg = True
"""
fine for is_sqrt = False
slightly wrong for is_sqrt = True but not for 311. Error doesn't seem to decrease appreciably with dt.
something wrong with maccormack? for sqrt. --don't think so: 414 shows the same behavior
not a dt thing... error doesn't decrease when reducing time step
I think maccormack should have a O(dt^2) discrepancy between sqrt and not-sqrt time stepping
1 time step:
dt = 0.005:  sqrt: px_e 0.004443638279162398 -1.1105775630843183e-05
             reg:  px_e 0.004499978912259819 -1.1249819698436637e-05
             delta ~ 0.00005
             sqrt 413 E -> E0: px_e 0.004498588577402748 -1.4313052127466524e-05
             sqrt 333 E -> E0: px_e 0.004443648997070124 -1.102215100840599e-05
dt = 0.001:  sqrt: px_e 0.000888732765738987 -4.442737180748169e-07
             reg:  px_e 0.0008999999301099957 -4.500444197095359e-07
             delta ~ 0.00001
"""

save_every_nt = 500   # 50 # 25000 # 1000 // 20 # // 5

# DMAX = 8   # 16  # None # 32 # None
# DMAX_F = DMAX
# comp_style = 3

VP_test = VlasovTest(Lxs=(Lx,), Lves=(Lve,Lve,Lve), Lvis=(Lvi,Lvi,Lvi), KX=KX, KV=KV, q=q,
                    #
                    ion_config = ion_config1, elc_config = elc_config1,
                    coll_type=None, do_adapt_coll = True, coll_coeff_e = 0.005, coll_rate_e = 0.01,
                    # DMAX = DMAX, DMAX_F = DMAX_F, compress_style = comp_style, compress_style_mod = 0,
                    do_adapt_dt=do_adapt_dt, dt_frac=dt_frac, dt=dt, T=T, te_order=te_order,
                    grid_layout=TN_layout, x_map_key = x_map_key, ve_map_key = v_map_key,
                    )

VP_test.initialize_axes(x_lims=(0, 2*np.pi), ve_lims=(-12*vth_e, 12*vth_e), vi_lims=(-12*vth_i, 12*vth_i),
                        x_map_key=x_map_key, ve_map_key=v_map_key, basis_xs = SpatialBasis(),)

VP_test.initialize_grid()

# # CUTOFF = 1.0e-12
# te_compress_i = VP_test.get_compression_config(DMAX=DMAX, cutoff=CUTOFF)
# te_compress_e = VP_test.get_compression_config(DMAX=DMAX, cutoff=CUTOFF)
# te_compress_E = VP_test.get_compression_config(DMAX=DMAX)
# te_compress_B = VP_test.get_compression_config(DMAX=DMAX)
#
# print('te compress fe', te_compress_e)
# print('te compress fi', te_compress_i)
# print('te compress E', te_compress_E)
# print('te compress B', te_compress_B)
#
# comp_levels = VP_test.get_compress_levels()
#
# print('compress', DMAX, VP_test.compress_style, VP_test.compress_style_mod)

#############################
## start building PDE system
#############################

## define axis objects; position -> k-space,
# ax_x, = VP_test.pos_axes
# ax_vx_e, ax_vy_e, = VP_test.ve_axes
ax_vx_e: 'Axis' = VP_test.ve_axes[0]
ax_vy_e: 'Axis' = VP_test.ve_axes[1]
print('ax map', ax_vx_e.map, ax_vy_e.map)
print('dvx', ax_vx_e.dx, 'dvy', ax_vy_e.dx)
print('cfl dt', ax_vx_e.dx / (Bz0 * np.max(ax_vx_e.xpts)), dt)
# exit()
# print('ax_x zero ind', ax_x.zero_ind)

# print('VP xlims', VP_test.x_lims)
# print('VP xlims', VP_test.x_lims[0])
# x_min, x_max = VP_test.x_lims[0]
# print('x range',ax_x.dx, x_min, x_max)
ve_x_min, ve_x_max = VP_test.ve_lims[0]
ve_y_min, ve_y_max = VP_test.ve_lims[1]
tot_x = 1  # x_max - x_min

# coords_x  = VP_test.pos_coordsys
# coords_ve = VP_test.ve_coordsys
# coords_vi = VP_test.vi_coordsys
#
# X, Y, Z = coords_x.coords
# VX, VY, VZ = coords_ve.coords

npts_x = 1 # ax_x.npts
npts_vxe = ax_vx_e.npts
npts_vye = ax_vy_e.npts

# ## define MPS grids
grid_i = VP_test.grid_i
grid_e = VP_test.grid_e
grid_X = VP_test.grid_X

x_vals = 0  # ax_x.xpts
xB_vals = x_vals

vx_e_vals = ax_vx_e.xpts
vy_e_vals = ax_vy_e.xpts
vx_mesh, vy_mesh = np.meshgrid(vx_e_vals, vy_e_vals, indexing='ij')
dV = ax_vx_e.dx * ax_vy_e.dx



####3 define initial gridTNs

## ion density:
# init_fi_gtn = grid_i.make_empty_gridTN()

## elec density
fe1_vx = helper_test.maxwellian(vx_e_vals, vth2=vth_e**2, density=1.0, flow=0.) * n0_e
fe1_vy = helper_test.maxwellian(vy_e_vals, vth2=vth_e**2, density=1.0, flow=0.)
if is_sqrt:
    fe1_vx = np.sqrt(fe1_vx)
    fe1_vy = np.sqrt(fe1_vy)

init_fe = np.outer(fe1_vx, fe1_vy)

# init_fe_gtn = grid_e.make_gtn_from_dicts([{ax_vx_e: fe1_vx, ax_vy_e: fe1_vy}],
#                                          data_type=DataType.MPS)


## EM forces
force_Bx = Bz0 * vy_mesh
force_By = -Bz0 * vx_mesh
force_Ex = Ex(x_vals, 0) * np.ones((ax_vx_e.npts, ax_vy_e.npts))


### time step info
num_tsteps = int(T / dt)
print('dt', dt, num_tsteps, plasma_config.c)


### get file naming string
fstr = f'w{omega}_' +  VP_test.get_fstr(extra_str=extra_str)
print('filename', fstr)

########################
#### run simulation ####
########################

### define v2 mpo for measurement ###
v2 = vx_mesh ** 2 + vy_mesh ** 2


##### initialize tests
ts = [0]
nt = 0

nrg_fe = np.sum(init_fe * v2) * ax_vx_e.dx * ax_vy_e.dx
print('tot_x', tot_x)


def compute_norm(dist, is_sqrt=False):
    dist = dist.conj() * dist if is_sqrt else dist.copy()
    # dist[ 0, :] /= 2
    # dist[-1, :] /= 2
    # dist[:,  0] /= 2
    # dist[:, -1] /= 2
    return np.sum(dist) * dV

def compute_m1(dist, is_sqrt=False):
    dist = dist.conj() * dist if is_sqrt else dist.copy()
    # dist[ 0, :] /= 2
    # dist[-1, :] /= 2
    # dist[:,  0] /= 2
    # dist[:, -1] /= 2

    vx_dist = vx_mesh * dist
    vy_dist = vy_mesh * dist

    return np.sum(vx_dist) * dV, np.sum(vy_dist) * dV

def compute_m2(dist, is_sqrt=False):
    dist = dist.conj() * dist if is_sqrt else dist.copy()
    # dist[ 0, :] /= 2
    # dist[-1, :] /= 2
    # dist[:,  0] /= 2
    # dist[:, -1] /= 2
    return np.sum(v2 * dist) * dV, np.sum(v2 * dist) * dV


### measure conservation ###
### note: conservation of momentum and energy don't hold because fields aren't evolved by Maxwell's eqns.
norm0 = compute_norm(init_fe, is_sqrt=False)

norm_fe = compute_norm(init_fe, is_sqrt=is_sqrt)
print('field norm', norm_fe)

px_e, py_e = compute_m1(init_fe, is_sqrt=is_sqrt)
print('init avg v', px_e, py_e)


nrg_fe_ts = [nrg_fe]
# nrg_E_ts  = [nrg_E]
avg_vx_ts = [px_e]
avg_vy_ts = [py_e]
errs_avg   = [np.linalg.norm([px_e-0,py_e-0])]
errs_dist  = [np.nan]     ## distribution - gaussian centered at theoretical px, py
errs_shape = [np.nan]     ## distribution - gaussian centered at measured px, py
## errors with respect to fitted distribution
errs_norm   = [np.nan]    ## distribution - gaussian centered at measured px, py
errs_drift  = [np.nan]    ## distribution - gaussian centered at measured px, py
errs_sigma2 = [np.nan]    ## distribution - gaussian centered at measured px, py
errs_fitted = [np.nan]    ## distribution - fitted gaussian
errs_deriv  = [np.nan]    ## (d/dx + d/dy) distribution - fitted gaussian; attempt to measure (lack of) smoothness



fig1, ax1 = plt.subplots(1, 1)
time_ = nt * dt

im1 = ax1.imshow(np.real(init_fe).T, extent=(ve_x_min, ve_x_max, ve_y_min, ve_y_max),
                    aspect='auto', origin='lower')

fig1.colorbar(im1, ax=ax1)
ax1.set_title(f'time {time_:3.2f}')
ax1.set_xlabel('vx'), ax1.set_ylabel('vy')

plt.show()

em_term = force_Ex + force_Bx
plt.figure()
plt.imshow(em_term)
plt.colorbar()
plt.show()

# deriv_vx = grid_e.get_firstderivative_mpo(ax_vx_e).get_data()
# deriv_vy = grid_e.get_firstderivative_mpo(ax_vy_e).get_data()

def euler(dist, dt, deriv0=None, **kwargs):
    if deriv0 is None:
        deriv0 = deriv(dist)
    return dist - dt * deriv0

def deriv(dist, time=None, **kwargs):
    time = ts[-1] if time is None else time
    # force_Ex_ = Ex(x_vals, time) * np.ones(force_Ex.shape)
    # print('time', time, Ex(0, time))
    force_Ex_ = force_Ex

    if order == 1:
        deriv_vx = (np.roll(dist, 1, axis=0) - np.roll(dist, -1, axis=0)) / 2 / ax_vx_e.dx
        deriv_vy = (np.roll(dist, 1, axis=1) - np.roll(dist, -1, axis=1)) / 2 / ax_vy_e.dx

    elif order == 2:
        deriv_vx = (np.roll(dist, 2, axis=0) * (-1/12) + np.roll(dist, 1, axis=0) * (2/3)
                    - np.roll(dist, -1, axis=0) * (2/3) - np.roll(dist, -2, axis=0) * (-1/12)) / ax_vx_e.dx
        deriv_vy = (np.roll(dist, 2, axis=1) * (-1/12) + np.roll(dist, 1, axis=1) * (2/3)
                    - np.roll(dist, -1, axis=1) * (2/3) - np.roll(dist, -2, axis=1) * (-1/12)) / ax_vy_e.dx

    elif order == 3:
        deriv_vx = (np.roll(dist, 3, axis=0) * (1/60) + np.roll(dist, 2, axis=0) * (-3/20)
                    + np.roll(dist, 1, axis=0) * (3/4) - np.roll(dist, -1, axis=0) * (3/4)
                    - np.roll(dist, -2, axis=0) * (-3/20) - np.roll(dist, -3, axis=0) * (1/60)) / ax_vx_e.dx
        deriv_vy = (np.roll(dist, 3, axis=1) * (1/60) + np.roll(dist, 2, axis=1) * (-3/20)
                    + np.roll(dist, 1, axis=1) * (3/4) - np.roll(dist, -1, axis=1) * (3/4)
                    - np.roll(dist, -2, axis=1) * (-3/20) - np.roll(dist, -3, axis=1) * (1/60)) / ax_vy_e.dx
    elif order == 4:
        ## this was unstable?
        deriv_vx = (np.roll(dist, 4, axis=0) * (-1/280) - np.roll(dist, -4, axis=0) * (-1/280)
                    + np.roll(dist, 3, axis=0) * (4/105) - np.roll(dist, -3, axis=0) * (4/105)
                    + np.roll(dist, 2, axis=0) * (-1/5)  - np.roll(dist, -2, axis=0) * (-1/5)
                    + np.roll(dist, 1, axis=0) * (4/5)   - np.roll(dist, -1, axis=0) * (4/5)) / ax_vx_e.dx

        deriv_vy = (np.roll(dist, 4, axis=1) * (-1/280) - np.roll(dist, -4, axis=1) * (-1/280)
                    + np.roll(dist, 3, axis=1) * (4/105) - np.roll(dist, -3, axis=1) * (4/105)
                    + np.roll(dist, 2, axis=1) * (-1/5)  - np.roll(dist, -2, axis=1) * (-1/5)
                    + np.roll(dist, 1, axis=1) * (4/5)   - np.roll(dist, -1, axis=1) * (4/5)) / ax_vy_e.dx
    else:
        raise NotImplementedError

    deriv_vx *= -1
    deriv_vy *= -1

    vx_comp = deriv_vx * (force_Ex_ + force_Bx) * charge_e / mass_e
    vy_comp = deriv_vy * force_By * charge_e / mass_e
    return vx_comp + vy_comp

def add(x, y, **kwargs):
    return x + y

def scale(x, const, **kwargs):
    return const * x

time1 = time.time()
fe_data = init_fe

print('starting time evolution')
while ts[-1] < T:

    dt_ = (dt * 0.1) if nt == 0 else dt
    print('t', ts[-1], dt_)

    ## manually update E
    force_Ex = Ex(x_vals, ts[-1] + dt_/2) * np.ones(force_Ex.shape)

    walltime = time.time()

    fe_data = helper_TE.rk4(fe_data, dt_, time=ts[-1],
                            euler_func=euler, deriv_func=deriv, add_func=add, scale_func=scale,)
    # fe_data = euler(fe_data, dt_, time=ts[-1])

    nrg_fe = compute_m2(fe_data, is_sqrt=is_sqrt)
    # energy_E_t = np.array(vm_sys.E.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_x  # electric field energy

    px_e, py_e = compute_m1(fe_data, is_sqrt=is_sqrt)
    # px_e = vm_sys.sys_fe.compute_moment([ax_vx_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x
    # py_e = vm_sys.sys_fe.compute_moment([ax_vy_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x
    # px_e = np.real(px_e)
    # py_e = np.real(py_e)

    print('px_e', px_e, py_e)

    nrg_fe_ts += [nrg_fe]
    # avg_vx_ts += [px_e]
    # avg_vy_ts += [py_e]
    # nrg_E_ts += [energy_E_t]

    ts += [ts[-1] + dt_]
    nt += 1

    # ## manually update E
    # force_Ex = Ex(x_vals, ts[-1]) * np.ones(force_Ex.shape)
    tot_mass = compute_norm(fe_data, is_sqrt=False)
    cons_error = np.abs(tot_mass - norm0)
    # plt.figure()
    # plt.plot(fe_data - init_fe)
    # plt.show()

    norm = compute_norm(fe_data, is_sqrt=is_sqrt)
    print('result norm', nt, ts[-1], dt, norm - 1, cons_error, # nrg_fe, nrg_fi,
           Lx, Lvi, Lve)

    ### compute errors
    if omega != 1.:
        z = elc_config1.charge
        vx_exact = Ex0 / (z ** 2 - omega ** 2) * (np.sin(z * ts[-1]) - z * omega * np.sin(omega * ts[-1]))
        vy_exact = Ex0 / (z ** 2 - omega ** 2) * (np.cos(z * ts[-1]) - np.cos(omega * ts[-1]))
    else:
        z = elc_config1.charge
        vx_exact = z * Ex0 / 2 * (ts[-1] * np.cos(ts[-1]) + np.sin(ts[-1]))
        vy_exact = - z**2 * Ex0 / 2 * ts[-1] * np.sin(ts[-1])

    errs_avg += [np.linalg.norm([px_e - vx_exact, py_e - vy_exact])]

    fe1_vx = helper_test.maxwellian(vx_e_vals, vth2=vth_e ** 2, density=1.0, flow=vx_exact) * n0_e
    fe1_vy = helper_test.maxwellian(vy_e_vals, vth2=vth_e ** 2, density=1.0, flow=vy_exact)
    if is_sqrt:
        fe1_vx = np.sqrt(fe1_vx)
        fe1_vy = np.sqrt(fe1_vy)

    theory_fe = np.outer(fe1_vx, fe1_vy)

    err = np.linalg.norm(theory_fe - fe_data)/ np.sqrt( ax_vx_e.npts * ax_vy_e.npts)
    errs_dist += [err]

    #### additional errors
    def maxwellian_func(xy_grid_point, x0, y0, n0, sig2x, sig2y):
        vx, vy = xy_grid_point
        gx = helper_test.maxwellian(vx, vth2=sig2x, density=1.0, flow=x0)
        gy = helper_test.maxwellian(vy, vth2=sig2y, density=1.0, flow=y0)
        out = gx * gy * n0
        return out

    def jac(xy_grid_point, x0, y0, n0, sig2x, sig2y):
        vx, vy = xy_grid_point

        gx = helper_test.maxwellian(vx, vth2=sig2x, density=1.0, flow=x0)
        gy = helper_test.maxwellian(vy, vth2=sig2y, density=1.0, flow=y0)

        ## df/d(x0) = (x-x0)/b * n0 * maxwellian
        d_x0 = (vx - x0) / sig2x * gx * gy * n0
        d_y0 = (vy - y0) / sig2y * gx * gy * n0

        ## df/d(n0) = maxwellian
        d_n0 = gx * gy

        ## df/d(sig2x) = (x-x0)^2 / 2 / b^2 * maxwellian
        d_sig2x = (vx-x0)**2 / 2 / sig2x**2 * gx * gy * n0
        d_sig2y = (vy-y0)**2 / 2 / sig2y**2 * gx * gy * n0

        return np.array([d_x0, d_y0, d_n0, d_sig2x, d_sig2y]).T

    dist_data = fe_data
    if is_sqrt:
        dist_data = np.conj(dist_data) * dist_data

    mesh_vx, mesh_vy = np.meshgrid(ax_vx_e.xpts, ax_vy_e.xpts, indexing='ij')
    grid_pts = (mesh_vx.reshape(-1), mesh_vy.reshape(-1))
    out =  maxwellian_func((mesh_vx, mesh_vy), *(px_e,py_e, 1., vth_e**2, vth_e**2))

    p_opt, p_cov = scipy.optimize.curve_fit(maxwellian_func, grid_pts, dist_data.reshape(-1),
                                            (px_e,py_e, 1., vth_e**2, vth_e**2),
                                            jac=jac
                                            )

    fit_vxs = maxwellian_func((mesh_vx, mesh_vy),
                                 px_e, py_e, 1.0, vth_e**2, vth_e**2)
    fit_opt = maxwellian_func((mesh_vx, mesh_vy), *p_opt)

    print('opt params', p_opt)
    print('theory params', vx_exact, vy_exact, 1.0, vth_e**2, vth_e**2)

    avg_vx_ts += [p_opt[0]]
    avg_vy_ts += [p_opt[1]]

    errs_shape  += [ np.linalg.norm(fit_vxs - dist_data) / np.sqrt(npts_vxe * npts_vye) ]
    errs_fitted += [ np.linalg.norm(fit_opt - dist_data) / np.sqrt(npts_vxe * npts_vye) ]
    errs_norm   += [ np.linalg.norm([p_opt[2] - 1.0]) ]
    errs_drift  += [ np.linalg.norm([p_opt[0] - vx_exact, p_opt[1] - vy_exact]) ]
    errs_sigma2 += [ np.linalg.norm([p_opt[3] - vth_e ** 2, p_opt[4] - vth_e ** 2])]

    dist_err = fit_opt - dist_data
    derr_dvx = np.abs(np.diff(dist_err, axis=0, append=0) / ax_vx_e.dx)
    derr_dvy = np.abs(np.diff(dist_err, axis=1, append=0) / ax_vy_e.dx)
    errs_deriv  += [ np.linalg.norm( derr_dvx + derr_dvy ) / np.sqrt(npts_vxe * npts_vye) ]

    if nt % save_every_nt == 0 or ts[-1] > T:

        fig1, ax1 = plt.subplots(1,1)
        time_ = ts[-1]

        fe_data_ = np.abs(fe_data)**2 if is_sqrt else fe_data
        im1 = ax1.imshow(np.real(fe_data_).T, extent=(ve_x_min, ve_x_max, ve_y_min, ve_y_max),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1)
        ax1.set_title(f'time {time_:3.2f}')
        ax1.xaxis.set_visible(False)
        ax1.set_xlabel('vx'), ax1.set_ylabel('vy')
        # plt.show()

        if omega != 1.:
            # vx_exact = Ex0 / (1 - omega ** 2) * (np.sin(ts) - omega * np.sin(omega * np.array(ts)))
            # vy_exact = Ex0 / (1 - omega ** 2) * (np.cos(ts) - np.cos(omega * np.array(ts)))
            z = elc_config1.charge
            print('charge', z)
            vx_exact = Ex0 / (z ** 2 - omega ** 2) * (np.sin(z * np.array(ts)) - z * omega * np.sin(omega * np.array(ts)))
            vy_exact = Ex0 / (z ** 2 - omega ** 2) * (np.cos(z * np.array(ts)) - z ** 2 * np.cos(omega * np.array(ts)))
        else:
            vx_exact = z * Ex0 / 2 * (np.array(ts) * np.cos(ts) + np.sin(ts))
            vy_exact = - z**2 * Ex0 / 2 * np.array(ts) * np.sin(ts)

        plt.figure()
        plt.plot(ts, avg_vx_ts, label='vx')
        plt.plot(ts, avg_vy_ts, label='vy')
        plt.plot(ts, vx_exact, 'k--')
        plt.plot(ts, vy_exact, 'k--')
        plt.xlabel('t'), plt.ylabel('v')
        plt.title('avg velocity')

        plt.figure()
        plt.plot(ts, avg_vx_ts - vx_exact, label='vx')
        plt.plot(ts, avg_vy_ts - vy_exact, label='vy')
        plt.plot(ts, errs_avg)
        plt.xlabel('t'), plt.ylabel('v')
        plt.title('avg velocity error')

        plt.figure()
        plt.semilogy(ts, errs_dist, label='dist')
        plt.semilogy(ts, errs_avg, label='avg')
        #
        plt.semilogy(ts, errs_shape, label='shape')
        plt.semilogy(ts, errs_fitted, label='fitted')
        plt.semilogy(ts, errs_norm, label='norm')
        plt.semilogy(ts, errs_drift, label='drift')
        plt.semilogy(ts, errs_sigma2, label='var')
        plt.semilogy(ts, errs_deriv, label='err deriv')

        plt.title('error')
        plt.xlabel('t'), plt.ylabel('error')
        plt.grid(color='gray',axis='y')
        plt.legend()

        plt.show()



print('wall time', time.time() - time1)

plt.figure()
plt.plot(ts, errs_avg)
plt.show()

if save_data:
    pickle.dump(np.array([ts, errs_avg, errs_dist, errs_shape,
                          errs_norm, errs_drift, errs_sigma2, errs_fitted, errs_deriv]),
                open(fdir + 'errs_' + fstr + '.pkl', 'wb'))
    print('saved', fdir + 'errs_' + fstr + '.pkl')


print('done')

