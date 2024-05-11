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
from basis.basis_k import FourierBasis
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
is_sqrt = False
is_darwin = False

if restart or save_data or save_figs:
    # sdir = '/pool001/erikaye/tns_pde_v2/data15-2/VM12-test/'
    # if not os.path.exists(sdir):
    #     os.makedirs(sdir)

    fdir = 'data22/VM12-test-k-full-F2/'
    # fdir = 'data22/VM12-test-k-full-Ft/'
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
dt = 0.00625 / 2  #  0.1 for L=7
is_yee = False

vmin, vmax = -12 * vth_e, 12 * vth_e

use_E0_as_bg = False
use_B0_as_bg = True

save_every_nt = 5000   # 50 # 25000 # 1000 // 20 # // 5

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
                    basis_ves=FourierBasis(), basis_vis=FourierBasis(),
                    )

dve = (vmax - vmin) / (2**Lve)
VP_test.initialize_axes(x_lims=(0, 2*np.pi), ve_lims=(-np.pi/dve, np.pi/dve), vi_lims=(-np.pi/dve, np.pi/dve),
                        x_map_key=x_map_key, ve_map_key=v_map_key, basis_xs = SpatialBasis(),
                        basis_ves=FourierBasis(), basis_vis=FourierBasis(),
                        )

# VP_test.initialize_axes(x_lims=(0, 2*np.pi), ve_lims=(-12*vth_e, 12*vth_e), vi_lims=(-12*vth_i, 12*vth_i),
#                         x_map_key=x_map_key, ve_map_key=v_map_key, basis_xs = SpatialBasis(),)

VP_test.initialize_grid()


#############################
## start building PDE system
#############################

## define axis objects; position -> k-space,
# ax_x, = VP_test.pos_axes
# ax_vx_e, ax_vy_e, = VP_test.ve_axes
ax_kvx_e: 'Axis' = VP_test.ve_axes[0]
ax_kvy_e: 'Axis' = VP_test.ve_axes[1]
print('ax map', ax_kvx_e.map, ax_kvy_e.map)
print('dvx', ax_kvx_e.dx, 'dvy', ax_kvy_e.dx)
print('cfl dt', ax_kvx_e.dx / (Bz0 * np.max(ax_kvx_e.xpts)), dt)
# exit()
# print('ax_x zero ind', ax_x.zero_ind)

ve_x_min, ve_x_max = vmin, vmax
ve_y_min, ve_y_max = vmin, vmax
vx_e_vals = np.linspace(vmin, vmax, 2**Lve, endpoint=False)
vy_e_vals = np.linspace(vmin, vmax, 2**Lve, endpoint=False)

ve_kx_min, ve_kx_max = VP_test.ve_lims[0]
ve_ky_min, ve_ky_max = VP_test.ve_lims[1]
tot_x = 1  # x_max - x_min

npts_x = 1 # ax_x.npts
npts_vxe = ax_kvx_e.npts
npts_vye = ax_kvy_e.npts

# ## define MPS grids
grid_i = VP_test.grid_i
grid_e = VP_test.grid_e
grid_X = VP_test.grid_X

x_vals = 0  # ax_x.xpts
xB_vals = x_vals

kvx_e_vals = ax_kvx_e.xpts
kvy_e_vals = ax_kvy_e.xpts
vx_mesh, vy_mesh = np.meshgrid(vx_e_vals, vy_e_vals, indexing='ij')
kvx_mesh, kvy_mesh = np.meshgrid(kvx_e_vals, kvy_e_vals, indexing='ij')
dV = ax_kvx_e.dx * ax_kvy_e.dx
dV_real = (vx_e_vals[1] - vx_e_vals[0]) * (vy_e_vals[1] - vy_e_vals[0])



####3 define initial gridTNs

## ion density:
# init_fi_gtn = grid_i.make_empty_gridTN()

## elec density
fe1_kvx = helper_test.maxwellian_k(kvx_e_vals, vth2=vth_e ** 2, density=1.0, flow=0., is_sqrt=is_sqrt) * n0_e
fe1_kvy = helper_test.maxwellian_k(kvy_e_vals, vth2=vth_e ** 2, density=1.0, flow=0., is_sqrt=is_sqrt)
init_fe = np.outer(fe1_kvx, fe1_kvy)

# init_fe_gtn = grid_e.make_gtn_from_dicts([{ax_vx_e: fe1_vx, ax_vy_e: fe1_vy}],
#                                          data_type=DataType.MPS)


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

nrg_fe = np.nan  # np.sum(init_fe * v2) * ax_kvx_e.dx * ax_kvy_e.dx
print('tot_x', tot_x)

def get_kspace(dist):
    return np.fft.fftshift(np.fft.fftn(dist, norm='forward'))

def get_realspace(dist):
    return np.fft.ifftn(np.fft.ifftshift(dist), norm='forward') # * npts_vye * npts_vxe

def compute_norm(dist, is_sqrt=False):
    if is_sqrt:
        return np.linalg.norm(dist) ** 2 * (vmax - vmin)**2
    else:
        return np.linalg.norm(dist[npts_vxe//2, npts_vye//2]) * (vmax - vmin)**2

def compute_m1(dist, is_sqrt=False):
    dist = get_realspace(dist)
    dist = dist.conj() * dist if is_sqrt else dist.copy()
    dist = np.real(dist)
    dist[ 0, :] /= 2
    dist[-1, :] /= 2
    dist[:,  0] /= 2
    dist[:, -1] /= 2

    # plt.figure()
    # plt.plot(dist * vx_mesh)
    # # plt.colorbar()
    # plt.show()

    vx_dist = vx_mesh * dist
    vy_dist = vy_mesh * dist

    return np.sum(vx_dist) * dV_real, np.sum(vy_dist) * dV_real

def compute_m2(dist, is_sqrt=False):
    dist = get_realspace(dist)
    dist = dist.conj() * dist if is_sqrt else dist.copy()
    dist = np.real(dist)
    dist[ 0, :] /= 2
    dist[-1, :] /= 2
    dist[:,  0] /= 2
    dist[:, -1] /= 2
    return np.sum(v2 * dist) * dV_real



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

fe_real = get_realspace(init_fe)
im1 = ax1.imshow(np.real(fe_real).T, extent=(ve_x_min, ve_x_max, ve_y_min, ve_y_max),
                    aspect='auto', origin='lower')

fig1.colorbar(im1, ax=ax1)
ax1.set_title(f'time {time_:3.2f}')
ax1.set_xlabel('vx'), ax1.set_ylabel('vy')

plt.figure()
plt.imshow(np.imag(fe_real))
plt.colorbar()

plt.show()

# deriv_vx = grid_e.get_firstderivative_mpo(ax_vx_e).get_data()
# deriv_vy = grid_e.get_firstderivative_mpo(ax_vy_e).get_data()

## EM forces
# v_kx = np.fftshift(np.fft.fft(vx_e_vals))
# v_ky = np.fftshift(np.fft.fft(vy_e_vals))
force_Bx =  Bz0 * vy_mesh   ## in real space
force_By = -Bz0 * vx_mesh
force_Ex =  Ex(x_vals, 0) * np.ones((ax_kvx_e.npts, ax_kvy_e.npts))

em_term = force_Ex + force_Bx
plt.figure()
plt.imshow(em_term)
plt.colorbar()
plt.show()

def euler(dist, dt, deriv0=None, **kwargs):
    if deriv0 is None:
        deriv0 = deriv(dist)
    return dist + dt * deriv0

def deriv_fft(dist, time=None, **kwargs):
    time = ts[-1] if time is None else time
    force_Ex_ = Ex(x_vals, time)
    # print('time', time, Ex(0, time))
    # force_Ex_ = force_Ex

    term_x = (dist.T * kvx_e_vals * -1.j).T
    term_x_ifft = get_realspace(term_x)
    term_x_ifft = term_x_ifft * (force_Ex_ + force_Bx)
    term_x = get_kspace(term_x_ifft)

    term_y = (dist * kvy_e_vals * -1.j)
    term_y_ifft = get_realspace(term_y)
    term_y_ifft = term_y_ifft * force_By
    term_y = get_kspace(term_y_ifft)

    return term_y  # (term_x + term_y) * charge_e / mass_e


def deriv(dist, time=None, **kwargs):
    time = ts[-1] if time is None else time
    # force_Ex_ = Ex(x_vals, time)
    # print('time', time, Ex(0, time))
    force_Ex_ = force_Ex

    # check = deriv_fft(dist, time=time)

    force_x = force_Ex_ + Bz0 * vy_e_vals
    force_y = -Bz0 * vx_e_vals

    # force_x_k = get_kspace(force_x)
    # force_y_k = get_kspace(force_y)

    # fx_mpo = ax_kvx_e.get_fmultiply_mpo(force_x_k)
    # fy_mpo = ax_kvy_e.get_fmultiply_mpo(force_y_k)
    fx_mpo = ax_kvx_e.get_fmultiply_mpo(force_x)
    fy_mpo = ax_kvy_e.get_fmultiply_mpo(force_y)
    fx_mat = ax_kvx_e.map_mpo_to_operator(fx_mpo)
    fy_mat = ax_kvy_e.map_mpo_to_operator(fy_mpo)

    term_x = (dist.T * kvx_e_vals * -1.j).T  ## d/dvx
    # term_x_ifft = get_realspace(term_x)
    # term_x_ifft = term_x_ifft * (force_Ex_ + force_Bx)
    # check_x = get_kspace(term_x_ifft)
    term_x = term_x @ (fx_mat.conj())  # @ term_x

    # plt.figure()
    # plt.imshow(np.real(check_x - term_x))
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(np.imag(check_x - term_x))
    # plt.colorbar()
    # plt.show()

    # dist_vec = dist[npts_vxe//2, :]
    # plt.figure()
    # plt.imshow(np.real(fx_mat))
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(np.imag(fx_mat))
    # plt.colorbar()
    # plt.show()

    # conv_out = np.convolve(force_x_k, dist_vec, mode='full')
    # plt.figure()
    # plt.plot(np.real(conv_out))
    # plt.plot(np.imag(conv_out))
    # plt.show()

    # dist_vec_r = get_realspace(dist_vec)
    # dist_vec_r = dist_vec_r * force_x
    # check_vec = get_kspace(dist_vec_r)
    #
    # plt.figure()
    # plt.plot(np.imag(fx_mat @ dist_vec))
    # plt.plot(np.imag(check_vec), 'k--')
    #
    # plt.figure()
    # plt.plot(np.real(fx_mat @ dist_vec))
    # plt.plot(np.real(check_vec), 'k--')
    # plt.show()
    # exit()

    term_y = (dist * kvy_e_vals * -1.j)     ## d/dvy
    # term_y_ifft = get_realspace(term_y)
    # term_y_ifft = term_y_ifft * force_By
    # check_y = get_kspace(term_y_ifft)
    term_y = fy_mat @ term_y

    ###
    # plt.figure()
    # plt.imshow(np.real(check_y - term_y))
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(np.imag(check_y - term_y))
    # plt.colorbar()
    # plt.show()
    # exit()

    return (term_x + term_y) * charge_e / mass_e


def add(x, y, **kwargs):
    return x + y

def scale(x, const, **kwargs):
    return const * x

time1 = time.time()
fe_data = init_fe

print('starting time evolution')
while ts[-1] < T:

    dt_ = (dt * 0.1) if nt == 0 else dt

    ## manually update E
    force_Ex = Ex(x_vals, ts[-1] + dt_/2)

    walltime = time.time()

    fe_data = helper_TE.rk4(fe_data, dt_, time=ts[-1],
                            euler_func=euler, deriv_func=deriv, add_func=add, scale_func=scale,)

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

    norm = compute_norm(fe_data, is_sqrt=is_sqrt)
    print('result norm', nt, ts[-1], dt, norm - 1, # nrg_fe, nrg_fi,
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

    fe1_kvx = helper_test.maxwellian_k(kvx_e_vals, vth2=vth_e ** 2, density=1.0, flow=vx_exact, is_sqrt=is_sqrt) * n0_e
    fe1_kvy = helper_test.maxwellian_k(kvy_e_vals, vth2=vth_e ** 2, density=1.0, flow=vy_exact, is_sqrt=is_sqrt)
    theory_fe = np.outer(fe1_kvx, fe1_kvy)

    err = np.linalg.norm(theory_fe - fe_data)/ np.sqrt(ax_kvx_e.npts * ax_kvy_e.npts)
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

    dist_data = get_realspace(fe_data)
    if is_sqrt:
        dist_data = np.conj(dist_data) * dist_data

    # mesh_vx, mesh_vy = np.meshgrid(vx_e_vals, vy_e_vals, indexing='ij')
    mesh_vx, mesh_vy = vx_mesh, vy_mesh
    grid_pts = (mesh_vx.reshape(-1), mesh_vy.reshape(-1))
    out =  maxwellian_func((mesh_vx, mesh_vy), *(px_e,py_e, 1., vth_e**2, vth_e**2))

    print('dtypes', grid_pts[0].dtype, dist_data.dtype)
    print('pxe ', px_e, py_e)
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
    derr_dvx = np.abs(np.diff(dist_err, axis=0, append=0) / (vx_e_vals[1]- vx_e_vals[0]))
    derr_dvy = np.abs(np.diff(dist_err, axis=1, append=0) / (vy_e_vals[1]- vy_e_vals[0]))
    errs_deriv  += [ np.linalg.norm( derr_dvx + derr_dvy ) / np.sqrt(npts_vxe * npts_vye) ]

    if nt % save_every_nt == 0 or ts[-1] > T:

        fig1, ax1 = plt.subplots(1,1)
        time_ = ts[-1]

        im1 = ax1.imshow(np.real(dist_data), extent=(ve_x_min, ve_x_max, ve_y_min, ve_y_max),
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

        if save_data:
            pickle.dump(np.array([ts, errs_avg, errs_dist, errs_shape,
                                  errs_norm, errs_drift, errs_sigma2, errs_fitted, errs_deriv]),
                        open(fdir + 'errs_' + fstr + '.pkl', 'wb'))
            print('saved', fdir + 'errs_' + fstr + '.pkl')



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

