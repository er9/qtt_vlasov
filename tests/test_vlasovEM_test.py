import os, sys, pickle, time, glob

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


"""
test distribution in the presence of time dependent electric field
http://ammar-hakim.org/sj/je/je32/je32-vlasov-test-ptcl.html
"""

save_figs = False
save_data = False
load_data = False
restart = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)
is_sqrt = True
is_darwin = False

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_pde_v2/data4/VM12-test/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data4/VM12-test/'
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
elc_config1 = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config1.vth  # electron thermal speed 1/2 m b^2 n0 = sqrt(5.0e-5)
wp_e = elc_config1.wp    # plasma e' frequency
lamD = elc_config1.lamD  # Debye length
wp_i = ion_config1.wp    # plasma i+ frequency
vth_i = ion_config1.vth  # ion thermal speed


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
        return Ex0 * np.ones(x.shape) * np.cos(omega * t)
    else:
        return Ex0 * np.cos( k * x - omega * t)


## runtime parameter
q = 2
KX = 1
KV = 2
Lx, Lve, Lvi = 6, 6, 6

# TN_layout = LayoutType.SEQUENTIAL
# x_map_key = 'F'
# v_map_key = 'B'

# TN_layout = LayoutType.PARALLEL_GROUP
# x_map_key = 'B'  # 'F'
# v_map_key = 'N'  # 'M'

TN_layout = LayoutType.COMB
x_map_key = 'B'
v_map_key = 'B'

order = 1   ## derivative order
T = 100
print('T', T)
do_adapt_dt = False # True  # False  # True
do_semiimplicit = False #True
te_order = 313  # 416  # 60  # 413 # 333 # 413 # 311 # 4
dt_frac = 0.9
dt = 0.1 #  0.1 # 25  # 0.05 # 0.005 # None # 0.005 # None

use_E0_as_bg = False
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

save_every_nt = 20 # 1000 // 20 # // 5

DMAX = 8 # 16  # None # 32 # None
DMAX_F = DMAX

VP_test = VlasovTest(Lxs=(Lx,), Lves=(Lve,Lve,Lve), Lvis=(Lvi,Lvi,Lvi), KX=KX, KV=KV, q=q,
                    #
                    ion_config = ion_config1, elc_config = elc_config1,
                    coll_type=None, do_adapt_coll = True, coll_coeff_e = 0.005, coll_rate_e = 0.01,
                    DMAX = DMAX, DMAX_F = DMAX_F, compress_style = 2, compress_style_mod = 0,
                    do_adapt_dt=do_adapt_dt, dt_frac=dt_frac, dt=dt, T=T, te_order=te_order,
                    grid_layout=TN_layout, x_map_key = x_map_key, ve_map_key = v_map_key,
                    )

VP_test.initialize_axes(x_lims=(0, 2*np.pi), ve_lims=(-10*vth_e, 10*vth_e), vi_lims=(-10*vth_i, 10*vth_i),
                        x_map_key=x_map_key, ve_map_key=v_map_key, basis_xs = SpatialBasis(),)

VP_test.initialize_grid()

# CUTOFF = 1.0e-12
te_compress_i = VP_test.get_compression_config(DMAX=DMAX, cutoff=CUTOFF)
te_compress_e = VP_test.get_compression_config(DMAX=DMAX, cutoff=CUTOFF)
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


## define axis objects; position -> k-space,
ax_x, = VP_test.pos_axes
ax_vx_e, ax_vy_e, = VP_test.ve_axes
print('ax_x zero ind', ax_x.zero_ind)

print('VP xlims', VP_test.x_lims)
print('VP xlims', VP_test.x_lims[0])
x_min, x_max = VP_test.x_lims[0]
print('x range',ax_x.dx, x_min, x_max)
ve_x_min, ve_x_max = VP_test.ve_lims[0]
ve_y_min, ve_y_max = VP_test.ve_lims[1]
tot_x = x_max - x_min

coords_x  = VP_test.pos_coordsys
coords_ve = VP_test.ve_coordsys
coords_vi = VP_test.vi_coordsys

X, Y, Z = coords_x.coords
VX, VY, VZ = coords_ve.coords

npts_x = ax_x.npts
npts_vxe = ax_vx_e.npts
npts_vye = ax_vy_e.npts

# ## define MPS grids
grid_i = VP_test.grid_i
grid_e = VP_test.grid_e
grid_X = VP_test.grid_X

x_vals = ax_x.xpts
xB_vals = x_vals + ax_x.dx/2    # for staggered Yee cell

vx_e_vals = ax_vx_e.xpts
vy_e_vals = ax_vy_e.xpts


####3 define initial gridTNs

## ion density:
init_fi_gtn = grid_i.make_empty_gridTN()

## elec density
fe1_vx = helper_test.maxwellian(vx_e_vals, vth2=vth_e**2, density=1.0, flow=0.) * n0_e
fe1_vy = helper_test.maxwellian(vy_e_vals, vth2=vth_e**2, density=1.0, flow=0.)
if is_sqrt:
    fe1_vx = np.sqrt(fe1_vx)
    fe1_vy = np.sqrt(fe1_vy)

init_fe_gtn = grid_e.make_gtn_from_dicts([{ax_vx_e: fe1_vx, ax_vy_e: fe1_vy}],
                                         data_type=DataType.MPS)


## E field: 0
Ex0_gtn = grid_X.map_state_to_mps(Ex(x_vals, 0))
Ex0_gtn.is_constant = (k == 0.)
Ex_gtn = grid_X.make_empty_gridTN()
Ey_gtn = grid_X.make_empty_gridTN()
Ez_gtn = grid_X.make_empty_gridTN()

## B field:
Bz0_gtn = grid_X.get_ones_mps().scalar_multiply(Bz0)
Bz0_gtn.is_constant = True
Bx_gtn = grid_X.make_empty_gridTN()
By_gtn = grid_X.make_empty_gridTN()
Bz_gtn = grid_X.make_empty_gridTN()

deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
deriv_v = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)

init_fe_gtn.ax_deriv_configs.update({ax_x: deriv_x, ax_vx_e: deriv_v, ax_vy_e: deriv_v,})
deriv_x_EM = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
Ex_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})
Ey_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})
Ez_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})
By_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})
Bz_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})

### time step info
if dt is None:
    if te_order == 413 or te_order == 414:
        Fmax_e = np.array([Ex0, 0.0, 0.0]) * charge / mass_e
        # Fmax_e = np.array([0.0, 0.0, 0.0]) * charge / mass_e
    else:
        Fmax_e = np.array([Ex0 + ve_y_max * Bz0, ve_x_max * Bz0, 0.0]) * charge / mass_e
    dt = VP_test.get_dt(Fmax_e)
num_tsteps = int(T / dt)
print('dt', dt, num_tsteps, plasma_config.c)


### get file naming string
fstr = VP_test.get_fstr(extra_str=extra_str)
print('filename', fstr)

########################
#### run simulation ####
########################

### define v2 mpo for measurement ###
v2e_mpo = test_setup.get_v2_mpo(grid_e, VP_test.ve_axes)


#### define fields
init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, is_sqrt=is_sqrt, compress_config=te_compress_e)
init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, is_sqrt=is_sqrt, compress_config=te_compress_i)
if use_E0_as_bg:
    init_E_field = Field('E', grid_X, data={X: Ex_gtn, Y: Ey_gtn, Z: Ez_gtn},
                         is_sqrt=True, compress_config=te_compress_E)
    E0_field = Field('E0', grid_X, data={X: Ex0_gtn}, is_sqrt=True)
else:
    init_E_field = Field('E', grid_X, data={X: Ex0_gtn, Y: Ey_gtn, Z: Ez_gtn},
                         is_sqrt=True, compress_config=te_compress_E)
    E0_field = None

init_B_field = Field('B', grid_X, data={X: Bx_gtn, Y: By_gtn, Z: Bz_gtn},
                     is_sqrt=True, compress_config=te_compress_B)
B0_field = Field('B0',grid_X, data={Z: Bz0_gtn}, is_sqrt=True)
curlB0_field = Field('B0', grid_X, data={})

##### initialize tests
ts = [0]
nt = 0

nrg_fe = init_fe_gtn.meas_expec(v2e_mpo, is_sqrt=is_sqrt) * mass_e / 2 / tot_x
nrg_E  = np.array(init_E_field.norms(compIDs=coords_x.coords))**2 / 2 / tot_x
nrg_B0  = np.array(B0_field.norms(compIDs=coords_x.coords))**2 / 2 / tot_x
print('tot_x', tot_x)
print('nrg E', nrg_E)
print('nrg B0', nrg_B0)


print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
max_bond_fe = [init_fe_gtn.max_bond()]

print('current nrgs', nrg_fe, nrg_E,)

norm_fe = init_fe_gtn.norm(is_sqrt=is_sqrt)
print('field norm', norm_fe)


### BUILD PDE
if is_darwin:
    em_sys = Darwin(init_E_field, None, init_B_field,  # phi=init_phi_field, psi=init_psi_field,
                     coords_x=coords_x, matl_params=plasma_config,
                     background_B0=B0_field, background_E0=E0_field,
                     normalize=False)
    em_sys.curlB0 = curlB0_field

    vm_sys = VlasovDarwin(init_fe_field, init_fi_field, em_sys,
                           coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                           elc_params=elc_config1, ion_params=ion_config1, evolve_ion=False, evolve_EM=False,
                           normalize=True, zipup=True, te_order=te_order, compress_levels=comp_levels, )
else:
    em_sys = Maxwell(init_E_field, init_B_field, # phi=init_phi_field, psi=init_psi_field,
                     coords_x=coords_x, matl_params=plasma_config,
                     background_B0=B0_field, background_E0=E0_field,
                     normalize=False, clean=False, is_yee=False)
    ## is_yee = True gives the wrong results because of current offset.
    em_sys.curlB0 = curlB0_field

    vm_sys = VlasovMaxwell(init_fe_field, init_fi_field, em_sys,
                           coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                           elc_params=elc_config1, ion_params=ion_config1, evolve_ion=False, evolve_EM=False,
                           normalize=True, zipup=True, te_order=te_order, compress_levels=comp_levels, )
print('initialized vm_sys')

vm_sys.semiimplicit_force = do_semiimplicit

vm_sys.collision = VP_test.collision_config
print('colls', vm_sys.collision.coll_params)

vm_sys.compress_F = VP_test.compress_F
vm_sys.compress_F_opts = VP_test.compress_F_opts
print('compress E?',  vm_sys.compress_F, vm_sys.compress_F_opts)

print('E nrg', vm_sys.E.norm()**2 * 1./ 2 / tot_x)

px_e = vm_sys.sys_fe.compute_moment([ax_vx_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x
py_e = vm_sys.sys_fe.compute_moment([ax_vy_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x
print('init avg v', px_e, py_e)

# norm_val = init_fe_gtn.norm() / tot_x
# print('norm val', norm_val)
#
# vx_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vx_e], x_power=1)
# vy_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vy_e], x_power=1)
# px_e = init_fe_gtn.meas_expec(vx_mpo) / tot_x
# py_e = init_fe_gtn.meas_expec(vy_mpo) / tot_x
# print('init avg v', px_e, py_e)
# exit()

nrg_fe_ts = [nrg_fe]
nrg_E_ts  = [nrg_E]
avg_vx_ts = [px_e]
avg_vy_ts = [py_e]

if do_adapt_dt:
    if te_order == 413 or te_order == 414:
        em_term = Field('EM', grid_e, data={})
        # em_term = vm_sys.compute_force_term(is_ion=False, background_force=False)
    else:
        em_term = vm_sys.compute_force_term(is_ion=False)

    max_em = []
    for coord in [X, Y, Z]:
        try:
            em_data = em_term.get_comp_data(coord, ax_select={ax_vx_e: 0, ax_vy_e: 0, })
            em_max = np.max(np.abs(em_data)) if em_data is not None else 0.0
        except KeyError:
            em_max = 0.
        max_em += [em_max]

    print('max em', max_em)
    Fe_maxs = np.array(max_em)
    dt = VP_test.get_dt(Fe_maxs)
    print('adapt dt', dt, plasma_config.c, elc_config1.vth)


fig1, ax1 = plt.subplots(2, 1)
time_ = nt * dt

fe_data = init_fe_gtn.get_data(ax_select={ax_x: 0, })
fe_data = ax_vy_e.basis.get_realspace_1D(fe_data, 1)
fe_data = ax_vx_e.basis.get_realspace_1D(fe_data, 0)
im1 = ax1[0].imshow(np.real(fe_data).T, extent=(x_min, x_max, ve_x_min, ve_x_max),
                    aspect='auto', origin='lower')

fe_data = init_fe_gtn.get_data(ax_select={ax_vy_e: ax_vy_e.npts // 2})
fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
fe_data = ax_vx_e.basis.get_realspace_1D(fe_data, 1)
im2 = ax1[1].imshow(np.real(fe_data).T,
                    extent=(x_min, x_max, ve_x_min, ve_x_max),
                    aspect='auto', origin='lower')

fig1.colorbar(im1, ax=ax1[0])
fig1.colorbar(im2, ax=ax1[1])
ax1[0].set_title(f'time {time_:3.2f}')
ax1[0].set_xlabel('vx'), ax1[0].set_ylabel('vy')
ax1[1].set_xlabel('x'), ax1[1].set_ylabel('vx')
# if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')
# plt.close(fig1)

plt.show()

time1 = time.time()

print('starting time evolution')
while ts[-1] < T:

    walltime = time.time()
    vm_sys = vm_sys.next_time_step(dt, compress_level=1, err_tol=1.0e-7, verbose_plot=False,
                                   is_first_time_step=(nt==0))
    print('wall time', time.time() - walltime)

    fe_field = vm_sys.fe
    max_bond_fe += [fe_field.component.max_bond()]
    nrg_fe = fe_field.component.meas_expec(v2e_mpo, is_sqrt=is_sqrt) * mass_e / 2 / tot_x
    energy_E_t = np.array(vm_sys.E.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_x  # electric field energy

    px_e = vm_sys.sys_fe.compute_moment([ax_vx_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x
    py_e = vm_sys.sys_fe.compute_moment([ax_vy_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x

    # px_e = vm_sys.sys_fe.compute_moment([ax_vx_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x
    # py_e = vm_sys.sys_fe.compute_moment([ax_vy_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x

    print('px_e', px_e, py_e)

    nrg_fe_ts += [nrg_fe]
    avg_vx_ts += [px_e]
    avg_vy_ts += [py_e]
    nrg_E_ts += [energy_E_t]

    ts += [ts[-1] + dt]
    nt += 1

    ## manually update E
    new_Ex = grid_X.map_state_to_mps(Ex(x_vals, ts[-1]))
    new_Ex.is_constant = (k == 0.)
    if is_darwin:
        if use_E0_as_bg:
            vm_sys.sys_ED.background_E0[X] = new_Ex
        else:
            vm_sys.ET[X] = new_Ex
    else:
        if use_E0_as_bg:
            vm_sys.EM_sys.background_E0[X] = new_Ex
            vm_sys.sys_fe.saved_SL_mpos = {ax: {} for ax in [ax_vx_e, ax_vy_e]}
            vm_sys.sys_fi.saved_SL_mpos = {ax: {} for ax in [ax_vx_e, ax_vy_e]}
        else:
            vm_sys.E[X] = new_Ex

    if do_adapt_dt:
        if te_order == 413 or te_order == 414:
            em_term = Field('EM', grid_e, data={})
            # em_term = vm_sys.compute_force_term(is_ion=False, background_force=False)
        else:
            em_term = vm_sys.compute_force_term(is_ion=False)
        max_em = []
        for coord in [X, Y, Z]:
            try:
                em_data = em_term.get_comp_data(coord, ax_select={ax_vx_e: 0, ax_vy_e: 0,})
                em_max = np.max(np.abs(em_data)) if em_data is not None else 0.0
            except KeyError:
                em_max = 0.
            max_em += [em_max]

        Fe_maxs = np.array(max_em)
        dt = VP_test.get_dt(Fe_maxs)
        # print('restarted dt', dt)

    norm = vm_sys.fe.norm() - vm_sys.fi.norm()
    if is_sqrt:
        norm = norm ** 2
    print('result norm', nt, ts[-1], dt, norm, # nrg_fe, nrg_fi,
          energy_E_t, fe_field.max_bond(), Lx, Lvi, Lve)

    if nt % save_every_nt == 0:

        fig1, ax1 = plt.subplots(2, 1)
        time_ = ts[-1]

        fe_data = fe_field.get_comp_data(ax_select={ax_x: 0,})
        fe_data = ax_vy_e.basis.get_realspace_1D(fe_data, 1)
        fe_data = ax_vx_e.basis.get_realspace_1D(fe_data, 0)
        if is_sqrt:
            fe_data = np.abs(fe_data)**2
        im1 = ax1[0].imshow(np.real(fe_data).T, extent=(x_min, x_max, ve_x_min, ve_x_max),
                            aspect='auto', origin='lower')

        fe_data = fe_field.get_comp_data(ax_select={ax_vy_e: npts_vye//2,})
        fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
        fe_data = ax_vx_e.basis.get_realspace_1D(fe_data, 1)
        if is_sqrt:
            fe_data = np.abs(fe_data)**2
        im2 = ax1[1].imshow(np.real(fe_data).T, extent=(x_min, x_max, ve_x_min, ve_x_max),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].xaxis.set_visible(False)
        ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
        ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
        plt.show()

        fig7, ax7 = plt.subplots()
        ax7.semilogy(ts, np.abs(nrg_fe_ts), label='elec')
        plt.xlabel('t'), plt.ylabel('fe nrg')
        plt.title('fe nrg')
        plt.show()

        if omega != 1.:
            # vx_exact = Ex0 / (1 - omega ** 2) * (np.sin(ts) - omega * np.sin(omega * np.array(ts)))
            # vy_exact = Ex0 / (1 - omega ** 2) * (np.cos(ts) - np.cos(omega * np.array(ts)))
            z = elc_config1.charge
            print('charge', z)
            vx_exact = Ex0 / (z ** 2 - omega ** 2) * (np.sin(z * np.array(ts)) - z * omega * np.sin(omega * np.array(ts)))
            vy_exact = Ex0 / (z ** 2 - omega ** 2) * (np.cos(z * np.array(ts)) - z ** 2 * np.cos(omega * np.array(ts)))
        else:
            vx_exact = Ex0 / 2 * (np.array(ts) * np.cos(ts) + np.sin(ts))
            vy_exact = - Ex0 / 2 * np.array(ts) * np.sin(ts)

        plt.figure()
        plt.plot(ts, avg_vx_ts, label='vx')
        plt.plot(ts, avg_vy_ts, label='vy')
        plt.plot(ts, vx_exact, 'k--')
        plt.plot(ts, vy_exact, 'k--')
        plt.xlabel('t'), plt.ylabel('v')
        plt.title('avg velocity')
        plt.show()



        #### compute moments
        density_e = vm_sys.sys_fe.compute_moment([])

        plt.figure()
        plt.plot(density_e.get_comp_data() * mass_e / tot_x, label='density e')
        plt.title('density i')

        px_e = vm_sys.sys_fe.compute_moment([ax_vx_e], powers=[1]).get_comp_data()
        py_e = vm_sys.sys_fe.compute_moment([ax_vy_e], powers=[1]).get_comp_data()
        p_e = np.sqrt( px_e ** 2 + py_e ** 2 ) * mass_e

        plt.figure()
        plt.plot(x_vals, p_e, label='p elec')
        plt.title('<v> elec')

        nrgx_e = vm_sys.sys_fe.compute_moment([ax_vx_e], powers=[2]).get_comp_data()
        nrgy_e = vm_sys.sys_fe.compute_moment([ax_vy_e], powers=[2]).get_comp_data()
        nrg_e = (nrgx_e + nrgy_e) * 0.5 * mass_e

        plt.figure()
        plt.plot(x_vals, nrg_e, label='nrg elec')
        plt.title('<v^2> elec')

        # print('vm sys fe', vm_sys.sys_fe.f.component)
        v_density = vm_sys.sys_fe.f.integrate(axes=[ax_x])
        # print('v density', v_density.component)
        vx_data = v_density.get_comp_data(ax_select={ax_vy_e: ax_vy_e.npts // 2}) / np.pi / 2
        plt.figure()
        plt.plot(vx_data)
        plt.title('vx, vy=0; integ x')

        plt.show()



print('wall time', time.time() - time1)


##############

fig7, ax7 = plt.subplots()
ax7.semilogy(ts, np.abs(nrg_fe_ts), label='elec')

ax7.legend()
ax7.set_xlabel('t')
ax7.set_ylabel('energy')
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

