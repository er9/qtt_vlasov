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
from vlasov_tests import RiemannEM

"""
EM Riemann problem
"""

save_figs = False
save_data = True  # False
load_data = True  # False
restart   = True  # False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_pde_v2/data8/VM13-riemann/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data8/VM13-riemann/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)

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

""" plasma 1
    electron thermal speed 1/2 m <|v|^2> = 1/2 m b^2 n0 = sqrt(5.0e-5)
        -> b^2 = 1.0e-4 -> vth = b = 1.0e-2
    ion thermal speed 1/2 mi b^2 n0 = sqrt(5.0e-5)
        -> b^2 = 1.0e-4 / mi -> vth = b = 1.0e-2 / sqrt(mi)
    recall vth2 = kb T / m
    vth / c = 1 / 8 --> c = 8.0e-2 
"""
n0_e1 = n0_i1 = 1.0
T_e1 = T_i1 = 1.0e-4
ion_config1 = IonConfiguration(n0=n0_i1, mass=mass_i, T=T_i1, units_config=plasma_config)
elc_config1 = ElcConfiguration(n0=n0_e1, mass=mass_e, T=T_e1, units_config=plasma_config)
vth_e1 = elc_config1.vth  # electron thermal speed 1/2 m b^2 n0 = sqrt(5.0e-5)
wp_e1 = elc_config1.wp    # plasma e' frequency
lamD1 = elc_config1.lamD  # Debye length
wp_i1 = ion_config1.wp    # plasma i+ frequency
vth_i1 = ion_config1.vth  # ion thermal speed

""" plasma 2
    electron thermal speed 1/2 m b^2 n0 = sqrt(5.0e-6)
        -> b^2 = 1.0e-5 / 0.125 = 8.0e-5 -> vth = b = sqrt(0.8) * 1e-2
    ion thermal speed 1/2 mi b^2 n0 = sqrt(5.0e-6)
        -> b^2 = 1.0e-5 / 0.125 / mi = 8.0e-5 / mi -> vth = b = sqrt(0.8) * 1e-2 / sqrt(mi)     
    recall vth2 = q T / m
"""
n0_e2 = n0_i2 = 0.125
T_e2 = T_i2 = 8.0e-5
ion_config2 = IonConfiguration(n0=n0_i2, mass=mass_i, T=T_i2, units_config=plasma_config)
elc_config2 = ElcConfiguration(n0=n0_e2, mass=mass_e, T=T_e2, units_config=plasma_config)
vth_e2 = elc_config2.vth  # electron thermal speed: 1/2 m b^2 n0 = sqrt(5.0e-6)
wp_e2 = elc_config2.wp    # plasma e' frequency
lamD2 = elc_config2.lamD  # Debye length
wp_i2 = ion_config2.wp    # plasma i+ frequency
vth_i2 = ion_config2.vth  # ion thermal speed


#######################
# standard parameters #
#######################

### initial B fields
Bx0 = 0.75 * 1.0e-2
Bz10, Bz20 = 1.0e-2, -1.0e-2
B0_mag = np.sqrt( Bx0**2 + Bz10**2 )

## runtime parameter
q = 2
KX = 1
KV = 3
## gkeyll:  64 x 32 x 32 x 32
Lx, Lve, Lvi = 7, 6, 6
# Lx, Lve, Lvi = 6, 5, 5



TN_layout = LayoutType.SEQUENTIAL
x_map_key = 'F'
v_map_key = 'B'

# TN_layout = LayoutType.COMB
# x_map_key = 'B'
# v_map_key = 'B'

order = 1   ## derivative order
T = 0.125 / ion_config1.cyclotron_frequency(B0_mag)
print('tot time', T)
do_adapt_dt = True  # False  # True
dt_frac = 0.9
te_order = 313 # 4
dt = None
dt_max = 0.25 * 2 * np.pi / elc_config1.plasma_frequency

te_order_EM = 22
is_yee = False

save_every_nt = 250

DMAX = 64
DMAX_F = DMAX
comp_style = 3

print('wci', ion_config1.cyclotron_frequency(B0_mag))
print('vth e', vth_e1, vth_e2)
print('vth i', vth_i1, vth_i2)
print('c', plasma_config.c)

VP_test = RiemannEM(Lxs=(Lx,), Lves=(Lve,Lve,Lve), Lvis=(Lvi,Lvi,Lvi), KX=KX, KV=KV, q=q,
                    x_map_key = x_map_key, ve_map_key = v_map_key,
                    basis_xs = SpatialBasis(),
                    #
                    ion_config = ion_config1, elc_config = elc_config1,
                    coll_type=None, do_adapt_coll = True, coll_coeff_e = 0.005, coll_rate_e = 0.01,
                    DMAX = DMAX, DMAX_F = DMAX_F, compress_style = comp_style, compress_style_mod = 0,
                    do_adapt_dt=do_adapt_dt, dt_frac=dt_frac, dt=dt, T=T, te_order=te_order, te_order_EM=te_order_EM,
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


## define axis objects; position -> k-space,
ax_x, = VP_test.pos_axes
ax_vx_i, ax_vy_i, ax_vz_i = VP_test.vi_axes
ax_vx_e, ax_vy_e, ax_vz_e = VP_test.ve_axes
print('ax_x zero ind', ax_x.zero_ind)

x_min, x_max = VP_test.x_lims[0]
print('x range',ax_x.dx, x_min, x_max, ion_config1.skin_depth)
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

####3 define initial gridTNs
H = lambda x:  np.heaviside(x, 0.0) # * dx

## ion density:
fi1_vx = helper_test.maxwellian(vx_i_vals, vth2=vth_i1**2, density=1.0, flow=0.0)
fi1_vy = helper_test.maxwellian(vy_i_vals, vth2=vth_i1**2, density=1.0, flow=0.0)
fi1_vz = helper_test.maxwellian(vz_i_vals, vth2=vth_i1**2, density=1.0, flow=0.0)

fi2_vx = helper_test.maxwellian(vx_i_vals, vth2=vth_i2**2, density=1.0, flow=0.0)
fi2_vy = helper_test.maxwellian(vy_i_vals, vth2=vth_i2**2, density=1.0, flow=0.0)
fi2_vz = helper_test.maxwellian(vz_i_vals, vth2=vth_i2**2, density=1.0, flow=0.0)

init_fi_gtn = grid_i.make_gtn_from_dicts([{ax_x: (1 - H(x_vals)) * n0_i1, ax_vx_i: fi1_vx, ax_vy_i: fi1_vy, ax_vz_i: fi1_vz},
                                          {ax_x: H(x_vals) * n0_i2, ax_vx_i: fi2_vx, ax_vy_i: fi2_vy, ax_vz_i: fi2_vz}],
                                         data_type=DataType.MPS)

## elec density
fe1_vx = helper_test.maxwellian(vx_e_vals, vth2=vth_e1**2, density=1.0, flow=0.0)
fe1_vy = helper_test.maxwellian(vy_e_vals, vth2=vth_e1**2, density=1.0, flow=0.0)
fe1_vz = helper_test.maxwellian(vz_e_vals, vth2=vth_e1**2, density=1.0, flow=0.0)

fe2_vx = helper_test.maxwellian(vx_e_vals, vth2=vth_e2**2, density=1.0, flow=0.0)
fe2_vy = helper_test.maxwellian(vy_e_vals, vth2=vth_e2**2, density=1.0, flow=0.0)
fe2_vz = helper_test.maxwellian(vz_e_vals, vth2=vth_e2**2, density=1.0, flow=0.0)

init_fe_gtn = grid_e.make_gtn_from_dicts([{ax_x: (1 - H(x_vals)) * n0_e1, ax_vx_e: fe1_vx, ax_vy_e: fe1_vy, ax_vz_e: fe1_vz},
                                          {ax_x: H(x_vals) * n0_e2, ax_vx_e: fe2_vx, ax_vy_e: fe2_vy, ax_vz_e: fe2_vz}],
                                         data_type=DataType.MPS)


## E field: 0
Ex_gtn = grid_X.make_empty_gridTN()
Ey_gtn = grid_X.make_empty_gridTN()
Ez_gtn = grid_X.make_empty_gridTN()

## B field:
Bx_gtn = grid_X.get_ones_mps().scalar_multiply(Bx0)
By_gtn = grid_X.make_empty_gridTN()
Bz_gtn = grid_X.make_gtn_from_dicts([{ax_x: (1 - H(x_vals)) * Bz10 + H(x_vals) * Bz20}], data_type=DataType.MPS)

init_phi_gtn = grid_X.make_empty_gridTN()
init_psi_gtn = grid_X.make_empty_gridTN()

deriv_x = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)
deriv_v = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)

init_fe_gtn.ax_deriv_configs.update({ax_x: deriv_x, ax_vx_e: deriv_v, ax_vy_e: deriv_v, ax_vz_e: deriv_v})
init_fi_gtn.ax_deriv_configs.update({ax_x: deriv_x, ax_vx_i: deriv_v, ax_vy_i: deriv_v, ax_vz_i: deriv_v})
deriv_x_EM = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)
Ex_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})
Ey_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})
Ez_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})
By_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})
Bz_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM})

# ### time step info
# if dt is None:
#     Fmax_e = np.array([ve_y_max*Bz10, ve_z_max*Bx0 + abs(ve_x_max*Bz10), Bx0*ve_y_max]) * charge / mass_e
#     Fmax_i = np.array([ve_y_max*Bz10, vi_z_max*Bx0 + abs(vi_x_max*Bz10), Bx0*vi_y_max]) * charge / mass_i
#     dt = VP_test.get_dt(Fmax_e, Fmax_i)
#     # dt = VP_test.get_dt([(B0 + A / dk) * charge / mass_e])
# num_tsteps = int(T / dt)
# print('dt', dt, num_tsteps, plasma_config.c)
# exit()

### get file naming string
fstr = VP_test.get_fstr()
print('filename', fstr)

########################
#### run simulation ####
########################

### define v2 mpo for measurement ###
v2e_mpo = test_setup.get_v2_mpo(grid_e, VP_test.ve_axes)
v2i_mpo = test_setup.get_v2_mpo(grid_i, VP_test.vi_axes)


#### define fields
init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, compress_config=te_compress_e)
init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, compress_config=te_compress_i)
init_E_field = Field('E', grid_X, data={X: Ex_gtn, Y: Ey_gtn, Z: Ez_gtn},
                     is_sqrt=True, compress_config=te_compress_E)
init_B_field = Field('B', grid_X, data={X: Bx_gtn, Y: By_gtn, Z: Bz_gtn},
                     is_sqrt=True, compress_config=te_compress_B)
init_phi_field = ScalarField('phi', grid_X, data=init_phi_gtn, is_sqrt=True, compress_config=te_compress_E)
init_psi_field = ScalarField('psi', grid_X, data=init_psi_gtn, is_sqrt=True, compress_config=te_compress_B)


### load data if restarting ###
try:
    out = test_setup.load_data_EM(VP_test,
                            init_fe_field, init_fi_field, init_E_field, init_B_field, init_phi_field, init_psi_field,
                            fdir=fdir, sdir=sdir, save_every_nt=save_every_nt,
                            load_data=load_data, restart=restart, restart_from_T=restart_from_T)

    (init_fe_field, init_fi_field, init_E_field, init_B_field, init_phi_field, init_psi_field,
     ts, nrg_fe_ts, nrg_fi_ts, nrg_E_ts, nrg_B_ts,
     max_bond_fe, max_bond_fi, max_bond_E, max_bond_B, max_bond_phi, max_bond_psi) = out

    # print('init fe', init_fe_field.component)
    # print('init fi', init_fi_field.component)
    # print('init E', init_E_field.components)
    # print('init B', init_B_field.components)

    nt = len(ts) - 1
    restarted = True
    print('restarted')

except(NameError, ValueError, FileNotFoundError):

    print('failed restart')

    ts = [0]
    nt = 0
    restarted = False

    nrg_fe = init_fe_gtn.meas_expec(v2e_mpo) * mass_e / 2 / tot_x
    nrg_fi = init_fi_gtn.meas_expec(v2i_mpo) * mass_i / 2 / tot_x
    # nrg_E  = init_E_field.norm() ** 2 * 1. / 2 / tot_x
    # nrg_B  = init_B_field.norm() ** 2 * 1. / 2 / tot_x
    nrg_E  = np.array(init_E_field.norms(compIDs=coords_x.coords))**2 / 2 / tot_x
    nrg_B  = np.array(init_B_field.norms(compIDs=coords_x.coords))**2 / 2 / tot_x
    print('tot_x', tot_x)
    print('nrg E', nrg_E, 'nrg B', nrg_B)

    ## maybe should initialize system, run EM_sys for half time step to start

    nrg_fe_ts = [nrg_fe]
    nrg_fi_ts = [nrg_fi]
    nrg_E_ts  = [nrg_E]
    nrg_B_ts  = [nrg_B]
    nrg_By_ts = [nrg_B]

    print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
    max_bond_fe = [init_fe_gtn.max_bond()]
    max_bond_fi = [init_fi_gtn.max_bond()]
    # max_bond_E = [init_E_field.max_bonds() if init_E_field is not None else np.nan]
    # max_bond_B = [init_B_field.max_bonds() if init_B_field is not None else np.nan]
    max_bond_E = [init_E_field.max_bonds(compIDs=coords_x.coords) if init_E_field is not None else np.nan]
    max_bond_B = [init_B_field.max_bonds(compIDs=coords_x.coords) if init_B_field is not None else np.nan]
    # max_bond_phi = [init_phi_field.max_bond() if init_phi_field is not None else np.nan]
    # max_bond_psi = [init_psi_field.max_bond() if init_psi_field is not None else np.nan]

print('current nrgs', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1], nrg_B_ts[-1])

norm_fe = init_fe_gtn.norm()
norm_fi = init_fi_gtn.norm()
print('field norm', norm_fe, norm_fi)


### BUILD PDE
em_sys = Maxwell(init_E_field, init_B_field, # phi=init_phi_field, psi=init_psi_field,
                 coords_x=coords_x, matl_params=plasma_config,
                 te_order=te_order_EM, is_yee=is_yee,
                 normalize=False, is_electrostatic=False, clean=False)
vm_sys = VlasovMaxwell(init_fe_field, init_fi_field, em_sys,
                       coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                       elc_params=elc_config1, ion_params=ion_config1, normalize=True, zipup=True,
                       te_order=te_order, compress_levels=comp_levels, )
print('initialized vm_sys')

vm_sys.collision = VP_test.collision_config
print('colls', vm_sys.collision.coll_params)

vm_sys.compress_F = VP_test.compress_F
vm_sys.compress_F_opts = VP_test.compress_F_opts
print('compress E?',  vm_sys.compress_F, vm_sys.compress_F_opts)

print('E nrg', vm_sys.E.norm()**2 * 1./ 2 / tot_x)
print('B nrg', vm_sys.B.norm()**2 * 1./ 2 / tot_x)

### time step info
if te_order_EM != 22:
    dt_max = min(dt_max, 0.1 * ax_x.dx / plasma_config.c)
print('EM dt max', dt_max, ax_x.dx / plasma_config.c, ax_x.dx, plasma_config.c)

if dt is None or do_adapt_dt:
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
    print('adapt dt', dt, plasma_config.c, elc_config1.vth)


if nt == 0:
    fig1, ax1 = plt.subplots(2, 1)
    time_ = nt * dt
    fi_data = init_fi_gtn.get_data(ax_select={ax_vz_i: ax_vz_i.npts // 2, ax_vy_i: ax_vy_i.npts // 2})
    fi_data = ax_x.basis.get_realspace_1D(fi_data, 0)
    fi_data = ax_vx_i.basis.get_realspace_1D(fi_data, 1)
    im1 = ax1[0].imshow(np.real(fi_data).T,
                        extent=(x_min, x_max, vi_x_min, vi_x_max),
                        aspect='auto', origin='lower')

    fe_data = init_fe_gtn.get_data(ax_select={ax_vx_e: ax_vx_e.npts // 2, ax_vy_e: ax_vy_e.npts // 2})
    fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
    fe_data = ax_vx_e.basis.get_realspace_1D(fe_data, 1)
    im2 = ax1[1].imshow(np.real(fe_data).T,
                        extent=(x_min, x_max, ve_x_min, ve_x_max),
                        aspect='auto', origin='lower')

    fig1.colorbar(im1, ax=ax1[0])
    fig1.colorbar(im2, ax=ax1[1])
    ax1[0].set_title(f'time {time_:3.2f}')
    ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
    ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
    # if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')
    # plt.close(fig1)

    plt.show()

    em_sys.current_density = vm_sys.compute_current(compress=4)
    em_sys.charge_density = vm_sys.compute_charge_density(compress=4)


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
    max_bond_E += [E_field.max_bonds(compIDs=coords_x.coords)]
    energy_E_t = np.array(E_field.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_x  # electric field energy

    B_field = vm_sys.B
    max_bond_B += [B_field.max_bonds(compIDs=coords_x.coords)]
    energy_B_t = np.array(B_field.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_x  # magnetic field energy
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

    if nt % save_every_nt == 0:

        if save_data:
            if not os.path.exists(sdir + 'restart/'):
                os.makedirs(sdir + 'restart/')

            test_setup.save_data_EM( VP_test, vm_sys, sdir+'restart/', add_fstr=f'-nt{nt}',
                                     ts=ts, nrg_fe_ts=nrg_fe_ts, nrg_fi_ts=nrg_fi_ts, nrg_E_ts=nrg_E_ts,
                                     nrg_B_ts=nrg_B_ts, max_bond_fe=max_bond_fe, max_bond_fi=max_bond_fi,
                                     max_bond_E=max_bond_E, max_bond_B=max_bond_B,
                                     # max_bond_phi=max_bond_phi, max_bond_psi=max_bond_psi
                                     )

        fig1, ax1 = plt.subplots(2, 1)
        time_ = ts[-1]
        fi_data = fi_field.get_comp_data(ax_select={ax_vy_i: npts_vyi//2, ax_vz_i: npts_vzi//2})
        fi_data = ax_x.basis.get_realspace_1D(fi_data, 0)
        fi_data = ax_vx_i.basis.get_realspace_1D(fi_data, 1)
        im1 = ax1[0].imshow(np.real(fi_data).T, extent=(x_min, x_max, vi_x_min, vi_x_max),
                            aspect='auto', origin='lower')

        fe_data = fe_field.get_comp_data(ax_select={ax_vy_e: npts_vye//2, ax_vz_e: npts_vze//2})
        fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
        fe_data = ax_vx_e.basis.get_realspace_1D(fe_data, 1)
        im2 = ax1[1].imshow(np.real(fe_data).T, extent=(x_min, x_max, ve_x_min, ve_x_max),
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

        ## velocity space cuts
        for div_x in [4, 2]:
            fig1, ax1 = plt.subplots(1, 3, figsize=(12,5))
            time_ = ts[-1]
            fe_data = fe_field.get_comp_data(ax_select={ax_x: npts_x // div_x, ax_vz_e: npts_vze // 2})
            fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
            fe_data = ax_vx_e.basis.get_realspace_1D(fe_data, 1)
            im1 = ax1[0].imshow(np.real(fe_data).T, extent=(ve_x_min, ve_x_max, ve_y_min, ve_y_max),
                                aspect='auto', origin='lower')

            ax1[0].set_xlabel('vx'), ax1[0].set_ylabel('vy')
            fig1.colorbar(im1, ax=ax1[0])

            fe_data = fe_field.get_comp_data(ax_select={ax_x: npts_x // div_x, ax_vx_e: npts_vxe // 2})
            fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
            fe_data = ax_vx_e.basis.get_realspace_1D(fe_data, 1)
            im2 = ax1[1].imshow(np.real(fe_data).T, extent=(ve_y_min, ve_y_max, ve_z_min, ve_z_max),
                                aspect='auto', origin='lower')

            ax1[1].set_xlabel('vy'), ax1[1].set_ylabel('vz')
            fig1.colorbar(im2, ax=ax1[1])

            fe_data = fe_field.get_comp_data(ax_select={ax_x: npts_x // div_x, ax_vy_e: npts_vye // 2})
            fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
            fe_data = ax_vx_e.basis.get_realspace_1D(fe_data, 1)
            im3 = ax1[2].imshow(np.real(fe_data).T, extent=(ve_x_min, ve_x_max, ve_z_min, ve_z_max),
                                aspect='auto', origin='lower')

            ax1[2].set_xlabel('vx'), ax1[2].set_ylabel('vz')
            fig1.colorbar(im3, ax=ax1[2])


            fig1.suptitle(f'time {time_:3.2f}, //{div_x}')

            if save_figs:
                fig1.savefig(fdir + f'plot_fv_' + fstr + f'_nt{nt}.png')
            else:
                plt.show()
            plt.close(fig1)


        ##
        real_xs = np.linspace(x_min, x_max, npts_x, endpoint=False)
        E_data = vm_sys.E.get_comp_data(Z)
        if np.all(E_data != 0):
            E_data = ax_x.basis.get_realspace_1D(E_data, 0)
            plt.figure()
            plt.plot(real_xs, np.real(E_data), label='E field real')
            plt.plot(real_xs, np.imag(E_data), label='E field imag')
            plt.legend()

        By_data = vm_sys.B.get_comp_data(Y)
        if np.all(By_data != 0):
            By_data = ax_x.basis.get_realspace_1D(By_data, 0)
            plt.figure()
            plt.plot(real_xs, np.real(By_data), label='B field real')
            plt.plot(real_xs, np.imag(By_data), label='B field imag')
            plt.legend()

        if not save_figs:
            plt.show()
        plt.close()

        fig7, ax7 = plt.subplots()
        ax7.plot(ts, np.abs([nrg[2] for nrg in nrg_B_ts]), label='Bz')
        ax7.legend()
        ax7.set_xlabel('t')
        ax7.set_ylabel('energy')
        ax7.legend()
        if save_figs:
            fig7.savefig(fdir + 'plot_nrg_Bz_' + fstr + '.png')
        else:
            plt.show()
        plt.close()

        #### compute moments
        density_e = vm_sys.sys_fe.compute_moment([])
        density_i = vm_sys.sys_fi.compute_moment([])

        plt.figure()
        plt.plot(density_i.get_comp_data() * mass_e / tot_x, label='density i')
        plt.title('density i')
        plt.figure()
        plt.plot(density_e.get_comp_data() * mass_i / tot_x, label='density e')
        plt.title('density e')
        plt.show()

        px_e = vm_sys.sys_fe.compute_moment([ax_vx_e], powers=[1]).get_comp_data()
        py_e = vm_sys.sys_fe.compute_moment([ax_vy_e], powers=[1]).get_comp_data()
        pz_e = vm_sys.sys_fe.compute_moment([ax_vz_e], powers=[1]).get_comp_data()
        p_e = np.sqrt( px_e ** 2 + py_e ** 2 + pz_e ** 2) * mass_e

        px_i = vm_sys.sys_fi.compute_moment([ax_vx_i], powers=[1]).get_comp_data()
        py_i = vm_sys.sys_fi.compute_moment([ax_vy_i], powers=[1]).get_comp_data()
        pz_i = vm_sys.sys_fi.compute_moment([ax_vz_i], powers=[1]).get_comp_data()
        p_i = np.sqrt(px_i ** 2 + py_i ** 2 + pz_i ** 2) * mass_i

        plt.figure()
        plt.plot(x_vals, p_e, label='p elec')
        plt.title('<v> elec')
        plt.figure()
        plt.plot(x_vals, p_i, label='p ion')
        plt.title('<v> ion')

        nrgx_e = vm_sys.sys_fe.compute_moment([ax_vx_e], powers=[2]).get_comp_data()
        nrgy_e = vm_sys.sys_fe.compute_moment([ax_vy_e], powers=[2]).get_comp_data()
        nrgz_e = vm_sys.sys_fe.compute_moment([ax_vz_e], powers=[2]).get_comp_data()
        nrg_e = (nrgx_e + nrgy_e + nrgz_e) * 0.5 * mass_e

        nrgx_i = vm_sys.sys_fi.compute_moment([ax_vx_i], powers=[2]).get_comp_data()
        nrgy_i = vm_sys.sys_fi.compute_moment([ax_vy_i], powers=[2]).get_comp_data()
        nrgz_i = vm_sys.sys_fi.compute_moment([ax_vz_i], powers=[2]).get_comp_data()
        nrg_i =  (nrgx_i + nrgy_i + nrgz_i) * 0.5 * mass_i

        plt.figure()
        plt.plot(x_vals, nrg_e, label='nrg elec')
        plt.title('<v^2> elec')
        plt.figure()
        plt.plot(x_vals, nrg_i, label='nrg ion')
        plt.title('<v^2> ion')

        plt.show()


if save_data:
    test_setup.save_data_EM(VP_test, vm_sys, fdir,
                            ts=ts, nrg_fe_ts=nrg_fe_ts, nrg_fi_ts=nrg_fi_ts, nrg_E_ts=nrg_E_ts,
                            nrg_B_ts=nrg_B_ts, max_bond_fe=max_bond_fe, max_bond_fi=max_bond_fi,
                            max_bond_E=max_bond_E, max_bond_B=max_bond_B,
                            # max_bond_phi=max_bond_phi, max_bond_psi=max_bond_psi
                            )

print('wall time', time.time() - time1)

fig1, ax1 = plt.subplots(2, 1)
time_ = ts[-1]
fi_data = vm_sys.fi.get_comp_data(ax_select={ax_vy_i: npts_vyi//2, ax_vz_i: npts_vzi//2})
fi_real = ax_x.basis.get_realspace_1D(fi_data, 0)
fi_real = ax_vz_i.basis.get_realspace_1D(fi_real, 1)
im1 = ax1[0].imshow(np.real(fi_real).T, extent=(x_min, x_max, vi_x_min, vi_x_max),
                    aspect='auto', origin='lower')

fe_data = vm_sys.fe.get_comp_data(ax_select={ax_vy_e: npts_vye//2, ax_vz_e: npts_vze//2})
fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
fe_real = ax_vx_e.basis.get_realspace_1D(fe_real, 1)
im2 = ax1[1].imshow(np.real(fe_real).T, extent=(x_min, x_max, ve_x_min, ve_x_max),
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

