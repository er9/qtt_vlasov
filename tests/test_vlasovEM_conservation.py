import os, sys, pickle, time, glob
import scipy.optimize
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from setup_.paths import save_dir, main_dir
from setup_.configs import *
import setup_.helper as helper_test
import helper_quimb
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
from vlasov_tests import VlasovTest

"""
check conservation of VM equation
commutation of low-rank projector with diag(v), diag(v^2)
with d/dvi v^2 = 2*vi
check that we get the expected expectation values 
"""

"""
diag(v) --> basis functions are the eigenvectors... which may or may not be low rank?
"""



is_sqrt = flags.get('is_sqrt', False)
do_tt = flags.get('do_tt', False)
do_full = flags.get('do_full', False)


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
elc_config1 = IonConfiguration(n0=n0_i, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config1.vth  # electron thermal speed 1/2 m b^2 n0 = sqrt(5.0e-5)
wp_e = elc_config1.wp    # plasma e' frequency
lamD = elc_config1.lamD  # Debye length
wp_i = ion_config1.wp    # plasma i+ frequency
vth_i = ion_config1.vth  # ion thermal speed

###################
# grid parameters #
###################

q = 2
KX = 1
KV = 1
Lx = flags.get('Lx', 4)
Lve = flags.get('Lve', 4)
Lvi = flags.get('Lvi', 4)

if (do_full or do_tt):
    TN_layout = LayoutType.PARALLEL_GROUP if do_full else LayoutType.SEQUENTIAL
    x_map_key = 'F'  # 'F'
    v_map_key = 'F'  # 'M'
else:
    TN_layout = flags.get('layout', LayoutType.SEQUENTIAL)
    x_map_key = flags.get('x_map_key', 'F')
    v_map_key = flags.get('v_map_key', 'B')

VP_test = VlasovTest(Lxs=(Lx,), Lves=(Lve,Lve,Lve), Lvis=(Lvi,Lvi,Lvi), KX=KX, KV=KV, q=q,
                     #
                     ion_config = ion_config1, elc_config = elc_config1,
                     # coll_type=None, do_adapt_coll = True, coll_coeff_e = 0.005, coll_rate_e = 0.01,
                     grid_layout=TN_layout, x_map_key = x_map_key, ve_map_key = v_map_key,
                     do_tt=(do_full or do_tt),
                     )

# dve = 1.0 / (2**Lve)
# dvi = 1.0 / (2**Lvi)
VP_test.initialize_axes(x_lims=(0, 2*np.pi), ve_lims=(-5*vth_e, 5*vth_e), vi_lims=(-5*vth_i, 5*vth_i),
                        x_map_key=x_map_key, ve_map_key=v_map_key, basis_xs = SpatialBasis(),
                        do_tt=(do_full or do_tt),
                        )

VP_test.initialize_grid()
grid_i = VP_test.grid_i
grid_e = VP_test.grid_e
grid_X = VP_test.grid_X

# ax_x, ax_y, ax_z, ax_vx, ax_vy, ax_vz = grid_e.axes
ax_x, ax_vx = grid_e.axes
x_vals = ax_x.xpts
vx_vals = ax_vx.xpts

coords_x  = VP_test.pos_coordsys
coords_ve = VP_test.ve_coordsys
coords_vi = VP_test.vi_coordsys

X, Y, Z = coords_x.coords
VX, VY, VZ = coords_ve.coords


order = 2
deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
deriv_v = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)

init_fe_gtn = grid_e.get_rand_state(8)
# init_fe_gtn = grid_e.make_mps_ndim({ax_x: 1 + np.cos(x_vals),
#                                     ax_vx: maxwellian(vx_vals, vth2=vth_e**2, flow=0.1)})
init_fe_gtn = grid_e.make_mps_ndim({ax_x: np.random.random(x_vals.shape),
                                    ax_vx: (maxwellian(vx_vals, vth2=vth_e**2, flow=0.1) * np.random.random(vx_vals.shape)
                                            + maxwellian(vx_vals, vth2=vth_e**2/4, flow=0.3) * np.random.random(vx_vals.shape)
                                            )
                                    })
init_fe_gtn_2 = grid_e.make_mps_ndim({ax_x: np.random.random(x_vals.shape),
                                    ax_vx: (maxwellian(vx_vals, vth2=vth_e**2/6, flow=-0.2) * np.random.random(vx_vals.shape)
                                            )
                                    })
init_fe_gtn = init_fe_gtn.add(init_fe_gtn_2)

print('vth_e', vth_e**2)

plt.figure()
plt.imshow(init_fe_gtn.get_data())
plt.show()


init_fe_gtn.ax_deriv_configs = {ax: deriv_x for ax in grid_e.axes}
init_fi_gtn = grid_i.make_empty_gridTN({ax: deriv_x for ax in grid_i.axes})
# helper.pad_mpx_virtuals(init_fe_gtn.data, max_bond=DMAX)
# init_fe_gtn.compress(compress_opts={'max_bond': DMAX})
# print('init_fe_gtn', init_fe_gtn.max_bond())


## E field: 0
x_vals = grid_e.axes[0].xpts
Ex_gtn = grid_X.get_rand_state(8)
Ex_gtn.ax_deriv_configs = {ax: deriv_x for ax in grid_X.axes}
Ey_gtn = grid_X.get_rand_state(8)
Ey_gtn.ax_deriv_configs = {ax: deriv_x for ax in grid_X.axes}
Ez_gtn = grid_X.get_rand_state(8)
Ez_gtn.ax_deriv_configs = {ax: deriv_x for ax in grid_X.axes}
# Ex_gtn = grid_X.make_empty_gridTN(ax_deriv_configs={ax: deriv_x for ax in grid_X.axes})
# Ey_gtn = grid_X.make_empty_gridTN(ax_deriv_configs={ax: deriv_x for ax in grid_X.axes})
# Ez_gtn = grid_X.make_empty_gridTN(ax_deriv_configs={ax: deriv_x for ax in grid_X.axes})
# Ex_gtn = grid_X.map_state_to_mps(1.0 * np.cos(x_vals))
# Ey_gtn = grid_X.map_state_to_mps(1.0 * np.sin(x_vals))
# Ez_gtn = grid_X.map_state_to_mps(0.5 * np.cos(x_vals))

## B field:
Bx_gtn = grid_X.map_state_to_mps(1.0 * np.cos(x_vals))
By_gtn = grid_X.map_state_to_mps(1.0 * np.sin(x_vals))
Bz_gtn = grid_X.map_state_to_mps(0.5 * np.cos(x_vals))

# init_E_field = Field('E', grid_X, data={X: Ex_gtn, Y: Ey_gtn, Z: Ez_gtn}, is_sqrt=True)
# init_B_field = Field('B', grid_X, data={X: Bx_gtn, Y: By_gtn, Z: Bz_gtn}, is_sqrt=True)

init_E_field = Field('E', grid_X, data={X: Ex_gtn}, is_sqrt=True)
init_B_field = Field('B', grid_X, data={}, is_sqrt=True)


em_sys = Maxwell(init_E_field, init_B_field,  # phi=init_phi_field, psi=init_psi_field,
                 coords_x=coords_x, matl_params=plasma_config,
                 normalize=False, clean=False, is_yee=False)

init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, is_sqrt=is_sqrt)
init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, is_sqrt=is_sqrt)

vm_sys = VlasovMaxwell(init_fe_field, init_fi_field, em_sys,
                           coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                           elc_params=elc_config1, ion_params=ion_config1, evolve_ion=False, evolve_EM=False,
                           normalize=False,  # True,
                           zipup=True,
                           upwind=False,
                           conservative=False,
                           )

vm_sys.compute_force_term()

####################################
## diag(v), diag(v^2) commutation ##
####################################

cur_orthog = 6
direction = 1 if cur_orthog > grid_e.L // 2 else -1
## otherwise is inaccurate, for whatever reason.

v_mps_gtn = grid_e.get_xmultiply_mps([ax_vx])
v_mpo_gtn = grid_e.get_xmultiply_mpo([ax_vx])

# init_fe_gtn = grid_e.make_mps_ndim({ax_x: ax_x.get_rand_state(8),
#                                     ax_vx: ax_vx.get_xmultiply_mps()})

v_mps = v_mps_gtn.data.copy()
v_mps.distribute_exponent()
init_fe_mps = init_fe_gtn.data.copy()
init_fe_mps.distribute_exponent()

v_mps = helper_quimb.canonize(v_mps, i=cur_orthog, scale=False)


# out = helper_quimb.add_MPS_target(v_mps, init_fe_gtn.data, # v_mps,
#                                   do_final_update=False, direction=-1)
out = helper_quimb.add_MPS_target(init_fe_mps, v_mps,
                                  do_final_update=False, direction=direction)
out = helper_quimb.canonize(out, scale=False, i=cur_orthog)


# print('--- add target ---')
# print('v mps', v_mps)
# print('init fe gtn', init_fe_gtn.data)
# print('out', out)

def proj_mps(mps, proj_mps, center):
    L_ = mps.L
    mps.mangle_inner_()
    proj_mps.site_ind_id = mps.site_ind_id

    envL = None
    for ix in range(center):
        # print('left: ix', ix)
        xtens = [mps[ix], proj_mps[ix]]
        if envL is not None:
            xtens += [envL]
        envL = qtn.tensor_contract(*xtens)

    envR = None
    for ix in range(mps.L - 1, center, -1):
        # print('right: ix', ix)
        xtens = [mps[ix], proj_mps[ix]]
        if envR is not None:
            xtens += [envR]
        envR = qtn.tensor_contract(*xtens)

    xtens = [mps[center]]
    if envL is not None:
        xtens += [envL]
    if envR is not None:
        xtens += [envR]
    mtens = qtn.tensor_contract(*xtens)
    mtens.drop_tags()
    mtens.add_tag(proj_mps.site_tag(cur_orthog))

    tensors = [proj_mps[ix] for ix in range(center)] + [mtens] + [proj_mps[ix] for ix in range(center + 1, L_)]

    new_mps = qtn.TensorNetwork(tensors)
    new_mps.view_like(proj_mps, inplace=True)
    new_mps.exponent = mps.exponent
    return new_mps


print('out mps exponent', out.exponent)
print('v mps exponent', v_mps.exponent)

proj_out = proj_mps(out.copy(), out, cur_orthog)
print('proj out - out', helper_quimb.distance(proj_out, out))

proj_v = proj_mps(v_mps, out, cur_orthog)
print('proj v - v', helper_quimb.distance(proj_v, v_mps))

####
## integ @ projector @ v * f
integ = grid_e.get_integrals_mpx()
integ.data.distribute_exponent()

out = helper_quimb.add_MPS_target(out.copy(), integ.data.copy(),
                                  do_final_update=False, direction=direction)
out = helper_quimb.canonize(out, scale=False, i=cur_orthog)

dx = ax_x.dx * ax_vx.dx
expec2 = helper_quimb.ovlp(v_mps, init_fe_mps, do_canonize=False) * dx
print('f expec 0:', expec2)

integ_proj = proj_mps(integ.data.copy(), out, cur_orthog)
print('integ exponent', integ.exponent, integ_proj.exponent )
print('proj integ - integ', helper_quimb.distance(integ_proj, integ.data))

v_fe_mps = helper_quimb.apply( helper_quimb.mps_to_diag_mpo(v_mps), init_fe_mps )
expec1 = helper_quimb.ovlp(integ_proj, v_fe_mps, do_canonize=False)
print('f expec 1: ', expec1)

### --> projector needs to include integ MPS (automatically satisfied for sqrt?)

## integ v_mps @ projector @ f
v_proj = proj_mps(v_mps.copy(), out, cur_orthog)
expec2 = helper_quimb.ovlp(v_proj, init_fe_mps, do_canonize=False) * dx
print('f expec 2: ', expec2)

print('')

#####
## integ @ projector @ v @ d/dv @ f
ddv_f_gtn = init_fe_gtn.take_firstderivative(ax_vx)
v_ddv_f_mps = helper_quimb.apply( helper_quimb.mps_to_diag_mpo(v_mps), ddv_f_gtn.data)
# helper_quimb.canonize(v_ddv_f_mps)

expec1 = helper_quimb.ovlp(integ.data, v_ddv_f_mps)
print('expec 0: ', expec1)


expec1 = helper_quimb.ovlp(integ_proj, v_ddv_f_mps)
print('expec 1: ', expec1)

## integ v mps @ projector @ d/dv @ f
### but I thought we already did this?  maybe difference in cur orthog?
out2 = helper_quimb.add_MPS_target(out, v_mps,
                                  do_final_update=False, direction=direction)
out2 = helper_quimb.canonize(out2, scale=False, i=cur_orthog)
print('distance out, out2', helper_quimb.distance(out2, out))

ddv_f_gtn = init_fe_gtn.take_firstderivative(ax_vx)
proj_ddv_f = proj_mps(ddv_f_gtn.data.copy(), out, cur_orthog)
dx = ax_x.dx * ax_vx.dx
expec2 = helper_quimb.ovlp(v_mps, proj_ddv_f) * dx
print('expec 2.1: ', expec2)

ddv_f_gtn = init_fe_gtn.take_firstderivative(ax_vx)
proj_ddv_f = proj_mps(ddv_f_gtn.data.copy(), out2, cur_orthog)
dx = ax_x.dx * ax_vx.dx
expec2 = helper_quimb.ovlp(v_mps, proj_ddv_f) * dx
print('expec 2.2: ', expec2)

# ## v @ d/dv = ?   not at the boundaries....
# ddv_v = v_mps_gtn.take_firstderivative(ax_vx)
# ddv_v_dense = ddv_v.get_data()
# plt.figure()
# plt.imshow(ddv_v_dense)
# plt.colorbar()
# plt.show()

#####
## compute <v|f> --> also target <v|f> * 1
## integ @ projector @ v @ d/dv @ f
## only works for product state fe
fv_moment = init_fe_gtn.meas_expec(v_mpo_gtn, integ_axes=[ax_vx], new_grid=grid_X)

plt.figure()
plt.plot(fv_moment.get_data())
plt.title('fv moment')
plt.show()

print('f expec (match above expecs?): ', fv_moment.integrate())
fv_gtn = grid_e.make_mps_ndim({ax_x: fv_moment.data, ax_vx: ax_vx.get_xmultiply_mps()})
out2 = helper_quimb.add_MPS_target(out, fv_gtn.data,
                                  do_final_update=False, direction=direction)
out2 = helper_quimb.canonize(out2, scale=False, i=cur_orthog)

# ## integ @ projector @ v @ d/dv @ f
# ddv_f_gtn = init_fe_gtn.take_firstderivative(ax_vx)
# v_ddv_f_mps = helper_quimb.apply( helper_quimb.mps_to_diag_mpo(v_mps), ddv_f_gtn.data)
# # helper_quimb.canonize(v_ddv_f_mps)
#
# expec1 = helper_quimb.ovlp(integ_proj, v_ddv_f_mps, do_canonize=False)
# print('expec 1: ', expec1)

## integ v mps @ projector @ d/dv @ f
ddv_f_gtn = init_fe_gtn.take_firstderivative(ax_vx)
proj_ddv_f = proj_mps(ddv_f_gtn.data.copy(), out2, cur_orthog)
dx = ax_x.dx * ax_vx.dx
expec2 = helper_quimb.ovlp(v_mps, proj_ddv_f) * dx
print('expec 3: ', expec2)   ## this is incorrect (only true for rank 1 TT)


## also target df/dv
## integ @ projector @ v @ d/dv @ f (expec 1) --> projector must capture integ mps

## integ v mps @ projector @ d/dv @ f --> projector must capture d/dv f
## ... or integ v mps
ddv_f_gtn = init_fe_gtn.take_firstderivative(ax_vx)
out2 = helper_quimb.add_MPS_target(out, ddv_f_gtn.data,
                                  do_final_update=False, direction=direction)
out2 = helper_quimb.canonize(out2, scale=False, i=cur_orthog)

proj_ddv_f = proj_mps(ddv_f_gtn.data.copy(), out2, cur_orthog)
dx = ax_x.dx * ax_vx.dx
expec2 = helper_quimb.ovlp(v_mps, proj_ddv_f) * dx
print('expec 4: ', expec2)

## projector targeting integ @ v
integ_v_mps = helper_quimb.apply( helper_quimb.mps_to_diag_mpo(v_mps), integ.data)
out2 = helper_quimb.add_MPS_target(out, integ_v_mps,
                                   do_final_update=False, direction=direction)
out2 = helper_quimb.canonize(out2, scale=False, i=cur_orthog)

proj_ddv_f = proj_mps(ddv_f_gtn.data.copy(), out2, cur_orthog)
dx = ax_x.dx * ax_vx.dx
expec2 = helper_quimb.ovlp(v_mps, proj_ddv_f) * dx
print('expec 5: ', expec2)


print('')

exit()


#######################
## mass conservation ##
#######################
em_force = vm_sys.compute_force_term()
vm_sys.sys_fe.set_force_term(em_force)
mpos = vm_sys.sys_fe._get_time_evolution_mpos()   ## [d/dx, d/dvx, d/dvy, d/dvz]
print('mpos', mpos)

print('\nmass conservation')
integ_mps = grid_e.get_iden_mps()

f = vm_sys.fe.component.data  #  grid_e.get_rand_state(64).data
# helper_quimb.canonize(f, i=0)
f = helper_quimb.add_MPS_target(f, integ_mps.data, direction=-1, inplace=True, do_final_update=False,
                                compress_opts={'cutoff': 1.0e-28})
proj_integ = helper_quimb.target_MPS(integ_mps.data, f, site=0)

tot_expec = 0.0
for it, mpo in enumerate(mpos):
    expec = helper_quimb.expectation_value(f, mpo.data, proj_integ)
    print('expec', it, expec)

###########################
## momentum conservation ##
###########################
# ## particle dist.
# ax_x, ax_vxe, ax_vye, ax_vze = grid_e.axes
# for ax_v in [ax_vxe, ax_vye, ax_vze]:
#     print('ax v', ax_v)
#     integ_mps = grid_e.get_xmultiply_mpo(ax_v)
#     for it, mpo in enumerate(mpos):
#         expec = helper_quimb.expectation_value(f, mpo, integ_mps)
#         print('expec', expec)
#
# ## maxwell eqn

#########################
## energy conservation ##
#########################
print('\nenergy conservation')
ax_xs = grid_e.axes[:KX]
ax_vs = grid_e.axes[KX:]

f = vm_sys.fe.component.data   # grid_e.get_rand_state(64).data
integ_gtn = None
for ax_v in ax_vs:
    print('ax v', ax_v)
    v2_mps = grid_e.make_mps_ndim({ax_v: ax_v.get_xmultiply_mps(x_power=2) for ax_v in ax_vs})
    if integ_gtn is None:
        integ_gtn = v2_mps
    else:
        integ_gtn = integ_gtn.add(v2_mps, inplace=True)


f = helper_quimb.add_MPS_target(f, integ_mps.data, direction=-1, inplace=True, do_final_update=False)
proj_integ = helper_quimb.target_MPS(integ_mps.data, f, site=0)

J = vm_sys.compute_current()
EJ_tot = None
for compID in [X, Y, Z]:
    try:
        EJ = vm_sys.E[compID].elemental_multiply(J[compID], compress=True)
    except KeyError:
        continue

    if EJ_tot is None:
        EJ_tot = EJ
    else:
        EJ_tot = EJ_tot.add(EJ, inplace=True)

print('integrated E dot J', EJ_tot.integrate(is_sqrt=False))


for it, mpo in enumerate(mpos):
    expec = helper_quimb.expectation_value(f, mpo.data, proj_integ)
    print('expec', expec)