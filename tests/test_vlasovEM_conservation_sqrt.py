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
use sqrt formalism 
"""

rng = np.random.default_rng(seed=42)

is_sqrt = True
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
                                    ax_vx: (maxwellian(vx_vals, vth2=vth_e**2, flow=0.1, is_sqrt=is_sqrt) * np.random.random(vx_vals.shape)
                                            + maxwellian(vx_vals, vth2=vth_e**2/4, flow=0.3, is_sqrt=is_sqrt) * np.random.random(vx_vals.shape)
                                            )
                                    })
init_fe_gtn_2 = grid_e.make_mps_ndim({ax_x: np.random.random(x_vals.shape),
                                    ax_vx: (maxwellian(vx_vals, vth2=vth_e**2/6, flow=-0.2, is_sqrt=is_sqrt) * np.random.random(vx_vals.shape)
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
v_mps_gtn.data.distribute_exponent()
v_mpo_gtn = grid_e.get_xmultiply_mpo([ax_vx])


# init_fe_gtn = grid_e.make_mps_ndim({ax_x: ax_x.get_rand_state(8),
#                                     ax_vx: ax_vx.get_xmultiply_mps()})

init_fe_mps = init_fe_gtn.data.copy()
init_fe_mps.distribute_exponent()

init_fe_v = init_fe_gtn.elemental_multiply(v_mps_gtn)
fe_v_mps = init_fe_v.data
fe_v_mps.distribute_exponent()
init_fe_v.canonize(i=cur_orthog, scale=False)

out = helper_quimb.add_MPS_target(init_fe_mps, init_fe_v.data,
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
print('fe v mps exponent', fe_v_mps.exponent)

proj_out = proj_mps(out.copy(), out, cur_orthog)
print('proj v - v', helper_quimb.distance(proj_out, out))

proj_v = proj_mps(fe_v_mps, out, cur_orthog)
print('proj v - v', helper_quimb.distance(proj_v, fe_v_mps))

dx = ax_x.dx * ax_vx.dx

## <g | v g >
print('f expec 0', helper_quimb.ovlp(init_fe_mps, fe_v_mps, do_canonize=False) * dx)

## <g | projector | v g >
integ = init_fe_gtn.copy()
out = helper_quimb.add_MPS_target(out, integ.data,
                                  do_final_update=False, direction=direction)
# cur_orthog = 0
out = helper_quimb.canonize(out, scale=False, i=cur_orthog)

integ_proj = proj_mps(integ.data.copy(), out, cur_orthog)
# v_fe_mps = helper_quimb.apply( helper_quimb.mps_to_diag_mpo(v_mps), init_fe_mps )
expec1 = helper_quimb.ovlp(integ_proj, fe_v_mps, do_canonize=False) * dx
print('f expec 1: ', expec1)

### --> projector needs to include integ MPS (automatically satisfied for sqrt?)

## <g v_mps | projector | g >
proj_init = proj_mps(init_fe_mps.copy(), out, cur_orthog)
expec2 = helper_quimb.ovlp(fe_v_mps, proj_init, do_canonize=False) * dx
print('f expec 2:', expec2)

print('')

#####
print('momentum conservation')
## <g |  projector @ v @ d/dv | g >
ddv_f_gtn = init_fe_gtn.take_firstderivative(ax_vx)
v_ddv_f_gtn = ddv_f_gtn.elemental_multiply(v_mps_gtn)
v_ddv_f_mps = v_ddv_f_gtn.data
# helper_quimb.canonize(v_ddv_f_mps)

expec0 = helper_quimb.ovlp(integ.data, v_ddv_f_mps) * dx
print('expec 0:', expec0)

expec1 = helper_quimb.ovlp(integ_proj, v_ddv_f_mps) * dx
print('expec 1: ', expec1)

## <g | v mps @ projector @ d/dv | g >
### but I thought we already did this?  maybe difference in cur orthog?
out2 = helper_quimb.add_MPS_target(init_fe_mps, fe_v_mps,
                                  do_final_update=False, direction=direction)
out2 = helper_quimb.canonize(out2, scale=False, i=cur_orthog)

ddv_f_gtn = init_fe_gtn.take_firstderivative(ax_vx)
proj_ddv_f = proj_mps(ddv_f_gtn.data.copy(), out2, cur_orthog)
expec2 = helper_quimb.ovlp(fe_v_mps, proj_ddv_f) * dx
print('expec 2 (should match): ', expec2)

# ## v @ d/dv = ?   not at the boundaries....
# ddv_v = v_mps_gtn.take_firstderivative(ax_vx)
# ddv_v_dense = ddv_v.get_data()
# plt.figure()
# plt.imshow(ddv_v_dense)
# plt.colorbar()
# plt.show()

## < g | v mps @ projector @ d/dv @ | g > --> projector must capture d/dv f
## ... or  v | g >
out2 = helper_quimb.add_MPS_target(init_fe_mps, ddv_f_gtn.data,
                                  do_final_update=False, direction=direction)
out2 = helper_quimb.canonize(out2, scale=False, i=cur_orthog)

proj_fe_v = proj_mps(fe_v_mps.copy(), out2, cur_orthog)
expec2 = helper_quimb.ovlp(proj_fe_v, ddv_f_gtn.data.copy()) * dx
print('expec 3 (should match): ', expec2)


###############################
print('\nenergy conservation')
J = vm_sys.compute_current()
em_force = vm_sys.compute_force_term()
vm_sys.sys_fe.set_force_term(em_force)
mpos = vm_sys.sys_fe._get_time_evolution_mpos()   ## [d/dx, d/dvx, d/dvy, d/dvz]
# print('mpos', mpos)

## energy conservation
ax_xs = grid_e.axes[:KX]
ax_vs = grid_e.axes[KX:]

g_gtn = vm_sys.fe.component   # grid_e.get_rand_state(64)
g = g_gtn.data

## get ||v^2| g> * me / 2
f_mps = g_gtn.elemental_multiply(g_gtn).data
v2g_gtn = None
v2_gtn = None
for ax_v in ax_vs:
    print('ax v', ax_v)
    v2i_mps = grid_e.make_mps_ndim({ax_v: ax_v.get_xmultiply_mps(x_power=2)})
    v2_g = g_gtn.elemental_multiply(v2i_mps)
    if v2g_gtn is None:
        v2g_gtn = v2_g
        v2_gtn = v2i_mps
    else:
        v2g_gtn = v2g_gtn.add(v2_g, inplace=True, compress=True)
        v2_gtn = v2_gtn.add(v2i_mps, inplace=True, compress=True)
v2g_gtn.scalar_multiply(0.5 * mass_e, inplace=True)
v2_gtn.scalar_multiply(0.5 * mass_e, inplace=True)

# f = helper_quimb.add_MPS_target(f, integ_gtn.data, direction=-1, inplace=True, do_final_update=False)
# proj_integ = helper_quimb.target_MPS(integ_gtn.data, g, site=cur_orthog)

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

# ax_v = ax_vs[0]
# v2_mpo = grid_e.make_mpo_ndim({ax_v: ax_v.get_xmultiply_mpo(x_power=2)}).data
# expec = helper_quimb.expectation_value(g, v2_mpo, g.conj()) * dx
# print('expec v2 ref g',  expec)
# expec = helper_quimb.expectation_value(f_mps, v2_mpo, grid_e.get_ones_mps().data) * dx
# print('expec v2 ref f',  expec)
#
# ones = grid_e.get_ones_mps().data
# ddv_mpo = grid_e.get_firstderivative_mpo(ax_v, deriv_config=deriv_v).data
# expec = helper_quimb.expectation_value(g, ddv_mpo, g.conj()) * dx * 2
# print('expec ddv ref g',  expec)
# expec1 = helper_quimb.expectation_value(g, ddv_mpo, g.conj()) * dx
# expec2 = helper_quimb.expectation_value(ones,
#                                         helper_quimb.apply(ddv_mpo, g_gtn.apply_elemental_multiply_op().data),
#                                         g.conj()) * dx
# print('expec ddv ref g',  expec1, expec2, expec1 + expec2)
# expec3 = helper_quimb.expectation_value(ones, ddv_mpo,
#                                         helper_quimb.apply(g_gtn.apply_elemental_multiply_op().data, g.conj()),
#                                         ) * dx
# expec4 = helper_quimb.expectation_value(ones, ddv_mpo, f_mps,) * dx
# expec5 = helper_quimb.expectation_value(f_mps, ddv_mpo, ones,) * dx
# f_mps_ = f_mps.copy()
# f_mps_.site_ind_id = 'blah{}'
# ddv_mpo_T = helper_quimb.mpo_conj_transpose(ddv_mpo)
# ddv_mpo_T.upper_ind_id = 'tmp{}111'
# ddv_mpo_T.lower_ind_id = 'tmp{}222'
# ddv_mpo.upper_ind_id = 'tmp{}111'
# ddv_mpo.lower_ind_id = 'tmp{}222'
#
# print('distance', helper_quimb.distance(ddv_mpo_T, ddv_mpo))
# print('distance', helper_quimb.distance(ddv_mpo_T * -1, ddv_mpo))
#
# gtn_ddv_mpo = grid_e.make_gridTN(ddv_mpo)
# gtn_ddv_mpo_T = grid_e.make_gridTN(ddv_mpo_T)
# plt.figure()
# plt.imshow(gtn_ddv_mpo.get_data(ax_select={ax_x: 0}))
# plt.colorbar()
#
# plt.figure()
# plt.imshow(gtn_ddv_mpo_T.get_data(ax_select={ax_x: 0}))
# plt.colorbar()
# plt.show()
#
# expec6 = helper_quimb.expectation_value(f_mps_, ddv_mpo_T, ones) * dx
# print('expec ddv ref gg', expec3, expec4, expec5, expec6)
# expec = helper_quimb.expectation_value(f_mps, ddv_mpo, grid_e.get_ones_mps().data) * dx
# print('expec ddv ref f',  expec)
# exit(0)

for it, mpo in enumerate(mpos):
    expec1 = helper_quimb.expectation_value(g, mpo.data, v2g_gtn.data) * dx
    print('expec mpo v2 ref g', it, expec1)
    expec2 = helper_quimb.expectation_value(v2g_gtn.data, mpo.data, g.conj()) * dx
    print('expec mpo v2.T ref g', it, expec2)
    print(expec1 - expec2, expec1 * 2)
    expec3 = helper_quimb.expectation_value(g.conj(),
                                            helper_quimb.apply(mpo.data, g_gtn.apply_elemental_multiply_op().data),
                                            v2_gtn.data) * dx
    print('expec v2 (mpo g) g', it, expec3)
    expec3 = helper_quimb.expectation_value(g.conj(),
                                            helper_quimb.apply(g_gtn.apply_elemental_multiply_op().data, mpo.data),
                                            v2_gtn.data) * dx
    print('expec v2 (g mpo) g', it, expec3)
    expec4 = helper_quimb.expectation_value(helper_quimb.apply(g_gtn.apply_elemental_multiply_op().data, g.conj()),
                                            mpo.data,
                                            v2_gtn.data,
                                            ) * dx
    print('expec v2 mpo (g g)', it, expec4)
    expec = helper_quimb.expectation_value(f_mps, mpo.data, v2_gtn.data) * dx
    print('expec mpo v2 ref f', it, expec)

    mpo_dat = mpo.data
    mpo_T = helper_quimb.mpo_conj_transpose(mpo_dat)

    mpo_g = helper_quimb.apply(mpo_dat, g)
    g_mpo = helper_quimb.apply(mpo_T, g)
    print('distance mpo @ g, g @ mpo', helper_quimb.distance(mpo_g, g_mpo * -1))

    ## (A diag(g)).T  = diag(g).T A.T = - diag(g) A  !=  - A diag(g)
    mpo_g_ = helper_quimb.apply(mpo_dat, g_gtn.apply_elemental_multiply_op().data)
    g_mpo_ = helper_quimb.apply(g_gtn.apply_elemental_multiply_op().data, mpo_dat)
    print('distance mpo @ diag(g), diag(g) @ mpo', helper_quimb.distance(mpo_g_, g_mpo_ * -1))

    mpo_g = grid_e.make_gridTN(mpo_g)
    g_mpo = grid_e.make_gridTN(g_mpo)

    tmp1 = helper_quimb.apply(mpo.data, g_gtn.apply_elemental_multiply_op().data)
    tmp2 = mpo_g.apply_elemental_multiply_op().data
    print('distance  tmp1 tmp2', helper_quimb.distance(tmp1, tmp2))

    expec3 = helper_quimb.expectation_value(g.conj(),
                                            mpo_g.apply_elemental_multiply_op().data,
                                            v2_gtn.data) * dx
    print('expec! v2 (mpo g) g', it, expec3)
    expec3 = helper_quimb.expectation_value(g.conj(),
                                            g_mpo.apply_elemental_multiply_op().data,
                                            v2_gtn.data) * dx
    print('expec! v2 (g mpo) g', it, expec3)


    mpo_T.upper_ind_id = 'tmp{}111'
    mpo_T.lower_ind_id = 'tmp{}222'
    mpo_dat.upper_ind_id = 'tmp{}111'
    mpo_dat.lower_ind_id = 'tmp{}222'

    print('distance', helper_quimb.distance(mpo_T, mpo_dat))
    print('distance', helper_quimb.distance(mpo_T * -1, mpo_dat))

    gtn_mpo = grid_e.make_gridTN(mpo_dat)
    gtn_mpo_T = grid_e.make_gridTN(mpo_T)
    plt.figure()
    plt.imshow(gtn_mpo.get_data(ax_select={ax_x: 0}))
    plt.colorbar()

    plt.figure()
    plt.imshow(gtn_mpo_T.get_data(ax_select={ax_x: 0}))
    plt.colorbar()
    plt.show()

out2 = helper_quimb.add_MPS_target(init_fe_mps, v2_gtn.data,
                                   do_final_update=False, direction=direction)
out2 = helper_quimb.canonize(out2, scale=False, i=cur_orthog)
proj_v2 = proj_mps(v2_gtn.data.copy(), out2, cur_orthog)

proj_g = proj_mps(g, out2, cur_orthog)
proj_g_gtn = grid_e.make_gridTN(proj_g)
proj_f_mps = g_gtn.elemental_multiply(proj_g_gtn).data

for it, mpo in enumerate(mpos):
    # expec2 = helper_quimb.expectation_value(g, mpo.data, proj_v2g) * dx
    # mpo_g_ = helper_quimb.apply(mpo.data, g_gtn.apply_elemental_multiply_op().data)
    expec2 = helper_quimb.expectation_value(proj_f_mps, mpo.data, proj_v2) * dx
    print('expec (should match): ', it, expec2)


################################
print('\nmomentum conservation')
J = vm_sys.compute_current()
rho = vm_sys.compute_charge_density()
mpos = vm_sys.sys_fe._get_time_evolution_mpos()   ## [d/dx, d/dvx, d/dvy, d/dvz]
# print('mpos', mpos)

ax_xs = grid_e.axes[:KX]
ax_vs = grid_e.axes[KX:]

g_gtn = vm_sys.fe.component   # grid_e.get_rand_state(64)
g = g_gtn.data

compIDs = [X, Y, Z]

for it, ax_v in enumerate(ax_vs):

    it1, it2 = (it + 1) % 3, (it + 2) % 3

    B_comp = init_B_field.components.get(compIDs[it1], None)
    J_comp = J.components.get(compIDs[it1], None)

    if B_comp is not None and J_comp is not None:
        BJ = B_comp.elemental_multiply(J_comp)
        expec_BJ = BJ.integrate(is_sqrt=False)
    else:
        expec_BJ = 0
    print('expec Bj', expec_BJ)

    E_comp = init_E_field.components.get(compIDs[it], None)
    if E_comp is not None:
        rhoE = E_comp.elemental_multiply(rho.component)
        expec_EJ = rhoE.integrate(is_sqrt=False)
    else:
        expec_EJ = 0
    print('expec rhoE', expec_EJ)

    ## get |vi > * me
    vi_mps = grid_e.make_mps_ndim({ax_v: ax_v.get_xmultiply_mps(x_power=1)})
    v_gtn = vi_mps.scalar_multiply(mass_e)

    for it, mpo in enumerate(mpos):
        expec = helper_quimb.expectation_value(f_mps, mpo.data, v_gtn.data) * dx
        print('expec ref', it, expec)

    out2 = helper_quimb.add_MPS_target(init_fe_mps, v_gtn.data,
                                       do_final_update=False, direction=direction)
    out2 = helper_quimb.canonize(out2, scale=False, i=cur_orthog)
    proj_v = proj_mps(v_gtn.data.copy(), out2, cur_orthog)

    proj_g = proj_mps(g, out2, cur_orthog)
    proj_g_gtn = grid_e.make_gridTN(proj_g)
    proj_f_mps = g_gtn.elemental_multiply(proj_g_gtn).data

    expec2 = helper_quimb.expectation_value(proj_f_mps, mpo.data, proj_v) * dx
    print('expec (should match): ', it, expec2)



exit()


#######################
## mass conservation ##
#######################
em_force = vm_sys.compute_force_term()
vm_sys.sys_fe.set_force_term(em_force)
mpos = vm_sys.sys_fe._get_time_evolution_mpos()   ## [d/dx, d/dvx, d/dvy, d/dvz]
print('mpos', mpos)

print('\nmass conservation')

g_gtn = grid_e.get_rand_state(64)
g = g_gtn.data
integ_mps = g_gtn.scalar_multiply(dx, inplace=False)
# helper_quimb.canonize(f, i=0)
g = helper_quimb.add_MPS_target(g, integ_mps.data, direction=-1, inplace=True, do_final_update=False,
                                compress_opts={'cutoff': 1.0e-28})
g = helper_quimb.canonize(g, scale=False, i=cur_orthog)
proj_integ = helper_quimb.target_MPS(integ_mps.data, g, site=cur_orthog)

tot_expec = 0.0
for it, mpo in enumerate(mpos):
    expec = helper_quimb.expectation_value(g, mpo.data, proj_integ)
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

g_gtn = vm_sys.fe.component   # grid_e.get_rand_state(64)
g = g_gtn.data

integ_gtn = None
for ax_v in ax_vs:
    print('ax v', ax_v)
    v2i_mps = grid_e.make_mps_ndim({ax_v: ax_v.get_xmultiply_mps(x_power=2)})
    v2_g = g_gtn.elemental_multiply(v2i_mps)
    if integ_gtn is None:
        integ_gtn = v2_g
    else:
        integ_gtn = integ_gtn.add(v2_g, inplace=True, compress=True)

# f = helper_quimb.add_MPS_target(f, integ_gtn.data, direction=-1, inplace=True, do_final_update=False)
proj_integ = helper_quimb.target_MPS(integ_gtn.data, g, site=0)

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
    expec = helper_quimb.expectation_value(g, mpo.data, proj_integ)
    print('expec', expec)