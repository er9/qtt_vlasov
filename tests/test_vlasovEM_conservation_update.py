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

from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from axis import Axis
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_EM import Maxwell
from pde_vlasovEM import VlasovMaxwell
from vlasov_tests import VlasovTest


"""
"""

def proj_mps(mps, proj_mps, center, verbose=False):
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

    if verbose:
        print('envL', envL)
        print('envR', envR)

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

def get_eff_mpo(mpo :qtn.MatrixProductOperator, proj_mps: qtn.MatrixProductState, center):
    L_ = mpo.L
    mpo.mangle_inner_()
    proj_mps.site_ind_id = mpo.lower_ind_id
    bra_mps = proj_mps.copy()
    bra_mps.mangle_inner_(append='_')
    bra_mps.site_ind_id = mpo.upper_ind_id

    envL = None
    for ix in range(center):
        # print('left: ix', ix)
        xtens = [bra_mps[ix].conj(), mpo[ix], proj_mps[ix]]
        if envL is not None:
            xtens += [envL]
        envL = qtn.tensor_contract(*xtens)

    envR = None
    for ix in range(mpo.L - 1, center, -1):
        # print('right: ix', ix)
        xtens = [bra_mps[ix].conj(), mpo[ix], proj_mps[ix]]
        if envR is not None:
            xtens += [envR]
        envR = qtn.tensor_contract(*xtens)


    xtens = [mpo[center]]
    if envL is not None:
        xtens += [envL]
    if envR is not None:
        xtens += [envR]
    mtens = qtn.tensor_contract(*xtens)
    mtens.drop_tags()
    mtens.add_tag(mpo.site_tag(center))

    mtens.modify(apply=lambda x: x * 10**mpo.exponent)
    return mtens

def get_eff_Ax(x: qtn.MatrixProductState, mpo: qtn.MatrixProductOperator, mps_proj: qtn.MatrixProductState, center:int):
    ref_Ax = helper_quimb.apply(mpo, x)
    eff_op = get_eff_mpo(mpo, mps_proj, center)
    proj_x = proj_mps(x, mps_proj, center)
    print('proj x err', helper_quimb.distance(proj_x, x))
    proj_Ax = proj_mps(ref_Ax, mps_proj, center)
    print('proj Ax err', helper_quimb.distance(proj_Ax, ref_Ax))

    print('x center', proj_x[center])
    print('eff op', eff_op)
    eff_op.reindex({mpo.lower_ind(center): x.site_ind(center)}, inplace=True)
    eff_op_x = qtn.tensor_contract(proj_x[center], eff_op)
    print('eff op @ x', eff_op_x)
    hbonds = [ind for ind in eff_op_x.inds if ind != mpo.upper_ind(center)]
    eff_op_x.reindex({mpo.upper_ind(center): x.site_ind(center),
                      **{ind: ind[:-1] for ind in hbonds}}, inplace=True)
    print('eff op @ x', eff_op_x)
    eff_op_x.drop_tags()
    eff_op_x.add_tag(x.site_tag(center))

    eff_Ax_tensors = [proj_x[ix] for ix in range(center)] + [eff_op_x] + [proj_x[ix] for ix in range(center+1, x.L)]
    eff_Ax_mps = qtn.TensorNetwork(eff_Ax_tensors)
    eff_Ax_mps.view_like(x, inplace=True)
    eff_Ax_mps.exponent = x.exponent

    print('eff Ax err', helper_quimb.distance(eff_Ax_mps, ref_Ax))
    return eff_Ax_mps

###################
# grid parameters #
###################

q = 2
KX = 1
KV = 1
Lx = flags.get('Lx', 4)
Lve = flags.get('Lve', 4)
Lvi = flags.get('Lvi', 4)

do_tt = flags.get('do_tt', False)
do_full = flags.get('do_full', False)

if (do_full or do_tt):
    TN_layout = LayoutType.PARALLEL_GROUP if do_full else LayoutType.SEQUENTIAL
    x_map_key = 'F'  # 'F'
    v_map_key = 'F'  # 'M'
else:
    TN_layout = flags.get('layout', LayoutType.SEQUENTIAL)
    x_map_key = flags.get('x_map_key', 'F')
    v_map_key = flags.get('v_map_key', 'B')
x_map, v_map = helper_test.get_maps(x_map_key + v_map_key)

X = Coordinate('X', CoordinateType.X)
Y = Coordinate('Y', CoordinateType.Y)
ax_x = Axis(Lx, q, coordinate=X, ax_map=x_map)
ax_y = Axis(Lx, q, coordinate=Y, ax_map=x_map)
grid_x = Grid1D('XY', [ax_x, ax_y], layout_type=TN_layout)

ddx = grid_x.get_firstderivative_mpo(ax_x)
ddy = grid_x.get_firstderivative_mpo(ax_y)

cur_orthog = 2
direction = 1 if cur_orthog > grid_x.L // 2 else -1

xvec = grid_x.get_rand_state(32)
xvec.canonize(i=cur_orthog)
Ax_ddx = xvec.apply(ddx)
Ax_ddy = xvec.apply(ddy)
proj_ddx_mps = helper_quimb.add_MPS_target(Ax_ddx.data, xvec.data, do_final_update=False, direction=direction)
proj_ddx = grid_x.make_gridTN(proj_ddx_mps)
# proj_ddx = Ax_ddx.copy()
proj_ddx.canonize(i=cur_orthog)
xvec_x = grid_x.make_gridTN(proj_mps(xvec.data, proj_ddx.data, cur_orthog))
Ax_ddx_x = proj_mps(Ax_ddx.data, proj_ddx.data, cur_orthog)
print('(1) proj x distance', helper_quimb.distance(xvec_x.data, xvec.data))
print('(1) proj Ax_ddx distance', helper_quimb.distance(Ax_ddx_x, Ax_ddx.data))

proj_ddy_mps = helper_quimb.add_MPS_target(Ax_ddy.data, xvec.data.copy(), do_final_update=False, direction=direction)
proj_ddy = grid_x.make_gridTN(proj_ddy_mps)
# proj_ddy = Ax_ddy.copy()
proj_ddy.canonize(i=cur_orthog)
xvec_y = grid_x.make_gridTN(proj_mps(xvec.data, proj_ddy.data, cur_orthog))
Ax_ddy_y = proj_mps(Ax_ddy.data, proj_ddy.data, cur_orthog)
print('(1) proj y distance', helper_quimb.distance(xvec_y.data, xvec.data))
print('(1) proj Ax_ddy distance', helper_quimb.distance(Ax_ddy_y, Ax_ddy.data))

proj_Axddx = proj_mps(Ax_ddx.data, proj_ddx.data, cur_orthog)
print('(2) proj Ax_ddx distance', helper_quimb.distance(proj_Axddx, Ax_ddx.data))
proj_xvec = proj_mps(xvec_x.data, proj_ddx.data, cur_orthog)
print('(2) proj x distance', helper_quimb.distance(proj_xvec, xvec.data))

proj_Axddy = proj_mps(Ax_ddy.data, proj_ddy.data, cur_orthog)
print('(2) proj Ax_ddy distance', helper_quimb.distance(proj_Axddy, Ax_ddy.data))
proj_xvec = proj_mps(xvec_y.data, proj_ddy.data, cur_orthog)
print('(2) proj x distance', helper_quimb.distance(proj_xvec, xvec.data))

## update xvec
dt = 10
xvec_x_ = xvec_x.copy()
rand_tens = xvec_x_.data[cur_orthog]
rand_tens.modify(apply= lambda x: x + dt * np.random.random(rand_tens.shape))
Ax_ddx_ = xvec_x_.apply(ddx, zipup=False, compress=False)

xvec_y_ = xvec_y.copy()
rand_tens = xvec_y_.data[cur_orthog]
rand_tens.modify(apply= lambda x: x + dt * np.random.random(rand_tens.shape))
Ax_ddy_ = xvec_y_.apply(ddy, zipup=False, compress=False)

print('diff x (should be nonzero)', xvec_x.distance(xvec_x_))
print('diff x (should be nonzero)', xvec_y.distance(xvec_y_))
print('diff Ax (should be nonzero)', helper_quimb.distance(Ax_ddx_.data, Ax_ddx_x))
print('diff Ax (should be nonzero)', helper_quimb.distance(Ax_ddy_.data, Ax_ddy_y))

# A_eff = get_eff_mpo(ddx.data, proj_Axddx, cur_orthog)
Ax_eff = get_eff_Ax(xvec_x.data, ddx.data, proj_Axddx, cur_orthog )
Ax_eff = get_eff_Ax(xvec_x_.data, ddx.data, proj_Axddx, cur_orthog )
exit()


## use original projectors
helper_quimb.check_orthog(proj_ddx.data)
proj_Axddx_ = proj_mps(Ax_ddx_.data, proj_ddx.data, cur_orthog, verbose=True)
print('proj new Ax_ddx distance (should be zero)', helper_quimb.distance(proj_Axddx_, Ax_ddx_.data))
proj_xvec_x_ = proj_mps(xvec_x_.data, proj_ddx.data, cur_orthog)
print('proj new x distance (should be zero)', helper_quimb.distance(proj_xvec_x_, xvec_x_.data))

proj_Axddy_ = proj_mps(Ax_ddy_.data, proj_ddy.data, cur_orthog)
print('proj new Ax_ddx distance (should be zero)', helper_quimb.distance(proj_Axddy_, Ax_ddy_.data))
proj_xvec_y_ = proj_mps(xvec_y_.data, proj_ddy.data, cur_orthog)
print('proj new x distance (should be zero)', helper_quimb.distance(proj_xvec_y_, xvec_y_.data))

## compare original and new projected result
print('diff Ax, Ax new (should be nonzero)', helper_quimb.distance(proj_Axddx_, proj_Axddx))
print('diff Ay, Ay new (should be nonzero)', helper_quimb.distance(proj_Axddy_, proj_Axddy))


## 1-site canonicalization to next site (with basis expansion)
