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
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_EM import Maxwell
from pde_vlasovEM import VlasovMaxwell
from vlasov_tests import VlasovTest


"""
"""

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
x_map, vmap = helper_test.get_maps(x_map_key + v_map_key)

ax_x = Axis(Lx, q, ax_map=x_map)
ax_y = Axis(Lx, q, ax_map=x_map)
grid_x = Grid1D('XY', [ax_x, ax_y], layout_type=TN_layout)

ddx = grid_x.get_firstderivative_mpo(ax_x)
ddy = grid_x.get_firstderivative_mpo(ax_y)

cur_orthog = 2
xvec = grid_x.get_rand_state(32)
xvec.canonize(i=cur_orthog)
Ax_ddx = xvec.apply(ddx)
Ax_ddy = xvec.apply(ddy)
proj_ddx = Ax_ddx.copy()
proj_ddx.canonize(i=cur_orthog)
proj_ddy = Ax_ddy.copy()
proj_ddy.canonize(i=cur_orthog)

proj_Axddx = proj_mps(Ax_ddx.data, proj_ddx.data, cur_orthog)
print('proj Ax_ddx distance', helper_quimb.distance(proj_Axddx, Ax_ddx.data))

proj_Axddy = proj_mps(Ax_ddy.data, proj_ddy.data, cur_orthog)
print('proj Ax_ddx distance', helper_quimb.distance(proj_Axddy, Ax_ddy.data))

## update xvec
dt = 10
xvec_ = xvec.copy()
rand_tens = xvec_.data[cur_orthog]
rand_tens.modify(apply= lambda x: x + dt * np.random.random(rand_tens.shape))

Ax_ddx_ = xvec_.apply(ddx)
Ax_ddy_ = xvec_.apply(ddy)

proj_Axddx_ = proj_mps(Ax_ddx_.data, proj_ddx.data, cur_orthog)
print('proj Ax_ddx distance', helper_quimb.distance(proj_Axddx_, Ax_ddx_.data))

proj_Axddy_ = proj_mps(Ax_ddy_.data, proj_ddy.data, cur_orthog)
print('proj Ax_ddx distance', helper_quimb.distance(proj_Axddy_, Ax_ddy_.data))

print('diff Ax, Ax new (should be nonzero)', helper_quimb.distance(proj_Axddx_, proj_Axddx))
print('diff Ay, Ay new (should be nonzero)', helper_quimb.distance(proj_Axddy_, proj_Axddy))

