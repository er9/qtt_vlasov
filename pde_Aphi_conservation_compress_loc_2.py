import pdb

import numpy as np
import os, sys, time, glob
sys.path.append('/')

from setup_.paths import save_dir, main_dir
from setup_.configs import *
import setup_test as test_setup

import helper_quimb
from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
import setup_.helper as helper_test
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
# from pde_vlasovEM import VlasovMaxwell
from pde_Aphi import VectorPotential


import helper_quadratic
# from local_solvers.blocks import Block, BlockVector_DMRG, BlockOperator_DMRG
from local_solvers.blocks_quad import BlockQuad

""" 
conservation at coarse grid.
optimize entire tensor instead of just singular values
"""



def E_nrg_blocks(coord_sys: CartesianCoordinateSpace, compID, num_coarse: int, dAdt: Field, phi: ScalarField,
                 cur_orthog: int):
    """ num_coarse: number of coarse grid cells per dimension
        Ei = < d/dxi phi - dAdt_i | d/dxi phi - dAdt_I >
    """
    ax = coord_sys.get_axis(compID)
    if ax is not None:
        ddx = phi.grid.get_firstderivative_mpo(ax).data
        # d2dx2 = helper_quimb.apply( helper_quimb.mpo_conj_transpose(ddx), ddx)

    # assert (num_coarse > 1), f'num coarse must be greater than 1, currently {num_coarse}'


    dAdt_data = dAdt[compID].data if compID in dAdt.components else None
    phi_data = phi.component.data
    ix = compID.type.value
    grid = dAdt.grid
    elem_mult: 'MatrixProductTensor' = grid.get_elemental_multiply_tn().data
    integ = grid.get_finescale_integrator(num_coarse).data

    dx = np.prod([ax.dx for ax in grid.axes])
    integ = helper_quimb.scalar_multiply(integ, 1./dx)
    # print('integ', integ)

    blocks = {}
    if dAdt_data is not None:
        blocks[(3 + ix, 3 + ix)] = [BlockQuad(dAdt_data, dAdt_data, integ, elem_mult, cur_orthog=cur_orthog)]
        # print('E block', blocks[(3 + ix, 3 + ix)][-1].norm())
        if phi_data is not None:
            ddx = ddx.scalar_multiply(-2)
            quad_op = helper_quadratic.make_quadratic_op(elem_mult,None, ddx.conj())
            blocks[(3 + ix, 7)] = [BlockQuad(dAdt_data, phi_data, integ, quad_op, cur_orthog=cur_orthog)]
    if phi_data is not None:
        quad_op = helper_quadratic.make_quadratic_op(elem_mult, ddx, ddx.conj())
        blocks[(7, 7)] = [BlockQuad(phi_data, phi_data, integ, quad_op, cur_orthog=cur_orthog)]
        # blocks[(7,7)] = [BlockOperator_DMRG(phi_data, phi_data, d2dx2, cur_orthog=cur_orthog)]
    return blocks

def B_nrg_blocks(coord_sys: CartesianCoordinateSpace, compID, num_coarse: int, A: Field,
                 cur_orthog: int):
    """ Bi = < curl(A)_i | curl(A)_i >
        Bx:  | d/dy Az - d/dz Ay |^2
        By:  | d/dz Ax - d/dx Az |^2
        Bz:  | d/dx Ay - d/dy Ax |^2
    """
    i = compID.type  # self.ax_coords[axID]
    i1, i2 = (i + 1) % 3, (i + 2) % 3

    c1_ = coord_sys.type_coords.get(i1, None)
    c2_ = coord_sys.type_coords.get(i2, None)

    grid = A.grid
    elem_mult: 'MatrixProductTensor' = grid.get_elemental_multiply_tn().data
    integ = grid.get_finescale_integrator(num_coarse).data

    dx = np.prod([ax.dx for ax in grid.axes])
    integ = helper_quimb.scalar_multiply(integ, 1. / dx)

    ax1_ = coord_sys.get_axis(c1_)
    vec1 = A[c2_]  # [ax2_.axID]
    vec1_data = None if vec1 is None or vec1.data is None else vec1.data
    ddx1 = A.grid.get_firstderivative_mpo(ax1_).data if ax1_ is not None else None
    # print('ddx1', c2_, ax1_)

    ax2_ = coord_sys.get_axis(c2_)
    vec2 = A[c1_]
    vec2_data = None if vec2 is None or vec2.data is None else vec2.data
    ddx2 = A.grid.get_firstderivative_mpo(ax2_).data if ax2_ is not None else None
    # print('ddx2', c1_, ax2_)

    blocks = {}
    if vec1_data is not None:
        if ddx1 is not None:
            # ddx1_2 = helper_quimb.apply(helper_quimb.mpo_conj_transpose(ddx1), ddx1)

            quad_op = helper_quadratic.make_quadratic_op(elem_mult, ddx1, ddx1.conj(),)

            # ax_y = grid.axes[1]
            # print('ddx1', ddx1.exponent, quad_op.exponent)
            # print('integ exponent', integ.exponent, elem_mult.exponent)
            # tmp = helper_quimb.apply(helper_quimb.mpo_conj_transpose(ddx1), ddx1)
            # print('ddx @ ddx norm', helper_quimb.norm(tmp), tmp.exponent)
            #
            # # ddx1_gtn = grid.make_gridTN(ddx1)
            # # print('ddx1', ddx1_gtn.get_data(ax_select={ax_y: (1, 1)})[:5, :5])
            # # print('ddx1', ddx1_gtn.get_data(ax_select={ax_y: (1, 2)})[:5, :5])
            # # print('ddx1', ddx1_gtn.get_data(ax_select={ax_y: (2, 1)})[:5, :5])
            # # print('ddx1', ddx1_gtn.get_data(ax_select={ax_y: (2, 2)})[:5, :5])
            #
            # tmp_gtn = grid.make_gridTN(tmp)
            # print('tmp', tmp_gtn.get_data(ax_select={ax_y: (0, 0)})[:5, :5])
            # print('tmp', tmp_gtn.get_data(ax_select={ax_y: (0, 1)})[:5, :5])
            # print('tmp', tmp_gtn.get_data(ax_select={ax_y: (1, 3)})[:5, :5])
            # print('tmp', tmp_gtn.get_data(ax_select={ax_y: (2, 2)})[:5, :5])
            # # print('quad op', quad_op)
            # # print('integ', integ)
            #
            #
            # tmp_integ = integ.copy()
            # tmp_integ.site_ind_id = 'o({})'
            #
            # out_tens = [qtn.tensor_contract(tmp_integ[i], quad_op[i]) for i in range(tmp_integ.L)]
            # out_tmp = qtn.TensorNetwork(out_tens)
            # out_tmp.exponent = tmp_integ.exponent + quad_op.exponent
            # out_tmp.view_as(qtn.MatrixProductOperator, L=tmp_integ.L, cyclic=False, inplace=True,
            #                 lower_ind_id='i({})[1]', upper_ind_id='i({})[2]', site_tag_id='T({})')
            # out_tmp.fuse_multibonds(inplace=True)
            # # print('out tmp', out_tmp)
            # print('out norm', helper_quimb.norm(out_tmp), tmp_integ.exponent, quad_op.exponent)
            # print('integ norm', helper_quimb.norm(quad_op))
            # out_tmp_gtn = grid.make_gridTN(out_tmp)
            # print('out tmp', out_tmp_gtn.get_data(ax_select={ax_y: (0, 0)})[:5, :5])
            # print('out tmp', out_tmp_gtn.get_data(ax_select={ax_y: (0, 1)})[:5, :5])
            # print('out tmp', out_tmp_gtn.get_data(ax_select={ax_y: (1, 3)})[:5, :5])
            # print('out tmp', out_tmp_gtn.get_data(ax_select={ax_y: (2, 2)})[:5, :5])
            # pdb.set_trace()
            #
            blocks[(i2, i2)] = [BlockQuad(vec1_data, vec1_data, integ, quad_op, cur_orthog=cur_orthog)]
            # print('block norm', i2, blocks[(i2, i2)][-1].norm())
            # print('B block i2', helper_quimb.norm(vec1_data))
    if vec2_data is not None:
        if ddx2 is not None:
            # tmp = helper_quimb.apply(ddx2, vec2_data)
            # print('tmp fro norm', helper_quimb.norm(tmp))
            # ddx2_2 = helper_quimb.apply(helper_quimb.mpo_conj_transpose(ddx2), ddx2)
            quad_op = helper_quadratic.make_quadratic_op(elem_mult, ddx2, ddx2.conj())
            blocks[(i1, i1)] = [BlockQuad(vec2_data, vec2_data, integ, quad_op, cur_orthog=cur_orthog)]
            # print('block norm', i1, blocks[(i1, i1)][-1].norm())
            # print('B block i1', helper_quimb.norm(vec2_data))
    if vec1_data is not None and vec2_data is not None:
        if ddx1 is not None and ddx2 is not None:
            # op = helper_quimb.apply(helper_quimb.mpo_conj_transpose(ddx1), ddx2)
            quad_op = helper_quadratic.make_quadratic_op(elem_mult, ddx2, ddx1.conj())
            helper_quimb.scalar_multiply(quad_op, -2, inplace=True)
            blocks[(i1, i2)] = [BlockQuad(vec2_data, vec1_data, integ, quad_op, cur_orthog=cur_orthog)]
            # blocks[(i1,i2)] = [BlockOperator_DMRG(vec2_data, vec1_data, op, cur_orthog=cur_orthog)]

    return blocks

def momentum_blocks(coord_sys: CartesianCoordinateSpace,
                    compID, num_coarse: int, A: Field, dAdt: Field, phi: ScalarField,
                    cur_orthog: int):
    """ integ (E x B)_i dx
        px:  <Ey | Bz> - <Ez | By>
        py:  <Ez | Bx> - <Ex | Bz>
        pz:  <Ex | By> - <Ey | Bx>

        Bx: d/dy Az - d/dz Ay
        By: d/dz Ax - d/dx Az
        Bz: d/dx Ay - d/dy Ax
    """
    i = compID.type  # self.ax_coords[axID]
    i1, i2 = (i + 1) % 3, (i + 2) % 3

    grid = A.grid
    elem_mult: 'MatrixProductTensor' = grid.get_elemental_multiply_tn().data
    integ = grid.get_finescale_integrator(num_coarse).data

    dx = np.prod([ax.dx for ax in grid.axes])
    integ = helper_quimb.scalar_multiply(integ, 1. / dx)

    c1_ = coord_sys.type_coords.get(i1, None)
    c2_ = coord_sys.type_coords.get(i2, None)
    # print('i0', i, 'i1', i1, 'i2', i2)
    # print('c0', compID, 'c1', c1_, 'c2', c2_)

    coords = coord_sys.coords
    axes = [coord_sys.get_axis(compID) for compID in coords]   # A.grid.axes + (None,)*(3-A.grid.ndim)
    ddxs = [A.grid.get_firstderivative_mpo(ax).data if ax is not None else None for ax in axes]

    phi_data = phi.component.data

    ## Bz = d/dx Ay - d/dy Ax
    ## Bx = d/dy Az - d/dz Ay
    ## By = d/dz Ax - d/dx Az
    k1, k2 = (i2 + 1) % 3, (i2 + 2) % 3
    op1 = ddxs[k1] if ddxs[k1] is not None else None
    A1 = A[coords[k2]] if coords[k2] is not None else None
    op2 = helper_quimb.scalar_multiply(ddxs[k2], -1) if ddxs[k2] is not None else None
    A2 = A[coords[k1]] if coords[k1] is not None else None

    ## Ey = d/dy phi - dA/dt_y
    op3 = ddxs[c1_.type.value]
    dAdt_4 = dAdt[c1_]

    blocks = {}
    if A1 is not None and A1.data is not None:
        if phi_data is not None and op3 is not None:
            # op31 = helper_quimb.apply( helper_quimb.mpo_conj_transpose(op3), op1)
            op31 = helper_quadratic.make_quadratic_op(elem_mult, op1, op3.conj())
            blocks[(7, coords[k2].type.value)] = [BlockQuad(A1.data, phi_data, integ, op31, cur_orthog=cur_orthog)]
            # blocks[(7, coords[k2].type.value)] = [BlockOperator_DMRG(A1.data, phi_data, op31, cur_orthog=cur_orthog)]
        if dAdt_4 is not None and dAdt_4.data is not None and op1 is not None:
            # op41 = helper_quimb.scalar_multiply(op1, -1, inplace=False)
            op41 = helper_quadratic.make_quadratic_op(elem_mult, op1, None)
            op41 = helper_quimb.scalar_multiply(op41, -1, inplace=False)
            blocks[(c1_.type.value + 3, coords[k2].type.value)] = [BlockQuad(A1.data, dAdt_4.data, integ, op41,
                                                                             cur_orthog=cur_orthog)]
            # blocks[(c1_.type.value + 3, coords[k2].type.value)] = [BlockOperator_DMRG(A1.data, dAdt_4.data, op41,
            #                                                                cur_orthog=cur_orthog)]

    if A2 is not None and A2.data is not None:
        if phi_data is not None and op3 is not None:
            # op32 = helper_quimb.apply(helper_quimb.mpo_conj_transpose(op3), op2)
            op32 = helper_quadratic.make_quadratic_op(elem_mult, op2, op3.conj())
            helper_quimb.scalar_multiply(op32, -1, inplace=True)
            blocks[(7, coords[k1].type.value)] = [BlockQuad(A2.data, phi_data, integ, op32, cur_orthog=cur_orthog)]
            # blocks[(7, coords[k1].type.value)] = [BlockOperator_DMRG(A2.data, phi_data, op32, cur_orthog=cur_orthog)]
        if dAdt_4 is not None and dAdt_4.data is not None and op2 is not None:
            # op42 = op2.copy()
            op42 = helper_quadratic.make_quadratic_op(elem_mult, op2,None)
            blocks[(c1_.type.value + 3, coords[k1].type.value)] = [BlockQuad(A2.data, dAdt_4.data, integ, op42,
                                                                            cur_orthog=cur_orthog)]
            # blocks[(c1_.type.value + 3, coords[k1].type.value)] = [BlockOperator_DMRG(A2.data, dAdt_4.data, op42,
            #                                                                 cur_orthog=cur_orthog)]

    ## By = d/dz Ax - d/dx Az
    ## Bz = d/dx Ay - d/dy Ax
    ## Bx = d/dy Az - d/dz Ay
    j1, j2 = (i1 + 1) % 3, (i1 + 2) % 3
    op1 = ddxs[j1] if ddxs[j1] is not None else None
    A1 = A[coords[j2]] if coords[j2] is not None else None
    op2 = helper_quimb.scalar_multiply(ddxs[j2], -1) if ddxs[j2] is not None else None
    A2 = A[coords[j1]] if coords[j1] is not None else None

    ## Ez = d/dz phi - dA/dt_z
    op3 = ddxs[c2_.type.value]
    dAdt_4 = dAdt[c2_]

    if A1 is not None and A1.data is not None:
        if phi_data is not None and op3 is not None:
            # op31 = helper_quimb.apply(helper_quimb.mpo_conj_transpose(op3), op1)
            op31 = helper_quadratic.make_quadratic_op(elem_mult, op1, op3.conj())
            key = (7, coords[j2].type.value)
            # blocks[key] = blocks.get(key, []) + [BlockOperator_DMRG(A1.data, phi_data, op31, cur_orthog=cur_orthog)]
            blocks[key] = blocks.get(key, []) + [BlockQuad(A1.data, phi_data, integ, op31, cur_orthog=cur_orthog)]
        if dAdt_4 is not None and dAdt_4.data is not None and op1 is not None:
            # op41 = helper_quimb.scalar_multiply(op1, -1, inplace=False)
            op41 = helper_quadratic.make_quadratic_op(elem_mult, op1, None)
            helper_quimb.scalar_multiply(op41, -1, inplace=True)
            key = (c2_.type.value + 3, coords[j2].type.value)
            blocks[key] = blocks.get(key, []) + [BlockQuad(A1.data, dAdt_4.data, integ, op41, cur_orthog=cur_orthog)]
            # blocks[key] = blocks.get(key, []) + [BlockOperator_DMRG(A1.data, dAdt_4.data, op41, cur_orthog=cur_orthog)]

    if A2 is not None and A2.data is not None:
        if phi_data is not None and op3 is not None:
            # op32 = helper_quimb.apply(helper_quimb.mpo_conj_transpose(op3), op2)
            # op32.scalar_multiply(-1, inplace=True)
            op32 = helper_quadratic.make_quadratic_op(elem_mult, op2, op3.conj())
            helper_quimb.scalar_multiply(op32, -1, inplace=True)
            key = (7, coords[j1].type.value)
            blocks[key] = blocks.get(key, []) + [BlockQuad(A2.data, phi_data, integ, op32, cur_orthog=cur_orthog)]
            # blocks[key] = blocks.get(key, []) + [BlockOperator_DMRG(A2.data, phi_data, op32, cur_orthog=cur_orthog)]
        if dAdt_4 is not None and dAdt_4.data is not None and op2 is not None:
            # op42 = op2.copy()
            op42 = helper_quadratic.make_quadratic_op(elem_mult, op2, None)
            key = (c2_.type.value + 3, coords[j1].type.value)
            blocks[key] = blocks.get(key, []) + [BlockQuad(A2.data, dAdt_4.data, integ, op42, cur_orthog=cur_orthog)]
            # blocks[key] = blocks.get(key, []) + [BlockOperator_DMRG(A2.data, dAdt_4.data, op42, cur_orthog=cur_orthog)]

    return blocks


def measure_errors(block_dict: dict[tuple[int,int], list[BlockQuad]], num_opt: int,
                   base_dict: dict[int, tuple[qtn.Tensor, qtn.Tensor]],
                   errs_dict: dict[int, tuple[qtn.Tensor, qtn.Tensor]],
                   cur_orthog: int):
    """
    returns (array, array), (tens, tens)
    obs = sum_ij < vec_i | A | vec_j > = sum_ij < vec_i + err_i | A | vec_j + err_j >
    = sum_ij < vec_i | A | vec_j > + < vec_i | A | err_j > + < err_i | A | vec_j > + < err_i | A | err_j >
    """
    print('measure num opt', num_opt)

    expec_dict = {}
    expec_weights = {}  ## expectation values of overlaps w/o weights

    # for k, tens in block_dict.items():
    #     print('block dict', k, tens)
    #
    # for k, tens in base_dict.items():
    #     print('base dict', k, tens)
    #
    # for k, tens in errs_dict.items():
    #     print('errs dict', k, tens)

    base_tens_dict = {k: qtn.tensor_contract(*tens) for k, tens in base_dict.items() if tens[0] is not None}
    # print('base tens dict', base_tens_dict)

    # try:
    #     base_tens_dict = {k: qtn.tensor_contract(*tens) for k, tens in base_dict.items() if tens[0] is not None}
    #     print('base tens dict', base_tens_dict)
    # except AttributeError:
    #     return (0.0, 0.0), {}

    for k, blocks in block_dict.items():
        # print('k blocks', k)
        bra_key, ket_key = k
        bra_errs = errs_dict.get(bra_key, (None, None))
        ket_errs = errs_dict.get(ket_key, (None, None))
        tot_expec_base, tot_expec_errs = 0.0, 0.0

        tot_bra_err, tot_ket_err, tot_bk_err = None, None, None

        for block in blocks:
            # print('block', block, 'expec', block.norm())

            integ = block.bra
            coarse_inds = tuple()      ## if empty, output of block is a scalar
            if isinstance(integ, qtn.MatrixProductOperator):
                all_inds = [integ.upper_ind(ix) for ix in range(integ.L)]
                coarse_inds = [ind + '_' for ix, ind in enumerate(all_inds) if (ind in integ[ix].inds)]
                coarse_inds = tuple(coarse_inds)

            # print('coarse inds', coarse_inds)

            ket_base_tens = base_tens_dict[ket_key]
            bra_base_tens = base_tens_dict[bra_key]

            ## expectation value at all coarse grid cells
            expec_base = block.compute_expec_i(cur_orthog, ket_base_tens, bra_base_tens)
            # if isinstance(expec_base, qtn.Tensor):
            #     expec_base = expec_base.fuse(coarse_inds)

            ## error terms + residual parameters
            if isinstance(expec_base, qtn.Tensor):
                expec_errs = qtn.Tensor(np.zeros_like(expec_base.data), inds=expec_base.inds)
            else:
                expec_errs = 0.0
            ket_err_tens, bra_err_tens = None, None
            ket_err_T1, ket_err_T2 = ket_errs
            bra_err_T1, bra_err_T2 = bra_errs

            # print('ket err T1', ket_err_T1, ket_err_T2)
            # print('bra err T1', bra_err_T1, ket_err_T2)

            if ket_err_T1 is not None:
                ket_err_tens = qtn.tensor_contract(ket_err_T1, ket_err_T2)

                # print('ket errs', ket_errs)
                # print('ket err T1', ket_err_T1, ket_err_T2)

                kT1_, kT2_ = ket_err_T1.copy(), ket_err_T2.copy()
                assert kT1_.inds[-1] == '__tmp__'
                kT1_.modify(data=kT1_.data[..., :num_opt])
                assert kT2_.inds[0] == '__tmp__'
                kT2_.modify(data=kT2_.data[:num_opt, ...])

            if bra_err_T1 is not None:
                bra_err_tens = qtn.tensor_contract(*bra_errs)
                bra_err_T1 = bra_err_T1.reindex({'__tmp__': '__tmp2__'})
                bra_err_T2 = bra_err_T2.reindex({'__tmp__': '__tmp2__'})

                # k1_to_k2_inds = block.get_ket2_inds(cur_orthog)
                # bra_err_T1 = bra_err_T1.reindex(k1_to_k2_inds)
                # bra_err_T2 = bra_err_T2.reindex(k1_to_k2_inds)

                bT1_, bT2_ = bra_err_T1.copy(), bra_err_T2.copy()
                assert bT1_.inds[-1] == '__tmp2__'
                # bT1_.modify(data=bT1_.data[..., :num_opt])
                assert bT2_.inds[0] == '__tmp2__'
                # bT2_.modify(data=bT2_.data[:num_opt, ...])

            if ket_err_T1 is not None:
                # print('ket err T1 is not None')
                # expec_no_T2 = block.compute_expec_i(cur_orthog, ket_err_T1, bra_base_tens)
                # errs = qtn.tensor_contract(expec_no_T2, ket_err_T2)
                # expec_no_T2_red = expec_no_T2.transpose(*coarse_inds, *ket_err_T2.inds)
                # expec_no_T2_red.modify(apply=lambda x: x[..., :num_opt])
                # print('expec no T2 red', expec_no_T2_red)

                errs = block.compute_expec_i(cur_orthog, ket_err_tens, bra_base_tens)
                expec_errs = expec_errs + errs
                # print('expec errs', expec_errs)

                expec_no_T2_red = block.compute_expec_i(cur_orthog, kT1_, bra_base_tens)
                expec_no_T2_red = expec_no_T2_red.transpose(*coarse_inds, *ket_err_T2.inds)
                # print('expec no T2 red', expec_no_T2_red)

                if tot_ket_err is None:
                    tot_ket_err = expec_no_T2_red
                else:
                    tot_ket_err += expec_no_T2_red

            if bra_err_T1 is not None:
                # expec_no_T2 = block.compute_expec_i(cur_orthog, ket_base_tens, bra_err_T1)
                # ## Tensor inds: (__tmp2__, hbond(i, i+1), coarse_inds)
                # errs = qtn.tensor_contract(expec_no_T2, bra_err_T2)
                # expec_errs = expec_errs + errs
                #
                # expec_no_T2_red = expec_no_T2.transpose(*coarse_inds, *bra_err_T2.inds)
                # expec_no_T2_red.modify(apply=lambda x: x[...,:num_opt, :])

                errs = block.compute_expec_i(cur_orthog, ket_base_tens, bra_err_tens)
                expec_errs = expec_errs + errs
                # print('expec errs', expec_errs)

                expec_no_T2_red = block.compute_expec_i(cur_orthog, ket_base_tens, bT1_)
                expec_no_T2_red = expec_no_T2_red.transpose(*coarse_inds,
                                                            *[ind + '_[2]' for ind in bra_err_T2.inds])
                # print('expec no T2 red', expec_no_T2_red)

                if tot_bra_err is None:
                    tot_bra_err = expec_no_T2_red
                else:
                    tot_bra_err += expec_no_T2_red

            if ket_err_tens is not None and bra_err_tens is not None:
                # expec_bk_no_T2 = block.compute_expec_i(cur_orthog, ket_err_T1, bra_err_T1)
                # errs = qtn.tensor_contract(expec_bk_no_T2, bra_err_T2, ket_err_T2)
                # expec_errs = expec_errs + errs
                #
                # # ind_k1, ind_k2 = ket_err_T2.inds    ## __tmp__, hbond
                # # ind_b1, ind_b2 = bra_err_T2.inds    ## __tmp2__, hbond
                # expec_bk_no_T2_red = expec_bk_no_T2.transpose(coarse_inds + bra_err_T2.inds + ket_err_T2.inds)
                # expec_bk_no_T2_red.modify(apply=lambda x: x[..., :num_opt, :, :num_opt, :])

                errs = block.compute_expec_i(cur_orthog, ket_err_tens, bra_err_tens)
                expec_errs = expec_errs + errs
                # print('expec errs', expec_errs)

                # print('bT1', bT1_, bT1_.data)
                # print('kT1', kT1_, kT1_.data)

                # print('get expec bk')
                expec_bk_no_T2_red = block.compute_expec_i(cur_orthog, kT1_, bT1_)
                expec_bk_no_T2_red = expec_bk_no_T2_red.transpose(*coarse_inds,
                                                                  *[ind + '_[2]' for ind in bra_err_T2.inds],
                                                                  *ket_err_T2.inds)
                # print('expec bk no T2 red', expec_bk_no_T2_red.data)

                if tot_bk_err is None:
                    tot_bk_err = expec_bk_no_T2_red
                else:
                    tot_bk_err += expec_bk_no_T2_red

                # print('exoec bk no T2 red', expec_bk_no_T2_red)
                # print('tot bk err', tot_bk_err)

            # print('expec base', expec_base, coarse_inds)
            if isinstance(expec_base, qtn.Tensor):
                expec_base.transpose(*coarse_inds, inplace=True)
                expec_errs.transpose(*coarse_inds, inplace=True)
                tot_expec_base = tot_expec_base + expec_base.data.reshape(-1)
                tot_expec_errs = tot_expec_errs + expec_errs.data.reshape(-1)
            else:
                tot_expec_base = tot_expec_base + expec_base
                tot_expec_errs = tot_expec_errs + expec_errs

        # print('k', k, tot_bk_err)
        expec_weights[k] = (tot_bra_err, tot_ket_err, tot_bk_err)
        expec_dict[k] = (tot_expec_base, tot_expec_errs)

    # print('expec dict', expec_dict)
    # tot_expec = np.sum([v[0] for v in expec_dict.values()])
    # tot_err = np.sum([v[1] for v in expec_dict.values()])
    tot_expec = [v[0] for v in expec_dict.values()]
    tot_err = [v[1] for v in expec_dict.values()]

    # print('tot expec', tot_expec)
    # print('tot err', tot_err)
    if len(tot_err) == 0:
        tot_err = 0.0
    else:
        tot_err = np.sum(np.array(tot_err), axis=0)

    if len(tot_expec) == 0:
        tot_expec = 0.0
    else:
        tot_expec = np.sum(np.array(tot_expec), axis=0)

    print('expec', tot_expec, tot_err, 'total', tot_expec + tot_err)

    return (tot_expec, tot_err), expec_weights



def get_cost_func(red_shapes: dict[int, tuple[int, int]], num_coarse: int, num_r: int,
                  target_val:Numeric,
                  *expec_weights: dict[tuple[int,int], tuple[qtn.Tensor, qtn.Tensor, qtn.Tensor]]):
    """
    cost func, jacobian, hessian for each QoI constraint
    ultimately want to solve a nonlinear least-squares problem

    cost: |expec(errs, errs) @ (coeffs, coeffs) - expec_err|^2  for each expectation value
          \sum_i (expec_i(c,c) expec_i(c,c) - 2 err_i expec_i(c,c)  + err_i**2)

    jac: d/dci ( expec(c)^2 - 2 err_j expec(c) + err^2 )
         = 2 expec(c) d/dci expec(c) - 2 expec_err d/dci expec(c)
         where d/dci expec(c) = d/dci ( c.T @ mat @ c + vec1 @ c + c.T @ vec2)
                = 2 mat_ii c_i + mat_ik c_k + vec1_k + vec2_k
    hess: d/dci d/dcj ()
         = 2 expec_i(c) d/dci d/dcj expec_i(c) + 2 d/dcj expec_i(c) d/dci expec_i(c)
           - 2 expec_err d/dci d/dcj expec_i(c)
    where d/dci d/dcj expec_i(c) = 2 mat_ii  if i == j else mat_ij

    balance between (1) choosing expectations with largest value
    well, it's hard to determine across all constraints
    and (2) choosing contributions that have the largest weights
    (meaning that they normally have some contribution).
    do (2) first choose contributions of comparable weight,
    also weights must be positive? Are negative weights ok?
    """
    len_coeffs_red = {k: min(num_r, sh[0]) * np.prod(sh[1:], dtype=int) for k, sh in red_shapes.items()}
    tot_len_coeffs = sum(len_coeffs_red.values())

    # print('len coeffs red', len_coeffs_red)
    # print('target val', target_val)

    bk_mat = np.zeros((num_coarse, tot_len_coeffs, tot_len_coeffs))
    k_vec = np.zeros((num_coarse, tot_len_coeffs))
    b_vec = np.zeros((num_coarse, tot_len_coeffs))

    # it_k, it_b, it_bk = 0, 0, 0
    for err_dict in expec_weights:
        for k, val in err_dict.items():
            # print('err dict items', k, val)
            bra_key, ket_key = k
            bra_errs, ket_errs, bk_errs = val

            if bra_errs is not None:

                bra_data = bra_errs.data[..., :num_r, :]

                num_rb = bra_data.shape[-2] * bra_data.shape[-1]
                xb = np.sum([len_coeffs_red.get(ix,0) for ix in range(bra_key)])
                # print('red shapes', red_shapes)
                # print('len coeffs red', len_coeffs_red)
                # print('num rb', xb, num_rb, num_coarse)
                # print('bvec shape', b_vec[:, xb: xb + num_rb].shape)
                b_vec[:, xb: xb + num_rb] = bra_data.reshape(num_coarse, num_rb)
                # if len(bra_errs) > 0:  it_b += 1

            if ket_errs is not None:

                ket_data = ket_errs.data[..., :num_r, :]

                num_rk = np.prod(ket_data.shape[-2:])
                xk = np.sum([len_coeffs_red.get(ix,0) for ix in range(ket_key)])
                k_vec[:, xk: xk + num_rk] = ket_data.reshape(num_coarse, num_rk)
                # if len(ket_errs) > 0:  it_k += 1

            if bk_errs is not None:

                bk_data = bk_errs.data[..., :num_r, :, :num_r, :]

                # print('xb', xb, xb + num_rb)
                # print('xk', xk, xk + num_rk)
                bk_mat[:, xb: xb + num_rb, xk: xk + num_rk] = bk_data.reshape(num_coarse, num_rb, num_rk)
                # if len(bk_errs) > 0:  it_bk += 1

    def cost(weights: np.ndarray):
        if (np.linalg.norm(target_val) == 0. and np.linalg.norm(bk_mat) == 0.
                and np.linalg.norm(b_vec) == 0. and np.linalg.norm(k_vec) == 0):
            return np.zeros(num_coarse)
        # return (weights @ bk_mat @ weights + b_vec @ weights + k_vec @ weights - target_val) / target_val
        # print('bk_mat', bk_mat.shape, weights.shape, k_vec.shape, b_vec.shape)
        out = (np.einsum('cab,a,b->c',bk_mat, weights, weights) + np.einsum('cb,b->c', k_vec, weights) +
               np.einsum('ca,a->c', b_vec, weights) - target_val)
        # print('out', out, target_val)
        # print('out', out / target_val)
        # if np.linalg.norm(out) < 1.0e-25:
        #     return np.zeros(num_coarse)
        return out / target_val

    def jac(weights: np.ndarray):
        """
        d/dci ( expec(c)^2 - 2 err expec(c) + err^2 )
        = 2 expec(c) d/dci expec(c) - 2 expec_err d/dci expec(c)
        where d/dci expec(c) = d/dci ( c.T @ mat @ c + vec1 @ c + c.T @ vec2)
                = 2 mat_ii c_i + mat_ik c_k + vec1_k + vec2_k

        d/dci (c A c + b c - d) = [2 A_ii c_i, A_ij c_j, ...] + b
        d/dci (c A c + b c - d)^2 = 2 ([cost]) d/dci (cAc + bc - d)
            = 2 ([cost]) ([2 A_ii c_i, A_ij c_j,...] + b)
        """
        if (np.linalg.norm(target_val) == 0. and np.linalg.norm(bk_mat) == 0.
                and np.linalg.norm(b_vec) == 0. and np.linalg.norm(k_vec) == 0):
            return np.zeros((num_coarse, tot_len_coeffs))

        ## diagonal terms are weighted by 2
        n1, n2, n3 = bk_mat.shape
        tmp = np.tensordot(np.ones(n1), (np.ones((n2,n2)) + np.eye(n2)), axes=0)
        # print('tmp', tmp.shape, bk_mat.shape)
        bk_mat_2 = bk_mat * tmp

        # print('bk mat 2', bk_mat.shape, bk_mat_2.shape, weights.shape)
        ddc_expec = np.einsum('cab,a->cb',bk_mat_2, weights) + b_vec + k_vec
        return ddc_expec / np.outer(target_val, np.ones(n2))

    # def hess(weights: np.ndarray):
    #     """
    #     d/dci d/dcj ()
    #     = 2 expec_i(c) d/dci d/dcj expec_i(c) + 2 d/dcj expec_i(c) d/dci expec_i(c)
    #        - 2 expec_err d/dci d/dcj expec_i(c)
    #     where d/dci d/dcj expec_i(c) = 2 mat_ii  if i == j else mat_ij
    #     """
    #     bk_mat_2 = bk_mat + np.diag(np.diag(bk_mat))  ## diagonal terms are weighted by 2
    #     expec = weights @ bk_mat @ weights
    #     ddc_expec = bk_mat_2 @ weights + b_vec + k_vec
    #     d2dc2_expec = bk_mat_2
    #     return 2 * expec * d2dc2_expec + 2 * np.outer(ddc_expec, ddc_expec) - 2 * target_val * d2dc2_expec

    return cost, jac


def compress_i_remainder(mps: 'qtn.MatrixProductState', cur_orthog:int, direction: int,
                         max_rank: int = None, cutoff: Numeric=CUTOFF, inplace=False):
    """ compress the MPS at site i and also return the remainder
        returns: compressed mps, remaining basis functions and weights
    """
    if mps is None:
        return None, None, # None

    tens = mps[cur_orthog]
    # new_mps = mps if inplace else mps.copy()
    # print('old mps', new_mps[cur_orthog: cur_orthog + 2])

    if direction > 0:  ## sweep left to right
        right_inds, left_inds = mps[cur_orthog].filter_bonds(mps[cur_orthog+1])
        TL, TS, TR = qtn.tensor_split(tens, left_inds, get='tensors', absorb=None, cutoff=CUTOFF,
                                      bond_ind='__tmp__')

        r = len(TS.data)
        if max_rank is None:
            s2_sum = np.cumsum(TS.data[::-1]**2)
            # print('s2 sum', s2_sum / s2_sum[-1])
            norm = s2_sum[-1]
            # cut_ind = np.count_nonzero(s2_sum/norm > cutoff)
            cut_ind = r - np.argmax(s2_sum/norm > cutoff)
            # print('cut ind', cut_ind)
            # cut_ind = np.count_nonzero(np.abs(TS.data)**2 > cutoff)
            cut_ind = max(min(r,MINBOND), cut_ind)
            # print('cut ind with min bond', cut_ind)
        else:
            cut_ind = max_rank

        print('error svals', TS.data[cut_ind:])

        TL.transpose(*left_inds, '__tmp__', inplace=True)
        TR.transpose('__tmp__', *right_inds, inplace=True)
        TR.modify(data=np.einsum('a,a...->a...', TS.data, TR.data))

        ## get remainder tens
        if cut_ind < r:
            # TL_ = TL.isel({'__tmp__': slice(cut_ind, r)})
            # TR_ = TR.isel({'__tmp__': slice(cut_ind, r)})
            TL_ = TL.copy()
            TL_.modify(data=TL.data[...,cut_ind:r])
            TR_ = TR.copy()
            TR_.modify(data=TR.data[cut_ind:r, ...])
            err_tens = TR_

            # err_tens = qtn.Tensor(data=np.einsum('a,a...->a...', TS.data[cut_ind:r], TR_.data),
            #                       inds=TR_.inds)

            # tens_ = qtn.Tensor(data=np.einsum('...a,ab->...ba', TL_.data, TR_.data),
            #                    inds=(*left_inds, *right_inds, '_weights_'))
            # TR_ = qtn.tensor_contract(TR_, new_mps[cur_orthog + 1])
            # weights_ = TS.data[cut_ind:]

            # print('weights', cut_ind, TS.data[cut_ind:], err_tens)
        else:
            weights_, TL_, err_tens = None, None, None

        ## update compressed result
        # T1 = TL.isel({'__tmp__': slice(cut_ind)})
        # T2 = TR.isel({'__tmp__': slice(cut_ind)})
        T1 = TL.copy()
        T1.modify(data=TL.data[..., :cut_ind])
        T2 = TR.copy()
        T2.modify(data=TR.data[:cut_ind, ...])
        # T2.modify(data=np.einsum('a,a...->a...', TS.data[:cut_ind], T2.data))
        # print('retained weights', TS.data[:cut_ind])

        # T1.transpose_like(new_mps[cur_orthog], inplace=True)
        # T2 = qtn.tensor_contract(T2, new_mps[cur_orthog + 1])
        # T2.transpose_like(new_mps[cur_orthog + 1], inplace=True)
        # new_mps[cur_orthog].modify(data=T1.data)
        # new_mps[cur_orthog + 1].modify(data=T2.data)

        # print('T1', T1, 'T2', T2)
        # print('TL', TL_, 'err tens', err_tens)

        return (T1, T2), (TL_, err_tens)  # , (tens_, weights_)

    elif direction < 0:

        right_inds, left_inds = mps[cur_orthog].filter_bonds(mps[cur_orthog - 1])
        TL, TS, TR = qtn.tensor_split(tens, left_inds, get='tensors', absorb=None, cutoff=CUTOFF,
                                      bond_ind='__tmp__')

        r = len(TS.data)
        if max_rank is None:
            s2_sum = np.cumsum(TS.data[::-1] ** 2)
            norm = s2_sum[-1]
            cut_ind = np.count_nonzero(s2_sum / norm > cutoff)
            # cut_ind = np.count_nonzero(np.abs(TS.data)**2 > cutoff)
            if r > MINBOND:
                cut_ind = max(MINBOND, cut_ind)
        else:
            cut_ind = max_rank

        TL.transpose(*left_inds, '__tmp__', inplace=True)
        TR.transpose('__tmp__', *right_inds, inplace=True)

        ## get remainder MPS
        if cut_ind < r:
            # TL_ = TL.isel({'__tmp__': slice(cut_ind, r)})
            # TR_ = TR.isel({'__tmp__': slice(cut_ind, r)})
            TL_ = TL.copy()
            TL_.modify(data=TL.data[..., cut_ind:r])
            TR_ = TR.copy()
            TR_.modify(data=TR.data[cut_ind:r])

            err_tens = qtn.Tensor(data=np.einsum('a,a...->a...', TS.data[cut_ind:r], TR_.data),
                                  inds=TR_.inds)

        else:
            weights_, TL_, err_tens = None, None, None

        ## update compressed result
        # T1 = TL.isel({'__tmp__': slice(cut_ind)})
        # T2 = TR.isel({'__tmp__': slice(cut_ind)})
        T1 = TL.copy()
        T1.modify(data=TL.data[..., :cut_ind])
        T2 = TR.copy()
        T2.modify(data=TR.data[:cut_ind, ...])
        T2.modify(data=np.einsum('a,a...->a...', TS.data[:cut_ind], T2.data))

        # T1.transpose_like(new_mps[cur_orthog], inplace=True)
        # T2 = qtn.tensor_contract(T2, new_mps[cur_orthog - 1])
        # T2.transpose_like(new_mps[cur_orthog - 1], inplace=True)
        # new_mps[cur_orthog].modify(data = T1.data)
        # new_mps[cur_orthog - 1].modify(data = T2.data)

        return (T1, T2), (TL_, err_tens,)

def compress_conservative(coord_sys:CartesianCoordinateSpace,
                          num_coarse: int,
                          A: Field, dAdt: Field, phi: ScalarField,
                          rank:int=None, cutoff:Numeric=CUTOFF,
                          num_opt: int=None, direction: Literal[1,-1] = 1):
    """ measure error after truncation
        compute contribution from error (to be optimized at single site)
        generally of the form  2 <const | x> + <x|x>^2
        find x st. min 2 < const | x> + <x|x>^2 - error for each constraint

        vec = x + error
        obs = sum_ij <vec_i | A | vec_j> = sum_ij <vec_i + err_i | A | vec_j + err_j>
            = sum_ij <vec_i | A | vec_j> + <vec_i | A | err_j > + <err_i | A | vec_j> + <err_i | A | err_j>

        if dAdt is None (not retained explicitly), then it's difficult to ensure E conservation
        but dAdt = (compress[A] - A[t-dt]) / dt
        so, any changes in A will be reflected in dAdt/dt as well?  like err[A]/dt
    """
    L = A.grid.L
    ndim = A.grid.ndim

    if A is not None:
        for compID, Acomp in A.components.items():
            Acomp.canonize(scale=False, i=0, inplace=True)
    if dAdt is not None:
        for compID, Acomp in dAdt.components.items():
            Acomp.canonize(scale=False, i=0, inplace=True)
    if phi is not None:
        phi.component.canonize(scale=False, i=0, inplace=True)

    X, Y, Z = coord_sys.coords
    i = 0 if direction > 0 else L - 1
    Ex_nrg_blocks = E_nrg_blocks(coord_sys, X, num_coarse, dAdt, phi, i)
    Ey_nrg_blocks = E_nrg_blocks(coord_sys, Y, num_coarse, dAdt, phi, i)
    Ez_nrg_blocks = E_nrg_blocks(coord_sys, Z, num_coarse, dAdt, phi, i)
    Bx_nrg_blocks = B_nrg_blocks(coord_sys, X, num_coarse, A, i)
    By_nrg_blocks = B_nrg_blocks(coord_sys, Y, num_coarse, A, i)
    Bz_nrg_blocks = B_nrg_blocks(coord_sys, Z, num_coarse, A, i)
    px_blocks = momentum_blocks(coord_sys, X, num_coarse, A, dAdt, phi, i)
    py_blocks = momentum_blocks(coord_sys, Y, num_coarse, A, dAdt, phi, i)
    pz_blocks = momentum_blocks(coord_sys, Z, num_coarse, A, dAdt, phi, i)

    print('Bx nrg blocks', Bx_nrg_blocks)
    print('By nrg blocks', By_nrg_blocks)
    print('Bz nrg blocks', Bz_nrg_blocks)

    print('Ex nrg blocks', Ex_nrg_blocks)
    print('Ey nrg blocks', Ey_nrg_blocks)
    print('Ez nrg blocks', Ez_nrg_blocks)

    print('px blocks', px_blocks)
    print('py blocks', py_blocks)
    print('pz blocks', pz_blocks)

    all_blocks = (Ex_nrg_blocks, Ey_nrg_blocks, Ez_nrg_blocks,
                  Bx_nrg_blocks, By_nrg_blocks, Bz_nrg_blocks,
                  px_blocks, py_blocks, pz_blocks)

    for i in (range(L - 1) if direction > 0 else range(L-1, 0, -1)):

        print(f'\nSite {i}')

        base_mps = {}
        base_tens: dict[int, tuple[qtn.Tensor, qtn.Tensor]] = {}
        ## "left" (canonicalized), "right" to merge with next tensor
        error_tens: dict[int, tuple[qtn.Tensor, qtn.Tensor]] = {}
        ## "left" (canonicalized), "right" (parameters to modify) to merge with next tensor
        error_terms: dict[int, qtn.Tensor] = {}
        # error_coeffs: dict[int, np.ndarray] = {}

        for compID, Acomp in A.components.items():
            Ac_base, Ac_err = compress_i_remainder(Acomp.data, i, direction, max_rank=rank, cutoff=cutoff)
            if Acomp is not None and Acomp.data is not None:
                base_mps[compID.type.value] = Acomp  ## mps
            if Ac_base is not None:
                base_tens[compID.type.value] = Ac_base  ## T1, T2 (iso, to contract with next site)
                error_tens[compID.type.value] = Ac_err  ## TL, TR (weighted)
                error_terms[compID.type.value] = Ac_err[1]  ## TL @ TR (not weighted, weights)

        for compID, Acomp in dAdt.components.items():
            Ac_base, Ac_err = compress_i_remainder(Acomp.data, i, direction, max_rank=rank, cutoff=cutoff)
            if Acomp is not None and Acomp.data is not None:
                base_mps[compID.type.value + 3] = Acomp
            if Ac_base is not None:
                base_tens[compID.type.value + 3] = Ac_base
                error_tens[compID.type.value + 3] = Ac_err
                error_terms[compID.type.value + 3] = Ac_err[1]

        phi_base, phi_err = compress_i_remainder(phi.component.data, i, direction, max_rank=rank, cutoff=cutoff, )
        if phi.component is not None and phi.component.data is not None:
            base_mps[6] = phi.component
        if phi_base is not None:
            base_tens[6] = phi_base
            error_tens[6] = phi_err
            error_terms[6] = phi_err[1]

        coeffs_inds = {ii: err_tens.inds for ii, err_tens in error_terms.items() if err_tens is not None}
        coeffs_shape = {ii: err_tens.shape for ii, err_tens in error_terms.items() if err_tens is not None}

        len_coeffs = {ii: err_tens.size for ii, err_tens in error_terms.items() if err_tens is not None}
        tot_len_coeffs = sum(len_coeffs.values())


        ## get enough d.o.f for 9 constraints
        ## but some of them might not matter? / be automatically satisfied bc 0.
        print('num opt', num_opt)
        print('len coeffs', len(len_coeffs))
        if num_opt is None:
            num_opt = np.ceil(9 / len(len_coeffs)) * 3
            print('num opt', num_opt)

        num_opt = 4
        print('num opt', num_opt, tot_len_coeffs)

        # error_terms_red = {}
        # for ii, tens in error_terms.items():
        #     if tens is not None:
        #         red_tens = tens.copy()
        #         assert red_tens.inds[0] == '__tmp__'
        #         red_tens.modify(apply=lambda x: x[:num_opt, :])
        #         error_terms_red[ii] = red_tens

        # error_tens_red = {}
        # for ii, (t1, t2) in error_tens.items():
        #     if t2 is not None:
        #         red_tens2 = t2.copy()
        #         assert red_tens2.inds[0] == '__tmp__'
        #         red_tens2.modify(apply=lambda x: x[:num_opt, :])
        #     if t1 is not None:
        #         red_tens1 = t1.copy()
        #         assert red_tens1.inds[-1] == '__tmp__'
        #         red_tens1.modify(apply=lambda x: x[..., :num_opt])
        #     error_tens_red[ii] = (red_tens1, red_tens2)

        coeffs_shape_red = {ii: (min(sh[0], num_opt), *sh[1:]) for ii, sh in coeffs_shape.items()}
        len_coeffs_red = {ii: np.prod(sh) for ii, sh in coeffs_shape_red.items()}
        tot_len_coeffs_red = sum(len_coeffs_red.values())

        ## catch: no compression
        # no_compress = True
        # for ii in range(7):
        #     ec = error_coeffs.get(ii, None)
        #     no_compress = no_compress and (ec is None or len(ec) == 0)

        if tot_len_coeffs_red > 0:  # > number of constraints

            for k, tens in error_tens.items():
                print('errs dict', k, tens)

            ### compute compression error
            print('i', i, 'direction', direction)
            print('Bx error')
            Bx_expec_vals, Bx_expec_weights = measure_errors(Bx_nrg_blocks, num_opt, base_tens, error_tens, i)
            print('By error')
            By_expec_vals, By_expec_weights = measure_errors(By_nrg_blocks, num_opt, base_tens, error_tens, i)
            print('Bz error')
            Bz_expec_vals, Bz_expec_weights = measure_errors(Bz_nrg_blocks, num_opt, base_tens, error_tens, i)

            print('Ex error')
            Ex_expec_vals, Ex_expec_weights = measure_errors(Ex_nrg_blocks, num_opt, base_tens, error_tens, i)
            print('Ey error')
            Ey_expec_vals, Ey_expec_weights = measure_errors(Ey_nrg_blocks, num_opt, base_tens, error_tens, i)
            print('Ez error')
            Ez_expec_vals, Ez_expec_weights = measure_errors(Ez_nrg_blocks, num_opt, base_tens, error_tens, i)

            print('px error')
            px_expec_vals, px_expec_weights = measure_errors(px_blocks, num_opt, base_tens, error_tens, i)
            print('py error')
            py_expec_vals, py_expec_weights = measure_errors(py_blocks, num_opt, base_tens, error_tens, i)
            print('pz error')
            pz_expec_vals, pz_expec_weights = measure_errors(pz_blocks, num_opt, base_tens, error_tens, i)


            nrg_Bx_base, nrg_Bx_err = Bx_expec_vals
            nrg_By_base, nrg_By_err = By_expec_vals
            nrg_Bz_base, nrg_Bz_err = Bz_expec_vals

            nrg_Ex_base, nrg_Ex_err = Ex_expec_vals
            nrg_Ey_base, nrg_Ey_err = Ey_expec_vals
            nrg_Ez_base, nrg_Ez_err = Ez_expec_vals

            px_base, px_err = px_expec_vals
            py_base, py_err = py_expec_vals
            pz_base, pz_err = pz_expec_vals

            print("Bx expec weights", Bx_expec_weights)
            print('px expec weights', px_expec_weights)

            ### ultimately want to solve a nonlinear least-squares problem
            ### |expec(errs, errs) @ (coeffs, coeffs) - expec_err|^2  for each expectation value
            ### jac: d/dci ( expec_i(c)^2 - 2 expec_err expec_i(c) + expec_err^2 )
            ###      = 2 expec_i(c) d/dci expec_i(c) - 2 expec_err d/dci expec_i(c)
            ###      where d/dci expec_i(c) = d/dci ( c.T @ mat @ c + vec1 @ c + c.T @ vec2)
            ###             = 2 mat_ii c_i + mat_ij c_j + vec1 + vec2
            ### hess: d/dci d/dcj ()
            ###      = 2 expec_i(c) d/dci d/dcj expec_i(c) + 2 d/dcj expec_i(c) d/dci expec_i(c)
            ###        - 2 expec_err d/dci d/dcj expec_i(c)
            ### where d/dci d/dcj expec_i(c) = 2 mat_ii  if i == j else mat_ij

            print('nrg Bx err', nrg_Bx_err)
            print('Bx expec vals', Bx_expec_vals)
            print('Bx expec weights', Bx_expec_weights)

            print('px base', px_base, np.sum(px_base))
            print('px err (target val)', px_err, np.sum(px_err))
            print('sum', px_base + px_err, np.sum(px_base + px_err))

            print('px base', py_base, np.sum(py_base))
            print('px err (target val)', py_err, np.sum(py_err))
            print('sum', py_base + py_err, np.sum(py_base + py_err))

            Bx_cost, Bx_jac = get_cost_func(coeffs_shape_red, num_coarse ** ndim, num_opt, nrg_Bx_err, Bx_expec_weights)
            By_cost, By_jac = get_cost_func(coeffs_shape_red, num_coarse ** ndim, num_opt, nrg_By_err, By_expec_weights)
            Bz_cost, Bz_jac = get_cost_func(coeffs_shape_red, num_coarse ** ndim, num_opt, nrg_Bz_err, Bz_expec_weights)
            Ex_cost, Ex_jac = get_cost_func(coeffs_shape_red, num_coarse ** ndim, num_opt, nrg_Ex_err, Ex_expec_weights)
            Ey_cost, Ey_jac = get_cost_func(coeffs_shape_red, num_coarse ** ndim, num_opt, nrg_Ey_err, Ey_expec_weights)
            Ez_cost, Ez_jac = get_cost_func(coeffs_shape_red, num_coarse ** ndim, num_opt, nrg_Ez_err, Ez_expec_weights)
            px_cost, px_jac = get_cost_func(coeffs_shape_red, num_coarse ** ndim, num_opt, px_err, px_expec_weights)
            py_cost, py_jac = get_cost_func(coeffs_shape_red, num_coarse ** ndim, num_opt, py_err, py_expec_weights)
            pz_cost, pz_jac = get_cost_func(coeffs_shape_red, num_coarse ** ndim, num_opt, pz_err, pz_expec_weights)

            def tot_cost(weights: np.ndarray):
                # return np.array([Bx_cost(weights), By_cost(weights), Bz_cost(weights),
                #                  Ex_cost(weights), Ey_cost(weights), Ez_cost(weights),
                #                  px_cost(weights), py_cost(weights), pz_cost(weights)])

                out = np.array([Bx_cost(weights), By_cost(weights), Bz_cost(weights),
                                 Ex_cost(weights), Ey_cost(weights), Ez_cost(weights),
                                 px_cost(weights), py_cost(weights), pz_cost(weights)])
                return out.reshape(-1)

            def tot_jac(weights: np.ndarray):
                # return np.array([Bx_jac(weights), By_jac(weights), Bz_jac(weights),
                #                  Ex_jac(weights), Ey_jac(weights), Ez_jac(weights),
                #                  px_jac(weights), py_jac(weights), pz_jac(weights)])

                jac = np.array([Bx_jac(weights), By_jac(weights), Bz_jac(weights),
                                Ex_jac(weights), Ey_jac(weights), Ez_jac(weights),
                                px_jac(weights), py_jac(weights), pz_jac(weights)])
                # print('jac.shape', jac.shape)
                return jac.reshape(jac.shape[0] * jac.shape[1], jac.shape[2])

            def coeffs_to_vec(coeffs: dict[int, qtn.Tensor]):
                vec = np.zeros(tot_len_coeffs_red)
                idx1 = 0
                for ix in range(7):
                    nc = len_coeffs_red.get(ix, 0)
                    if nc > 0:
                        # print('coeffs', ix, coeffs[ix])
                        coeffs[ix].transpose(*coeffs_inds[ix], inplace=True)
                        assert(coeffs[ix].inds[0] == '__tmp__')
                        vec[idx1: idx1 + nc] = coeffs[ix].data[:num_opt, :].reshape(-1)
                    idx1 += nc
                return vec

            def vec_to_coeffs(vec):
                new_coeff_dict = {}
                idx1 = 0
                for ix in range(7):
                    nc = len_coeffs_red.get(ix, 0)
                    if nc > 0:
                        new_coeff_dict[ix] = qtn.Tensor(vec[idx1: idx1 + nc].reshape(coeffs_shape_red[ix]),
                                                        inds=coeffs_inds[ix])
                    idx1 += nc
                return new_coeff_dict

            # print('error terms', error_terms)
            x0 = coeffs_to_vec(error_terms)
            print('init tot cost', tot_cost(x0), 'x0 shape', x0.shape)

            if np.linalg.norm(tot_cost(x0)) > 1.0e-12:
                # print('error coeffs', error_coeffs)
                # x0 = coeffs_to_vec(error_coeffs)
                # print('x0', x0, 'tot cost', tot_cost(x0))
                # result = scipy.optimize.minimize(tot_cost, x0, bounds=[(0, x0i*100) for x0i in x0],
                #                                  jac=tot_jac,  #hess=tot_hess)
                #                                  tol=1.0e-12,
                #                                  )
                # print('opt result', result.x, 'tot cost', tot_cost(result.x))
                # print('opt result', result.x, result.fun, result.success)

                lb = np.minimum(-1.0e-16, np.abs(x0) * -1000)
                ub = np.maximum(1.0e-16, np.abs(x0) * 1000)
                # lb = np.abs(x0) * -10
                # ub = np.abs(x0) * 10
                # print('lb', lb, 'ub', ub)
                result = scipy.optimize.least_squares(tot_cost, x0,
                                                      # bounds=(lb, ub),
                                                      jac=tot_jac,
                                                      )
                # print('opt result', result.x, 'tot cost', tot_cost(result.x))
                print('opt result', np.linalg.norm(result.x), result.cost, result.fun, # tot_cost(result.x),
                      np.linalg.norm(result.fun)**2 / len(result.x), result.success)

                opt_weights = vec_to_coeffs(result.x)

            else:
                opt_weights = vec_to_coeffs(x0)

            # # opt_weights = vec_to_coeffs(result.x)
            # if np.linalg.norm(x0)**2 < result.cost:
            #     opt_weights = error_coeffs
            #     print('using initial error coeffs')
            # else:
            #     opt_weights = vec_to_coeffs(result.x)

            # base_tens_copy = {k: (t1.copy() if t1 is not None else None,
            #                       t2.copy() if t2 is not None else None) for k, (t1, t2) in base_tens.items()}

            ### pad compressed MPS with contributions in opt weights
            ### comment out to test without correction
            for ii in range(7):
                TL_, TR_ = error_tens.get(ii, (None, None))
                if TL_ is None:
                    continue

                # TR_ = opt_weights[ii]   ## new TR_
                TR_.modify(data=opt_weights[ii].data)
                TL_.modify(apply=lambda x: x[..., :TR_.shape[0]])
                # print('TL', TL_)
                # print('TR', TR_)

                ## update MPS
                TL, TR = base_tens[ii]
                # print('base TL', TL)
                # print('base TR', TR)

                ### add correction to base tensors
                TL_.transpose_like(TL, inplace=True)
                TL_.modify(inds = TL.inds)
                TR_.transpose_like(TR, inplace=True)
                TR_.modify(inds = TR.inds)
                _, left_inds = TL.filter_bonds(TR)
                _, right_inds = TR.filter_bonds(TL)
                TL.direct_product(TL_, sum_inds=left_inds, inplace=True)
                TR.direct_product(TR_, sum_inds=right_inds, inplace=True)
                # print('padded TL', TL)
                # print('padded TR', TR)

            # print('post update')
            # # _, _ = measure_errors(Bx_nrg_blocks, num_opt, base_tens, {}, i)
            # # _, _ = measure_errors(By_nrg_blocks, num_opt, base_tens, {}, i)
            # # _, _ = measure_errors(Ez_nrg_blocks, num_opt, base_tens, {}, i)
            # # chk_1 = measure_errors(px_blocks, num_opt, base_tens_copy, error_tens, i)
            # # print('px before base update', chk_1[0])
            # # print('px new err', np.sum(chk_1[0][1]))
            # chk_base, _ = measure_errors(px_blocks, num_opt, base_tens, {}, i)
            # print('px', chk_base)
            # print('px', np.sum(chk_base[0]))
            #
            # chk_base, _ = measure_errors(py_blocks, num_opt, base_tens, {}, i)
            # print('py', chk_base)
            # print('py', np.sum(chk_base[0]))
            pdb.set_trace()

        #### perform base compression using base tensors
        print('update mps site i', i)
        for ii in range(7):
            mps = base_mps.get(ii, None)
            if mps is None:
                continue

            tens1, tens2 = mps[i], mps[i + direction]
            T1, T2 = base_tens[ii]
            T2 = qtn.tensor_contract(T2, tens2)
            T1.transpose_like(tens1, inplace=True)
            # print('T1', T1, 'tens1', tens1)
            tens1.modify(data=T1.data)
            T2.transpose_like(tens2, inplace=True)
            # print('T2', T2, 'tens2', tens2)
            tens2.modify(data=T2.data)

            helper_quimb.check_orthog(mps)

        for blocks_dict in all_blocks:
            for k, blocks in blocks_dict.items():
                for bl in blocks:
                    bl.extend_env(i, direction)
                    # print('bl cur orthog', bl.cur_orthog)

    return



