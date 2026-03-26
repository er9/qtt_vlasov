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



from local_solvers.blocks import Block, BlockVector_DMRG, BlockOperator_DMRG


def E_nrg_blocks(coord_sys: CartesianCoordinateSpace, compID, dAdt: Field, phi: ScalarField,
                 cur_orthog: int):
    """ Ei = < d/dxi phi - dAdt_i | d/dxi phi - dAdt_I >
    """
    ax = coord_sys.get_axis(compID)
    if ax is not None:
        ddx = phi.grid.get_firstderivative_mpo(ax).data
        d2dx2 = helper_quimb.apply( helper_quimb.mpo_conj_transpose(ddx), ddx)

    dAdt_data = dAdt[compID].data if compID in dAdt.components else None
    phi_data = phi.component.data
    ix = compID.type.value

    blocks = {}
    if dAdt_data is not None:
        blocks[(3+ix, 3+ix)] = [BlockVector_DMRG(dAdt_data, dAdt_data, cur_orthog=cur_orthog)]
        # print('E block', blocks[(3 + ix, 3 + ix)][-1].norm())
        if phi_data is not None:
            ddx = ddx.scalar_multiply(-2)
            blocks[(3+ix, 7)] = [BlockOperator_DMRG(dAdt_data, phi_data, ddx, cur_orthog=cur_orthog)]
    if phi_data is not None:
        blocks[(7,7)] = [BlockOperator_DMRG(phi_data, phi_data, d2dx2, cur_orthog=cur_orthog)]
    return blocks

def B_nrg_blocks(coord_sys: CartesianCoordinateSpace, compID, A: Field,
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
            # tmp = helper_quimb.apply(ddx1, vec1_data)
            # print('tmp fro norm', helper_quimb.norm(tmp))
            ddx1_2 = helper_quimb.apply(helper_quimb.mpo_conj_transpose(ddx1), ddx1)
            blocks[(i2,i2)] = [BlockOperator_DMRG(vec1_data, vec1_data, ddx1_2, cur_orthog=cur_orthog)]
            # print('block norm', blocks[(i2,i2)][-1].norm())
            # print('B block i2', helper_quimb.norm(vec1_data))

    if vec2_data is not None:
        if ddx2 is not None:
            # tmp = helper_quimb.apply(ddx2, vec2_data)
            # print('tmp fro norm', helper_quimb.norm(tmp))
            ddx2_2 = helper_quimb.apply(helper_quimb.mpo_conj_transpose(ddx2), ddx2)
            blocks[(i1,i1)] = [BlockOperator_DMRG(vec2_data, vec2_data, ddx2_2, cur_orthog=cur_orthog)]
            # print('block norm', blocks[(i1, i1)][-1].norm())
            # print('B block i2', helper_quimb.norm(vec2_data))

    if vec1_data is not None and vec2_data is not None:
        if ddx1 is not None and ddx2 is not None:
            op = helper_quimb.apply(helper_quimb.mpo_conj_transpose(ddx1), ddx2)
            helper_quimb.scalar_multiply(op, -2, inplace=True)
            blocks[(i1,i2)] = [BlockOperator_DMRG(vec2_data, vec1_data, op, cur_orthog=cur_orthog)]

    return blocks

def momentum_blocks(coord_sys: CartesianCoordinateSpace,
                    compID, A: Field, dAdt: Field, phi: ScalarField,
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

    c1_ = coord_sys.type_coords.get(i1, None)
    c2_ = coord_sys.type_coords.get(i2, None)
    # print('i0', i, 'i1', i1, 'i2', i2)
    # print('c0', compID, 'c1', c1_, 'c2', c2_)

    # ax1 = coord_sys.get_axis(compID)
    # ax2 = coord_sys.get_axis(c1_)
    # ax3 = coord_sys.get_axis(c2_)
    # ddx1 = A.grid.get_firstderivative_mpo(ax1).data if ax1 is not None else None
    # ddx2 = A.grid.get_firstderivative_mpo(ax2).data if ax2 is not None else None
    # ddx3 = A.grid.get_firstderivative_mpo(ax3).data if ax3 is not None else None

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
            op31 = helper_quimb.apply( helper_quimb.mpo_conj_transpose(op3), op1)
            blocks[(7, coords[k2].type.value)] = [BlockOperator_DMRG(A1.data, phi_data, op31, cur_orthog=cur_orthog)]
        if dAdt_4 is not None and dAdt_4.data is not None and op1 is not None:
            op41 = helper_quimb.scalar_multiply(op1, -1, inplace=False)
            # op41 = op1.copy()
            blocks[(c1_.type.value + 3, coords[k2].type.value)] = [BlockOperator_DMRG(A1.data, dAdt_4.data, op41,
                                                                           cur_orthog=cur_orthog)]

    if A2 is not None and A2.data is not None:
        if phi_data is not None and op3 is not None:
            op32 = helper_quimb.apply(helper_quimb.mpo_conj_transpose(op3), op2)
            helper_quimb.scalar_multiply(op32, -1, inplace=True)
            blocks[(7, coords[k1].type.value)] = [BlockOperator_DMRG(A2.data, phi_data, op32, cur_orthog=cur_orthog)]
        if dAdt_4 is not None and dAdt_4.data is not None and op2 is not None:
            # op42 = helper_quimb.scalar_multiply(op2, -1, inplace=False)
            op42 = op2.copy()
            blocks[(c1_.type.value + 3, coords[k1].type.value)] = [BlockOperator_DMRG(A2.data, dAdt_4.data, op42,
                                                                            cur_orthog=cur_orthog)]

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
            op31 = helper_quimb.apply(helper_quimb.mpo_conj_transpose(op3), op1)
            key = (7, coords[j2].type.value)
            blocks[key] = blocks.get(key, []) + [BlockOperator_DMRG(A1.data, phi_data, op31, cur_orthog=cur_orthog)]
        if dAdt_4 is not None and dAdt_4.data is not None and op1 is not None:
            op41 = helper_quimb.scalar_multiply(op1, -1, inplace=False)
            key = (c2_.type.value + 3, coords[j2].type.value)
            blocks[key] = blocks.get(key, []) + [BlockOperator_DMRG(A1.data, dAdt_4.data, op41, cur_orthog=cur_orthog)]

    if A2 is not None and A2.data is not None:
        if phi_data is not None and op3 is not None:
            op32 = helper_quimb.apply(helper_quimb.mpo_conj_transpose(op3), op2)
            helper_quimb.scalar_multiply(op32, -1, inplace=True)
            key = (7, coords[j1].type.value)
            blocks[key] = blocks.get(key, []) + [BlockOperator_DMRG(A2.data, phi_data, op32, cur_orthog=cur_orthog)]
        if dAdt_4 is not None and dAdt_4.data is not None and op2 is not None:
            # op42 = helper_quimb.scalar_multiply(op2, -1, inplace=False)
            op42 = op2.copy()
            key = (c2_.type.value + 3, coords[j1].type.value)
            blocks[key] = blocks.get(key, []) + [BlockOperator_DMRG(A2.data, dAdt_4.data, op42, cur_orthog=cur_orthog)]

    return blocks


def measure_errors(block_dict: dict[tuple[int,int], list[Block]],
                   base_dict: dict[int, tuple[qtn.Tensor, qtn.Tensor]],
                   errs_dict: dict[int, tuple[qtn.Tensor, np.array]],
                   cur_orthog: int):
    """
    obs = sum_ij < vec_i | A | vec_j > = sum_ij < vec_i + err_i | A | vec_j + err_j >
    = sum_ij < vec_i | A | vec_j > + < vec_i | A | err_j > + < err_i | A | vec_j > + < err_i | A | err_j >
    """
    expec_dict = {}
    expec_errs_dict = {}  ## expectation values of overlaps w/o weights

    base_tens_dict = {k: qtn.tensor_contract(*tens) for k, tens in base_dict.items()}
    # print('block dict', block_dict)

    for k, blocks in block_dict.items():
        # print('k blocks', k)
        bra_key, ket_key = k
        bra_errs = errs_dict.get(bra_key, None)
        ket_errs = errs_dict.get(ket_key, None)
        tot_expec_base, tot_expec_errs = 0.0, 0.0

        tot_bra_err, tot_ket_err, tot_bk_err = None, None, None

        for block in blocks:
            # print('block', block, 'expec', block.norm())

            ket_err_tn = None
            if ket_errs is not None:
                ket_err_tens, ket_err_weights = ket_errs
                if ket_err_tens is not None:
                    ket_err_tn = block.get_projected_Ax(cur_orthog, 1, site_tens=ket_err_tens)

            ket_base = base_tens_dict[ket_key]
            ket_base_tn = block.get_projected_Ax(cur_orthog, 1, site_tens=ket_base)

            b2k = block.projected_bra_to_ket(cur_orthog, 1)
            k2b = {v: k for k, v in b2k.items()}
            if block.operator is None:
                k2b.pop(block.ket.site_ind(cur_orthog))
            bra_base_tens = base_tens_dict[bra_key]
            bra_base_tens = bra_base_tens.reindex(k2b)
            if bra_errs is not None:
                bra_err_tens, bra_err_weights = bra_errs
                if bra_err_tens is not None:
                    bra_err_tens = bra_err_tens.reindex(k2b)
            else:
                bra_err_tens = None

            # print('bra', bra_err_weights)
            # print('ket', ket_err_weights)

            # if bra_errs is not None:
            #     bra_err_tens, weights = bra_errs
            #     bra_err_tn = block.get_projected(cur_orthog, 1, site_tens=bra_err_tens)
            # else:
            #     bra_err_tn = None
            # bra_base_tn = block.get_projected(cur_orthog, 1, site_tens=None)

            expec_base = qtn.tensor_contract(*ket_base_tn.tensors, bra_base_tens)
            expec_base *= 10**(ket_base_tn.exponent + block.bra.exponent)

            expec_errs = 0.0
            if bra_err_tens is not None:
                err = qtn.tensor_contract(*ket_base_tn.tensors, bra_err_tens).data
                err *= 10**(ket_base_tn.exponent + block.bra.exponent)

                expec_errs += err @ bra_err_weights
                if tot_bra_err is None:
                    tot_bra_err = err
                else:
                    tot_bra_err += err

            if ket_err_tn is not None:
                err = qtn.tensor_contract(*ket_err_tn.tensors, bra_base_tens).data
                err *= 10**(ket_base_tn.exponent + block.bra.exponent)

                expec_errs += err @ ket_err_weights
                if tot_ket_err is None:
                    tot_ket_err = err
                else:
                    tot_ket_err += err

            if bra_err_tens is not None and ket_err_tn is not None:
                bra_err_tens_ = bra_err_tens.reindex({'_weights_': '_weights_b_'}, inplace=False)
                # print('bra tens', bra_err_tens_, bra_err_tens_.data)
                # print('ket tens', ket_err_tens, ket_err_tens.data)
                proj_ket = qtn.tensor_contract(*ket_err_tn.tensors)
                proj_ket = proj_ket * 10 ** ket_err_tn.exponent
                # print('proj ket norm', proj_ket.norm(), ket_err_tn.exponent)
                err = qtn.tensor_contract(*ket_err_tn.tensors, bra_err_tens_)
                err = err * 10**(ket_err_tn.exponent + block.bra.exponent)
                # print('bk err', err)

                err.transpose('_weights_b_', '_weights_', inplace=True)
                expec_errs += np.einsum('a,ab,b->', bra_err_weights, err.data, ket_err_weights)
                if tot_bk_err is None:
                    tot_bk_err = err.data
                else:
                    tot_bk_err += err.data

            tot_expec_base += expec_base
            tot_expec_errs += expec_errs

        # print('tot bra err', tot_bra_err)
        # print('tot ket err', tot_ket_err)
        # print('to bk err', tot_bk_err)

        # tmp = bra_err_weights @ tot_bk_err @ ket_err_weights + tot_bra_err @ bra_err_weights + tot_ket_err @ ket_err_weights
        # print('target val', tot_expec_errs, tmp)

        # print('k', k, tot_bk_err)
        expec_errs_dict[k] = (tot_bra_err, tot_ket_err, tot_bk_err)
        expec_dict[k] = (tot_expec_base, tot_expec_errs)

    tot_expec = np.sum([v[0] for v in expec_dict.values()])
    tot_err = np.sum([v[1] for v in expec_dict.values()])

    print('expec', tot_expec, tot_err, 'total', tot_expec + tot_err)

    return (tot_expec, tot_err), expec_errs_dict


def compress_conservative(coord_sys:CartesianCoordinateSpace,
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

    for compID, Acomp in A.components.items():
        Acomp.canonize(scale=False, i=0, inplace=True)
    for compID, Acomp in dAdt.components.items():
        Acomp.canonize(scale=False, i=0, inplace=True)
    phi.component.canonize(scale=False, i=0, inplace=True)

    X, Y, Z = coord_sys.coords
    i = 0 if direction > 0 else L - 1
    Ex_nrg_blocks = E_nrg_blocks(coord_sys, X, dAdt, phi, i)
    Ey_nrg_blocks = E_nrg_blocks(coord_sys, Y, dAdt, phi, i)
    Ez_nrg_blocks = E_nrg_blocks(coord_sys, Z, dAdt, phi, i)
    Bx_nrg_blocks = B_nrg_blocks(coord_sys, X, A, i)
    By_nrg_blocks = B_nrg_blocks(coord_sys, Y, A, i)
    Bz_nrg_blocks = B_nrg_blocks(coord_sys, Z, A, i)
    px_blocks = momentum_blocks(coord_sys, X, A, dAdt, phi, i)
    py_blocks = momentum_blocks(coord_sys, Y, A, dAdt, phi, i)
    pz_blocks = momentum_blocks(coord_sys, Z, A, dAdt, phi, i)

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
        error_tens: dict[int, tuple[qtn.Tensor, qtn.Tensor]] = {}
        error_terms: dict[int, tuple[qtn.Tensor, np.ndarray]] = {}
        error_coeffs: dict[int, np.ndarray] = {}

        for compID, Acomp in A.components.items():
            Ac_base, Ac_err, Ac_weights = compress_i_remainder(Acomp.data, i, direction, max_rank=rank, cutoff=cutoff)
            if Acomp is not None and Acomp.data is not None:
                base_mps[compID.type.value] = Acomp  ## mps
            if Ac_base is not None:
                base_tens[compID.type.value] = Ac_base  ## T1, T2 (iso, to contract with next site)
                error_tens[compID.type.value] = Ac_err  ## TL, TR (not weighted)
                error_terms[compID.type.value] = Ac_weights  ## TL @ TR (not weighted, weights)
                error_coeffs[compID.type.value] = Ac_weights[1]   ## coeff weights

        for compID, Acomp in dAdt.components.items():
            Ac_base, Ac_err, Ac_weights = compress_i_remainder(Acomp.data, i, direction, max_rank=rank, cutoff=cutoff)
            if Acomp is not None and Acomp.data is not None:
                base_mps[compID.type.value + 3] = Acomp
            if Ac_base is not None:
                base_tens[compID.type.value + 3] = Ac_base
                error_tens[compID.type.value + 3] = Ac_err
                error_terms[compID.type.value + 3] = Ac_weights
                error_coeffs[compID.type.value + 3] = Ac_weights[1]

        phi_base, phi_err, phi_weights = compress_i_remainder(phi.component.data, i, direction, max_rank=rank,
                                                              cutoff=cutoff, )
        if phi.component is not None and phi.component.data is not None:
            base_mps[6] = phi.component
        if phi_base is not None:
            base_tens[6] = phi_base
            error_tens[6] = phi_err
            error_terms[6] = phi_weights
            error_coeffs[6] = phi_weights[1]

        len_coeffs = {ii: len(ec) for ii, ec in error_coeffs.items() if ec is not None}
        tot_len_coeffs = sum(len_coeffs.values())


        # print('A', A.components if A is not None else None)
        # print('dAdt', dAdt.components if dAdt is not None else None)

        ## get enough d.o.f for 9 constraints
        ## but some of them might not matter? / be automatically satisfied bc 0.
        print('num opt', num_opt)
        print('len coeffs', len(len_coeffs))
        if num_opt is None:
            num_opt = np.ceil(9 / len(len_coeffs)) * 3
            print('num opt', num_opt)

        num_opt = 12  # 10  # 6  # min(6, max(2, tot_len_coeffs - 2) )
        print('num opt', num_opt, tot_len_coeffs)

        len_coeffs_red = {ii: min(num_opt, lc) for ii, lc in len_coeffs.items()}
        tot_len_coeffs_red = sum(len_coeffs_red.values())


        ## catch: no compression
        # no_compress = True
        # for ii in range(7):
        #     ec = error_coeffs.get(ii, None)
        #     no_compress = no_compress and (ec is None or len(ec) == 0)

        if tot_len_coeffs_red > 0:  # > number of constraints

            ### compute compression error
            print('Bx error')
            Bx_expec_vals, Bx_expec_errs_dict = measure_errors(Bx_nrg_blocks, base_tens, error_terms, i)
            print('By error')
            By_expec_vals, By_expec_errs_dict = measure_errors(By_nrg_blocks, base_tens, error_terms, i)
            print('Bz error')
            Bz_expec_vals, Bz_expec_errs_dict = measure_errors(Bz_nrg_blocks, base_tens, error_terms, i)

            print('Ex error')
            Ex_expec_vals, Ex_expec_errs_dict = measure_errors(Ex_nrg_blocks, base_tens, error_terms, i)
            print('Ey error')
            Ey_expec_vals, Ey_expec_errs_dict = measure_errors(Ey_nrg_blocks, base_tens, error_terms, i)
            print('Ez error')
            Ez_expec_vals, Ez_expec_errs_dict = measure_errors(Ez_nrg_blocks, base_tens, error_terms, i)

            print('px error')
            px_expec_vals, px_expec_errs_dict = measure_errors(px_blocks, base_tens, error_terms, i)
            print('py error')
            py_expec_vals, py_expec_errs_dict = measure_errors(py_blocks, base_tens, error_terms, i)
            print('pz error')
            pz_expec_vals, pz_expec_errs_dict = measure_errors(pz_blocks, base_tens, error_terms, i)


            nrg_Bx_base, nrg_Bx_err = Bx_expec_vals
            nrg_By_base, nrg_By_err = By_expec_vals
            nrg_Bz_base, nrg_Bz_err = Bz_expec_vals

            nrg_Ex_base, nrg_Ex_err = Ex_expec_vals
            nrg_Ey_base, nrg_Ey_err = Ey_expec_vals
            nrg_Ez_base, nrg_Ez_err = Ez_expec_vals

            px_base, px_err = px_expec_vals
            py_base, py_err = py_expec_vals
            pz_base, pz_err = pz_expec_vals

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

            # Bx_cost, Bx_jac, Bx_hess = get_cost_func(len_coeffs, 100, nrg_Bx_err, Bx_expec_errs_dict)
            # By_cost, By_jac, By_hess = get_cost_func(len_coeffs, 100, nrg_By_err, By_expec_errs_dict)
            #
            # def coeffs_to_vec_tmp(coeffs):
            #     vec = np.zeros(tot_len_coeffs)
            #     idx1 = 0
            #     for ix in range(7):
            #         nc = len_coeffs.get(ix, 0)
            #         if nc > 0:
            #             vec[idx1: idx1 + nc] = coeffs[ix][:nc]
            #         idx1 += nc
            #     return vec
            #
            # weights = coeffs_to_vec_tmp(error_coeffs)
            #
            # print('full error Bx cost', Bx_cost(weights))
            # print('full error By cost', By_cost(weights))

            Bx_cost, Bx_jac, Bx_hess = get_cost_func(len_coeffs, num_opt, nrg_Bx_err, Bx_expec_errs_dict)
            By_cost, By_jac, By_hess = get_cost_func(len_coeffs, num_opt, nrg_By_err, By_expec_errs_dict)
            Bz_cost, Bz_jac, Bz_hess = get_cost_func(len_coeffs, num_opt, nrg_Bz_err, Bz_expec_errs_dict)
            Ex_cost, Ex_jac, Ex_hess = get_cost_func(len_coeffs, num_opt, nrg_Ex_err, Ex_expec_errs_dict)
            Ey_cost, Ey_jac, Ey_hess = get_cost_func(len_coeffs, num_opt, nrg_Ey_err, Ey_expec_errs_dict)
            Ez_cost, Ez_jac, Ez_hess = get_cost_func(len_coeffs, num_opt, nrg_Ez_err, Ez_expec_errs_dict)
            px_cost, px_jac, px_hess = get_cost_func(len_coeffs, num_opt, px_err, px_expec_errs_dict)
            py_cost, py_jac, py_hess = get_cost_func(len_coeffs, num_opt, py_err, py_expec_errs_dict)
            pz_cost, pz_jac, pz_hess = get_cost_func(len_coeffs, num_opt, pz_err, pz_expec_errs_dict)

            def tot_cost(weights: np.ndarray):
                # return Bx_cost(weights)
                # return (Bx_cost(weights) + By_cost(weights) + Bz_cost(weights) +
                #         Ex_cost(weights) + Ey_cost(weights) + Ez_cost(weights) +
                #         px_cost(weights) + py_cost(weights) + pz_cost(weights))
                return np.array([Bx_cost(weights), By_cost(weights), Bz_cost(weights),
                                 Ex_cost(weights), Ey_cost(weights), Ez_cost(weights),
                                 px_cost(weights), py_cost(weights), pz_cost(weights)])

            def tot_jac(weights: np.ndarray):
                # return Bx_jac(weights)
                # return (Bx_jac(weights) + By_jac(weights) + Bz_jac(weights) +
                #         Ex_jac(weights) + Ey_jac(weights) + Ez_jac(weights) +
                #         px_jac(weights) + py_jac(weights) + pz_jac(weights))
                return np.array([Bx_jac(weights), By_jac(weights), Bz_jac(weights),
                                 Ex_jac(weights), Ey_jac(weights), Ez_jac(weights),
                                 px_jac(weights), py_jac(weights), pz_jac(weights)])

            def tot_hess(weights: np.ndarray):
                return (Bx_hess(weights) + By_hess(weights) + Bz_hess(weights) +
                        Ex_hess(weights) + Ey_hess(weights) + Ez_hess(weights) +
                        px_hess(weights) + py_hess(weights) + pz_hess(weights))

            def coeffs_to_vec(coeffs):
                vec = np.zeros(tot_len_coeffs_red)
                idx1 = 0
                for ix in range(7):
                    nc = len_coeffs_red.get(ix, 0)
                    if nc > 0:
                        vec[idx1: idx1 + nc] = coeffs[ix][:nc]
                    idx1 += nc
                return vec

            def vec_to_coeffs(vec):
                new_coeff_dict = {}
                idx1 = 0
                for ix in range(7):
                    nc = len_coeffs_red.get(ix, 0)
                    if nc > 0:
                        new_coeff_dict[ix] = vec[idx1: idx1 + nc]
                    idx1 += nc
                return new_coeff_dict

            print('error coeffs', error_coeffs)
            x0 = coeffs_to_vec(error_coeffs)
            print('x0', x0, 'tot cost', tot_cost(x0))

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

                result = scipy.optimize.least_squares(tot_cost, x0, bounds=(np.zeros(tot_len_coeffs_red), x0 * 100),
                                                      jac=tot_jac,
                                                      )
                # print('opt result', result.x, 'tot cost', tot_cost(result.x))
                print('opt result', result.x, result.cost, result.fun, tot_cost(result.x),
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

            ### pad compressed MPS with contributions in opt weights
            for ii in range(7):
                TL_, TR_ = error_tens.get(ii, (None, None))
                if TL_ is None:
                    continue

                # TL_.isel('__tmp__', slice(num_opt), inplace=True)
                # TR_.isel('__tmp__', slice(num_opt), inplace=True)

                new_coeffs = opt_weights[ii]
                num_r = len(new_coeffs)
                s_ind = '__tmp__'
                l_ind = [ii for ii in TL_.inds if ii != s_ind]
                r_ind = [ii for ii in TR_.inds if ii != s_ind]
                TL_.transpose(s_ind, *l_ind, inplace=True)
                TL_.modify(data = TL_.data[:num_r])
                TR_.transpose(s_ind, *r_ind,  inplace=True)
                TR_.modify(data = np.einsum('ab,a-> ab', TR_.data[:num_r], new_coeffs))

                ## update MPS
                TL, TR = base_tens[ii]

                ### add correction to base tensors
                TL_.transpose_like(TL, inplace=True)
                TL_.modify(inds = TL.inds)
                TR_.transpose_like(TR, inplace=True)
                TR_.modify(inds = TR.inds)
                _, left_inds = TL.filter_bonds(TR)
                _, right_inds = TR.filter_bonds(TL)
                TL.direct_product(TL_, sum_inds=left_inds, inplace=True)
                TR.direct_product(TR_, sum_inds=right_inds, inplace=True)

            print('post update')
            _, _ = measure_errors(Bx_nrg_blocks, base_tens, {}, i)
            _, _ = measure_errors(By_nrg_blocks, base_tens, {}, i)
            _, _ = measure_errors(Ez_nrg_blocks, base_tens, {}, i)

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


def get_cost_func(len_coeffs: dict[int, int], num_r: int, target_val:Numeric,
                  *expec_errs_dict: dict[tuple[int,int], tuple[np.ndarray, np.ndarray, np.ndarray]]):
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
    len_coeffs_red = {k: min(num_r, lc) for k, lc in len_coeffs.items() if (lc is not None and lc != 0)}
    tot_len_coeffs = sum(len_coeffs_red.values())

    # print('len coeffs red', len_coeffs_red)
    # print('target val', target_val)

    bk_mat = np.zeros((tot_len_coeffs,) * 2)
    k_vec = np.zeros((tot_len_coeffs,))
    b_vec = np.zeros((tot_len_coeffs,))

    # it_k, it_b, it_bk = 0, 0, 0
    for err_dict in expec_errs_dict:
        for k, val in err_dict.items():
            bra_key, ket_key = k
            bra_errs, ket_errs, bk_errs = val

            if bra_errs is not None:
                num_rb = min(num_r, len(bra_errs))
                xb = np.sum([len_coeffs_red.get(ix,0) for ix in range(bra_key)])
                b_vec[xb: xb + num_rb] = bra_errs[:num_rb]
                # if len(bra_errs) > 0:  it_b += 1

            if ket_errs is not None:
                num_rk = min(num_r, len(ket_errs))
                xk = np.sum([len_coeffs_red.get(ix,0) for ix in range(ket_key)])
                k_vec[xk: xk + num_rk] = ket_errs[:num_rk]
                # if len(ket_errs) > 0:  it_k += 1

            if bk_errs is not None:
                # print('xb', xb, xb + num_rb)
                # print('xk', xk, xk + num_rk)
                bk_mat[xb: xb + num_rb, xk: xk + num_rk] = bk_errs[:num_rb, :num_rk]
                # if len(bk_errs) > 0:  it_bk += 1

    # print('bk mat', bk_mat)
    # print('b vec', b_vec)
    # print('k vec', k_vec)

    def cost(weights: np.ndarray):
        if (target_val == 0. and np.linalg.norm(bk_mat) == 0. and np.linalg.norm(b_vec) == 0.
                and np.linalg.norm(k_vec) == 0):
            return 0.0

        # print('weights', weights)

        # print('cost', (weights @ bk_mat @ weights + b_vec @ weights + k_vec @ weights - target_val), target_val)
        # print('cost', (weights @ bk_mat @ weights + b_vec @ weights + k_vec @ weights - target_val) / target_val)
        return (weights @ bk_mat @ weights + b_vec @ weights + k_vec @ weights - target_val) / target_val

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
        if (target_val == 0. and np.linalg.norm(bk_mat) == 0. and np.linalg.norm(b_vec) == 0.
                and np.linalg.norm(k_vec) == 0):
            return np.zeros(tot_len_coeffs)

        bk_mat_2 = bk_mat + np.diag(np.diag(bk_mat))  ## diagonal terms are weighted by 2
        ddc_expec = weights @ bk_mat_2 + b_vec + k_vec
        # expec = cost(weights)
        # return 2 * expec * ddc_expec # - 2 * target_val * ddc_expec
        return ddc_expec / target_val

    def hess(weights: np.ndarray):
        """
        d/dci d/dcj ()
        = 2 expec_i(c) d/dci d/dcj expec_i(c) + 2 d/dcj expec_i(c) d/dci expec_i(c)
           - 2 expec_err d/dci d/dcj expec_i(c)
        where d/dci d/dcj expec_i(c) = 2 mat_ii  if i == j else mat_ij
        """
        bk_mat_2 = bk_mat + np.diag(np.diag(bk_mat))  ## diagonal terms are weighted by 2
        expec = weights @ bk_mat @ weights
        ddc_expec = bk_mat_2 @ weights + b_vec + k_vec
        d2dc2_expec = bk_mat_2
        return 2 * expec * d2dc2_expec + 2 * np.outer(ddc_expec, ddc_expec) - 2 * target_val * d2dc2_expec

    return cost, jac, hess


def compress_i_remainder(mps: 'qtn.MatrixProductState', cur_orthog:int, direction: int,
                         max_rank: int = None, cutoff: Numeric=CUTOFF, inplace=False):
    """ compress the MPS at site i and also return the remainder
        returns: compressed mps, remaining basis functions and weights
    """
    if mps is None:
        return None, None, None

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
            print('s2 sum', s2_sum / s2_sum[-1])
            norm = s2_sum[-1]
            # cut_ind = np.count_nonzero(s2_sum/norm > cutoff)
            cut_ind = r - np.argmax(s2_sum/norm > cutoff)
            print('cut ind', cut_ind)
            # cut_ind = np.count_nonzero(np.abs(TS.data)**2 > cutoff)
            cut_ind = max(min(r,MINBOND), cut_ind)
            print('cut ind with min bond', cut_ind)
        else:
            cut_ind = max_rank

        TL.transpose(*left_inds, '__tmp__', inplace=True)
        TR.transpose('__tmp__', *right_inds, inplace=True)

        ## get remainder MPS
        if cut_ind < r:
            # TL_ = TL.isel({'__tmp__': slice(cut_ind, r)})
            # TR_ = TR.isel({'__tmp__': slice(cut_ind, r)})
            TL_ = TL.copy()
            TL_.modify(data=TL.data[...,cut_ind:r])
            TR_ = TR.copy()
            TR_.modify(data=TR.data[cut_ind:r])
            tens_ = qtn.Tensor(data=np.einsum('...a,ab->...ba', TL_.data, TR_.data),
                               inds=(*left_inds, *right_inds, '_weights_'))
            # TR_ = qtn.tensor_contract(TR_, new_mps[cur_orthog + 1])
            weights_ = TS.data[cut_ind:]
            # print('weights', cut_ind, weights_, TS.data)
        else:
            weights_, TL_, TR_, tens_ = None, None, None, None

        ## update compressed result
        # T1 = TL.isel({'__tmp__': slice(cut_ind)})
        # T2 = TR.isel({'__tmp__': slice(cut_ind)})
        T1 = TL.copy()
        T1.modify(data=TL.data[..., :cut_ind])
        T2 = TR.copy()
        T2.modify(data=TR.data[:cut_ind, ...])
        T2.modify(data=np.einsum('a,a...->a...', TS.data[:cut_ind], T2.data))
        print('retained weights', TS.data[:cut_ind])

        # T1.transpose_like(new_mps[cur_orthog], inplace=True)
        # T2 = qtn.tensor_contract(T2, new_mps[cur_orthog + 1])
        # T2.transpose_like(new_mps[cur_orthog + 1], inplace=True)
        # new_mps[cur_orthog].modify(data=T1.data)
        # new_mps[cur_orthog + 1].modify(data=T2.data)

        # print('T1', T1, 'T2', T2)
        # print('new mps', new_mps[cur_orthog: cur_orthog + 2])
        # print('TL_', TL_, 'TR_', TR_)

        return (T1, T2), (TL_, TR_), (tens_, weights_)

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
            tens_ = qtn.Tensor(data=np.einsum('...a,ba->...ba', TL_.data, TR_.data),
                               inds=(*left_inds, *right_inds, '_weights_'))
            # TR_ = qtn.tensor_contract(TR_, new_mps[cur_orthog - 1])
            weights_ = TS.data[cut_ind:]
        else:
            weights_, TL_, TR_, tens_ = None, None, None, None

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

        return (T1, T2), (TL_, TR_), (tens_, weights_,)



