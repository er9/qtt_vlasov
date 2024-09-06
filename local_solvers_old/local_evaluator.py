import numpy as np
from abc import ABC

import helper_quimb
from setup_.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from local_solvers_old.environments_v2 import Environment, EnvironmentSide
from local_solvers_old.environments_v2 import TargetEnv_DMRG, TargetEnv_Cross
from local_solvers_old.environments_v2 import OperatorEnv_DMRG, OperatorEnv_Cross
from local_solvers_old.local_solver import SweepDirection
from local_solvers_old.blocks import EnvironmentType, init_block, BlockDiagKet_DMRG
from local_solvers_old.helper_tn import get_mps_matching_inds, sum_tens

from gridTN_1D import GridTN1D



"""
borrows machinery of LocalSolver
but solves nonlinear functions f(A1 * x1, A2 * x2, ...) using projectors
out = f(P1 A1 P1^* P1 x1, P2 A2 P2^* P2 x2, ...)
where Ai are the "operators", xi are the "targets"

todo: let Ai be polynomial ops, e.g. a list specifying the powers of x
    define a special DMRG TargetBlock
Ai can also be a linear func (eg. rk4, solve Ax=b)
element-wise functions don't work.
"""

def get_ket_to_bra_inds(ket: qtn.MatrixProductState, bra: qtn.MatrixProductState, i:int):
    return get_mps_matching_inds(ket, bra, i)


def local_dmrg_evaluator(*terms: tuple[ Sequence[Union[GridTN1D, Sequence, Callable[[np.ndarray], np.ndarray]]],
                                        GridTN1D ],
                         direction: SweepDirection.LEFT, max_bond=None):
    """ terms are sequences of ((A, B, ...), x)
        where A, B are operators or functions that are applied to x in
        the order A(B(x))

        algorithm: compute rdm for each term
        sum together terms by summing rdms together (state-averaging method)
        projecting next site onto new rdm
    """
    opx, ref_x = terms[0]
    L = ref_x.L

    ref_x_data: 'qtn.MatrixProductState' = ref_x.data
    ref_x_conj: 'qtn.MatrixProductState' = ref_x.data.conj(mangle_inner=False)
    for ix in range(L):
        tens = ref_x_conj[ix]
        tens.reindex({ind: ind + '_' for ind in tens.inds if ind != ref_x_conj.site_ind(ix)}, inplace=True)
    ## also need to adjust ancilliary bonds

    ### build projector envs
    term_envs = []
    x_conjs = []
    for ops, x in terms:

        start = 0 if direction == SweepDirection.RIGHT else x.L - 1
        x.canonize(inplace=True, i=start)
        x[start].modify(apply=lambda data: data * 10**x.exponent)
        x.data.exponent = 0.0

        ### compute envs
        envs: Sequence[Optional['Block']] = []
        x_data: qtn.MatrixProductState = x.data
        helper_quimb.match_inner_inds(x_data, ref_x_data, inplace=True)
        x_conj = x_data.conj()  ## also need to reindex ancillas
        for ix in range(L):
            tens = x_conj[ix]
            tens.reindex({ind: ind + '_' for ind in tens.inds if ind != x_conj.site_ind(ix)}, inplace=True)
        x_conjs += [x_conj]
        for op in ops:
            if isinstance(op, GridTN1D):
                op_data: qtn.MatrixProductOperator = op.data
                op_block = init_block(EnvironmentType.DMRG, x_data, x_conj, op_data, cur_orthog=start)
                envs += [op_block]
            elif isinstance(op, (tuple, list, np.ndarray)):  ##
                op_block = BlockDiagKet_DMRG(x.L, x_data, cur_orthog=start)
                envs += [op_block]
            else:
                envs += [None]

        term_envs += [envs]

    ### compute func at each site
    site_inds = range(L) if direction == SweepDirection.RIGHT else range(L-1,-1,-1)
    for ix in site_inds:
        at_end = (ix == L - 1) if direction == SweepDirection.RIGHT else (ix == 0)
        ix2 = ix + 1 if direction == SweepDirection.RIGHT else ix - 1

        ### compute new site
        tot_rdm = None
        tot_site = None
        for x_conj, envs, (ops, x) in zip(x_conjs, term_envs, terms):
            site_tens = x.data[ix]
            # site_tens.modify(apply=lambda data: data * 10**x.exponent)

            for env, op in zip(envs, ops):
                if isinstance(op, GridTN1D):     ## operator is not None, apply projected operator
                    eff_op = env.get_projected(ix, 1, return_combined=False)
                    site_tens = qtn.tensor_contract(*eff_op.tensors, site_tens)
                    site_tens.modify(apply=lambda x: x * 10**eff_op.exponent)
                    site_tens.reindex( env.projected_bra_to_ket(ix, 1), inplace=True )
                elif isinstance(op, (list, tuple, np.ndarray)):  ## polynomial function
                    eff_op = env.get_projected(ix, 1, return_combined=False)
                    pow_terms = []
                    # ## zero
                    # if op[0] != 0:
                    #     pow_term = qtn.Tensor(np.ones(site_tens.shape) * op[0], inds=site_tens.inds)
                    #     pow_terms += [pow_term]
                    ## power of 1+
                    for power, coeff in enumerate(op):
                        pow_term = site_tens.copy()
                        eff_op = env.get_projected(ix, 1, return_combined=False)
                        if coeff != 0:
                            it = 0
                            while it < power:
                                it += 1
                                pow_term = qtn.tensor_contract(pow_term, *eff_op.tensors)
                                pow_term.modify(apply=lambda x: x * 10 ** eff_op.exponent)
                                pow_term.reindex(env.projected_bra_to_ket(ix, 1), inplace=True)
                            pow_coeff = pow_term.copy()
                            pow_coeff.modify(apply = lambda data: data * coeff)
                            pow_terms += [pow_coeff]
                    site_tens = sum_tens(pow_terms)
                    # site_tens.reindex(env.projected_bra_to_ket(ix, 1), inplace=True)
                elif op is not None:       ## perform elemental operation
                    site_data = site_tens.data
                    site_data = op(site_data)
                    site_tens = site_tens.copy()
                    site_tens.modify(data=site_data)
                else:
                    pass

            if at_end:
                if tot_site is None:
                    tot_site = site_tens
                else:
                    tot_site = helper_quimb.add_tensors(tot_site, site_tens, inplace=True)

            else:    # if not at_end:
                # horizontal bond to contract over (right ind if sweep l2r, left ind if sweep r2l)
                # x_bond = ref_x_data.bond(ix, ix2)
                if direction == SweepDirection.RIGHT:
                    x_bond = ref_x_data.bond(ix, ix + 1) if not at_end else None
                else:
                    x_bond = ref_x_data.bond(ix, ix - 1) if not at_end else None

                ## bonds to not contract over
                # ket_to_bra = get_ket_to_bra_inds(ref_x_data, ref_x_conj, ix)
                ket_iso = [ind for ind in ref_x_data[ix].inds if ind != x_bond]
                # bra_iso = [ket_to_bra[k] for k in ket_iso]
                new_bra_iso = [k + '_' for k in ket_iso]

                ## compute rdm
                site_conj = site_tens.conj()
                site_conj.reindex({k:b for k,b in zip(ket_iso, new_bra_iso)}, inplace=True)

                rdm = qtn.tensor_contract(site_conj, site_tens)
                if tot_rdm is None:
                    tot_rdm = rdm
                else:
                    tot_rdm = helper_quimb.add_tensors(tot_rdm, rdm)

                # print('modify tot rdm')
                # tot_rdm.modify(apply=lambda data: data * 2)

        if not at_end:
            ## eigendecomposition to obtain canonical form
            tot_rdm.transpose(*ket_iso, *new_bra_iso, inplace=True)
            nb = len(ket_iso)

            denmat_shape = tot_rdm.shape
            tens_shape = denmat_shape[:nb]
            sq_shape = np.prod(denmat_shape[:nb])

            # test = tot_denmat.data.reshape(sq_shape, sq_shape)
            # print('rho is H', np.linalg.norm(test - test.T.conj()))
            eigval, eigvec = np.linalg.eigh(tot_rdm.data.reshape(sq_shape, sq_shape))

            sort_inds = np.argsort(np.abs(eigval))[::-1]
            ev_max = eigval[sort_inds[0]]
            # eigval_inds = np.where(np.abs(eigval) > 1.0e-12 * ev_max)
            new_sort_inds = []
            for si in sort_inds:
                if np.abs(eigval[si]) > CUTOFF * np.abs(ev_max):
                    new_sort_inds += [si]
                else:
                    break
            sort_inds = new_sort_inds

            if max_bond is not None:
                eigvec = eigvec[:, sort_inds[:max_bond]]
            else:
                eigvec = eigvec[:, sort_inds]

            eigvec = eigvec.reshape(*tens_shape, -1)
            eigvec = qtn.Tensor(data=eigvec, inds=tuple(ket_iso) + (x_bond + '_',))

            ## update next site
            for x_conj, envs, (ops, x) in zip(x_conjs, term_envs, terms):
                eigvec.transpose_like(x.data[ix], inplace=True)
                next_site = qtn.tensor_contract(eigvec.conj(), x.data[ix], x.data[ix2])
                next_site.transpose_like(x.data[ix2], inplace=True)
                x.data[ix].modify(data=eigvec.data)
                x.data[ix2].modify(data=next_site.data)

                ket_to_bra = get_mps_matching_inds(x.data, x_conj, ix)
                # x_conj[ix].transpose_like(eigvec, inplace=True)
                x_conj[ix].transpose(*[ket_to_bra[ind] for ind in x.data[ix].inds], inplace=True)
                x_conj[ix].modify(data=eigvec.data.conj())
                ket_to_bra = get_mps_matching_inds(x.data, x_conj, ix2)
                x_conj[ix2].transpose(*[ket_to_bra[ind] for ind in x.data[ix2].inds], inplace=True)
                x_conj[ix2].modify(data=next_site.data.conj())

                for env in envs:
                    if env is not None:
                        env.extend_env(ix, direction=direction)
        else:
            print('tot_site at_end norm', tot_site.norm())
            pass

    tot_site.drop_tags()
    tot_site.add_tag(ref_x_data.site_tag_id.format(site_inds[-1]))
    out_mps = [ref_x_data[ix] for ix in site_inds][:-1] + [tot_site]
    if direction == SweepDirection.LEFT:
        out_mps = out_mps[::-1]

    out_tn = qtn.TensorNetwork([*out_mps])
    out_tn.view_as(qtn.MatrixProductState, inplace=True, L=L, cyclic=False, site_ind_id=ref_x_data.site_ind_id,
                   site_tag_id=ref_x_data.site_tag_id)
    # out_tn.exponent = ref_x_data.exponent

    out_gtn = GridTN1D(x.grid, out_tn)
    return out_gtn



