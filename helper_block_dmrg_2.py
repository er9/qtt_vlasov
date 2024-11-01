import numpy as np
import itertools
import quimb

from setup_.defaults import *
import quimb.tensor as qtn
from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
import helper_quimb as helper
import helper_dmrg
import helper_dmrg_2

from helper_dmrg import *
from helper_block_dmrg import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gridTN import GridTN

""" implicit solver for Ax = b, where x, b are composed of multiple components
"""


class BaseSolver(Enum):
    LOCAL = LocalSolver
    DMRG = DMRGSolver
    LINEAR = LinearSolver
    LINEAR2 = helper_dmrg_2.LinearSolver2


def implicit_solver_2(ncomps: int, A_mats: dict[tuple[int, int], Sequence['MPO_type']],
                      b_vecs: dict[int, Sequence['MPS_type']],
                      init_guess: dict[int, 'qtn.MatrixProductState'] = None, verbose_output=False,
                      nsites=2, init_direction=SweepDirection.RIGHT,
                      **solver_kwargs):
    print('BLOCK DMRG 2')
    solver = BlockedLinearSolver2(ncomps, init_guess, b_vecs, A_mats, **solver_kwargs)
    solver.solve(nsites, init_direction=init_direction)

    if verbose_output:
        return solver.kets, solver.err, solver.is_conv
    else:
        return solver.kets


class BlockedLinearSolver2(BlockedLinearSolver):
    def __init__(self, ncomps, trial_state: dict[int, 'qtn.MatrixProductState'],
                 targets: Optional[dict[int, Sequence['MPS_type']]] = None,
                 operators: Optional[dict[tuple[int, int], Sequence['MPO_type']]] = None,
                 is_H=True,
                 mps_inds: Sequence[int] = None,
                 norm_env0_Ls: dict[int, Sequence['qtn.Tensor']] = None,
                 norm_env0_Rs: dict[int, Sequence['qtn.Tensor']] = None,
                 ovlp_env0_Ls: dict[tuple[int, int], Sequence['qtn.Tensor']] = None,
                 ovlp_env0_Rs: dict[tuple[int, int], Sequence['qtn.Tensor']] = None,
                 err_env0_Ls: dict[int, Sequence['qtn.Tensor']] = None,
                 err_env0_Rs: dict[int, Sequence['qtn.Tensor']] = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_bond=None,
                 solve_type: SolveMethod = DEFAULT_SOLVE,
                 solvers=None,
                 ):

        # operators = [] if operators is None else list(operators)
        self.A_envsLs, self.A_envsRs = [], []
        self.b_envsLs, self.b_envsRs = [], []

        if solvers is not None:
            super(BlockedLinearSolver, self).__init__(ncomps, None, solvers=solvers)
        else:
            ## key (i,j). assumes only one operator per key
            ## -> operators   : [A_kj for all k]
            ## -> operators_H : [A_ki.H for all k]

            max_key = max(*targets.keys(), *[k[0] for k in operators.keys()])
            out_keys = range(max_key + 1)

            solver_keys = itertools.product(out_keys, out_keys)  ## oo, ii
            operators2 = {key: [operators.get((ok, key[1]), [None])[0] for ok in out_keys]
                          for key in solver_keys}

            operators2_H = {(key[1], key[0]): [helper.mpo_conj_transpose(mpo)
                                               if mpo is not None else None for mpo in mpos]
                            for key, mpos in operators2.items()}

            ## input / output inds for the solver, where i is connected to j
            # solver_keys = itertools.product(out_keys, out_keys)  ## oo, ii
            for key in list(operators2.keys()):
                is_none_1 = np.array([op is None for op in operators2[key]])
                is_none_2 = np.array([op is None for op in operators2_H[key]])
                either_none = [n1 or n2 for (n1, n2) in zip(is_none_1, is_none_2)]
                if all(either_none):
                    operators2.pop(key)
                    operators2_H.pop(key)

            init_state = trial_state
            super(BlockedLinearSolver, self).__init__(ncomps, init_state, targets=targets,
                                                      operators=operators2, operators_H=operators2_H,
                                                      mps_inds=mps_inds, conv_tol=conv_tol,
                                                      max_iter=max_iter, max_bond=max_bond, max_tot_iter=max_tot_iter,
                                                      base_solver=BaseSolver.LINEAR2,
                                                      norm_env0_Rs=norm_env0_Rs, norm_env0_Ls=norm_env0_Ls,
                                                      ovlp_env0_Rs=ovlp_env0_Rs, ovlp_env0_Ls=ovlp_env0_Ls,
                                                      err_env0_Rs=err_env0_Rs, err_env0_Ls=err_env0_Ls,
                                                      solvers=solvers)

            for k, solver in self.solvers.items():
                solver.is_H = True  # is_H
                solver.solve_type = solve_type

            self.solve_type = solve_type
            self.is_H = True  # is_H

    combined_ind_o = 'out_ind'
    combined_ind_i = 'in_ind'

    def _get_b_eff_dict_(self, sites, bra_inds):
        """
        BL * B * BR =  d/dT*[s] <x_j| A_ji.T | b_i>
        dict indexes xj
        """
        b_eff_dict = {}
        ### d/dT*[s] <x_j| A_ji.T | b_i>
        for key, solver in self.solvers.items():
            oo, ii = key
            # print('get b eff', key)
            out = solver._get_b_eff(sites)
            if out is not None:
                out.transpose(*bra_inds, inplace=True)
            else:
                shape = [solver.bra.ind_size(b_ind) for b_ind in bra_inds]
                out = qtn.Tensor(np.zeros(tuple(shape)), inds=bra_inds)

            ## \sum_i <x_j | A_ji.T | b_i>
            if oo not in b_eff_dict:
                b_eff_dict[oo] = out
            else:
                b_eff_dict[oo] += out

        return b_eff_dict

    def check_err_np(self):
        """ check err using numpy """
        As, bs, xs = {}, {}, {}
        iis, oos = [], []
        for (oo, ii), solver in self.solvers.items():

            for op_ind, op in enumerate(solver.operators):
                if op is None:  continue
                op_tens = op.contract(all) * 10 ** op.exponent
                op_tens.transpose(*[op.upper_ind_id.format(i) for i in solver.mps_inds],
                                  *[op.lower_ind_id.format(i) for i in solver.mps_inds],
                                  inplace=True)
                op_mat = op_tens.data.reshape(2 ** self.L, 2 ** self.L)
                # A = np.dot(op_mat, A)
                As[op_ind, ii] = op_mat

            if oo == ii:
                b = np.zeros(2 ** self.L)
                for target in solver.targets:
                    target_tens = target.contract(all) * 10 ** target.exponent
                    # print('target tens?', target_tens)
                    target_tens.transpose(*[target.site_ind_id.format(i) for i in solver.mps_inds], inplace=True)
                    target_vec = target_tens.data.reshape(-1)
                    b = b + target_vec

                bs[oo] = b
                oos += [oo]

                x_tens = solver.out.contract(all) * 10 ** solver.out.exponent
                x_tens.transpose(*[solver.out.site_ind_id.format(i) for i in solver.mps_inds], inplace=True)
                x_vec = x_tens.data.reshape(-1)
                xs[ii] = x_vec
                iis += [ii]

        # xAAx_ii = {}
        # Ax_s = {ii: {oo: np.dot(As.get((oo, ii), 0), xs[ii]) for oo in oos} for ii in iis}
        # for ii, iii in np.ndindex(6, 6):
        #     xAAx_ii[ii, iii] = np.sum([np.dot(Ax_s[iii][oo], Ax_s[ii][oo]) for oo in oos])
        # print('xAAx ii')
        # for k, val in xAAx_ii.items():
        #     print(k, val)

        errors2 = {}
        for oo in oos:
            # Ax_i = {ii: np.dot(As.get((oo, ii), 0), xs[ii]) for ii in iis}
            # print('b', oo, 'Ax', iis, [np.dot(Ax_i[ii].conj(), bs[oo]) for ii in iis])
            # print('xAAx', oo, iis, [np.linalg.norm(np.dot(As.get((oo, ii), 0), xs[ii])) ** 2 for ii in iis])
            # xAAx_ii = {(iii, ii): np.dot(Ax_i[iii], Ax_i[ii]) for ii, iii in np.ndindex(6, 6)}
            # print('xAAx_ii')
            # for k, val in xAAx_ii.items():
            #     print(k, val)
            Ax = np.sum([np.dot(As.get((oo, ii), 0), xs[ii]) for ii in iis], axis=0)
            errors2[oo] = np.linalg.norm(bs[oo] - Ax) ** 2
        # print('err2s', errors2)

        b2norms = {oo: np.linalg.norm(bs[oo]) ** 2 for oo in oos}
        err2 = np.sum([errors2[oo] for oo in oos]) / np.sum([b2norms[oo] for oo in oos])
        err = np.sqrt(err2)
        return err

    # @profile
    def check_err(self, dense=False) -> float:
        """ calculates sqrt(<b-Ax|b-Ax>) = sqrt(<Ax|Ax> + <b|b> - 2Re[<Ax|b>])
            for each output component:
                0 = \sum_i A_oi |x_i> - |b_o>
                \sum_i <A_oi x_i | A_oi x_i> - 2Re[<b_o|A_oi|x_i>] + <b_o|b_o>
        """
        # print('np err', self.check_err_np())
        dense = False
        if dense:
            raise NotImplementedError

        else:
            ## || |bo> - sum_i Aoi |xi> ||^2 =
            ## <bo|bo> - sum_i <bo | Aoi | xi> * 2 + sum_i,i' <xi'|Aoi'.T Aoi|xi>
            normAAs = {}  # key o: list [<A_oi x_i | A_oi x_i>]
            ovlps = {}  # key o: list [<b_o|A_oi|x_i>]
            b2_norms = {}  # key o

            # def compute_cross_Ax(solver1: LocalSolver, solver2: LocalSolver):
            #     ### <solver 2 | solver 1>
            #     A1, x1 = solver1.operators[0].copy(), solver1.ket.copy()
            #     A2, x2 = solver2.operators_H[0].copy(), solver2.ket.conj()
            #     x2.site_ind_id = A2.upper_ind_id
            #     ovlp = qtn.TensorNetwork([A1, x1, A2, x2])
            #     exponent = ovlp.exponent
            #     ovlp = ovlp.contract_tags(all)
            #     ovlp = ovlp * 10 ** exponent
            #     return ovlp

            for (oo, ii), solver in self.solvers.items():

                ### \sum_o <A_oj x_j | A_oi x_i>
                if ii not in normAAs:  normAAs[ii] = {}
                normAAs[ii][oo] = solver._get_xAAx_()  ## includes exponent
                # print('norm AA', oo, ii, normAAs[ii][oo])

                ### get < x_o | AH_oi | b_i >
                ovlp_b_tot = solver._get_xAb_()  ## includes exponent
                # print('ovlp b (x)', oo, '(b)', ii, ovlp_b_tot)
                ### \sum_o < x_o | AH_oi | b_i >
                if ii not in ovlps.keys():
                    ovlps[ii] = 2 * np.real(ovlp_b_tot)
                else:
                    ovlps[ii] += 2 * np.real(ovlp_b_tot)

                if ii not in b2_norms:
                    b_norm = solver.targets_norm  ## includes exponent
                    b2_norms[ii] = b_norm ** 2

            # for oo, normAAs_o in normAAs.items():
            #     print('norm AA ', oo, normAAs[oo].keys(), [norm_AA_i for ii, norm_AA_i in normAAs_o.items()])
            # print('ovlps', ovlps)

            normAAs = {oo: np.sum([norm_AA_i for ii, norm_AA_i in normAAs_o.items()]) for oo, normAAs_o in
                       normAAs.items()}
            # print('norm AAs', normAAs)
            # print('residual^2', (norm_Ax + self.targets_norm ** 2 - 2 * np.real(ovlp_b_tot)) / self.targets_norm**2)
            err2_o = {oo: (normAAs[oo] + b2_norms[oo] - ovlps[oo]) for oo in b2_norms.keys()}
            b2_norm = np.sum([(b2norm) for oo, b2norm in b2_norms.items()])

            err2 = np.sum([(val) for oo, val in err2_o.items()]) / b2_norm
            err = np.sqrt(np.abs(err2))

        # print('err2s', err2_o)
        print('err', err, self.conv_tol)
        self.err = err
        self.is_conv = err < self.conv_tol
        # exit()
        return err
