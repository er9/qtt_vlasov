import numpy as np
from abc import ABC

from setup_.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from setup_.quimb_TN1D import IndexedMPS, IndexedMPO
import helper_quimb as helper

from local_solvers_old.local_solver import LocalSolver, MPS_type, MPO_type, SweepDirection
from local_solvers_old.local_solver import DEFAULT_MAX_TOT_ITER, DEFAULT_CONV_TOL, DEFAULT_MAX_ITER, DEFAULT_MAX_WRONG_ITER
from local_solvers_old.environments_v2 import TargetEnv_DMRG, OperatorEnv_DMRG


class EigenSolver(LocalSolver, ABC):

    def __init__(self, ket_state: 'qtn.MatrixProductState',
                 operators: Sequence['MPO_type'],
                 operators_H: Sequence['MPO_type'] = None,
                 bra_state: Optional['qtn.MatrixProductState'] = None,
                 in_ind: int = 0, out_ind: int = 0,
                 op_env0_Ls: Sequence['qtn.Tensor'] = None,
                 op_env0_Rs: Sequence['qtn.Tensor'] = None,
                 A_envsLs: Optional[Sequence['OperatorEnv']] = None,
                 A_envsRs: Optional[Sequence['OperatorEnv']] = None,
                 normAA_env0_Ls: Sequence['qtn.Tensor'] = None,
                 normAA_env0_Rs: Sequence['qtn.Tensor'] = None,
                 mps_inds: Optional[tuple[int]] = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_wrong_iter=DEFAULT_MAX_WRONG_ITER, max_bond=None):

        super().__init__(ket_state, targets=None, operators=operators, operators_H=operators_H, bra_state=bra_state,
                         in_ind=in_ind, out_ind=out_ind, mps_inds=mps_inds,
                         op_env0_Rs=op_env0_Rs, op_env0_Ls=op_env0_Ls, A_envsRs=A_envsRs, A_envsLs=A_envsLs,
                         conv_tol=conv_tol, max_iter=max_iter, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter,
                         max_bond=max_bond)



        self.normAA_env0_Ls = [None] * (self.num_operators ** 2) if normAA_env0_Ls is None else normAA_env0_Ls
        self.normAA_env0_Rs = [None] * (self.num_operators ** 2) if normAA_env0_Rs is None else normAA_env0_Rs

        # self.A2_envsRs, self.A2_envsLs = [], []
        # for i, j in np.ndindex(self.num_operators, self.num_operators):
        #     A, AT = self.operators[i], self.operators_H[j]
        #     if isinstance(A, qtn.MatrixProductOperator):
        #         AAT = MatrixProductOperatorTN([A,AT])
        #     else:
        #         AAT = A.apply(AT)
        #
        #     self.A2_envsLs += [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, AAT,
        #                                    init_env=norm_env0_Ls[i])]
        #     self.A2_envsRs += [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, AAT,
        #                                    init_env=norm_env0_Rs[i])]
        #     ## init env only works if MPOs don't have ancillas
        #
        # # self.A2_envsLs = [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, [self.A, self.AT],
        # #                               init_env=norm_env0_Ls[i]) for i in range(self.num_operators)]
        # # self.A2_envsRs = [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, [self.A, self.AT],
        # #                               init_env=norm_env0_Rs[i]) for i in range(self.num_operators)]
        #
        # # self.left_envs = [self.A_envsL, self.A2_envsL]
        # # self.right_envs = [self.A_envsR, self.A2_envsR]

        # self.left_envs = self.A_envsLs # + self.A2_envsLs
        # self.right_envs = self.A_envsRs # + self.A2_envsRs

        ### initial results
        self.eigval = self.estimate_eigval()
        self.prev_eigval = None
        # self.update_results()


    def eigsolve(self, target_eig=0):
        """ solve via numpy/scipy?
        """
        op = np.empty(self.operators[0].shape)
        for mpo in self.operators:
            op_mat = mpo.to_dense()  # default inds seq is upper, lower
            op += op_mat

        npts = int(np.sqrt(op.size))
        op = op.reshape(npts, npts)
        print('op shape', op.shape)

        plt.imshow(op)
        plt.show()

        eigvals = np.linalg.eigvals(op)
        sort_inds = np.argsort(np.imag(eigvals))
        print('np eigvals', eigvals[sort_inds])
        eigval = eigvals[sort_inds[target_eig]]
        print('np eigval', eigval)

        return eigval


    def get_outer_inds(self, left_site_pos: int, nsites: int, is_bra=False):
        """ get outer inds of a section of an MPS
        """
        site_pos = list(range(left_site_pos, left_site_pos + nsites))

        if is_bra:
            outer_inds = [self.bra.site_ind_id.format(i) for i in site_pos]
            anc_L = self.A_envsLs[0].bra_horizontal_bonds[site_pos[0]]  ## all envs should be the same
            anc_R = self.A_envsRs[0].bra_horizontal_bonds[site_pos[-1]]
        else:
            outer_inds = [self.ket.site_ind_id.format(i) for i in site_pos]
            anc_L = self.A_envsLs[0].ket_horizontal_bonds[site_pos[0]]
            anc_R = self.A_envsRs[0].ket_horizontal_bonds[site_pos[-1]]

        if isinstance(anc_L, (list, tuple)):
            outer_inds += [*anc_L]  # a tuple / list for MPSTN
        elif isinstance(anc_L, str):
            outer_inds += [anc_L]  # norm str for qtn.MPS
        ## else is None

        if isinstance(anc_R, (list, tuple)):
            outer_inds += [*anc_R]
        elif isinstance(anc_R, str):
            outer_inds += [anc_R]
        ## else is None

        return outer_inds


    def _site_solve(self, left_site_pos: int, nsites: int, **kwargs) -> tuple[qtn.Tensor, Numeric]:
        """ site optimization for eigensolver. ie: minimizes <psi|H|psi> / <psi|psi>
            H_eff = (d/dT*[i] d/dT*[i] <psi|H|psi>)
            N_eff = d/dT*[i] d/dT[i] <psi|psi> = I if system is properly canonicalized
            --> minimize <psi|H|psi> - lambda <psi|psi>
            --> d/dT*[i] <psi|H|psi> - lambda d/dT*[i] <psi|psi> = 0
            --> H_eff * T[i] - lambda * N_eff * T[i] = 0
            lambda will be the energy of the state

            sites: qtn.Tensor, to be optimized (of the desired shape)
            left: qtn.Tensor, left environment <psi|H|psi>
            right: qtn.Tensor, right environment <psi|H|psi>
            H: Sequence[qtn.Tensor] or qtn.TensorNetwork, corresponding MPO site of operator H
        """
        site_pos = list(range(left_site_pos, left_site_pos + nsites))
        site_inds = [self.mps_inds[ix] for ix in site_pos]
        # site_inds = slice(site_ind_0, site_ind_0 + len(site_pos))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|A|x>
        H_eff_tot = None
        input_inds = self.get_outer_inds(left_site_pos, nsites)
        output_inds = self.get_outer_inds(left_site_pos, nsites, is_bra=True)

        for i in range(self.num_operators):
            A_left = self.A_envsLs[i][site_pos[0]]
            A_right = self.A_envsRs[i][site_pos[-1]]

            H_eff = qtn.TensorNetwork([])
            H_eff.add([self.operators[i][ix] for ix in site_inds])
            if A_left is not None:
                H_eff.add(A_left)
            if A_right is not None:
                H_eff.add(A_right)

            H_eff_exponent = self.A_envsLs[i].exponent

            H_eff = H_eff.contract_tags(all)
            H_eff.transpose(*output_inds, *input_inds, inplace=True)
            H_eff.modify(apply=lambda data: data * 10 ** H_eff_exponent)

            if H_eff_tot is None:
                H_eff_tot = H_eff
            else:
                H_eff_tot.modify(apply=lambda data: data + H_eff.data)

        # ### L * H * R = d/dT*[i] d/dT[i] <psi|H|psi>
        # H_eff = qtn.TensorNetwork([left, right])
        # H_eff.add(H)
        # H_eff = H_eff.contract_tags(all)
        # H_eff.transpose(*output_inds, *input_inds, inplace=True)

        ### H_eff[i] * T[i] = lambda T[i]  (assumes in canonical form)
        shape_out = H_eff_tot.shape[:len(output_inds)]
        shape_in = H_eff_tot.shape[len(output_inds):]  ## shape_in = shape_out
        H_eff_data = H_eff_tot.data.reshape(np.prod(shape_out), -1)
        if self.is_H:
            eigvals, eigvecs = linalg.eigh(H_eff_data)
        else:
            eigvals, eigvecs = linalg.eig(H_eff_data)

        eigval = eigvals[0]
        eigvec = eigvecs[:, 0]

        eigvec = eigvec.reshape(shape_in)
        eigvec_tens = qtn.Tensor(data=eigvec, inds=tuple(input_inds))
        return eigvec_tens, eigval

    # def dmrg_1site(self, **dmrg_kwargs):
    #     """ dmrg sweep with 1-site optimization
    #     """
    #     return self.solve(1, **dmrg_kwargs)
    #
    # def dmrg_2site(self, **dmrg_kwargs):
    #     """ dmrg sweep with 2-site optimization
    #     """
    #     return self.solve(1, **dmrg_kwargs)

    def sweep_l2r(self, nsites, *args, verbose=False, **kwargs):

        self.prev_eigval = self.eigval

        for i in range(self.L - nsites):
            # ix = self.get_mps_ind(i)

            site_i, eigval = self._site_solve(i, nsites)
            # site_i = self._site_solve(inds)

            ## update ket (+ canonicalization)
            if nsites == 1:
                self._update_1site(i, site_i, direction=SweepDirection.RIGHT)
            elif nsites == 2:
                self._update_2site(i, site_i, direction=SweepDirection.RIGHT)
            else:
                raise NotImplementedError

            ## update A, b envs (left)
            self._update_envs_left(i, canonize=False)

        self.eigval = eigval
        return

    def sweep_r2l(self, nsites, *args, verbose=False, **kwargs):

        self.prev_eigval = self.eigval

        ## left sweep
        for i in range(self.L - 1, nsites - 1, -1):

            eigval, site_i = self._site_solve(i - nsites + 1, nsites)

            ## update ket, bra
            if nsites == 1:
                self._update_1site(i, site_i, SweepDirection.LEFT)
            elif nsites == 2:
                self._update_2site(i, site_i, SweepDirection.LEFT)
            else:
                raise NotImplementedError

            ## update A, b envs (right)
            self._update_envs_right(i, canonize=False)

        # err = self.check_err(0, nsites)
        self.eigval = eigval
        return


    # def solve(self, nsites, *args, init_direction=SweepDirection.RIGHT, canonize=False, verbose=False, **kwargs):
    #
    #     L = self.ket.L
    #     canon_site = 0 if init_direction == SweepDirection.RIGHT else L - 1
    #     self.canonize(canon_site)  # canonicalizees ket and bra
    #
    #     ## ovlp <Ax|x>, <Ax|Ax> init envs
    #     if init_direction == SweepDirection.RIGHT:
    #         self._build_all_envs_right(nsites, canonize=False)
    #     else:
    #         self._build_all_envs_left(nsites, canonize=False)
    #
    #     ## iterative solver
    #     it = 0
    #     err = self.check_err(get_approx_err=True)
    #     conv_it, prev_err = 0, err
    #     eigval = self.estimate_eigval()
    #     self.eigval = eigval
    #     self.prev_eigval = eigval
    #
    #     # print('init err', err)
    #     # print('init eigval', eigval)
    #     get_approx_err = True
    #     min_ket, min_err, min_eigval = self.ket, err, eigval  ## save minimum error solns
    #     min_solver = self
    #     direction = init_direction
    #     while err > self.conv_tol and conv_it < self.max_iter and it < self.max_tot_iter:
    #
    #
    #
    #         ## right sweep
    #         if direction == SweepDirection.RIGHT:
    #             self.sweep_l2r(nsites, verbose=verbose, **kwargs)
    #             err = self.check_err(get_approx_err, 0, nsites)
    #
    #         else:
    #             self.sweep_r2l(nsites, verbose=verbose, **kwargs)
    #             err = self.check_err(get_approx_err, 0, nsites)
    #
    #         if verbose:
    #             print('err', it, err)
    #
    #         if np.abs((prev_err - err) / err) < 1.0e-4:
    #             conv_it += 1
    #
    #         prev_err = err
    #
    #         it += 1
    #
    #         if err < min_err:
    #             min_solver = self.copy(deep=True)
    #             min_err = err
    #         else:
    #             print('Warning: solve error went up', err, min_err)
    #
    #         direction *= -1  ## swaps sweep direction
    #
    #     ## revert to optimal results
    #     if err > min_err:
    #         return min_solver.eigval, min_solver.ket
    #         # self._ket = min_solver.ket
    #         # self._bra = min_solver.bra
    #         # self.A_envsRs = min_solver.A_envsRs
    #         # self.A_envsLs = min_solver.A_envsLs
    #         # self.err = min_solver.err
    #         # self.eigval = min_solver.eigval
    #
    #     return self.eigval, self.ket

    def check_err(self, get_approx_err: bool=True, left_site_ind: int=0, nsites: int=1) -> 'float':
        if get_approx_err:
            # eigval = self.estimate_eigval()
            # err = np.abs(eigval - self.eigval)
            err = np.abs(self.prev_eigval - self.eigval)
        else:
            err = self.check_err_exact(left_site_ind, nsites)
        return err

    def check_err_exact(self, left_ind: int, nsites: int):
        """ error = <x|A^T A|x> - <x|A|x>^2
        """
        old_site_ind = self.bra.site_ind_id

        if self.num_operators > 0:
            self.bra.site_ind_id = self.operators_H[-1].upper_ind_id
            # self.bra.site_ind_id = self.operators_H[-1].lower_ind_id

        ## <xA|Ax>
        exp_AA_tot = 0.0
        for i, j in np.ndindex(self.num_operators, self.num_operators):
            normAA_L = self.normAA_env0_Ls[i * self.num_operators + j]
            normAA_R = self.normAA_env0_Rs[i * self.num_operators + j]
            # norm_Ax = qtn.TensorNetwork([self.ket, self.bra] + [mpo for mpo in self.operators] +
            #                             [mpo for mpo in self.operators_H], virtual=True)
            norm_Ax = qtn.TensorNetwork([self.ket, self.bra, self.operators[i], self.operators_H[j]], virtual=True)
            if normAA_L is not None:
                norm_Ax.add(normAA_L)
            if normAA_R is not None:
                norm_Ax.add(normAA_R)

            A_eff_exponent = norm_Ax.exponent  ## self.A_envsL equivalent exponent included in normAA_env0_Ls?
            exp_AA = norm_Ax.contract_tags(all)
            exp_AA *= 10 ** A_eff_exponent
            exp_AA_tot += exp_AA

        self.bra.site_ind_id = old_site_ind

        ## <x|A|x>
        exp_A_tot = 0.0
        for i in range(self.num_operators):
            env_tens = []
            if self.A_envsLs[i][left_ind] is not None:
                env_tens += [self.A_envsLs[i][left_ind]]
            if self.A_envsRs[i][left_ind + nsites - 1] is not None:
                env_tens += [self.A_envsRs[i][left_ind + nsites - 1]]
            # env_tens = [self.A_envsLs[i][left_ind], self.A_envsRs[i][left_ind + nsites - 1]]
            ket_tens = [t for t in self.ket[left_ind:left_ind + nsites]]
            bra_tens = [t for t in self.bra[left_ind:left_ind + nsites]]
            mpo = self.operators[i]
            if isinstance(mpo, qtn.MatrixProductOperator):
                mpo_tens = [t for t in mpo[left_ind:left_ind + nsites]]
            else:
                mpo_tens = []
                for x in range(left_ind, left_ind + nsites):
                    mpo_tens += mpo[x]

            exp_A = qtn.TensorNetwork(env_tens + ket_tens + bra_tens + mpo_tens)
            # exp_A = qtn.TensorNetwork([self.A_envsL[left_ind], self.A_envsR[left_ind + nsites - 1],
            #                            self.ket[left_ind:left_ind + nsites], self.bra[left_ind:left_ind+nsites]] +
            #                           [mpo[left_ind:left_ind + nsites] for mpo in self.operators])
            #### !!! CHANGED HERE
            A_eff_exponent = self.A_envsLs[i].exponent
            exp_A = exp_A.contract_tags(all)
            exp_A *= 10 ** A_eff_exponent
            exp_A_tot += exp_A

        err = np.abs(exp_A_tot ** 2 - exp_AA_tot)
        self.err = err
        self.is_conv = err < self.conv_tol
        return err


    def estimate_eigval(self):
        eigval = 0
        for i in range(self.num_operators):
            eigval += helper.expectation_value(self.ket, self.operators[i], self.bra)
        norm = helper.ovlp(self.ket, self.bra)
        return eigval / norm


    def solve(self, nsites, init_direction=SweepDirection.RIGHT, canonize=False, verbose=False, **kwargs):
        out_solver = super().solve(nsites, init_direction=init_direction, canonize=canonize, verbose=verbose)

        if out_solver.err < self.err:
            self._ket = out_solver.bra
            self._bra = out_solver.bra
            self.A_envsRs = out_solver.A_envsRs
            self.A_envsLs = out_solver.A_envsLs
            self.err = out_solver.err
            self.eigval = out_solver.eigval

        return out_solver.bra, out_solver.eigval


