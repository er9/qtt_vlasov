import numpy as np
from abc import ABC

from setup.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
from setup.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from setup.quimb_TN1D import IndexedMPS, IndexedMPO
import helper_quimb as helper
import helper_TE

import local_solvers_old.helper_tn as helper_tn
from local_solvers_old.local_solver import LocalSolver, MPS_type, MPO_type, SweepDirection
from local_solvers_old.local_solver import DEFAULT_MAX_TOT_ITER, DEFAULT_CONV_TOL, DEFAULT_MAX_ITER, DEFAULT_MAX_WRONG_ITER
from local_solvers_old.environments_v2 import TargetEnv_DMRG, OperatorEnv_DMRG
from local_solvers_old.linear_solver import qtn_conjugate_gradient_descent_1site, qtn_conjugate_gradient_squared_1site

"""
what about burgers df/dt = -1/3 d/dx f^3
explicit: f(t+1) = f(t) - 1/3 d/dx f(t)^3 * dt
                 ~ P f(t) - 1/3 P d/dx f(t)^3 dt
                 ~ P f(t) - 1/3 P d/dx PT P f(t) PT P f(t) PT P f(t) * dt
implicit: (f(t+1) - f(t))/dt = -1/3 d/dx (f(t+1)^3 + f(t)^3)/2
           f(t+1) / dt + d/dx (f(t+1)^3) / 6 = f(t) / dt - d/dx f(t)^3 / 6
           i don't even know how to solve this
           exact solution with lax-pairs?
        
To do: 
implement and verify P f(t) PT P f(t) PT P f(t) ~ P f(t)^3
with naive and "informed" implementation
implement in "local_evaluator.py" ?  a "power" function for "informed" implementation
"""

class TimeSolver(LocalSolver):

    def __init__(self, ket_state: 'qtn.MatrixProductState',
                 operators: Optional[Sequence['MPO_type']] = None,
                 sources: Optional[Sequence['MPS_type']] = None,
                 bra_state: Optional['qtn.MatrixProductState'] = None,
                 in_ind: int = 0, out_ind: int = 0,
                 mps_inds: Sequence[int] = None,
                 # op_env0_Ls: Sequence['qtn.Tensor'] = None,
                 # op_env0_Rs: Sequence['qtn.Tensor'] = None,
                 # target_env0_Ls: Sequence['qtn.Tensor'] = None,
                 # target_env0_Rs: Sequence['qtn.Tensor'] = None,
                 A_envsLs: Optional[Sequence['OperatorEnv']] = None,
                 A_envsRs: Optional[Sequence['OperatorEnv']] = None,
                 b_envsLs: Optional[Sequence['TargetEnv']] = None,
                 b_envsRs: Optional[Sequence['TargetEnv']] = None,
                 te_order=0, dt=0.0, is_H=False, backprop_edge=False,
                 **env_kwargs):

        super().__init__(ket_state, operators=operators, targets=sources, bra_state=bra_state,
                         in_ind=in_ind, out_ind=out_ind, mps_inds=mps_inds,
                         A_envsLs=A_envsLs, A_envsRs=A_envsRs, b_envsLs=b_envsLs, b_envsRs=b_envsRs,
                         )
        self.te_order = te_order
        self.dt = dt
        self.is_H = is_H
        self.backprop_edge = backprop_edge


    def _site_solve(self, left_site_pos: int, nsites: int, sweep_direction=SweepDirection.RIGHT, **kwargs
                    ) -> tuple[qtn.Tensor, Numeric]:

        return self.site_time_evolution(self.dt, left_site_pos=left_site_pos, nsites=nsites,
                                        sweep_direction=sweep_direction)

    def site_time_evolution(self, dt, left_site_pos, nsites=1,
                            sweep_direction=SweepDirection.RIGHT,
                            ) -> tuple[qtn.Tensor, Numeric]:
        raise NotImplementedError

    def take_time_step(self, dt, grid=None, do_adapt=False, **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        # L = self.ket.L
        # max_bond = self.max_bond
        # do_adapt = False

        self.canonize(0)

        ## ovlp <Ax|b>, <Ax|x> init envs
        self._build_all_envs_right(1, canonize=False)
        self.take_time_step_l2r(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)
        # exit()

        ### right to left sweep
        self.take_time_step_r2l(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)

        print('done sweep')
        # exit()
        return self.ket


    def take_time_step_l2r(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, verbose=False,
                           **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        max_bond = self.max_bond
        do_adapt = False
        print('l2r tdvp V2', dt, 'do adapt', do_adapt)
        # print('ket L', L)
        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if canonize:
            self.canonize(0)   ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_right(1, canonize=False)

        ## right sweep
        nsites = 0
        direction = SweepDirection.RIGHT
        site_ind = 0
        cur_orthog = 0
        while site_ind < L:

            if site_ind == L - 1 and nsites == 2:
                ## previously updated L-2, L-1
                if verbose:  print('continue', site_ind)
                site_ind += 1
                continue

            ## num sites for update
            if site_ind == L - 1:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                max_bond_ = min(max_bond, 2**(site_ind+1), 2**(L-site_ind-1))
                adapt = do_adapt and self.ket.bond_size(site_ind, site_ind + 1) < max_bond_

            nsites = 2 if adapt else 1

            ## forward propagation
            if verbose:  print('FORWARD prop sites (LR)', list(range(site_ind,site_ind+nsites)) )
            self.site_time_evolution(dt, left_site_pos=site_ind, nsites=nsites, compress_direction=direction)
                ## updates environment within function to site_ind + nsites - 1
            site_ind = site_ind + nsites - 1 if nsites > 1 else site_ind + nsites

        if self.backprop_edge:
            raise NotImplementedError('backprop of last site in TDVP not implemented')

        print('done left to right sweep')
        return self.ket


    def take_time_step_r2l(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, verbose=False,
                           **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        do_adapt = False
        max_bond = self.max_bond
        print('r2l tdvp V2', dt, 'do adapt', do_adapt)

        if canonize:
            self.canonize(L-1)  ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_left(1, canonize=False)

        ### right to left sweep
        nsites = 0
        direction = SweepDirection.LEFT
        site_ind = L - 1
        cur_orthog = L - 1
        while site_ind >= 0:

            if site_ind == 0 and nsites == 2:
                ## previously updated 0, 1
                if verbose:   print('continue', site_ind)
                site_ind -= 1
                continue

            if site_ind == 0:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                max_bond_ = min(max_bond, 2 ** site_ind, 2 ** (L - site_ind))  # if site_ind != L-1 else 2
                adapt = do_adapt and self.ket.bond_size(site_ind, site_ind - 1) < max_bond_

            nsites = 2 if adapt else 1
            cur_orthog = site_ind - nsites + 1    ## get left site ind

            if verbose:  print('FORWARD prop sites (RL)', list(range(cur_orthog,cur_orthog+nsites)) )
            self.site_time_evolution(dt, cur_orthog, nsites=nsites, sweep_direction=direction)
            site_ind = site_ind - nsites + 1 if nsites > 1 else site_ind - nsites

        if self.backprop_edge:
            raise NotImplementedError('backprop of last site in TDVP not implemented')

        print('done right to left sweep')
        return self.ket



    def get_time_evolved_site(self, dt, left_site_pos, nsites=1, T1=None, te_order=4) -> 'qtn.Tensor':

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))
        bra_to_ket_inds = {b: k for k, b in ket_to_bra_inds.items()}

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self.get_A_effs(left_site_pos, nsites)    # A exponent only
        # b_effs = self.get_b_eff(left_site_pos, nsites)     # source exponent only
        # b_effs_scaled = [helper.scalar_multiply(b_eff, 10 ** (-self.ket.exponent)) for b_eff in b_effs]
        b_eff = self.get_b_eff(left_site_pos, nsites)  # source exponent only
        if b_eff is not None:
            b_eff_scaled = b_eff.copy()
            b_eff_scaled.modify(apply=lambda x: 10 ** (-self.ket.exponent) * x)
            # b_eff = helper.scalar_multiply(b_eff, 10 ** (-self.ket.exponent))
        else:
            b_eff_scaled = None

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        def deriv_func(dist_submpx_, dt=1, **kwargs) -> 'qtn.Tensor':
            """ for df/dt = Af + b
            """
            eff_Ax = []
            for deriv_tn in A_effs:
                eff_Ax += [qtn.TensorNetwork([deriv_tn, dist_submpx_])]
                ## dist_submpx exponent should equal 0

            if b_eff is not None:
                eff_Ax += [qtn.TensorNetwork([b_eff])]
            # for source_tens in b_effs_scaled:
            #     eff_Ax += [qtn.TensorNetwork([source_tens])]

            out = helper_tn.sum_eff_TNs(eff_Ax, transpose_bonds=bonds_o)
            # out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
            out.reindex({bo: bi for bo, bi in zip(bonds_o, bonds_i)}, inplace=True)
            out.modify(apply=lambda x: x * dt)
            return out

        def add_func(tens1, tens2, inplace=False, is_list=True, **kwargs) -> 'qtn.Tensor':
            if is_list:
                sum = [t.copy() for t in tens1] if inplace else tens1
            else:
                sum = tens1 if inplace else tens1.copy()

            if is_list:
                sum += tens2
                return sum
            else:
                if not isinstance(tens2, list):
                    tens2 = [tens2]
                for t2 in tens2:
                    if isinstance(t2, qtn.TensorNetwork):
                        t2 = t2.contract()
                        t2.reindex(bra_to_ket_inds, inplace=True)
                    sum = helper.add_tensors(sum, t2)
                return sum

        def scale_func(tens, scale_val, inplace=False, **kwargs) -> 'qtn.Tensor':
            out = tens if inplace else tens.copy()
            out.modify(apply=lambda x: x * scale_val)
            return out

        def euler_func(state, dt, deriv0, **kwargs) -> 'qtn.Tensor':
            return add_func(state, scale_func(deriv0, dt), is_list=False)


        def scale_tn_func(tens, scale_val, inplace=False, **kwargs) -> 'qtn.TensorNetwork':
            # out = tens if inplace else tens.copy()
            if isinstance(tens, list):
                tens = [helper.scalar_multiply(t, scale_val, inplace=inplace) for t in tens]
            else:
                tens = helper.scalar_multiply(tens, scale_val, inplace=inplace)
            # out.modify(apply=lambda x: x * scale_val)
            return tens

        output_to_input_inds = {bo: bi for bo, bi in zip(bonds_o, bonds_i)}

        ## implicit solver
        def solve_func_cgd(deriv_ops, explicit_contribution, init_guess=None,
                           max_iter=None, conv_tol=None, **kwargs):

            conv_kwargs = {}
            conv_kwargs['max_iter'] = 500 if max_iter is None else max_iter
            conv_kwargs['conv_tol'] = 1.0e-6 if conv_tol is None else conv_tol

            init_guess = T1.copy() if init_guess is None else init_guess
            out, err = qtn_conjugate_gradient_squared_1site(deriv_ops, explicit_contribution, init_guess,
                                                            output_to_input_inds, **conv_kwargs)
            print('solve err', err)
            return out

        def solve_func_lsq(deriv_ops, explicit_contribution, init_guess=None,
                           max_iter=None, conv_tol=None, **kwargs):

            shape_ = explicit_contribution.shape
            size_ = int(np.prod(shape_))

            conv_kwargs = {}
            conv_kwargs['max_iter'] = 500 if max_iter is None else max_iter
            conv_kwargs['conv_tol'] = 1.0e-6 if conv_tol is None else conv_tol

            A_eff_tens = helper_tn.sum_eff_TNs(deriv_ops, [*bonds_o, *bonds_i])
            explicit_contribution = explicit_contribution.transpose(*bonds_i)

            Amat = A_eff_tens.data.reshape(size_,size_)
            bvec = explicit_contribution.data.reshape(size_)

            out_data = np.linalg.solve(Amat, bvec)
            out = qtn.Tensor(out_data.reshape(shape_), inds=bonds_i)
            return out

        def identity_func(state: Union['qtn.Tensor', 'qtn.TensorNetwork']):
            shape_ = state.shape
            size_ = int(np.prod(shape_))
            iden = qtn.Tensor( np.eye(size_).reshape(*shape_, *shape_),
                               inds=(*bonds_o,*bonds_i) )
            return qtn.TensorNetwork([iden])

        if te_order == 0:
            print('exact te', site_inds)
            out = helper_TE.exact(T1, dt, exact_func)
        elif te_order == 4:
            out = helper_TE.rk4(T1, dt, euler_func, deriv_func, add_func, scale_func)
        elif te_order == 22:
            print('cn te', site_inds)
            ## beff_dict:  (implied) exponent = x.exponent
            ## Aeff_dict:  exponent = A.exponent
            out = helper_TE.solve_crank_nicolson(T1, euler_func, deriv_func, add_func, scale_tn_func,
                                                 A_effs, b_eff_scaled, identity_func, solve_func_cgd, dt)
                                                 # Aeff_dict, identity_func, solve_func_lsq, dt)
        elif te_order == 228:
            print('cn te 0.8', site_inds)
            out = helper_TE.solve_crank_nicolson(T1, euler_func, deriv_func, add_func, scale_tn_func,
                                                 A_effs, b_eff_scaled, identity_func, solve_func_cgd, dt, weight=0.8)
        else:
            raise NotImplementedError

        return out