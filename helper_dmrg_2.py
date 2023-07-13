from helper_dmrg import *

class LinearSolver2(LinearSolver):
    """ solve Ax - b by actually minimizing || Ax - b || = <xA|Ax> - <xA|b> - <b|Ax> + <b|b>
        meaning that we need to solve
    """
    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 targets: Optional[Sequence['MPS_type']] = None,
                 operators: Optional[Sequence['MPO_type']] = None,
                 is_H = True,
                 mps_inds: Sequence[int] = None,
                 norm_env0_Ls: Sequence['qtn.Tensor'] = None,
                 norm_env0_Rs: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 err_env0_Ls: Sequence['qtn.Tensor'] = None,
                 err_env0_Rs: Sequence['qtn.Tensor'] = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_bond=None,
                 solve_type:SolveMethod=DEFAULT_SOLVE):

        # operators = [] if operators is None else list(operators)
        self.A_envsLs, self.A_envsRs = [], []
        self.b_envsLs, self.b_envsRs = [], []

        init_state = trial_state
        super().__init__(init_state, targets=targets, operators=operators, mps_inds=mps_inds, conv_tol=conv_tol,
                         max_iter=max_iter, max_bond=max_bond, max_tot_iter=max_tot_iter)

        if self.num_operators > 1:
            raise NotImplementedError


        norm_env0_Ls = [None] * self.num_operators if norm_env0_Ls is None else norm_env0_Ls
        norm_env0_Rs = [None] * self.num_operators if norm_env0_Rs is None else norm_env0_Rs

        opH_ops = [MatrixProductOperatorTN([self.operators[i], self.operators_H[i]]) for i in range(self.num_operators)]

        self.A_envsLs = [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, opH_ops[i],
                                     init_env=norm_env0_Ls[i], mps_inds=self.mps_inds)
                            for i in range(self.num_operators)]
        self.A_envsRs = [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, opH_ops[i],
                                     init_env=norm_env0_Rs[i], mps_inds=self.mps_inds)
                            for i in range(self.num_operators)]
        # self.A_envsL = Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, self.operators,
        #                            init_env = norm_env0_L)
        # self.A_envsR = Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, self.operators,
        #                            init_env = norm_env0_R)

        #### only works for num_operators == 1
        self.b_envsLs, self.b_envsRs = [], []
        for ib in range(self.num_targets):
            b_envL = Environment(self.L, EnvironmentSide.LEFT, self.targets[ib], self.bra, self.operators_H[0],
                                 mps_inds = self.mps_inds,
                                 init_env = ovlp_env0_Ls[ib] if ovlp_env0_Ls is not None else None )
            b_envR = Environment(self.L, EnvironmentSide.RIGHT, self.targets[ib], self.bra, self.operators_H[0],
                                 mps_inds = self.mps_inds,
                                 init_env=ovlp_env0_Rs[ib] if ovlp_env0_Rs is not None else None )
            self.b_envsLs += [b_envL]
            self.b_envsRs += [b_envR]

        # self.left_envs  = self.A_envsLs + self.b_envsLs
        # self.right_envs = self.A_envsRs + self.b_envsRs

        self.solve_type = solve_type
        self.is_H = True  # is_H
        self.err_env0_Ls, self.err_env0_Rs = err_env0_Ls, err_env0_Rs
        ### each:  <xA|Ax>, <b|Ax>, <b|b> ancilla envs

        self.canon_direction = 0

    @property
    def bra_site_ind(self):
        """ desired bra site index """
        return self.operators_H[0].upper_ind_id

    @property
    def target_site_ind(self):
        return self.operators_H[0].lower_ind_id  # self.ket.site_ind_id

    def _get_A_effs(self, sites: Union[int, slice], return_sum=False) \
            -> Optional[Union[qtn.Tensor,Sequence[qtn.TensorNetwork]]]:
        """
        sites: int or slice(start, stop, step)
        """
        site_pos = list(range(self.L))[sites]
        site_inds = self.mps_inds[sites]

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = []
        for i in range(self.num_operators):
            A_left = self.A_envsLs[i][site_pos[0]]
            A_right = self.A_envsRs[i][site_pos[-1]]

            A_eff = qtn.TensorNetwork([])
            for A, AH in zip(self.operators, self.operators_H):
                if isinstance(site_inds, (list, tuple)):
                    for si in site_inds:
                        A_eff.add(A[si])
                        A_eff.add(AH[si])
                else:
                    A_eff.add(A[site_inds])
                    A_eff.add(AH[site_inds])
            if A_left is not None:
                A_eff.add(A_left)
            if A_right is not None:
                A_eff.add(A_right)
            # print('A eff exponent', A_eff.exponent, self.A_envsL.exponent)
            #### !!! CHANGED HERE
            A_eff.exponent = self.A_envsLs[i].exponent + A_eff.exponent
            A_effs += [A_eff]

        def sum_Aeffs():
            A_eff_ = None
            for A_eff_tn in A_effs:
                A_eff_tens = A_eff_tn.contract_tags(all)
                A_eff_tens.modify(apply=lambda data: data * 10 ** A_eff_tn.exponent)

                if A_eff_ is None:
                    A_eff_ = A_eff_tens
                else:
                    A_eff_tens.transpose_like(A_eff_, inplace=True)
                    A_eff_.modify(apply=lambda data: data + A_eff_tens.data)
            return A_eff_

        if len(A_effs) == 0:
            return None

        if return_sum:
            return sum_Aeffs()
        else:
            return A_effs


    def _get_b_eff(self, sites: Union[int,slice]) -> Optional[qtn.Tensor]:
        """
        sites: int or slice(start, stop, step)
        """
        site_pos = list(range(self.L))[sites]
        site_inds = self.mps_inds[sites]

        ### BL * B * BR = d/dT*[i] <x|b>
        b_eff: Optional['qtn.Tensor'] = None
        for ti in range(self.num_targets):
            b_left = self.b_envsLs[ti][site_pos[0]]
            b_right = self.b_envsRs[ti][site_pos[-1]]

            b_eff_t = qtn.TensorNetwork([])
            if b_left is not None:
                b_eff_t.add(b_left)
            if b_right is not None:
                b_eff_t.add(b_right)

            if isinstance(site_inds, (list, tuple)):
                for si in site_inds:
                    b_eff_t.add(self.operators_H[0][si])
            else:
                b_eff_t.add(self.operators_H[0][site_inds])

            if isinstance(site_inds, (list, tuple)):
                for si in site_inds:
                    b_eff_t.add(self.targets[ti][si])
            else:
                b_eff_t.add(self.targets[ti][site_inds])

            #### !!! CHANGED HERE
            b_eff_exponent = self.b_envsLs[ti].exponent + b_eff_t.exponent
            b_eff_t = b_eff_t.contract_tags(all)
            # print('b eff exponent', b_eff_exponent)
            b_eff_t.modify(apply=lambda x: 10 ** b_eff_exponent * x)

            if b_eff is None:
                b_eff = b_eff_t
            else:
                b_eff_t.transpose_like(b_eff, inplace=True)
                b_eff.modify(apply=lambda x: x + b_eff_t.data)

        return b_eff


    # # @profile
    # def _site_solve(self, sites: Union[int, slice], ) -> qtn.Tensor:
    #     """
    #     sites: int or slice(start, stop, step)
    #     """
    #     # print("SITE SOLVE", self.solve_type, sites)
    #     site_pos = list(range(self.L))[sites]
    #     site_inds = self.mps_inds[sites]
    #
    #     ### AL * A * AR = d/dT*[i] d/dT[i] <xA|Ax>
    #     A_effs = []
    #     for i in range(self.num_operators):
    #         A_left = self.A_envsLs[i][site_pos[0]]
    #         A_right = self.A_envsRs[i][site_pos[-1]]
    #
    #         A_eff = qtn.TensorNetwork([])
    #         for A, AH in zip(self.operators, self.operators_H):
    #             if isinstance(site_inds, (list, tuple)):
    #                 for si in site_inds:
    #                     A_eff.add(A[si])
    #                     A_eff.add(AH[si])
    #             else:
    #                 A_eff.add(A[site_inds])
    #                 A_eff.add(AH[site_inds])
    #         if A_left is not None:
    #             A_eff.add(A_left)
    #         if A_right is not None:
    #             A_eff.add(A_right)
    #         # print('A eff exponent', A_eff.exponent, self.A_envsL.exponent)
    #         #### !!! CHANGED HERE
    #         A_eff.exponent = self.A_envsLs[i].exponent + A_eff.exponent
    #         A_effs += [A_eff]
    #
    #     ### BL * B * BR = d/dT*[i] <xA|b>
    #     b_eff: Optional['qtn.Tensor'] = None
    #     for ti in range(self.num_targets):
    #         b_left = self.b_envsLs[ti][site_pos[0]]
    #         b_right = self.b_envsRs[ti][site_pos[-1]]
    #
    #         b_eff_t = qtn.TensorNetwork([])
    #         if b_left is not None:
    #             b_eff_t.add(b_left)
    #         if b_right is not None:
    #             b_eff_t.add(b_right)
    #
    #         if isinstance(site_inds, (list, tuple)):
    #             for si in site_inds:
    #                 b_eff_t.add(self.operators_H[0][si])
    #         else:
    #             b_eff_t.add(self.operators_H[0][site_inds])
    #
    #         # print('site inds', site_inds, self.targets[ti])
    #         if isinstance(site_inds, (list, tuple)):
    #             for si in site_inds:
    #                 b_eff_t.add(self.targets[ti][si])
    #         else:
    #             b_eff_t.add(self.targets[ti][site_inds])
    #
    #         #### !!! CHANGED HERE
    #         b_eff_exponent = self.b_envsLs[ti].exponent + b_eff_t.exponent
    #         b_eff_t = b_eff_t.contract_tags(all)
    #         # print('b eff exponent', b_eff_exponent)
    #         b_eff_t.modify(apply=lambda x: 10 ** b_eff_exponent * x)
    #
    #         if b_eff is None:
    #             b_eff = b_eff_t
    #         else:
    #             b_eff_t.transpose_like(b_eff, inplace=True)
    #             b_eff.modify(apply=lambda x: x + b_eff_t.data)
    #
    #     ## change indices from bra (missing T*[i]) to ket
    #     bra_to_ket_inds = {}  # self.get_bra_to_ket_inds(site_inds[0])
    #     for site_p in site_pos:
    #         bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))
    #
    #     # print('self.ket', self.ket)
    #     # print('bra to ket inds', bra_to_ket_inds)
    #     bra_inds = b_eff.inds if b_eff is not None else []
    #     ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]
    #
    #     def sum_Aeffs():
    #         A_eff_ = None
    #         for A_eff_tn in A_effs:
    #             A_eff_tens = A_eff_tn.contract_tags(all)
    #             A_eff_tens.modify(apply=lambda data: data * 10 ** A_eff_tn.exponent)
    #             A_eff_tens.transpose(*bra_inds, *ket_inds, inplace=True)
    #
    #             if A_eff_ is None:
    #                 A_eff_ = A_eff_tens
    #             else:
    #                 A_eff_.modify(apply=lambda data: data + A_eff_tens.data)
    #         return A_eff_
    #
    #     ## solve for T[i]
    #     if len(A_effs) == 0:
    #         x_eff = b_eff.reindex(bra_to_ket_inds)
    #         x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent - self.bra.exponent))
    #         ## b_eff_exponent already in b_eff tensor
    #     else:
    #         if self.solve_type is SolveMethod.CGD:
    #             # A_eff_exponent = A_eff.exponent
    #             if isinstance(site_inds, (list, tuple)):
    #                 ket_tensors = [self.ket[si] for si in site_inds]
    #             else:
    #                 ket_tensors = self.ket[site_inds]
    #             x_tens = qtn.tensor_contract(*ket_tensors)
    #             x_tens.transpose(*ket_inds, inplace=True)
    #             # x_tens.modify(apply=lambda x: x * 0.)
    #             if self.is_H:
    #                 x_eff, error = qtn_conjugate_gradient_descent_1site(A_effs, b_eff, x_tens, bra_to_ket_inds)
    #             else:
    #                 x_eff, error = qtn_conjugate_gradient_squared_1site(A_effs, b_eff, x_tens, bra_to_ket_inds)
    #             # x_eff, error = qtn_conjugate_gradient_descent_1site(A_eff, b_eff, None, bra_to_ket_inds)
    #             # x_eff.modify(apply=lambda x: x * 10 ** (-A_eff_exponent))
    #
    #         elif self.solve_type is SolveMethod.CGDx:
    #
    #             if isinstance(site_inds, (list, tuple)):
    #                 ket_tensors = [self.ket[si] for si in site_inds]
    #             else:
    #                 ket_tensors = self.ket[site_inds]
    #             x_tens = qtn.tensor_contract(*ket_tensors)
    #             x_tens.transpose(*ket_inds, inplace=True)
    #
    #             # A_eff_exponent = A_eff.exponent
    #             A_eff = sum_Aeffs()
    #             out_shape = A_eff.shape[:len(bra_inds)]
    #             in_shape = A_eff.shape[len(bra_inds):]
    #             A_eff_data = A_eff.data.reshape(np.prod(out_shape), -1)
    #
    #             if b_eff is not None:
    #                 b_eff_data = b_eff.data.reshape(-1)
    #             else:
    #                 b_eff_data = np.zeros((A_eff_data.shape[0],))
    #
    #             if self.is_H:
    #                 x_eff_data, error = conjugate_gradient_descent(A_eff_data, b_eff_data,
    #                                                                x=x_tens.data.reshape(-1))
    #             else:
    #                 x_eff_data, error = conjugate_gradient_squared(A_eff_data, b_eff_data,
    #                                                                x=x_tens.data.reshape(-1))
    #             # x_eff_data = np.linalg.solve(A_eff_data, b_eff_data)
    #             x_eff_data = x_eff_data.reshape(in_shape)
    #             # x_eff_data *= 10 ** (-A_eff_exponent)
    #             ## b_eff_exponent already in b_eff, A_eff already includes ket, bra exponents
    #             # print('ket inds', ket_inds, A_eff)
    #             # exit()
    #             x_eff = qtn.Tensor(data=x_eff_data, inds=ket_inds)
    #
    #         else:  ## ALS optimzation
    #             # A_eff_exponent = A_eff.exponent
    #             # A_eff = A_eff.contract_tags(all)
    #             # ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]
    #             # # print('bra inds', bra_inds, 'ket inds', ket_inds)
    #             # A_eff.transpose(*bra_inds, *ket_inds, inplace=True)
    #             # out_shape = A_eff.shape[:len(bra_inds)]
    #             # in_shape = A_eff.shape[len(bra_inds):]
    #
    #             A_eff = sum_Aeffs()
    #             out_shape = A_eff.shape[:len(bra_inds)]
    #             in_shape = A_eff.shape[len(bra_inds):]
    #             A_eff_data = A_eff.data.reshape(np.prod(out_shape), -1)
    #
    #             if b_eff is not None:
    #                 b_eff_data = b_eff.data.reshape(-1)
    #             else:
    #                 b_eff_data = np.zeros((A_eff_data.shape[0],))
    #
    #             x_eff_data = np.linalg.solve(A_eff_data, b_eff_data)
    #             x_eff_data = x_eff_data.reshape(in_shape)
    #             # x_eff_data *= 10 ** (-A_eff_exponent)
    #             ## b_eff_exponent already in b_eff, A_eff already includes ket, bra exponents
    #             x_eff = qtn.Tensor(data=x_eff_data, inds=ket_inds)
    #
    #     return x_eff


    def check_err_np(self):
        """
        """
        if self.num_operators > 0:
            A = np.zeros((2**self.L, 2**self.L))
            for op in self.operators:
                op_tens = op.contract(all) * 10**op.exponent
                op_tens.transpose(*[op.upper_ind_id.format(i) for i in self.mps_inds],
                                  *[op.lower_ind_id.format(i) for i in self.mps_inds],
                                  inplace=True)
                op_mat = op_tens.data.reshape(2**self.L, 2**self.L)
                # A = np.dot(op_mat, A)
                A += op_mat
        else:
            A = np.eye(2**self.L)

        b = np.zeros(2**self.L)
        for target in self.targets:
            target_tens = target.contract(all) * 10**target.exponent
            # print('target tens?', target_tens)
            target_tens.transpose(*[target.site_ind_id.format(i) for i in self.mps_inds], inplace=True)
            target_vec = target_tens.data.reshape(-1)
            b = b + target_vec

        x_tens = self.ket.contract(all) * 10**self.ket.exponent
        x_tens.transpose(*[self.ket.site_ind_id.format(i) for i in self.mps_inds], inplace=True)
        x_vec = x_tens.data.reshape(-1)

        # Ax = np.dot(A, x_vec)
        # print('<xA|Ax>', np.dot(Ax, Ax), '<b|Ax>', np.dot(b, Ax), '<b|b>', self.targets_norm**2)

        err = np.linalg.norm( np.dot(A, x_vec) - b )
        return err / self.targets_norm


    # @profile
    def check_err(self, dense=False) -> float:
        """ calculates sqrt(<b-Ax|b-Ax>) = sqrt(<Ax|Ax> + <b|b> - 2Re[<Ax|b>])
        """
        # print('np err', self.check_err_np())

        dense = False
        if dense or self.canon_direction == 0:

            ## combine err envs
            ### L, R:  <xA|Ax>, <b|Ax>, <b|b> ancilla envs; ordered by bra_ind, ket_ind
            def combine_err_envs(err_envs: Sequence['qtn.Tensor']):
                xAAx, bAx, bb = err_envs
                bb = bb.reindex({bb.inds[i]: xAAx.inds[i] for i in range(len(bb.inds))})
                tot_env = qtn.tensor_direct_product(xAAx, bb, inplace=False)

                b_size, Ax_size = bb.shape[0], xAAx.shape[0]
                tot_size = tot_env.shape[0]
                bAx = bAx.copy()
                for ind in bAx.inds:
                    bAx.expand_ind(ind, tot_size)

                    if ind == 0:    ## move b inds to correct position
                        reorder_tens = np.zeros(tot_size)
                        reorder_tens[-b_size-1:,:b_size] = np.eye(b_size)
                        bAx.modify(apply=lambda x: np.tensordot(reorder_tens, x, axes=1), inplace=True)
                tot_env.modify(apply = lambda x: x + bAx.data)
                return tot_env

            err_env_L = None if self.err_env0_Ls is None else combine_err_envs(self.err_env0_Ls)
            err_env_R = None if self.err_env0_Rs is None else combine_err_envs(self.err_env0_Rs)

            ## sum |Ax>
            if self.num_operators > 0:
                ket_copy = self.ket.copy()
                Ax = None
                for i in range(self.num_operators):
                    Ax_i = helper.apply(self.operators[i], ket_copy)
                    if Ax is None:
                        Ax = Ax_i
                    else:
                        Ax = helper.add_MPS(Ax, Ax_i, inplace=True)
            else:
                Ax = self.ket.copy()

            ## sum |b>
            total_b = self.targets[0].copy()
            # print('total b', total_b)
            for ib in range(1,self.num_targets):
                # print('add b', self.targets[ib])
                total_b = helper.add_MPS(total_b, self.targets[ib], inplace=True)

            Ax.site_ind_id = total_b.site_ind_id
            # print('Ax', Ax)
            # print('total b', total_b)
            ## method=overlap method generally does not give as accurate results
            if err_env_L is None and err_env_R is None:
                err = helper.distance(Ax, total_b, method='auto') / self.targets_norm
            else:
                helper.match_inner_inds(total_b, Ax, inplace=True)
                total_b.site_ind_id = Ax.site_ind_id
                Ax.distribute_exponent()
                total_b.distribute_exponent()

                tens_list = []
                if err_env_L is not None:
                    tens_list += [err_env_L]
                    shared_, ancL_b  = total_b[0].filter_bonds(Ax[0])
                    shared_, ancL_Ax = Ax[0].filter_bonds(total_b[0])
                    total_b[0].reindex({anc1: anc2 for anc1, anc2 in zip(ancL_b, ancL_Ax)}, inplace=True)

                if err_env_R is not None:
                    tens_list += [err_env_R]
                    shared_, ancR_b  = total_b[-1].filter_bonds(Ax[-1])
                    shared_, ancR_Ax = Ax[-1].filter_bonds(total_b[-1])
                    total_b[-1].reindex({anc1: anc2 for anc1, anc2 in zip(ancR_b, ancR_Ax)}, inplace=True)

                diff = helper.add_MPS(Ax, total_b.scalar_multiply(-1, inplace=True))
                diff_conj = diff.conj()
                if err_env_L is not None:
                    diff_conj[0].reindex({anc1: anc1 + '_' for anc1 in ancL_Ax}, inplace=True)
                if err_env_L is not None:
                    diff_conj[-1].reindex({anc1: anc1 + '_' for anc1 in ancR_Ax}, inplace=True)

                tens_list += [diff, diff_conj]
                err = qtn.tensor_contract(tens_list)

        else:   ## contract tags doesn't give as small errors as helper.distance()

            old_site_ind_id = self.bra.site_ind_id

            if self.num_operators > 0:
                self.bra.site_ind_id = self.operators_H[-1].upper_ind_id

            ## <xA|Ax>
            if self.num_operators == 1:
                norm_Ax = self.A_envsLs[0].norm() if self.canon_direction > 0 else self.A_envsRs[0].norm()
            elif self.num_operators == 0:
                norm_Ax = helper.norm(self.ket) ** 2
            else:
                raise ValueError("linear solver shouldn't have multiplie operators")

            ## <xA|b>
            ovlp_b_tot = 0.
            for ib in range(self.num_targets):
                ovlp_b = self.b_envsLs[ib].norm() if self.canon_direction > 0 else self.b_envsRs[ib].norm()
                ovlp_b_tot += ovlp_b

            self.bra.site_ind_id = old_site_ind_id

            # if self.num_operators == 0 and self.num_targets == 1:
            #     print('err dist b', helper.distance(self.bra.conj(), self.targets[0]) / self.targets_norm)
            #     print('err dist b', helper.distance(self.bra, self.targets[0]) / self.targets_norm)
            #     print('err dist k', helper.distance(self.ket, self.targets[0]) / self.targets_norm)
            #     print('self.targets norm', self.targets_norm, helper.norm(self.targets[0].copy()))
            #     print('norm', helper.norm(self.ket)**2)
            #     print('ovlp', helper.ovlp(self.bra, self.targets[0]))
            #     print('ovlp', helper.ovlp(self.ket, self.targets[0]))
            # elif self.num_operators == 1 and self.num_targets == 1:
            #     Ax = helper.apply(self.operators[0], self.ket)
            #     print('err dist Ax', helper.distance(Ax, self.targets[0]) / self.targets_norm)
            #     print('self.targets norm', self.targets_norm, helper.norm(self.targets[0].copy()))
            #     print('norm Ax', helper.norm(Ax)**2)
            #     print('ovlp', helper.ovlp(Ax, self.targets[0]))


            # print('residual^2', (norm_Ax + self.targets_norm ** 2 - 2 * np.real(ovlp_b_tot)) / self.targets_norm**2)
            err2 = (norm_Ax + self.targets_norm ** 2 - 2 * np.real(ovlp_b_tot))
            # print(norm_Ax, self.targets_norm**2, ovlp_b_tot)
            # if err2 < 0 and np.abs(err2) < 1.0e-12:
            #     err2 = np.abs(err2)
            err2 = np.abs(err2)
            err = np.sqrt(err2) / self.targets_norm

        self.err = err
        self.is_conv = err < self.conv_tol
        return err
