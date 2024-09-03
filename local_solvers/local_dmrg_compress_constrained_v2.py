import scipy.optimize

import helper_quimb
from local_solvers.local_dmrg_eval import *

""" different  
"""


### scipy optimizer only works for real-valued objects ###
def real_to_complex(z):  # real vector of length 2n -> complex of length n
    return z[:len(z) // 2] + 1j * z[len(z) // 2:]

def complex_to_real(z):  # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))

def local_constrained_compress(ref_state: Union['MPS','qtn.MatrixProductState'],
                               # orthog_states: Sequence[Union['MPS', 'qtn.MatrixProductState']],
                               # lin_constraints: Sequence[tuple[Union['MPS', 'qtn.MatrixProductState'], Numeric]],
                               quad_constraints: Sequence[tuple[Optional['qtn.MatrixProductOperator'],
                                                                Union['MPS', 'qtn.MatrixProductState'], Numeric]],
                               penalty: Sequence[tuple['qtn.MatrixProductOperator', Numeric]] = None,
                               init_guess: Optional['qtn.MatrixProductState'] = None,
                               max_bond=None, combine_terms_func:Callable=None,
                               nsites=1, direction: SweepDirection = SweepDirection.RIGHT,
                               ):
    """ use OrthogonalDMRGEvaluator with first projecting out orthog_states from ref_state
        quadratic constraint:
        -
    """
    init_guess = ref_state.copy() if init_guess is None else init_guess
    # print('init guess', init_guess.max_bond())
    # if max_bond is not None and init_guess.max_bond() > max_bond:
    #     helper_quimb.compress(init_guess, compress_opts={'max_bond': max_bond})
    #     helper_quimb.match_inner_inds(init_guess, ref_state)
    #     # helper_quimb.check_orthog(init_guess)
    #
    #     init_guess[0].modify(apply=lambda x: x * 10**(init_guess.exponent - ref_state.exponent))
    #     init_guess.exponent = ref_state.exponent

    print('init guess', init_guess.max_bond())

    term = Term_DMRG(ref_state, bra=init_guess)
    solver = ConstrainedDMRGEvaluator(init_guess, [term],
                                      quad_constraints = quad_constraints,
                                      direction=direction, max_bond=max_bond,
                                      max_wrong_iter=2,
                                      )
    nsites = 1
    solver.solve(nsites)
    return solver.solution


class ConstrainedDMRGEvaluator(DMRGEvaluator):

    def __init__(self,
                 ket_state: Union['MPS', 'qtn.MatrixProductState'],
                 terms: Sequence['Term_DMRG'],
                 direction: SweepDirection = SweepDirection.RIGHT,
                 max_bond: int = None,
                 conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
                 max_tot_iter: int = DEFAULT_MAX_TOT_ITER, max_wrong_iter: int = DEFAULT_MAX_WRONG_ITER,
                 copy_obj: 'ConstrainedDMRGEvaluator' = None,
                 combine_terms_func: 'Callable' = None,
                 grid: 'Grid' = None, ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None,
                 #
                 # quad_constraints: Sequence[tuple[Optional[Term], Optional[Term],'MPS', 'qtn.MatrixProductState']] = None,
                 quad_constraints: Sequence[tuple[Optional['qtn.MatrixProductOperator'],
                                                  Union['MPS', 'qtn.MatrixProductState'], Numeric]] = None,
                 ):

        print('ket state max bond', ket_state.max_bond())
        super().__init__(ket_state, terms, direction=direction, max_bond=max_bond, conv_tol=conv_tol,
                         max_iter=max_iter, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter,
                         copy_obj=copy_obj, combine_terms_func=combine_terms_func,
                         grid=grid, ax_deriv_configs=ax_deriv_configs)

        print('done super init')

        if copy_obj:
            self.eq_constraints = [[qterm.copy(bra_copy=self.ket) for qterm in qcterms[:1]] + [qcterms[-1]]
                                   for qcterms in copy_obj.eq_constraints]
        else:
            qc_terms_list = []
            for qc in quad_constraints:
                ## constraint of the form x^T Q x + A x + b = 0
                ## Q can be list of MPOs
                Qs, A, b = qc

                qc_terms = []
                ## quadratic constraint
                if (Qs is not None) and (not isinstance(Qs, (tuple, list))):
                    Qs = [Qs]
                    term_q = Term_DMRG(ket_state, operators=Qs, bra=ket_state, direction=self.direction,
                                       cur_orthog=self.cur_orthog)  ## checks self.cur_orthog here
                    term_q.initialize()
                    qc_terms += [term_q]
                else:
                    qc_terms += [None]

                ## linear constraint
                if A is not None:
                    helper_quimb.match_inner_inds(A, ket_state)
                    A.site_ind_id = ket_state.site_ind_id
                    term_a = Term_DMRG(A, bra=ket_state, direction=self.direction, cur_orthog=self.cur_orthog)
                    term_a.initialize()
                    qc_terms += [term_a]
                else:
                    qc_terms += [None]

                if len(qc_terms) > 0:
                    ## scalar constraint
                    qc_terms += [b if b is not None else 0.0]

                    ## add to list
                    qc_terms_list += [qc_terms]

            self.eq_constraints: list[tuple[Optional[Term], Optional[Term], Numeric]] = qc_terms_list


    def _site_solve(self, left_site_pos: int, nsites: int, site_tens: qtn.Tensor = None, return_intermediates=False
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        solve but with scipy optimizer.
        """
        print('ket max bond', self.ket.max_bond(), self.out.max_bond(), self.terms[0].ket.max_bond())

        site_list, site_err = super()._site_solve(left_site_pos, nsites, site_tens,
                                                  return_intermediates=return_intermediates)
        is_complex = np.any(np.iscomplex(site_list[0].data))
        num_constraints = len(self.eq_constraints)

        ### doing the compression here is fine, but need to ensure that another 1/2 sweep is performed.
        ## if output_tensor bond dimension is larger than the desired rank, then perform low-rank approximation.
        out_tens = site_list[0]
        # print('self.out.max bond', self.out.max_bond(), self.max_bond)
        if np.any([out_tens.ind_size(ind) > self.max_bond for ind in out_tens.inds]):
            # print('compress site_list')
            left_inds = [self.ket.bond(left_site_pos, left_site_pos -1)] if left_site_pos > 0 else []
            left_inds += [self.ket.site_ind_id.format(left_site_pos)]
            ltens, rtens = qtn.tensor_split(out_tens, left_inds, cutoff=CUTOFF, max_bond=self.max_bond, renorm=False)
            comp_out_tens = qtn.tensor_contract(ltens, rtens)
            out_tens.modify(data=comp_out_tens.data, inds=comp_out_tens.inds)

        ## transpose all targets so that they are the same
        for xtens in site_list[1:]:
            xtens.transpose_like(site_list[0], inplace=True)
        site_shape = site_list[0].shape
        site_inds = site_list[0].inds

        bra_to_ket = self.terms[0].projected_bra_to_ket(left_site_pos, nsites)
        ket_to_bra = {v: k for k,v in bra_to_ket.items()}
        bra_inds = [ket_to_bra[k] for k in site_inds]

        ## evaulate constraints at the active site
        ## equality constraints
        eff_constraints = []
        for constraint in self.eq_constraints:
            Q, A, b = constraint

            Q_effs = None
            if Q is not None:
                Q_effs = Q.get_eff_operator(left_site_pos, nsites, return_combined=False)[0]  ## 0th tier
                # print('Q_Effs id', id(Q_effs))

            A_eff = None
            if A is not None:
                A_eff = A.get_evaluated_site(left_site_pos, nsites)
                A_eff.transpose(site_inds, inplace=True)

            eff_constraints += [(Q_effs, A_eff, b)]


        ## def constraint cost/jacobian functions
        ## define cost object
        def constraint_fun(x: 'np.ndarray', index: int):

            Q_effs_, A_eff_, b_ = eff_constraints[index]

            if is_complex:
                x = real_to_complex(x)

            x_tens = qtn.Tensor(x.reshape(site_shape), inds=site_inds)
            x_conj = x_tens.conj()
            x_conj.reindex(ket_to_bra, inplace=True)

            tot_cost = 0
            for Q_eff in Q_effs_:
                out = qtn.tensor_contract(x_conj, *Q_eff.tensors, x_tens)
                out *= 10 ** (Q_eff.exponent + self.ket.exponent * 2)
                tot_cost = tot_cost + out

            if A_eff_ is not None:
                return tot_cost + qtn.tensor_contract(x_conj, A_eff_) * self.ket.exponent - b_
            else:
                return np.real(tot_cost - b_)


        ## define cost jacobian
        def constraint_jac(x: 'np.ndarray', index: int):

            Q_effs_, A_eff_, b_ = eff_constraints[index]

            # print('Qeffs', id(Q_effs_))

            if is_complex:
                x = real_to_complex(x)

            x_tens = qtn.Tensor(x.reshape(site_shape), inds=site_inds)
            x_conj = x_tens.conj()
            x_conj.reindex(ket_to_bra, inplace=True)

            out_list = []
            for Q_eff in Q_effs_:
                out_tens = qtn.tensor_contract(x_tens, *Q_eff.tensors)
                out_tens.modify(apply=lambda x: x * 10 ** (Q_eff.exponent + self.ket.exponent * 2))
                ## * 2 because x_conj exponent is not accounted for otherwise
                out_tens.reindex(bra_to_ket, inplace=True)
                out_tens.transpose(*site_inds, inplace=True)
                out_list += [out_tens]

            sum_xQ = out_list[0].copy()
            sum_xQ.modify(data=sum([ot.data for ot in out_list]))
            sum_xQ_data = sum_xQ.data.reshape(-1)

            if A_eff_ is not None:
                A_eff_conj_data = A_eff_.data.reshape(-1).conj()
                jac = sum_xQ + A_eff_conj_data  ## not tested
                if is_complex:
                    jac = complex_to_real(jac)
                return jac
            else:
                jac = sum_xQ_data
                if is_complex:
                    jac = complex_to_real(jac)
                return jac

        ## equality constraints
        # constraint_funs = [lambda x: cost_fun(x, i) for i in range(num_constraints)]
        # constraint_jacs = [lambda x: cost_jac(x, i) for i in range(num_constraints)]
        constraint_funs = [lambda x: constraint_fun(x, 0), lambda x: constraint_fun(x,1), lambda x: constraint_fun(x,2)]
        constraint_jacs = [lambda x: constraint_jac(x, 0), lambda x: constraint_jac(x,1), lambda x: constraint_jac(x,2)]
        constraints = []
        for ix, (Q_effs, A_eff, b) in enumerate(eff_constraints):

            if Q_effs is not None:   ## quadratic constraint

                # ## check init error
                # xtens = site_list[0]
                # res_x = xtens.data.reshape(-1)
                # if is_complex:
                #     res_x = complex_to_real(res_x)
                # print('!! init constraint error', cost_fun(res_x, ix), cost_fun)

                # cost_fun_Q = constraint_funs[ix]
                # cost_jac_Q = constraint_jacs[ix]
                if ix == 0:
                    cost_fun_Q = lambda x: constraint_fun(x, 0)  # constraint_funs[ix]
                    cost_jac_Q = lambda x: constraint_jac(x, 0)  # constraint_jacs[ix]
                elif ix == 1:
                    cost_fun_Q = lambda x: constraint_fun(x, 1)  # constraint_funs[ix]
                    cost_jac_Q = lambda x: constraint_jac(x, 1)  # constraint_jacs[ix]
                elif ix == 2:
                    cost_fun_Q = lambda x: constraint_fun(x, 2)  # constraint_funs[ix]
                    cost_jac_Q = lambda x: constraint_jac(x, 2)  # constraint_jacs[ix]
                else:
                    raise NotImplementedError

                constraint = scipy.optimize.NonlinearConstraint( cost_fun_Q, jac=cost_jac_Q,
                                                                 lb=0, ub=0)
                constraints += [constraint]

            else:   ## linear constraint

                assert(A_eff is not None),'A cannot be None if Q is None'
                constraint = scipy.optimize.LinearConstraint(A_eff.data.reshape(-1), lb=b, ub=b)
                constraints += [constraint]

        ## initial constraint errors
        for xtens in site_list:
            res_x = xtens.data.reshape(-1)
            if is_complex:
                res_x = complex_to_real(res_x)
            # print('init error', [np.abs(constraint_func(res_x)) for constraint_func in constraint_funs])
            print('init error', [np.abs(constraint_fun(res_x, ix)) for ix in range(num_constraints)])
            # print('init error', [constraint_func for constraint_func in constraint_funs])

        # exit()

        ## impose boundary conditions on all outputs, including intermediate states
        constrained_tens_list = []
        tot_err = site_err
        for ix, new_tens in enumerate(site_list):
            c = new_tens.data.reshape(-1)
            if is_complex:
                c = complex_to_real(c)

            fun = lambda x: np.linalg.norm(c - x)
            jac = lambda x: np.conj(x - c)
            ## with constraints, can be COBYLA, COBYQA, SLSQP, trust-constr. default: SLSQP
            ## COBYLA can't do equality constraints
            ## COBYQA not known
            init_guess = c + (np.random.random(c.shape) + np.random.random(c.shape) * 1.j) * 0.01
            init_guess = c + (np.random.random(c.shape)) * 0.02
            # target_norm = self.eq_constraints[0][-1]
            # print('target norm', target_norm, helper_quimb.norm(self.ket), np.linalg.norm(c), constraint_funs[0](c))
            # # init_guess = c * target_norm/(constraint_funs[0](c) + target_norm)
            # print('init error', [np.abs(constraint_func(init_guess)) for constraint_func in constraint_funs])
            res = scipy.optimize.minimize(fun, init_guess, jac=jac, constraints=constraints,
                                          # method='trust-constr')
                                          method='SLSQP',
                                          # options={'maxiter': 1000}
                                          )

            print('res', res.success, res.message)
            # if res.message == 'Singular matrix C in LSQ subproblem':
            #     for xx, constraint in enumerate(self.eq_constraints):
            #         Q, A, b = constraint
            #
            #         print('Q', Q)
            #         tmpQ = Q.get_eff_operator(left_site_pos, nsites, return_combined=True)[0]  ## 0th tier
            #         print('tmp Q', tmpQ)
            #         tmpQ.transpose(*bra_inds, *site_inds, inplace=True)
            #         sq_size = np.prod(site_shape)
            #
            #         tmpQ_data = tmpQ.data.reshape(-1, sq_size)
            #
            #         plt.figure()
            #         plt.imshow(np.real(tmpQ_data))
            #         plt.colorbar()
            #         plt.title(f're Q {xx}')
            #         plt.figure()
            #         plt.imshow(np.imag(tmpQ_data))
            #         plt.colorbar()
            #         plt.title(f'im Q {xx}')
            #
            #         u,s,vt = np.linalg.svd(tmpQ_data)
            #         print('sing vals', xx, s, tmpQ_data.shape)
            #
            #     plt.show()

            # if not res.success:
            #     print('unsuccesful optimization')
            #     exit()

            if ix == 0:
                print('errors', site_err, fun(res.x))
                # constraint_errs = [np.abs(constraint_func(res.x)) for constraint_func in constraint_funs]
                constraint_errs = [constraint_fun(res.x, ix) for ix in range(num_constraints)]
                print('constraint errors', constraint_errs)
                error = fun(res.x) + np.sum(constraint_errs)
                tot_err += error
                # tot_err = np.sum(constraint_errs)
                print('error', site_err, tot_err)

            res_x = real_to_complex(res.x) if is_complex else res.x
            x_tens = new_tens.copy()
            x_tens.modify(data=res_x.reshape(new_tens.shape))

            constrained_tens_list += [x_tens]

        return constrained_tens_list, tot_err



    def _bond_solve(self, left_site_pos: int, site_tens: qtn.Tensor=None, return_intermediates=True,
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        update bond between left_site_pos, left_site_pos + 1
        """
        site_list, site_err = super()._bond_solve(left_site_pos, site_tens,
                                                  return_intermediates=return_intermediates)

        raise NotImplementedError



    def update_blocks(self, i: int, direction: SweepDirection): #, new_ket_site=None):

        super().update_blocks(i, direction)

        for term in self.eq_constraints:
            Q, A, b = term
            if Q is not None:
                Q.extend_env(i, direction=direction)
            if A is not None:
                A.extend_env(i, direction=direction)

            # # print('term extend env', term)
            # term.extend_env(i, direction=direction) #, new_ket_site=new_ket_site)