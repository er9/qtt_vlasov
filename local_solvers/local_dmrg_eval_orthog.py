import helper_quimb
from local_solvers.local_dmrg_eval import *



def local_orthogonal_compress_2(ref_state: Union['MPS','qtn.MatrixProductState'],
                              orthog_states: Sequence[Union['MPS', 'qtn.MatrixProductState']],
                              init_guess: Optional['qtn.MatrixProductState'] = None,
                              max_bond=None, combine_terms_func:Callable=None,
                              nsites=1, direction: SweepDirection = SweepDirection.RIGHT,
                              ):
    """ compress ref_state while orthogonalizing against orthog_states
        project out orthog states before compressing.
    """
    if max_bond is None:
        out = ref_state.copy()
        form = 'left' if direction == SweepDirection.RIGHT else 'right'
        return helper_quimb.compress(out, compress_opts={'max_bond': max_bond, 'form': form})

    remainder = ref_state.copy()
    proj_vals = []
    orthog_bond = 0
    for ostate in orthog_states:
        proj_val = helper_quimb.ovlp(ref_state, ostate) / helper_quimb.norm(ostate)**2
        proj_vals += [proj_val]
        remainder = helper_quimb.add_MPS( remainder, helper_quimb.scalar_multiply(ostate, -proj_val) )
        # orthog_bond += ostate.max_bond()
        orthog_bond += helper_quimb.max_inner_bond(ostate)

    max_bond_r = max_bond # - orthog_bond if max_bond is not None else None
    print('target max bond r', max_bond_r)

    init_guess = remainder.copy() if init_guess is None else init_guess
    print('init guess', init_guess.max_bond())
    if max_bond_r is not None and init_guess.max_bond() > max_bond_r:
        helper_quimb.compress(init_guess, compress_opts={'max_bond': max_bond_r})
        helper_quimb.match_inner_inds(init_guess, ref_state)
        print('init guess', init_guess.max_bond())

    for ostate in orthog_states:
        helper_quimb.match_inner_inds(ostate, remainder)
        ostate.site_ind_id = remainder.site_ind_id
        print('remainder ovlp', helper_quimb.ovlp(remainder, ostate))

    if max_bond_r > 0:
        term = Term_DMRG(remainder) # , bra=init_guess)
        solver = OrthogonalDMRGEvaluator(init_guess, [term], combine_terms_func=combine_terms_func,
                                         direction=direction, max_bond=max_bond_r,
                                         orthog_states=orthog_states,
                                         max_wrong_iter=2
                                         )
        solver.solve(nsites)
        out = solver.solution

        # for ostate in orthog_states:
        #     print('out ovlp', helper_quimb.ovlp(out, ostate))

    else:
        out = None

    tot_mps = out
    for proj_val, ostate in zip(proj_vals, orthog_states):
        tot_mps = helper_quimb.add_MPS( tot_mps, helper_quimb.scalar_multiply(ostate, proj_val) , inplace=True)

    for proj_val, ostate in zip(proj_vals, orthog_states):
        print('tot ovlp', helper_quimb.ovlp(tot_mps, ostate) - proj_val, proj_val)

    return tot_mps


def local_orthogonal_compress(ref_state: Union['MPS','qtn.MatrixProductState'],
                              orthog_states: Sequence[Union['MPS', 'qtn.MatrixProductState']],
                              init_guess: Optional['qtn.MatrixProductState'] = None,
                              max_bond=None, combine_terms_func:Callable=None,
                              nsites=1, direction: SweepDirection = SweepDirection.RIGHT,
                              ):
    """ use OrthogonalDMRGEvaluator with first projecting out orthog_states from ref_state
    """
    init_guess = ref_state.copy() if init_guess is None else init_guess
    # print('init guess', init_guess.max_bond())
    # if max_bond is not None and init_guess.max_bond() > max_bond:
    #     helper_quimb.compress(init_guess, compress_opts={'max_bond': max_bond})
    #     helper_quimb.match_inner_inds(init_guess, ref_state)

    print('init guess', init_guess.max_bond())

    for ostate in orthog_states:
        helper_quimb.match_inner_inds(ostate, ref_state)
        ostate.site_ind_id = ref_state.site_ind_id

    term = Term_DMRG(ref_state, bra=init_guess)
    solver = OrthogonalDMRGEvaluator(init_guess, [term], combine_terms_func=combine_terms_func,
                                     direction=direction, max_bond=max_bond,
                                     orthog_states=orthog_states,
                                     max_wrong_iter=2
                                     )
    solver.solve(nsites)

    # out_gtn = GridTN1D(ref_x.grid, solver.ket)
    # return out_gtn
    return solver.solution


class OrthogonalDMRGEvaluator(DMRGEvaluator):

    def __init__(self,
                 ket_state: Union['MPS', 'qtn.MatrixProductState'],
                 # bra_state: 'qtn.MatrixProductState',
                 terms: Sequence['Term_DMRG'],
                 # cur_orthog: int=None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 max_bond: int = None,
                 conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
                 max_tot_iter: int = DEFAULT_MAX_TOT_ITER, max_wrong_iter: int = DEFAULT_MAX_WRONG_ITER,
                 copy_obj: 'OrthogonalDMRGEvaluator' = None,
                 combine_terms_func: 'Callable' = None,
                 grid: 'Grid' = None, ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None,
                 #
                 orthog_states: Sequence[Union['MPS', 'qtn.MatrixProductState']] = None,
                 ):

        print('ket state max bond', ket_state.max_bond())
        super().__init__(ket_state, terms, direction=direction, max_bond=max_bond, conv_tol=conv_tol,
                         max_iter=max_iter, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter,
                         copy_obj=copy_obj, combine_terms_func=combine_terms_func,
                         grid=grid, ax_deriv_configs=ax_deriv_configs)

        print('ket state max bond', self.ket.max_bond())

        if copy_obj:
            self.orthog_terms = [oterm.copy(bra_copy=self.ket) for oterm in copy_obj.orthog_terms]
        else:
            self.orthog_terms = [Term_DMRG(ostate, bra=ket_state, cur_orthog=self.cur_orthog)
                                 for ostate in orthog_states]
            for oterm in self.orthog_terms:
                oterm.direction = self.direction
                oterm.max_bond = self.max_bond
                oterm.initialize()


    def _site_solve(self, left_site_pos: int, nsites: int, site_tens: qtn.Tensor = None, return_intermediates=False
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        """
        # site_inds = list(range(left_site_pos, left_site_pos + nsites))
        # direction = self.direction
        # at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

        # bra_to_ket = self.terms[0].projected_bra_to_ket(left_site_pos, nsites)
        print('ket max bond', self.ket.max_bond(), self.out.max_bond(), self.terms[0].ket.max_bond())
        # exit()

        site_list, site_err = super()._site_solve(left_site_pos, nsites, site_tens,
                                                  return_intermediates=return_intermediates)

        ### doing compression here is fine, but need to ensure that another 1/2 sweep is performed.
        ## if output_tensor bond dimension is larger than the desired rank, then perform low-rank approximation.
        out_tens = site_list[0]
        # print('self.out.max bond', self.out.max_bond(), self.max_bond)
        if np.any([out_tens.ind_size(ind) > self.max_bond for ind in out_tens.inds]):
            # print('site solve compress site_list')
            # exit()
            left_inds = [self.ket.bond(left_site_pos, left_site_pos -1)] if left_site_pos > 0 else []
            left_inds += [self.ket.site_ind_id.format(left_site_pos)]
            ltens, rtens = qtn.tensor_split(out_tens, left_inds, cutoff=CUTOFF, max_bond=self.max_bond)
            comp_out_tens = qtn.tensor_contract(ltens, rtens)
            out_tens.modify(data=comp_out_tens.data, inds=comp_out_tens.inds)

        ## ket environment projected onto
        orth_vecs = [oterm.get_evaluated_site(left_site_pos, nsites) for oterm in self.orthog_terms]  ## <x|b>
        # print('ov', self.orthog_terms[0].ket is self.ket, self.orthog_terms[0].bra is self.ket)

        # ## orth vecs are not necessarily orthogonal to each other
        # for n in range(len(orth_vecs)):
        #     for m in range(n + 1):
        #         print('orth vecs', m, n, qtn.tensor_contract(orth_vecs[m].conj(), orth_vecs[n]))

        # print()
        ## orthogonalize orth vecs against each other
        for n in range(len(orth_vecs)):
            ov_n = orth_vecs[n]
            # print('tmp', qtn.tensor_contract(ov_n.conj(), ov_n))
            for m in range(n):
                ov_m = orth_vecs[m]
                ov_m.transpose_like(ov_n, inplace=True)
                ovlp = qtn.tensor_contract(ov_m.conj(), ov_n)
                orth_vecs[n].modify(apply = lambda x: x - (ovlp/ov_m.norm()**2) * ov_m.data)

        # ## orth vecs should now be orthogonal to each other
        # for n in range(len(orth_vecs)):
        #     for m in range(n + 1):
        #         print('orth vecs', m, n, qtn.tensor_contract(orth_vecs[m].conj(), orth_vecs[n]))
        #
        # print()

        # site_tens = self.terms[0].get_evaluated_site(left_site_pos, nsites,)
        # proj_val = qtn.tensor_contract(orth_vecs[0], site_tens.conj())
        # # print('ovlp',proj_val, helper_quimb.ovlp(self.orthog_terms[0].bra, self.orthog_terms[0].ket))
        # # print('ovlp', proj_val, helper_quimb.ovlp(self.orthog_terms[0].bra.conj(), self.orthog_terms[0].ket))
        # # print('ovlp', proj_val * 10**(self.ket.exponent))
        # # exit()

        ## orthogonal projection
        orthog_site_list = []
        tot_err = site_err
        for ix, new_tens in enumerate(site_list):
            orth_tens = new_tens.copy()
            for ov in orth_vecs:
                proj_val = qtn.tensor_contract(ov, new_tens.conj())  ### <x|b>
                if ix == 0:
                    tot_err += np.abs(proj_val)
                ov.transpose_like(orth_tens,inplace=True)
                # print('ov', ov.norm())
                # print('orth tens', orth_tens)
                # print('proj val', proj_val, ov.norm())

                orth_tens.modify(apply=lambda x: x - ov.data * np.conj(proj_val) / ov.norm()**2)
                print('site is orthog?', qtn.tensor_contract(ov, orth_tens.conj()))

            orthog_site_list += [orth_tens]

        return orthog_site_list, tot_err



    def _bond_solve(self, left_site_pos: int, site_tens: qtn.Tensor=None, return_intermediates=True,
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        update bond between left_site_pos, left_site_pos + 1
        """
        site_list, site_err = super()._bond_solve(left_site_pos, site_tens,
                                                  return_intermediates=return_intermediates)

        ## ket environment projected onto
        orth_vecs = [oterm.get_evaluated_site(left_site_pos, 0) for oterm in self.orthog_terms]

        ## orthogonal projection
        orthog_site_list = []
        tot_err = site_err
        for ix, new_tens in enumerate(site_list):
            orth_tens = new_tens.copy()
            for ov in orth_vecs:
                proj_val = qtn.tensor_contract(ov, new_tens)
                if ix == 0:
                    tot_err -= proj_val
                ov.transpose_like(orth_tens, inplace=True)
                orth_tens.modify(apply=lambda x: x - ov * proj_val)
            orthog_site_list += [orth_tens]

        return orthog_site_list, tot_err


    def update_blocks(self, i: int, direction: SweepDirection): #, new_ket_site=None):

        super().update_blocks(i, direction)

        for term in self.orthog_terms:
            # print('term extend env', term)
            term.extend_env(i, direction=direction) #, new_ket_site=new_ket_site)