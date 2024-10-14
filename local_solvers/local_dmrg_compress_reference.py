import numpy as np
import scipy.optimize
import cvxpy as cp
import time

import helper_quimb
from local_solvers.local_dmrg_eval import *

""" optimal compression with respect to some operator
"""

tol = 1e-8

def local_reference_compress(ref_state: Union['MPS','qtn.MatrixProductState'],
                               ref_mpo: 'qtn.MatrixProductOperator',
                               init_guess: Optional['qtn.MatrixProductState'] = None,
                               max_bond=None, cutoff:Numeric=CUTOFF, combine_terms_func:Callable=None,
                               nsites=1, direction: SweepDirection = SweepDirection.RIGHT, grid=None,
                               ):
    """ use OrthogonalDMRGEvaluator with first projecting out orthog_states from ref_state
        quadratic constraint:
    """
    init_guess = ref_state.copy() if init_guess is None else init_guess
    print('init guess', init_guess.max_bond())
    # if max_bond is not None and init_guess.max_bond() > max_bond:
    #     helper_quimb.compress(init_guess, compress_opts={'max_bond': max_bond})
    #     helper_quimb.match_inner_inds(init_guess, ref_state)
    #     # helper_quimb.check_orthog(init_guess)
    #
    #     init_guess[0].modify(apply=lambda x: x * 10**(init_guess.exponent - ref_state.exponent))
    #     init_guess.exponent = ref_state.exponent

    if ref_mpo is not None:
        ref_mpo_T = ref_mpo.copy()
        # ref_mpo_T.exponent = ref_mpo.exponent
        helper_quimb.mpo_flip_upper_lower(ref_mpo_T, inplace=True)
        # helper_quimb.mpo_transpose(ddx_mpo_T.data, inplace=True)
        AT_A = helper_quimb.apply(ref_mpo_T, ref_mpo)
        # AT_A = helper_quimb.apply(ref_mpo, ref_mpo_T)
        print('AT_A', AT_A)
        print('expec ATA', helper_quimb.expectation_value(init_guess, AT_A))

        # plt.figure()
        # plt.imshow(np.real(grid.axes[0].map_mpo_to_operator(ref_mpo_T)))
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(np.real(grid.axes[0].map_mpo_to_operator(AT_A)))
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(np.imag(grid.axes[0].map_mpo_to_operator(AT_A)))
        # plt.colorbar()
        # plt.show()
        # tmp = np.real(grid.axes[0].map_mpo_to_operator(ref_mpo))
        # tmp = tmp.T @ tmp
        # plt.figure()
        # plt.imshow(tmp)
        # plt.colorbar()
        # plt.show()


        print('init guess', init_guess.max_bond())
        term_op = Term_DMRG(init_guess, operators=[AT_A], bra=init_guess)
        term_vec = Term_DMRG(ref_state, operators=[AT_A], bra=init_guess)
        terms = [term_op, term_vec]
    else:
        terms = [Term_DMRG(ref_state, bra=init_guess)]

    solver = ReferenceDMRGEvaluator(init_guess, terms,
                                      # ref_mpo=ref_mpo,
                                      direction=direction, max_bond=max_bond, cutoff=cutoff,
                                      max_wrong_iter=2,
                                      grid=grid,
                                      )
    nsites = 1
    solver.solve(nsites)
    return solver.solution


class ReferenceDMRGEvaluator(DMRGEvaluator):

    def __init__(self,
                 ket_state: Union['MPS', 'qtn.MatrixProductState'],
                 terms: Sequence['Term_DMRG'],
                 direction: SweepDirection = SweepDirection.RIGHT,
                 max_bond: int = None, cutoff: Numeric=CUTOFF,
                 conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
                 max_tot_iter: int = DEFAULT_MAX_TOT_ITER, max_wrong_iter: int = DEFAULT_MAX_WRONG_ITER,
                 copy_obj: 'ConstrainedDMRGEvaluator' = None,
                 combine_terms_func: 'Callable' = None,
                 grid: 'Grid' = None, ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None,
                 #
                 # ref_mpo: Optional['qtn.MatrixProductOperator'] = None,
                 ):
        """
        if ref_mpo is not None:
            terms:  eff_op term <ket | O | ket>, eff_vec term <ket | O |ref ket>
        else:
            terms: eff_vec = <ket | ref_ket>  (eff_op term is the identity)
        """

        print('ket state max bond', ket_state.max_bond())
        super().__init__(ket_state, terms, direction=direction, max_bond=max_bond, conv_tol=conv_tol,
                         max_iter=max_iter, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter,
                         copy_obj=copy_obj, combine_terms_func=combine_terms_func,
                         grid=grid, ax_deriv_configs=ax_deriv_configs)

        self.cutoff = cutoff
        print('done super init')


    def _site_solve(self, left_site_pos: int, nsites: int, site_tens: qtn.Tensor = None, return_intermediates=False
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        solve but with scipy optimizer.
        """
        print('ket max bond', self.ket.max_bond(), self.out.max_bond(), self.terms[0].ket.max_bond())

        print('len self.terms', len(self.terms))
        if len(self.terms) == 1:
            site_list, site_err = super()._site_solve(left_site_pos, nsites, None, # site_tens,
                                                      return_intermediates=return_intermediates)
        else:
            # tmp = self.terms[0].get_eff_operator(left_site_pos, nsites, return_combined=True)[0]
            #
            # bra_to_ket = self.terms[0].projected_bra_to_ket(left_site_pos, nsites)
            # ket_to_bra = {v: k for k, v in bra_to_ket.items()}
            # bra_inds = [bi for bi in bra_to_ket.keys() if bi is not None]
            # ket_inds = [bra_to_ket[bi] for bi in bra_inds]
            # print('bra inds', bra_inds)
            # print('ket inds', ket_inds)
            # print('tmp', tmp)
            # tmp.transpose(*bra_inds, *ket_inds, inplace=True)
            # tmp_sz = int(np.round(np.sqrt(np.prod(tmp.shape))))
            # tmp_data = tmp.data.reshape(tmp_sz, tmp_sz)

            # plt.figure()
            # plt.imshow(np.real(tmp_data))
            # plt.colorbar()
            # plt.figure()
            # plt.imshow(np.imag(tmp_data))
            # plt.colorbar()
            # plt.show()


            eff_AA_ops = self.terms[0].get_eff_operator(left_site_pos, nsites, return_combined=False)[0]
            for eff_AA_op in eff_AA_ops:
                eff_AA_op.exponent += 2 * self.ket.exponent
            eff_AAx_vec = self.terms[1].get_evaluated_site(left_site_pos, nsites)
            eff_AAx_vec.modify(apply=lambda x: x * 10 ** self.ket.exponent)

            bra_to_ket = self.terms[0].projected_bra_to_ket(left_site_pos, nsites)
            ket_to_bra = {v: k for k, v in bra_to_ket.items()}

            ket_conj = self.ket[left_site_pos].conj()
            ket_conj = ket_conj.reindex(ket_to_bra)
            print('expec', np.sum( [qtn.tensor_contract(*eff_AA_op.tensors, self.ket[left_site_pos], ket_conj)
                                    * 10 ** eff_AA_op.exponent for eff_AA_op in eff_AA_ops] ))

            from helper_dmrg import qtn_conjugate_gradient_descent_1site
            soln_tens, site_err = qtn_conjugate_gradient_descent_1site(eff_AA_ops, eff_AAx_vec, self.ket[left_site_pos],
                                                                       bra_to_ket)
            site_list = [soln_tens]

            # exit()

        ### doing the compression here is fine, but need to ensure that another 1/2 sweep is performed.
        ## if output_tensor bond dimension is larger than the desired rank, then perform low-rank approximation.
        for out_tens in site_list:
            # print('self.out.max bond', self.out.max_bond(), self.max_bond)
            # if self.cutoff > CUTOFF or np.any([out_tens.ind_size(ind) > self.max_bond for ind in out_tens.inds]):
            if self.max_bond is not None and np.any([out_tens.ind_size(ind) > self.max_bond for ind in out_tens.inds]):
                # print('compress site_list')
                left_inds = [self.ket.bond(left_site_pos, left_site_pos -1)] if left_site_pos > 0 else []
                left_inds += [self.ket.site_ind_id.format(left_site_pos)]
                ltens, rtens = qtn.tensor_split(out_tens, left_inds, cutoff=self.cutoff, max_bond=self.max_bond, renorm=False)
                comp_out_tens = qtn.tensor_contract(ltens, rtens)
                comp_out_tens.transpose_like(out_tens, inplace=True)
                err = np.linalg.norm(comp_out_tens.data - out_tens.data)
                out_tens.modify(data=comp_out_tens.data) #, inds=comp_out_tens.inds)
                site_err += err

        return site_list, site_err



    def _bond_solve(self, left_site_pos: int, site_tens: qtn.Tensor=None, return_intermediates=True,
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        update bond between left_site_pos, left_site_pos + 1
        """
        site_list, site_err = super()._bond_solve(left_site_pos, site_tens,
                                                  return_intermediates=return_intermediates)

        raise NotImplementedError