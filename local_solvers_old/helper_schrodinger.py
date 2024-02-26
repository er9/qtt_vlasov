from setup.configs import *
import scipy.linalg
import helper_quimb as helper
from helper_dmrg import *
import helper_TE
import local_solvers_old.helper_tn as helper_tn
from setup.quimb_TN1D import MatrixProductStateUSVT as MPS_USVT


def _update_tn_with_tensor(dist_submpx: 'qtn.TensorNetwork', active_tens: 'qtn.Tensor',
                           submpx_tags: Sequence['str'], compress_direction=SweepDirection.RIGHT,
                           compress_opts=None):
    """ inplace operation
    """
    nsites = dist_submpx.num_tensors
    compress_opts = {'cutoff': CUTOFF, 'cutoff_mode': CUTOFF_MODE} if compress_opts is None else compress_opts

    T1 = active_tens
    s_tags_list = submpx_tags if compress_direction > 0 else submpx_tags[::-1]
    # print('update tn', submpx_tags, s_tags_list, compress_direction)
    for ix in range(nsites - 1):
        # print('s_ind', s_ind)
        dist_gtn_tens: 'qtn.Tensor' = next(iter(dist_submpx.select(s_tags_list[ix])))
        neighbor_tens: 'qtn.Tensor' = next(iter(dist_submpx.select(s_tags_list[ix + 1])))
        rix, lix = dist_gtn_tens.filter_bonds(neighbor_tens)
        # rix, lix = dist_submpx[s_tag].filter_bonds(dist_submpx[s_ind + 1 * compress_direction])
        TL, T1 = T1.split(lix, get='tensors', absorb='right', **compress_opts)

        TL.transpose_like(dist_gtn_tens, inplace=True)
        dist_gtn_tens.modify(data=TL.data)

    # print('update', s_tags_list[-1])
    dist_gtn_tens = dist_submpx[s_tags_list[-1]]
    T1 = T1.transpose_like(dist_gtn_tens, inplace=True)
    dist_gtn_tens.modify(data=T1.data)

    return dist_submpx



def _exactTE_site(dist_submpx: 'qtn.TensorNetwork', dt, deriv_tens: 'qtn.Tensor',
                  submpx_tags: Sequence[str], bonds_i: Sequence['str'], bonds_o: Sequence['str'],
                  inplace=False, compress_direction=SweepDirection.RIGHT, compress_opts=None,
                  verbose_plot=False
                  ) -> Optional['qtn.TensorNetwork1D']:
    """ compute projection of MPS onto site(s) [specified by nsites] or bond (right of site_ind) [nsites=0]
    """
    dist_submpx = dist_submpx if inplace else dist_submpx.copy()

    deriv_tens.transpose(*bonds_o, *bonds_i, inplace=True)

    if len(bonds_o) > 0:
        fuse_map = {'out': tuple(bonds_o), 'in': tuple(bonds_i)}
        proj_op = deriv_tens.fuse(fuse_map, inplace=False)
        proj_op.transpose('out','in', inplace=True)
    else:
        proj_op = deriv_tens

    # print('proj op', proj_total.shape)
    if len(proj_op.inds) > 0:
        sp_expm = scipy.linalg.expm(proj_op.data * dt)
        ### line 331 in scipy.linalg._matfuncs.py: change empty to zeros

        print('is H', np.linalg.norm(proj_op.data - proj_op.data.conj().T),
              'is AH', np.linalg.norm(proj_op.data + proj_op.data.conj().T))

        # evals, evecs = np.linalg.eig(proj_op.data)
        # exp_data =  (np.exp(evals * dt) * evecs) @ np.linalg.inv(evecs)
        # # exp_data = (np.exp(evals * dt) * evecs) @ evecs.conj().T    ## for unitary systems
        # error = np.linalg.norm(exp_data - sp_expm)
        # print('proj_op conditioning:', np.linalg.cond(proj_op.data, 1), np.linalg.cond(proj_op.data), error)

        # if error > 1.0e-8:
        #     print(np.linalg.cond(proj_op.data,np.inf), np.linalg.cond(proj_op.data,'fro'), np.linalg.cond(proj_op.data,-2))
        #     # print('evals', evals)
        #     plt.figure()
        #     plt.semilogy(np.abs(evals), 'r')
        #     plt.show()
        # else:
        #     plt.figure()
        #     plt.semilogy(np.abs(evals))
        #     plt.show()

        # proj_op.modify(data=exp_data)
        proj_op.modify(data=sp_expm)
        # proj_op.modify(apply=lambda x: scipy.linalg.expm(x * dt))

    else:
        proj_op.modify(apply=lambda x: np.exp(x * dt))

    if verbose_plot:
        plt.figure()
        plt.imshow(np.real(proj_op.data))
        plt.colorbar()
        plt.figure()
        plt.imshow(np.imag(proj_op.data))
        plt.colorbar()
        plt.title('exponentiated')
        plt.show()

    if len(bonds_o) > 0:
        shape_i = [deriv_tens.ind_size(bi) for bi in bonds_i]
        shape_o = [deriv_tens.ind_size(bo) for bo in bonds_o]
        exp_total = proj_op.unfuse(fuse_map, {'out': shape_o, 'in': shape_i})

        reindex_map = {bo: bi for bo, bi in zip(bonds_o, bonds_i)}
        active_tens: 'qtn.Tensor' = qtn.tensor_contract(*dist_submpx, exp_total)
        active_tens.reindex(reindex_map, inplace=True)

        _update_tn_with_tensor(dist_submpx, active_tens, submpx_tags,
                               compress_direction=compress_direction, compress_opts=compress_opts)
    else:
        ## dist_submpx is a scalar Tensor
        dist_submpx[submpx_tags[0]].modify(apply=lambda x: x * proj_op.data)

    return dist_submpx



######

def _euler_site(dist_submpx: 'qtn.TensorNetwork', dt, deriv: 'qtn.Tensor', submpx_tags=Sequence[str],
                inplace=False, compress_direction=SweepDirection.RIGHT, compress_opts=None
                ) -> Optional['GridTN']:
    """ df/dt = ...
        projected onto qtn manifold defined remaining sites
        for backwards TE, set -dt
        direction: canonicalization direction of output (+1: left canon, -1: right canon)
    """
    dist_submpx = dist_submpx if inplace else dist_submpx.copy()

    active_tens: 'qtn.Tensor' = dist_submpx.contract_tags(all)

    deriv.transpose_like(active_tens, inplace=True)
    active_tens.modify(apply=lambda data: data + dt * deriv.data)  # * 10**(-dist_submpx.exponent))

    _update_tn_with_tensor(dist_submpx, active_tens, submpx_tags,
                           compress_direction=compress_direction, compress_opts=compress_opts)

    return dist_submpx


# @profile
def site_time_evolution(dist_submpx: 'qtn.TensorNetwork', dt, # deriv_tens: 'qtn.Tensor',
                        deriv_tns: Sequence['qtn.TensorNetwork'],
                        submpx_tags: Sequence[str], bonds_i: Sequence[str], bonds_o: Sequence[str],
                        sources: Optional[Sequence['qtn.TensorNetwork']] = None,
                        te_order = 4, inplace=False, compress_direction = CompressDirection.RIGHT,
                        compress_level = 1, compress_opts_dict = None) -> 'qtn.TensorNetwork1D':
    """ time integration of specific site, as dictated by te_order. limited to RK TE methods
    """
    dist_submpx = dist_submpx if inplace else dist_submpx.copy()

    # deriv_total = _sum_eff_TNs(deriv_tns, transpose_bonds=list(bonds_o) + list(bonds_i))
    # sq_shape = int(np.sqrt(np.prod(deriv_total.shape)))
    # deriv_total_data = deriv_total.data.reshape(sq_shape,-1)
    # evals = np.linalg.eigvals(deriv_total_data)
    # print('is AH', np.linalg.norm(deriv_total_data + deriv_total_data.T.conj()))
    # print('deriv evals',np.max(np.abs(evals)), dt, np.max(np.abs(evals)) * dt)
    # out_exact = _exactTE_site(dist_submpx, dt, deriv_total, submpx_tags, bonds_i, bonds_o,
    #                      inplace=False, compress_direction=compress_direction,
    #                      compress_opts=compress_opts_dict.get(compress_level, None))

    if te_order == 0:
        deriv_total = helper_tn.sum_eff_TNs(deriv_tns, transpose_bonds=list(bonds_o) + list(bonds_i))
        return _exactTE_site(dist_submpx, dt, deriv_total, submpx_tags, bonds_i, bonds_o,
                             inplace=True, compress_direction=compress_direction,
                             compress_opts=compress_opts_dict.get(compress_level, None))

    def euler_func(dist_submpx_, dt_, deriv0=None, compress_level=1, inplace=False, **kwargs):
        return _euler_site(dist_submpx_, dt_, deriv0, submpx_tags, inplace=inplace,
                           compress_direction=compress_direction,
                           compress_opts=compress_opts_dict.get(compress_level, None))

    # @profile
    def deriv_func(dist_submpx_, **kwargs) -> 'qtn.Tensor':

        eff_Ax = []
        for deriv_tn in deriv_tns:
            eff_Ax += [qtn.TensorNetwork([deriv_tn, dist_submpx_])]
            # print('eff_ax', eff_Ax[-1].exponent, deriv_tn.exponent)
            # print('dist_submpx exponent', dist_submpx_.exponent)
            ## dist_submpx exponent should equal 0

        if sources is not None:
            eff_Ax += sources

        out = helper_tn.sum_eff_TNs(eff_Ax, transpose_bonds=bonds_o)
        # out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
        out.reindex({bo: bi for bo, bi in zip(bonds_o, bonds_i)}, inplace=True)
        return out

    def add_func(t1, t2, inplace=False, **kwargs):
        return helper.add_tensors(t1, t2, inplace=inplace)

    deriv0 = deriv_func(dist_submpx)
    # print('site time evolution', compress_opts_dict)
    out = helper_TE.time_integration(dist_submpx, dt, euler_func, deriv_func, add_func, helper.scale_tensors,
                                     te_order=te_order, deriv0=deriv0, compress_level=compress_level,
                                     compress_opts_dict=compress_opts_dict)
    if inplace:
        for stags in submpx_tags:
            t1 = next(iter(dist_submpx.select_tensors(stags)))
            t2 = next(iter(out.select_tensors(stags)))
            t1.modify(data=t2.data, inds=t2.inds)
        return dist_submpx
    else:
        return out

