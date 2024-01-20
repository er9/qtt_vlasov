from setup.configs import *
import time
import scipy.linalg as linalg
import quimb.tensor as qtn
# import helper_quimb as helper
from local_solvers.defaults import *
import local_solvers.helper_tn as helper_tn

MPO_type = Union['qtn.MatrixProductOperator']
MPS_type = Union['qtn.MatrixProductState']


def get_tens_rdm(tens: 'qtn.Tensor', ket_iso: Sequence[str], bra_iso: Optional[Sequence[str]]=None,
                 transpose_inds: Optional[Sequence[str]]=None):
    """ tens.conj() * tens, contracted over inds not specified in bra to ket inds
    """
    if bra_iso is None:
        bra_iso = [ind + '_' for ind in ket_iso]

    if transpose_inds is None:
        transpose_inds = [*ket_iso, *bra_iso]

    tens_conj = tens.conj()
    tens_conj.reindex({k: b for k, b in zip(ket_iso, bra_iso)}, inplace=True)

    rdm = qtn.tensor_contract(tens_conj, tens, output_inds=transpose_inds)

    return rdm


def get_tot_rdm(tensors: Sequence['qtn.Tensor'], ket_iso: Sequence[str], bra_iso: Optional[Sequence[str]]=None,
                transpose_inds: Optional[Sequence[str]]=None):

    rdms = [get_tens_rdm(tens, ket_iso, bra_iso, transpose_inds=transpose_inds) for tens in tensors]
    tot_rdm = helper_tn.sum_tens(rdms)
    return tot_rdm


def get_rdm_eig(tot_rdm: 'qtn.Tensor', ket_iso: Sequence[str], bra_iso: Optional[Sequence[str]]=None,
                new_ind: str=None, max_bond=None):
    """ A = Q Lambda Q.T  --> return Q
    """

    if bra_iso is None:
        bra_iso = [ind + '_' for ind in ket_iso]

    # if transpose_inds is None:
    #     transpose_inds = [*ket_iso, *bra_iso]

    tot_rdm.transpose(*ket_iso, *bra_iso, inplace=True)
    nb = len(ket_iso)

    denmat_shape = tot_rdm.shape
    tens_shape = denmat_shape[:nb]
    sq_shape = np.prod(denmat_shape[:nb])

    eigval, eigvec = np.linalg.eigh(tot_rdm.data.reshape(sq_shape, sq_shape))

    # sort_inds = np.argsort(np.abs(eigval))[::-1]

    ## apply cut-off
    # ev_max = eigval[sort_inds[0]]
    # new_sort_inds = []
    # for si in sort_inds:
    #     if np.abs(eigval[si]) > CUTOFF * np.abs(ev_max):
    #         new_sort_inds += [si]
    #     else:
    #         break
    # sort_inds = new_sort_inds

    if max_bond is not None:
        sort_inds = np.argsort(np.abs(eigval))[::-1]
        eigvec = eigvec[:, sort_inds[:max_bond]]
    # else:
    #     eigvec = eigvec[:, sort_inds]

    eigvec = eigvec.reshape(*tens_shape, -1)
    eigvec = qtn.Tensor(data=eigvec, inds=tuple(ket_iso) + (new_ind,))

    return eigvec
