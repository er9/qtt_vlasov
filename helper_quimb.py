# from setup.configs import *
from setup.defaults import *
import time
import quimb.tensor as qtn

""" helper functions for working with quimb Tensor or Tensor Network objects
"""

def mpx_add_label(mpx,label,check_exist=True) -> qtn.TensorNetwork1D:
    """ add dims to all labels in mpx (inplace)
    """
    if not (check_exist and label in mpx.site_tag_id):
        mpx.site_tag_id = mpx.site_tag_id + label

    if isinstance(mpx,qtn.MatrixProductState):
        if not (check_exist and label not in mpx.site_ind_id):
            mpx.site_ind_id = mpx.site_ind_id + label
    elif isinstance(mpx,qtn.MatrixProductOperator):
        if not (check_exist and label not in mpx.upper_ind_id):
            mpx.upper_ind_id = mpx.upper_ind_id + label
            mpx.lower_ind_id = mpx.lower_ind_id + label

    return mpx


def norm(mpx: qtn.TensorNetwork):
    """ return norm of mps/mpo
    """
    # return qtn.expec_TN_1D(mpx.H, mpx) * 10**(mpx.exponent)
    return mpx.norm() * 10 ** mpx.exponent


def ovlp(mps1: 'MPSType', mps2: 'MPSType'):
    tn = mps1.view_as(qtn.TensorNetwork)
    mps2 = mps2.reindex_sites(mps1.site_ind_id)
    tn.add(mps2)
    return tn.contract() * 10**(mps1.exponent + mps2.exponent)


def expectation_value(mpx: 'MPSType', obs_mpo: 'MPOType', bra: 'MPSType' = None):
    """ return norm of mps/mpo
    """
    if bra is None:  bra = mpx.conj()
    bra.site_ind_id = obs_mpo.upper_ind_id
    obs_mpo.lower_ind_id = mpx.site_ind_id
    tn_ = qtn.TensorNetwork([bra, obs_mpo, mpx])
    return tn_.contract() * 10**(mpx.exponent + obs_mpo.exponent + bra.exponent)
    # return qtn.expec_TN_1D(bra, obs_mpo, mpx) * 10 ** mpx.exponent
    # return mpx.comp_norms() * 10 ** (mpx.exponent)


def singular_values(mpx: Union['MPSType','MPOType'],i: int,cur_orthog=None):
    """ measures EE at bond i (between site i-1, i)
        von Neumann entropy:  S = -tr (rho*ln(rho)), where rho is s.t. <B> = tr(rho*B)
        quantum:  - \sum_i |a_i|^2 ln( |a_i|^2 )
        classical:  ...
    """
    sing_vals = mpx.singular_values(i,cur_orthog=cur_orthog)
    return sing_vals


def singular_values_all(mpx: Union['MPSType','MPOType']) -> list:
    """ measures EE at bond i (between site i-1, i)
        von Neumann entropy:  S = -tr (rho*ln(rho)), where rho is s.t. <B> = tr(rho*B)
        quantum:  - \sum_i |a_i|^2 ln( |a_i|^2 )
        classical:  ...
    """
    sing_vals = []
    mpx.right_canonize()
    for i in range(1,mpx.L):
        sing_vals += [singular_values(mpx,i,cur_orthog=i-1)]
    return sing_vals


def entanglement_entropy(mpx: Union['MPSType','MPOType'],i: int,cur_orthog=None):
    """ measures EE at bond i (between site i-1, i)
        von Neumann entropy:  S = -tr (rho*ln(rho)), where rho is s.t. <B> = tr(rho*B)
        quantum:  - \sum_i |a_i|^2 ln( |a_i|^2 )
        classical:  ...
    """
    sing_vals = mpx.singular_values(i,cur_orthog=cur_orthog)
    sing_vals *= 1./np.linalg.norm(sing_vals)
    EE = -1 * np.sum(sing_vals**2 * np.log( sing_vals**2 ))
    # print('mpx EE', mpx.calc_current_orthog_center(), sing_vals, EE)
    return EE


def entanglement_entropy_all(mpx: Union['MPSType','MPOType']) -> list:
    """ measures EE at bond i (between site i-1, i)
        von Neumann entropy:  S = -tr (rho*ln(rho)), where rho is s.t. <B> = tr(rho*B)
        quantum:  - \sum_i |a_i|^2 ln( |a_i|^2 )
        classical:  ...
    """
    EEs = []
    mpx.right_canonize()
    for i in range(1,mpx.L):
        EEs += [entanglement_entropy(mpx,i)] # ,cur_orthog=i-1)]
    return EEs


def distance(tn1,tn2):
    """ Frobenius norm of tn1-tn2
    """
    tn1 = tn1.copy()
    tn2 = tn2.copy()
    tn1.distribute_exponent()
    tn2.distribute_exponent()
    return qtn.tensor_network_distance(tn1,tn2)


def pad_mps(mps: 'MPSType', mps_inds: Sequence[int], mps_L: int, inplace=False, pad_ind_size=None):
    """ pad mps (with 1's) where padded mps is of length mps_L
        and original mps corresponds to indices specified by mps_inds
        mps_inds needs to be in increasing order
        not an inplace operation
    """
    if mps_inds is None or list(mps_inds) == list(range(mps_L)):
        return mps

    if len(mps_inds) > 1:
        assert (np.all(np.diff(mps_inds) > 0)), 'mps_inds must be in increasing order'

    site_tag_id = mps.site_tag_id
    site_ind_id = mps.site_ind_id

    padded_mps = renumber_mps(mps, list(range(mps.L)), mps_inds, site_tag_id=site_tag_id, site_ind_id=site_ind_id,
                              inplace=inplace)

    # if pad_ind_size is None:
    #     pad_ind_size = mps.ind_size(0)

    ind1 = 0
    for ind2 in mps_inds:

        if ind2 == ind1:   # no padded tensors:
            ind1 += 1
            continue

        ## else pad from ind1 to ind2
        if ind1 == 0:
            if pad_ind_size is None:
                ones = ones_mps(ind2, 1, site_ind_id=site_ind_id, site_tag_id=site_tag_id)
                for idx in range(ind2):
                    ones[idx].isel({site_ind_id.format(idx):0}, inplace=True)
            else:
                ones = ones_mps(ind2, pad_ind_size, site_ind_id=site_ind_id, site_tag_id=site_tag_id)
            padded_mps.add(ones)
            padded_mps.new_bond(site_tag_id.format(ind1), site_tag_id.format(ind2))
        else:
            for i in range(ind1,ind2-1):
                tag1, tag2 = site_tag_id.format(i), site_tag_id.format(ind2)
                tens1, tens2 = padded_mps.select(tag1), padded_mps.select(tag2)
                bond_dim = padded_mps.bond_size(tens1, tens2)
                padded_mps.insert_operator(np.eye(bond_dim[0]), site_tag_id.format(i), site_tag_id.format(ind2),
                                           tags=(site_tag_id.format(i+1),), inplace=True )
                if pad_ind_size:
                    tens = padded_mps.select_tensors((site_tag_id.format(i+1),))[0]
                    tens.new_ind(site_ind_id.format(i+1),pad_ind_size)

    if ind2 < mps_L:
        if pad_ind_size is None:
            ones = ones_mps(mps_L - ind2 - 1, 1, site_ind_id=site_ind_id, site_tag_id=site_tag_id)
            for idx in range(mps_L-ind2):
                ones[idx].isel({site_ind_id.format(idx): 0}, inplace=True)
            renumber_mps(ones, list(range(mps_L - ind2 - 1)), list(range(ind2 + 1, mps_L)), site_ind_id=site_ind_id,
                         site_tag_id=site_tag_id, inplace=True)
        else:
            ones = ones_mps(mps_L-ind2-1, pad_ind_size, site_ind_id=site_ind_id, site_tag_id=site_tag_id)
            renumber_mps(ones,list(range(mps_L-ind2-1)), list(range(ind2+1,mps_L)), site_ind_id=site_ind_id,
                         site_tag_id=site_tag_id, inplace=True)
        padded_mps.add(ones)
        padded_mps.new_bond(site_tag_id.format(ind2),site_tag_id.format(ind2+1))

    padded_mps._L = padded_mps.num_tensors

    return padded_mps


def mps_outerproduct(mps1: 'MPSType', mps2: 'MPSType', site_ind_id=None, site_tag_id=None):
    """ take outer product of mps1, mps2; fuse physical bonds into new physical bonds
    """
    assert(mps1.L == mps2.L),'mps1 and mps2 need to have the same length'

    if site_ind_id is None:  site_ind_id = mps1.site_ind_id
    if site_tag_id is None:  site_tag_id = mps1.site_tag_id

    if mps1.site_ind_id == mps2.site_ind_id:
        raise RuntimeError('mps1, mps2 must have different site_ind_id')

    new_mps = qtn.TensorNetwork([])
    for x in range(mps1.L):
        tens = mps1[x].contract(mps2[x])
        ind1, ind2 = mps1.site_ind(x), mps2.site_ind(x)
        if ind2 in tens.inds:
            if ind1 in tens.inds:
                tens.fuse({site_ind_id.format(x):(ind1, ind2)}, inplace=True)
            else:
                tens.reindex({ind2: site_ind_id.format(x)}, inplace=True)
        tens.drop_tags()
        tens.add_tag(site_tag_id.format(x))
        new_mps.add(tens)

    new_mps.view_like(mps1, site_ind_id=site_ind_id, site_tag_id=site_tag_id, inplace=True)
    new_mps.fuse_multibonds(inplace=True)

    return new_mps



def mpo_transpose(mpo_, mangle_inner=True, inplace=False):
    """ switch upper and lower ind labels
    """
    mpo_ = mpo_ if inplace else mpo_.copy()
    uid = mpo_.upper_ind_id
    lid = mpo_.lower_ind_id
    mpo_.lower_ind_id = '_tmp{}_'
    mpo_.upper_ind_id = lid
    mpo_.lower_ind_id = uid
    if mangle_inner:  mpo_.mangle_inner_()
    return mpo_


def mps_flip_lr(mps: 'qtn.MatrixProductState', inplace=True):
    mps = mps if inplace else mps.copy()
    L = mps.L
    ind_map = {mps.site_ind_id.format(i): mps.site_ind_id.format(L - 1 - i) for i in range(L)}
    tag_map = {mps.site_tag_id.format(i): mps.site_tag_id.format(L - 1 - i) for i in range(L)}
    mps.retag(tag_map, inplace=True)
    mps.reindex(ind_map, inplace=True)
    return mps


def mpo_flip_lr(mpo: 'qtn.MatrixProductOperator', inplace=True):
    mpo = mpo if inplace else mpo.copy()
    L = mpo.L
    upper_map = {mpo.upper_ind_id.format(i): mpo.upper_ind_id.format(L - 1 - i) for i in range(L)}
    lower_map = {mpo.lower_ind_id.format(i): mpo.lower_ind_id.format(L - 1 - i) for i in range(L)}
    tag_map = {mpo.site_tag_id.format(i): mpo.site_tag_id.format(L - 1 - i) for i in range(L)}
    mpo.retag(tag_map, inplace=True)
    mpo.reindex(upper_map, inplace=True)
    mpo.reindex(lower_map, inplace=True)
    return mpo


def renumber_mps(tn: 'MPSType', old_inds, new_inds, site_tag_id=None, site_ind_id=None, inplace=False) -> 'MPSType':
    """ eg. when some sites are removed/integrated out of mps, renumber tensors
        note: not designed to change site_tag_id, site_ind_id
    """
    tn = tn if inplace else tn.copy()
    site_tag_id = tn.site_tag_id if site_tag_id is None else site_tag_id
    site_ind_id = tn.site_ind_id if site_ind_id is None else site_ind_id

    for x in range(len(old_inds)):
        tens = tn.select_tensors(site_tag_id.format(x))[0]
        tens.reindex({site_ind_id.format(old_inds[x]): site_ind_id.format(new_inds[x])}, inplace=True)
        tens.retag  ({site_tag_id.format(old_inds[x]): site_tag_id.format(new_inds[x])}, inplace=True)
    return tn


def renumber_mpo(tn: 'MPOType', old_inds, new_inds, site_tag_id=None, upper_ind_id=None, lower_ind_id=None,
                 inplace=False) -> 'MPOType':
    """ eg. when some sites are removed/integrated out of mps, renumber tensors
        note: not designed to change site_tag_id, upper_ind_id, lower_ind_id
    """
    tn = tn if inplace else tn.copy()
    site_tag_id = tn.site_tag_id if site_tag_id is None else site_tag_id
    upper_ind_id = tn.upper_ind_id if upper_ind_id is None else upper_ind_id
    lower_ind_id = tn.lower_ind_id if lower_ind_id is None else lower_ind_id

    for x in range(len(old_inds)):
        tens = tn.select_tensors(site_tag_id.format(x))[0]
        tens.reindex({upper_ind_id.format(old_inds[x]): upper_ind_id.format(new_inds[x]),
                      lower_ind_id.format(old_inds[x]): lower_ind_id.format(new_inds[x]),}, inplace=True)
        tens.retag({site_tag_id.format(old_inds[x]): site_tag_id.format(new_inds[x])}, inplace=True)
    return tn


def mps_to_diag_mpo(mps: 'MPSType', upper_ind_id=None, lower_ind_id=None) -> 'MPOType':
    """ change MPS to MPO with elements of MPS along the diagonal
    """
    bond_name = mps.site_ind_id
    tens_name = mps.site_tag_id

    qs = mps.shape    # physical bond dimensions; physical bonds are all external dims

    out = mps.view_as(qtn.TensorNetwork,inplace=False)
    
    bond_name_1 = bond_name+'[1]'
    bond_name_2 = bond_name+'[2]'
    for i in range(mps.num_tensors):

        d_ijk = qtn.tensor_core.COPY_tensor(qs[i], 
                                (bond_name.format(i), bond_name_1.format(i), bond_name_2.format(i)),
                                tags=(f'd_ijk({i})',))
        out.add_tensor( d_ijk )
        out.contract( tags=(tens_name.format(i),f'd_ijk({i})'), inplace=True )

    out.view_as(qtn.MatrixProductOperator, inplace=True, like=mps,
                                           upper_ind_id=bond_name_1, lower_ind_id=bond_name_2)
    out.drop_tags([f'd_ijk({i})' for i in range(mps.L)])

    if upper_ind_id is not None:   out.upper_ind_id = upper_ind_id
    if lower_ind_id is not None:   out.lower_ind_id = lower_ind_id

    return out


# @profile
def apply( mpo1, mpx2, compress=False, verbose=False, compress_opts: dict = None ) -> Union['MPSType', 'MPOType']:
    """ allows for mpo1, mpx2 to have different site_tag_id's, and be of length 1
    """
    mpo1.site_tag_id = mpx2.site_tag_id

    if mpo1.L == 1:
        A, x = mpo1.copy(), mpx2.copy()

        # align the indices
        A.lower_ind_id = "__tmp{}__"
        if isinstance(mpx2,qtn.MatrixProductState):
            A.upper_ind_id = x.site_ind_id
            x.reindex_sites_("__tmp{}__")
        elif isinstance(mpx2,qtn.MatrixProductOperator):
            A.upper_ind_id = x.upper_ind_id
            x.reindex_upper_sites_("__tmp{}__")

        # form total network and contract each site
        x.add(A)  # adding tensors to contract with each other A*x
        x.exponent = mpo1.exponent + mpx2.exponent
        x.contract_ind("__tmp0__")
        new_mpx = x

    else:
        A, x = mpo1.copy(), mpx2.copy()

        # align the indices
        A.lower_ind_id = "__tmp{}__"
        if isinstance(mpx2,qtn.MatrixProductState):
            A.upper_ind_id = x.site_ind_id
            x.reindex_sites_("__tmp{}__")
        elif isinstance(mpx2,qtn.MatrixProductOperator):
            A.upper_ind_id = x.upper_ind_id
            x.reindex_upper_sites_("__tmp{}__")

        new_mpx = A.apply(x, inplace=False)  ## takes exponents into account
        ## write this out explicitly?

        # print('apply', x.exponent, A.exponent, new_mpx.exponent)
        if compress:  compress_func(new_mpx, scale=True, verbose=verbose, compress_opts=compress_opts)

    return new_mpx


# @profile
def apply_zipup(mpo1: 'MPOType', mpx2: Union['MPSType', 'MPOType'],
                compress=False, verbose=False, compress_opts: dict = None ) -> Union['MPSType', 'MPOType']:
    """ apply mpo1 to mpx2; perform approximate contraction
    """
    mpo1.site_tag_id = mpx2.site_tag_id

    if compress_opts is None:   compress_opts = {}
    else:                       compress_opts = compress_opts.copy()
    compress_opts.setdefault('cutoff',CUTOFF)
    compress_opts.setdefault('cutoff_mode',CUTOFF_MODE)

    form = compress_opts.get('form','right')
    if not compress:
        form = 'right' if form == 'left' else 'left'
        # we don't perform the last compression to put it in the desired form

    if mpo1.L == 1:
        A, x = mpo1.copy(), mpx2.copy()

        # align the indices
        A.lower_ind_id = "__tmp{}__"
        if isinstance(mpx2, qtn.MatrixProductState):
            A.upper_ind_id = x.site_ind_id
            x.reindex_sites_("__tmp{}__")
        elif isinstance(mpx2, qtn.MatrixProductOperator):
            A.upper_ind_id = x.upper_ind_id
            x.reindex_upper_sites_("__tmp{}__")

        # form total network and contract each site
        x.add(A)  # adding tensors to contract with each other A*x
        x.exponent = mpo1.exponent + mpx2.exponent
        x.contract_ind("__tmp0__")
        new_mpx = x

    else:
        A, x = mpo1.copy(), mpx2.copy().mangle_inner_()
        ## perhaps optional? bc i take norm out separately?
        # canonize(A, i=0)
        # canonize(x, i=0)

        # align the indices
        A.lower_ind_id = "__tmp{}__"
        if isinstance(mpx2, qtn.MatrixProductState):
            A.upper_ind_id = x.site_ind_id
            x.reindex_sites_("__tmp{}__")
        elif isinstance(mpx2, qtn.MatrixProductOperator):
            A.upper_ind_id = x.upper_ind_id
            x.reindex_upper_sites_("__tmp{}__")

        new_mpx = qtn.TensorNetwork([])

        if form == 'right':   # first put in left canonical form
            new_tens = qtn.tensor_contract(A[0], x[0])
            new_tens.drop_tags()
            A_idx, open_idx = new_tens.filter_bonds(A[1])
            x_idx, open_idx = new_tens.filter_bonds(x[1])
            lix = [idx for idx in open_idx if idx not in A_idx + x_idx]
            tensL, tensR = new_tens.split(lix, absorb='right', cutoff_mode=CUTOFF_MODE, right_inds=A_idx + x_idx,
                                          cutoff=compress_opts.get('cutoff', CUTOFF), ltags=(x.site_tag(0)))
            new_mpx.add(tensL)

            for i in range(1, x.L-1):
                # print('contract', tensR, A[i], x[i])
                new_tens = qtn.tensor_contract(tensR,x[i])
                new_tens = qtn.tensor_contract(new_tens,A[i])
                new_tens.drop_tags()
                A_idx, open_idx = new_tens.filter_bonds(A[i+1])
                x_idx, open_idx = new_tens.filter_bonds(x[i+1])
                lix = [idx for idx in open_idx if idx not in A_idx + x_idx]
                tensL, tensR = new_tens.split(lix, absorb='right', cutoff_mode=CUTOFF_MODE, right_inds=A_idx + x_idx,
                                              cutoff=compress_opts.get('cutoff', CUTOFF), ltags=(x.site_tag(i)))
                new_mpx.add(tensL)

            new_tens = qtn.tensor_contract(tensR, A[-1], x[-1])
            new_mpx.add(new_tens)

        else:
            new_tens = qtn.tensor_contract(A[x.L-1], x[x.L-1])
            new_tens.drop_tags()
            A_idx, open_idx = new_tens.filter_bonds(A[x.L-2])
            x_idx, open_idx = new_tens.filter_bonds(x[x.L-2])
            lix = [idx for idx in open_idx if idx not in A_idx + x_idx]
            tensL, tensR = new_tens.split(lix, absorb='right', cutoff_mode=CUTOFF_MODE, right_inds=A_idx + x_idx,
                                          cutoff=compress_opts.get('cutoff', CUTOFF), ltags=(x.site_tag(x.L-1)))
            new_mpx.add(tensL)

            for i in range(x.L-2, 0, -1):
                # print('contract', tensR, A[i], x[i])
                new_tens = qtn.tensor_contract(tensR, x[i])
                new_tens = qtn.tensor_contract(new_tens, A[i])
                new_tens.drop_tags()
                A_idx, open_idx = new_tens.filter_bonds(A[i-1])
                x_idx, open_idx = new_tens.filter_bonds(x[i-1])
                lix = [idx for idx in open_idx if idx not in A_idx + x_idx]
                # print('new_tens', new_tens.norm())
                tensL, tensR = new_tens.split(lix, absorb='right', cutoff_mode=CUTOFF_MODE, right_inds=A_idx + x_idx,
                                              cutoff=compress_opts.get('cutoff', CUTOFF), ltags=(x.site_tag(i)))
                new_mpx.add(tensL)
                # print('tensL', tensL.norm())

            new_tens = qtn.tensor_contract(tensR, A[0], x[0])
            new_mpx.add(new_tens)

        new_mpx = new_mpx.view_like(x, inplace=True)
        new_mpx.exponent = A.exponent + x.exponent

        # print('zip up before', new_mpx.max_bond())
        if compress:
            compress_opts['form'] = form
            compress_func(new_mpx, scale=True, verbose=verbose, canonize=False, compress_opts=compress_opts)
            # print('zip up after', new_mpx.max_bond())

    return new_mpx


def add_MPS(mps1: qtn.MatrixProductState, mps2: qtn.MatrixProductState, inplace=False,
            compress=False, compress_opts:dict = None):
    """ add MPS while accounting for their norms contained in mps.exponent
    """
    if mps1 is None or np.isneginf(mps1.exponent):   return mps2.copy()
    if mps2 is None or np.isneginf(mps2.exponent):   
        if inplace:    return mps1
        else:          return mps1.copy()

    out = mps1 if inplace else mps1.copy()
    tmp = mps2.reindex_sites(mps1.site_ind_id)

    # out.distribute_exponent()
    tmp.exponent -= out.exponent
    tmp.distribute_exponent()
    
    out.add_MPS(tmp, inplace=True)
    if compress:  compress_func(out, compress_opts=compress_opts)
    # out.equalize_norms()

    return out


def add_MPO(mpo1: qtn.MatrixProductOperator, mpo2: qtn.MatrixProductOperator,
            inplace=False, compress=False, compress_opts:dict=None):
    """ add MPOs while accounting for their norms contained in mps.exponent
    """
    if mpo1 is None or np.isneginf(mpo1.exponent):   return mpo2.copy()
    if mpo2 is None or np.isneginf(mpo2.exponent):   
        if inplace:    return mpo1
        else:          return mpo1.copy()

    out = mpo1 if inplace else mpo1.copy()
    tmp = mpo2.copy()
    if tmp.upper_ind_id != mpo1.upper_ind_id:
        if mpo1.upper_ind_id == tmp.lower_ind_id:
            tmp.lower_ind_id = '_tmp{}_'
            tmp.upper_ind_id = mpo1.upper_ind_id
            tmp.lower_ind_id = mpo1.lower_ind_id
        else:
            tmp.upper_ind_id = mpo1.upper_ind_id

    if tmp.lower_ind_id != mpo1.lower_ind_id:
        tmp.lower_ind_id = mpo1.lower_ind_id


    # out.distribute_exponent()
    tmp.exponent -= out.exponent
    tmp.distribute_exponent()
    
    out.add_MPO(tmp, inplace=True)
    if compress:  compress_func(out, compress_opts=compress_opts)
    # out.equalize_norms()

    return out


def canonize(mps: Union['MPSType', 'MPOType'], scale=True, form='right', i=None, cur_orthog='calc'):
    """ canonize MPS around site i
    """
    if mps is None or np.isneginf(mps.exponent):   return None

    if i is None:
        if form == 'right':   i = 0
        elif form == 'left':  i = mps.L-1

    if cur_orthog == 'calc':
        cur_orthog = mps.calc_current_orthog_center()
    if not isinstance(cur_orthog, tuple):
        if cur_orthog < 0:
            cur_orthog = (0, mps.L-1)
        else:
            cur_orthog = (cur_orthog, cur_orthog)

    try:
        if not scale:         return mps.canonize(i)
        else:

            ## helper function to normalize site i, shift weight into mps.exponent
            def site_norm_to_exponent(ix):
                tag = mps.site_tag(ix)
                tens = mps[tag] 
                # tens = mps.select_tensors(tag)
                mps.strip_exponent(tens)  # normalizes tensor, adds norm to exponent

            ## canonicalize
            for x in range(cur_orthog[0],i):
                mps.left_canonize_site(x)
                # mps.left_compress_site(x, max_bond=None, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                site_norm_to_exponent(x+1)

            for x in range(cur_orthog[1],i,-1):
                mps.right_canonize_site(x)
                # mps.right_compress_site(x, max_bond=None, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                site_norm_to_exponent(x-1)

    except np.linalg.LinAlgError:
        if np.isinf(mps.exponent):    mps = None
        else:                         raise np.linalg.LinAlgError

    if mps.exponent < -30:   return None
    return mps


def compress(mps: Union[MPSType, MPOType], scale=True, verbose=False, canonize=True, compress_opts:dict = None):
    """ canonicalize and then compress MPS while absorbing norm of singular values/tensors
        into mps.exponent norm
        inplace operation
    """
    if compress_opts is None:   compress_opts = {}
    else:                       compress_opts = compress_opts.copy()
    compress_opts.setdefault('cutoff',CUTOFF)
    compress_opts.setdefault('cutoff_mode',CUTOFF_MODE)

    # if compress_opts.get('max_bond',None) is None:
    #     canonize = False

    # if mps is None or np.isneginf(mps.exponent):   return None
    if mps is None or mps.exponent < -30:   return None

    if verbose:   print('compress',compress_opts)

    do_midpt = compress_opts.pop('do_midpt', False)
    if do_midpt:
        midpt = compress_opts.pop('midpt', mps.L//2)
        compress_midpt(mps, midpt, scale=scale, verbose=verbose, compress_opts=compress_opts)
        return mps

    try:
        if not scale:        return mps.compress(**compress_opts)
        else:
            form = compress_opts.pop('form','right')

            ## helper function to normalize site i, shift weight into mps.exponent
            def site_norm_to_exponent(ix):
                tag = mps.site_tag_id.format(ix)
                tens = mps[tag] 
                mps.strip_exponent(tens)  # normalizes tensor, adds norm to exponent

            if form == 'left':
                ## canonicalize
                if canonize:
                    for i in range(mps.L-1,0,-1):
                        mps.right_canonize_site(i)
                        site_norm_to_exponent(i-1)

                ## compress
                for i in range(mps.L-1):
                    mps.left_compress_site(i, **compress_opts)
                    # mps.left_canonize_site(i)
                    # site_norm_to_exponent(i+1)

            elif form == 'right' or form is None:
                ## canonicalize
                if canonize:
                    for i in range(mps.L-1):
                        mps.left_canonize_site(i)
                        site_norm_to_exponent(i+1)

                ## compress
                for i in range(mps.L-1,0,-1):
                    if verbose:
                        print('before orthog i',i,mps.calc_current_orthog_center())
                        print('before',i,mps.singular_values(i,cur_orthog=i))
                    mps.right_compress_site(i, **compress_opts)
                    if verbose:
                        print('after',i,mps.singular_values(i,cur_orthog=i-1))
                        mps.right_canonize_site(i)
                        print('after orthog i',i,mps.calc_current_orthog_center())
                    # site_norm_to_exponent(i-1)

    except np.linalg.LinAlgError:
        print('error', i, mps.singular_values(i), mps.exponent)
        if np.isinf(mps.exponent):
            mps = None
        elif mps.singular_values[i] == [0]:
            mps = None
        else:
            raise np.linalg.LinAlgError

    # print('mps exp',mps.exponent)
    if mps is None or mps.exponent < -30:   return None
    return mps

compress_func = compress


def compress_midpt(mps, mid_pt, scale=True, verbose=False, timeit=False, mid_compress_opts=None, compress_opts=None):
    """ compress MPS while absorbing norm of singular values/tensors into mps.exponent norm
        inplace operation
    """
    # print('compress midpt')
    if mps is None or np.isneginf(mps.exponent):   return None

    if compress_opts is None:   compress_opts = {}
    else:                       compress_opts = compress_opts.copy()

    if mid_compress_opts is None:  mid_compress_opts = compress_opts
    else:                          mid_compress_opts = mid_compress_opts.copy()

    if mps.exponent < -30:   return None

    if verbose:   print('compress', compress_opts)
    if timeit:    canon_times, comp_times = [], []

    try:

        form = compress_opts.pop('form', 'right')  ## toss this

        ## helper function to normalize site i, shift weight into mps.exponent
        def site_norm_to_exponent(i):
            tag = mps.site_tag(i)
            tens = mps[tag]
            mps.strip_exponent(tens)  # normalizes tensor, adds norm to exponent

        ## canonicalize
        ## canonicalize LHS to left canonical
        for i in range(mid_pt):
            time1 = time.time()
            mps.left_canonize_site(i)
            if timeit:   canon_times += [time.time() - time1]
            if scale:    site_norm_to_exponent(i + 1)

        ## canonicalize RHS to right canonical
        for i in range(mps.L - 1, mid_pt, -1):
            time1 = time.time()
            mps.right_canonize_site(i)
            if timeit:   canon_times += [time.time() - time1]
            if scale:    site_norm_to_exponent(i - 1)

        # print('check orthog i',mid_pt,mps.calc_current_orthog_center())

        ## compress mid point --> M = midpt -> M = midpt - 1
        svals = right_compress_site(mps, mid_pt, return_svals=True, **mid_compress_opts)
        # print('svals',svals)
        # print(mps.exponent)
        # print([mps[i].norm() for i in range(mps.L)])
        if verbose:
            print('mid compress shape', mps[mid_pt].shape, mid_compress_opts)


        ## compression scheme 1
        mps.insert_operator(np.diag(1. / svals), mps.site_tag(mid_pt - 1), mps.site_tag(mid_pt),
                            tags='svals_inv', inplace=True)
        mps.insert_operator(np.diag(svals), 'svals_inv', mps.site_tag(mid_pt),
                            tags='svals', inplace=True)

        ## compress left hand side -> right canonical
        for i in range(mid_pt - 1, 0, -1):
            if verbose:
                print('before orthog i', i, mps.calc_current_orthog_center())
                print('before', i, mps.singular_values(i, cur_orthog=i))

            time1 = time.time()
            right_compress_site(mps, i, **compress_opts)
            if timeit:   comp_times += [time.time() - time1]

            if verbose:
                print('after', i, mps.singular_values(i, cur_orthog=i - 1))
                mps.right_canonize_site(i)
                print('after orthog i', i, mps.calc_current_orthog_center())

        ## compress right hand side -> to the left canonical
        mps.contract_between('svals', mps.site_tag(mid_pt))

        for i in range(mid_pt, mps.L - 1):
            # mps.left_compress_site(i, **compress_opts)
            time1 = time.time()
            left_compress_site(mps, i, **compress_opts)
            if timeit:   comp_times += [time.time() - time1]

        mps.contract_between('svals_inv', mps.site_tag(mid_pt))
        mps[mid_pt].drop_tags(('svals', 'svals_inv'))


    except(np.linalg.LinAlgError, ValueError):
        print('error', i, mps.singular_values(i), mps.exponent)
        if np.isinf(mps.exponent):
            mps = None
        elif mps.singular_values(i)[0] <= 1e-12:
            mps = None
        else:
            raise np.linalg.LinAlgError

    # print('mps exp',mps.exponent)

    if timeit:
        return mps, canon_times, comp_times

    if mps.exponent < -30:   return None
    return mps


def left_compress_site(mps, i, return_svals=False, **compress_opts):
    """ inplace compression of ith site of mps
    """
    max_bond = compress_opts.get('max_bond', -1)
    do_adapt = compress_opts.pop('adapt', False)
    adapt_cutoff = compress_opts.pop('adapt_cutoff', 0.1)  # cutoff fraction below DMAX singular val

    if return_svals and not do_adapt:
        do_adapt = True
        adapt_cutoff = 0.0

    if do_adapt:  # and max_bond is not None:
        if max_bond is None:
            max_bond = -1

        T1, T2 = mps[i], mps[i + 1]
        rix, lix = T1.filter_bonds(T2)
        U, svals, VT = T1.split(lix, get='arrays', absorb=None, cutoff_mode=CUTOFF_MODE,
                                cutoff=compress_opts.get('cutoff', CUTOFF))

        try:
            # U = U[...,:max_bond]
            # svals = svals[:max_bond]
            # VT = VT[:max_bond,...]

            if max_bond > len(svals):
                svals_ref = svals[-1] / svals[0]
            else:
                svals_ref = svals[max_bond] / svals[0]
            U, svals, VT = qtn.decomp._trim_and_renorm_SVD(U, svals, VT,
                                                           (svals_ref * adapt_cutoff) ** 2, 4,  # cutoff, 'rsum2'
                                                           max_bond, None, 1)  # max_bond, absorb, renormalize
            # # keep_ind = np.argmin(svals >= svals_ref*adapt_cutoff)
            # # if keep_ind == 0:   raise(IndexError)  # keep all singular values
            # # U = np.take(U, range(keep_ind), axis=-1)
            # # svals = svals[:keep_ind]
            # # VT = VT[:keep_ind]
        except IndexError:
            pass

        VT = (svals * (VT.T)).T
        TL = qtn.Tensor(data=U, inds=(*lix, 'tmp'))
        TR = qtn.Tensor(data=VT, inds=('tmp', *rix))
        TR = TR.contract(T2)

        TL.transpose_like(T1, inplace=True)
        TR.transpose_like(T2, inplace=True)
        T1.modify(data=TL.data)
        T2.modify(data=TR.data)

        if return_svals:
            return svals

    else:
        mps.left_compress_site(i, **compress_opts)


def right_compress_site(mps, i, return_svals=False, **compress_opts):
    """ inplace compression of ith site of mps
    """
    max_bond = compress_opts.get('max_bond', -1)
    do_adapt = compress_opts.pop('adapt', False)
    adapt_cutoff = compress_opts.pop('adapt_cutoff', 0.1)  # cutoff fraction below DMAX singular val

    if return_svals and not do_adapt:
        do_adapt = True
        adapt_cutoff = 0.0

    if do_adapt:  # and max_bond is not None:
        if max_bond is None:
            max_bond = -1

        T1, T2 = mps[i - 1], mps[i]
        # print('T1',T1, 'T2',T2)
        lix, rix = T2.filter_bonds(T1)
        U, svals, VT = T2.split(lix, get='arrays', absorb=None, cutoff_mode=CUTOFF_MODE, right_inds=rix,
                                cutoff=compress_opts.get('cutoff', CUTOFF))

        try:
            # U = U[...,:max_bond]
            # svals = svals[:max_bond]
            # VT = VT[:max_bond,...]

            if max_bond > len(svals):
                svals_ref = svals[-1] / svals[0]
            else:
                svals_ref = svals[max_bond] / svals[0]
            U, svals, VT = qtn.decomp._trim_and_renorm_SVD(U, svals, VT,
                                                           (svals_ref * adapt_cutoff) ** 2, 4,  # cutoff, 'rsum2'
                                                           max_bond, None, 1)  # max_bond, absorb, renormalize
            # print(svals_ref,len(svals))
            # # keep_ind = np.argmin(svals >= svals_ref*adapt_cutoff)
            # # # print('r keep ind',max_bond,len(svals),keep_ind, svals_ref, np.min(svals))
            # # if keep_ind == 0:   raise(IndexError)  # keep all singular values
            # # U = np.take(U, range(keep_ind), axis=-1)
            # # svals = svals[:keep_ind]
            # # VT = VT[:keep_ind]
        except IndexError:  ## max_bond > len(svals)
            pass

        U = U * svals
        TL = qtn.Tensor(data=U, inds=(*lix, 'tmp'))
        TR = qtn.Tensor(data=VT, inds=('tmp', *rix))
        TL = TL.contract(T1)

        TL.transpose_like(T1, inplace=True)
        TR.transpose_like(T2, inplace=True)
        T1.modify(data=TL.data)
        T2.modify(data=TR.data)

        if return_svals:
            return svals

    else:
        mps.right_compress_site(i, **compress_opts)


def canonize_tens_list(*tens, inplace=False):
    """ canonize this list of tensors from left to right
    """
    return compress_tens_list(*tens, inplace=inplace, compress_opts={'method':'qr'})


def compress_tens_list(*tens, inplace=False, compress_opts=None, two_site=False):
    """ compress this list of tensors from left to right
        assumes already in canonical form
    """
    if compress_opts is None:   compress_opts = {}
    else:                       compress_opts = compress_opts.copy()
    compress_opts['absorb'] = 'right'
    compress_opts.setdefault('cutoff', CUTOFF)
    compress_opts.setdefault('cutoff_mode', CUTOFF_MODE)

    if not inplace:
        tens = tuple([t.copy() for t in tens])

    for i in range(len(tens) - 1):

        if two_site:   ## contracting tensors together before splitting
            _, bonds_L = tens[i].filter_bonds(tens[i + 1])
            # _, bonds_R = tens[i+1].filter_bonds(tens[i])
            active_tens = tens[i].contract(tens[i+1])
            TL, TR = active_tens.split(bonds_L, **compress_opts)
        else:
            bonds_R, bonds_L = tens[i].filter_bonds(tens[i+1])
            TL, TR = tens[i].split(bonds_L, **compress_opts)
            TR = TR.contract(tens[i+1])

        if TL.ndim == tens[i].ndim:
            TL.transpose_like(tens[i],inplace=True)
            TR.transpose_like(tens[i+1],inplace=True)
            tens[i].modify(data=TL.data)
            tens[i+1].modify(data=TR.data)
        else:
            tens[i].modify(data=TL.data, inds=TL.inds)
            tens[i+1].modify(data=TR.data, inds=TR.inds)

    return tens


def scalar_multiply(mps: 'qtn.TensorNetwork', scalar_const: Numeric ,inplace=False):
    """ scalar multiplication
    """
    val = np.abs(scalar_const)
    if np.isreal(scalar_const):
        if scalar_const < 0:  sign = -1
        else:                 sign =  1
    else:
        sign = np.exp(-1.j*np.angle(scalar_const))

    mps = mps if inplace else mps.copy()

    if val != 0:
        mps.exponent += np.log10(val)    # change total norm of MPS
    else:
        mps.exponent += -np.inf

    mps.multiply(sign,inplace=True,spread_over=1)
    return mps


def scalar_add(mps,scalar_val,inplace=False,compress=False,compress_opts: dict = None):
    """ add scalar value to mps
    """
    mps = mps if inplace else mps.copy()

    scalar_val_mps = sum_tensornetwork(like_tn=mps, scale=scalar_val)
    scalar_val_mps = scalar_val_mps.view_like(mps, inplace=True)  # view as MPS

    add_MPS(mps, scalar_val_mps, inplace=True, compress=compress, compress_opts=compress_opts)
    return mps


# def elemental_multiply(mps1,mps2,inplace=False,compress=False,compress_opts=None):
#     """ h(x) = f(x) * g(x)
#         elemental multiplication of the two MPS
#         for actual elemental_multiply, cannot do inplace. (inplace is for scalar mult)
#     """
#     if np.isscalar(mps1):
#         return scalar_multiply(mps2,mps1,inplace=inplace)
#     if np.isscalar(mps2):
#         return scalar_multiply(mps1,mps2,inplace=inplace)
#
#     if (mps1 is None or np.isneginf(mps1.exponent)) or (mps2 is None or np.isneginf(mps2.exponent)):
#         return None
#
#     # mps1.distribute_exponent()
#     # mps2.distribute_exponent()
#
#     bond_name = mps1.site_ind_id
#     tens_name = mps1.site_tag_id
#
#     if mps1.L == 1:
#         out = mps1.copy()
#         t1 = out.select_tensors(mps1.site_tag(0))[0]
#         t2 = mps2.select_tensors(mps2.site_tag(0))[0]
#         new_data = t1.data * t2.data * 10 ** mps2.exponent
#         t1.modify(data=new_data)
#         return out
#
#
#     # physical bond dimensions; physical bonds are all external dims
#     qs = mps1.shape
#
#     out = mps1.view_as(qtn.TensorNetwork,inplace=False)
#
#     bond_name_1 = bond_name+'[1]'
#     bond_name_2 = bond_name+'[2]'
#     for i in range(mps1.num_tensors):
#
#         d_ijk = qtn.tensor_core.COPY_tensor(qs[i],
#                                 (bond_name.format(i), bond_name_1.format(i), bond_name_2.format(i)),
#                                 tags=(f'd_ijk({i})',))
#         out.add_tensor( d_ijk )
#         out.contract( tags=(tens_name.format(i),f'd_ijk({i})'), inplace=True )
#
#     out.view_as(qtn.MatrixProductOperator, inplace=True, like=mps1,
#                                            upper_ind_id=bond_name_1, lower_ind_id=bond_name_2)
#     out.drop_tags([f'd_ijk({i})' for i in range(mps1.L)])
#
#     out = apply(out,mps2)
#     if compress:   compress_func(out, compress_opts=compress_opts)
#
#     return out


# def get_convolve_op(L,q=2,out_id=None,in1_id=None,in2_id=None,site_tag_id=None):
#     """ get operator that convolves two input vectors -> output
#         f(i) = sum_k g(k) h(i-k)
#         output is of the same length as the input
#         a 1D TN with 3 i/o legs
#     """
#     if q != 2:  raise NotImplementedError('convolution for q!=2 not implemented')
#     conv_op_tn = qtn.TensorNetwork([])
#
#     out_id = 'ind1,{}' if out_id is None else out_id
#     in1_id = 'ind1,{}' if in1_id is None else in1_id
#     in2_id = 'ind1,{}' if in2_id is None else in2_id
#     site_tag_id = 'site{}' if site_tag_id is None else site_tag_id
#
#     T0 = qtn.Tensor( np.array([[[[0.,1.],[1.,0.]],[[1.,0.],[0.,0.]]],
#                                [[[0.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]]),
#                      inds=(out_id.format(0),'b0',in1_id.format(0),in2_id.format(0)),
#                      tags=(site_tag_id.format(0),) )
#     conv_op_tn.add(T0)
#
#     for x in range(1,L-1):
#         T1 = qtn.Tensor( np.array([[[[[1.,0.],[0.,0.]],[[0.,0.],[0.,0.]]],
#                                     [[[0.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]],
#                                    [[[[0.,1.],[1.,0.]],[[1.,0.],[0.,0.]]],
#                                     [[[0.,0.],[0.,0.]],[[0.,0.],[0.,1.]]]]]),
#                          inds=(out_id.format(x),f'b{x-1}',f'b{x}',in1_id.format(x),in2_id.format(x)),
#                          tags=(site_tag_id.format(x),) )
#         conv_op_tn.add(T1)
#
#     TL = qtn.Tensor( np.array([[[[1.,0.],[0.,0.]],[[0.,1.],[1.,0.]]],
#                                [[[0.,0.],[0.,1.]],[[0.,0.],[0.,0.]]]]),
#                      inds=(f'b{L-2}', out_id.format(L-1),in1_id.format(L-1),in2_id.format(L-1)),
#                      tags=(site_tag_id.format(L-1),) )
#     conv_op_tn.add(TL)
#     return conv_op_tn


# def convolve(mps1,mps2,compress=False,site_ind_id=None,site_tag_id=None,**compress_opts):
#     """ h(x) = convolve( f(x), g(x) )
#         convolution of the two MPS
#     """
#     ### This with probably assume binary ordering
#
#     if site_ind_id is None:  site_ind_id = mps1.site_ind_id
#     if site_tag_id is None:  site_tag_id = mps1.site_tag_id
#
#     convolve_op = get_convolve_op(mps1.L,site_tag_id=site_tag_id,out_id='out{}',
#                                          in1_id=mps1.site_ind_id,in2_id=mps2.site_ind_id)
#     convolve_op.add(mps1,mps2)
#     new_mps = convolve_op.contract()
#     new_mps.view_like(mps1, site_ind_id='out{}', site_tag_id=site_tag_id, inplace=True)
#     new_mps.site_ind_id = site_ind_id
#     return new_mps


def tn1D_from_dense(tensor: qtn.Tensor, ns: int, site_nlegs: Sequence[int], site_tag_id='I{}',
                    direction=0, split_opts=None) -> qtn.TensorNetwork:
    """ qtn.Tensor with correctly order inds --> MPX with nlegs for all ns sites
        site_nlegs:  can dictate where splitting
        site_inds:  for labeling tensors via site_tag_id (from left to right)
        inds are determined from tensor
    """
    tensor_list = []

    if split_opts is None:   split_opts = {}
    else:                    split_opts=split_opts.copy()
    split_opts['absorb'] = 'right'
    split_opts.setdefault('cutoff', CUTOFF)
    split_opts.setdefault('cutoff_mode', CUTOFF_MODE)

    inds_list = tensor.inds
    ind0 = 0

    norm = tensor.norm()
    if norm**2 < CUTOFF:
        for i in range(ns):
            ind_names = tuple(inds_list[ind0:ind0+site_nlegs[i]])
            ind_sizes = tuple([tensor.ind_size(ind_name) for ind_name in ind_names])
            ind0 += site_nlegs[i]

            if i==0:
                ztens = np.zeros(ind_sizes+(1,))
                tmp_inds = (f'_tmp_{i}',)
            elif i==ns-1:
                ztens = np.zeros(ind_sizes+(1,))
                tmp_inds = (f'_tmp_{i-1}',)
            else:
                ztens = np.zeros(ind_sizes+(1,1))
                tmp_inds = (f'_tmp_{i-1}',f'_tmp_{i}')

            tensor_list += [qtn.Tensor(ztens, inds=ind_names+tmp_inds, tags=(site_tag_id.format(i)))]
        norm = 1.

    else:
        tensor.modify(data=tensor.data*(1./norm))   # (1./norm) * tensor

        if direction == 0:
            sites_n = range(ns-1)
            indL = ns-1
            ind0 = 0
        else:
            sites_n = range(ns-1, 0, -1)
            indL = 0
            ind0 = len(inds_list)

        TM = tensor
        iso_inds = ()
        for i in sites_n:
            if direction == 0:
                iso_inds += inds_list[ind0:ind0+site_nlegs[i]]
                ind0 += site_nlegs[i]
            else:
                iso_inds += inds_list[ind0-site_nlegs[i]:ind0]
                ind0 -= site_nlegs[i]
            # print('TM', TM, i)
            # TL, TM = TM.split(left_inds=iso_inds,bond_ind=f'vb{i}',get='tensors',**compress_opts)
            TL, TM = TM.split(left_inds=iso_inds, get='tensors', **split_opts)
            TL.add_tag(site_tag_id.format(i))
            iso_inds = tuple(TL.bonds(TM))  # (f'vb{i}',)
            tensor_list += [TL]
        # print('TM', TM, indL)
        TM.add_tag(site_tag_id.format(indL))
        tensor_list += [TM]

    tn = qtn.TensorNetwork(tensor_list)
    tn.exponent += np.log10(norm)
    return tn


def mpx_from_dense(tensor: qtn.Tensor, ns: int, site_inds_list: Sequence[str], site_tag_id='I{}',
                   direction=0, return_mpx=True, split_opts=None):
    """ ndarray block with legs ordered by site --> MPX with nlegs for all ns sites
        site_inds:  for labeling tensors via site_tag_id (from left to right)
        inds are determined from tensor
    """
    tensor_list = []
    errs = []

    if split_opts is None:      split_opts = {}
    else:                       split_opts = split_opts.copy()
    split_opts['absorb'] = 'right'
    split_opts.setdefault('cutoff', CUTOFF)
    split_opts.setdefault('cutoff_mode', CUTOFF_MODE)

    if direction == 0:
        site_n = range(ns - 1)
        indL = ns-1
    else:    # direction == 1:
        site_n = range(ns - 1, 0, -1)
        indL = 0


    norm = tensor.norm()
    if norm**2 < CUTOFF:
        for i in range(ns):
            ind_names = tuple([ind_name.format(i) for ind_name in site_inds_list])
            ind_sizes = tuple([tensor.ind_size(ind_name) for ind_name in ind_names])

            if i == 0:
                ztens = np.zeros(ind_sizes + (1,))
                tmp_inds = (f'_tmp_{i}',)
            elif i == ns - 1:
                ztens = np.zeros(ind_sizes + (1,))
                tmp_inds = (f'_tmp_{i - 1}',)
            else:
                ztens = np.zeros(ind_sizes + (1, 1))
                tmp_inds = (f'_tmp_{i - 1}', f'_tmp_{i}')

            tensor_list += [qtn.Tensor(ztens, inds=ind_names + tmp_inds, tags=(site_tag_id.format(i)))]

        norm = 1.
    else:
        tensor = 1. / norm * tensor

        TM = tensor
        left_inds = []
        for i in site_n:
            left_inds += [ind_name.format(i) for ind_name in site_inds_list]
            TL, TM = TM.split(left_inds=left_inds, bond_ind=f'vb{i}', get='tensors', **split_opts)
            TL.add_tag(site_tag_id.format(i))
            left_inds = [f'vb{i}']
            tensor_list += [TL]
        TM.add_tag(site_tag_id.format(indL))
        tensor_list += [TM]

    tn = qtn.TensorNetwork(tensor_list)
    tn.exponent += np.log10(norm)
    if return_mpx:
        if len(site_inds_list) == 1:
            qtn.MatrixProductState.from_TN(tn, inplace=True, cyclic=False, L=ns,
                                           site_tag_id=site_tag_id,
                                           site_ind_id=site_inds_list[0])
        elif len(site_inds_list) == 2:
            qtn.MatrixProductOperator.from_TN(tn, inplace=True, cyclic=False, L=ns,
                                              site_tag_id=site_tag_id,
                                              upper_ind_id=site_inds_list[0],
                                              lower_ind_id=site_inds_list[1])
        return tn
    else:
        scale = norm ** (1. / ns)
        for i in range(ns):     tensor_list[i] *= scale
        return tensor_list


def append_mpx(mpx1,mpx2,inplace=False):
    """ append mpx2 to the end of mpx1. 
        generate bond of bond dimension 1 between last tensor of mpx1, first tensor of mpx2
        retag all tensors in mpx
        return mpx object
        mpx1 and mpx2 have to both be MatrixProductOperators or MatrixProductStates
    """
    L1, L2 = mpx1.nsites, mpx2.nsites

    if L2 == 0:  return mpx1

    # mpx2 = mpx2 if inplace else mpx2.copy()
    # mpx2.retag_sites(mpx1.site_tag_id)

    mpx2 = mpx2.retag({**{mpx2.site_tag_id.format(i): mpx1.site_tag_id.format(i+L1) \
                       for i in range(L2)}}, inplace=False)

    if isinstance(mpx2,qtn.MatrixProductState):
        mpx2 = mpx2.reindex({**{mpx2.site_ind_id.format(i): mpx1.site_ind_id.format(i+L1) \
                             for i in range(L2)}}, inplace=True)
    elif isinstance(mpx2,qtn.MatrixProductOperator):
        mpx2 = mpx2.reindex({**{mpx2.upper_ind_id.format(i): mpx1.upper_ind_id.format(i+L1) \
                            for i in range(L2)}}, inplace=True)
        mpx2 = mpx2.reindex({**{mpx2.lower_ind_id.format(i): mpx1.lower_ind_id.format(i+L1) \
                            for i in range(L2)}}, inplace=True)

    new = mpx1 if inplace else mpx1.copy()
    new.add_tensor_network(mpx2)   # checks for collisions
    new._L = L1+L2
    if L1 != 0:  new.new_bond(new.site_tag(L1-1), new.site_tag(L1))

    return new


def ones_mps(L,q,site_ind_id='i{}',site_tag_id='T{}'):
    """ make an MPS of 1's
    """
    if not (isinstance(q,tuple) or isinstance(q,list)):  q = (q,)*L
    ones = sum_tensornetwork(L=L, qs=q, inds=site_ind_id, tags=site_tag_id)
    ones.view_as(qtn.MatrixProductState, inplace=True, cyclic=False, L=L,
                 site_ind_id=site_ind_id, site_tag_id=site_tag_id)
    return ones


# def sum(tn,sum_tn=None):   ### previously L1norm
#     tn = tn.view_as(qtn.TensorNetwork)
#     if sum_tn is None:   sum_tn = sum_tensornetwork(like_tn=tn, L=tn.num_tensors)
#     tn.add(sum_tn)
#     return tn.contract() * 10**tn.exponent


def sum_tensornetwork(like_tn=None, scale=1., L=1, qs=None, inds=None, tags=None):
    """ tensor product of just one vectors (scaled by scale**(1./L))
    """
    if like_tn is not None:
        inds_dims_list = like_tn.outer_dims_inds()
        tensor_list = []
        i = 0
        for q, ind_name in inds_dims_list:
            tensor_list += [qtn.Tensor(np.ones(q),inds=(ind_name,),
                            tags=(like_tn.site_tag_id.format(i),))]
            i += 1
        sum_tn = qtn.TensorNetwork( tensor_list )
        for i in range(len(tensor_list)-1):
            tensL = sum_tn.select_tensors(like_tn.site_tag_id.format(i))[0]
            tensR = sum_tn.select_tensors(like_tn.site_tag_id.format(i+1))[0]
            tensL.new_bond(tensR)
    else:

        if isinstance(qs,int):    qs = [qs]*L

        if isinstance(tags,list):   tags_list = tags
        elif isinstance(tags,str):  tags_list = [tags.format(i) for i in range(L)]
        else:                       tags_list = [f'L1({i})' for i in range(L)]

        if like_tn is not None:     inds_list = like_tn.outer_inds()
        elif isinstance(inds,list): inds_list = inds
        elif isinstance(inds,str):  inds_list = [inds.format(i) for i in range(L)]
        else:                       inds_list = [f'tmp{(i)}' for i in range(L)]

        sum_tn = qtn.TensorNetwork( [qtn.Tensor(np.ones(qs[i]), inds=(inds_list[i],),
                                                tags=(tags_list[i],)) for i in range(L)] )
        for i in range(len(tags_list)-1):
            tensL = sum_tn.select_tensors(tags_list[i])[0]
            tensR = sum_tn.select_tensors(tags_list[i+1])[0]
            tensL.new_bond(tensR)

    if scale != 1.:   scalar_multiply(sum_tn,scale,inplace=True)
    return sum_tn

#
# def field_to_tensornetwork(field,field_name='',addstr='comp{}'):
#     field_tn = qtn.TensorNetwork([])
#     for i, field_data in field._components.items():
#         tmp = field_data.copy()
#         tmp.distribute_exponent()
#
#         # addstr_i = addstr.format(i)
#         # tmp.add_tag(addstr_i)
#         # tmp.reindex_all(field_data.site_ind_id+field_name+addstr_i,inplace=True)
#         field_tn.add( tmp.view_as(qtn.TensorNetwork, inplace=True) )
#
#     return field_tn
#
#
# def tensornetwork_to_field(field_tn,ref_field,field_name='',addstr='comp{}'):
#     field = ref_field.create_like()
#     for i in ref_field._components.keys():
#         compstr_i = f'comp{i}'
#         field_tn, tmp = field_tn.partition((compstr_i,),which='all')
#         tmp.view_like(ref_field[i], inplace=True)
#                       # site_ind_id=ref_field[i].site_ind_id+field_name+compstr_i,
#         # tmp.reindex_all(ref_field[i].site_ind_id,inplace=True)
#         tmp.exponent = field_tn.exponent   # partition doesn't keep the exponent
#         # field.components[i] = tmp # .copy()
#         field[i] = tmp # .copy()
#     return field
#
#
# def state_to_tensornetwork(state):
#     state_tn = qtn.TensorNetwork([])
#     for fname, field in state.fields.items():
#         field_tn = field_to_tensornetwork(field)
#         # field_tn.add_tag(fname[:3])
#         state_tn.add( field_tn )
#     # state_tn.equalize_norms(inplace=True)   ## introduces some error in going back and forth ~1e-15
#     return state_tn
#
#
# def tensornetwork_to_state(state_tn,ref_state):
#     new_fields = []
#     for fname in ref_state.field_names:
#         ref_field = ref_state.fields[fname]
#         state_tn, tmp = state_tn.partition((fname,),which='all')
#         # tmp.drop_tags(fname[:3])
#         field = tensornetwork_to_field(tmp,ref_field)
#         new_fields += [field.copy()]   ## why do we need this copy?
#     state = ref_state.create_like(*new_fields)
#     return state




