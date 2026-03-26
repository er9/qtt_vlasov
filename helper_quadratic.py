import quimb

from setup_.defaults import *
import time
import quimb.tensor as qtn
import helper_quimb as helper

def make_quadratic_op(elem_mult: 'MatrixProductTensor', in1_mpo: 'qtn.MatrixProductOperator',
                      in2_mpo: 'qtn.MatrixProductOperator',
                      out_mpo: Optional['qtn.MatrixProductOperator'] = None,
                      in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})', site_tag_id='T({})'
                      ) -> 'MatrixProductTensor':

    print('elem mult exp', elem_mult.exponent)
    exponent = elem_mult.exponent # 0.0      ## TT is scaled by 10^exponent
    ## copying + renaming tensor indices so that they are all distinct
    if in1_mpo is not None:
        in1_mpo = in1_mpo.copy()
        in1_mpo.mangle_inner_()
        in1_mpo.upper_ind_id = in1_mpo.upper_ind_id + "_1"
        in1_mpo.lower_ind_id = in1_ind_id  ## input 1 index
        exponent += in1_mpo.exponent
        print('in1 exp', in1_mpo.exponent)
    if in2_mpo is not None:
        in2_mpo = in2_mpo.copy()
        in2_mpo.mangle_inner_()
        in2_mpo.upper_ind_id = in2_mpo.upper_ind_id + "_2"
        in2_mpo.lower_ind_id = in2_ind_id  ## input 2 index
        exponent += in2_mpo.exponent
        print('in2 exp', in2_mpo.exponent)
    if out_mpo is not None:
        out_mpo = out_mpo.copy()
        out_mpo.mangle_inner_()
        out_mpo.upper_ind_id = out_ind_id  ## output index
        out_mpo.lower_ind_id = out_mpo.lower_ind_id + "_o"
        exponent += out_mpo.exponent
        print('out exp', out_mpo.exponent)
    print('exponent', exponent)

    in1_tmp_id = in1_mpo.upper_ind_id if in1_mpo is not None else in1_ind_id
    in2_tmp_id = in2_mpo.upper_ind_id if in2_mpo is not None else in2_ind_id
    out_tmp_id = out_mpo.upper_ind_id if out_mpo is not None else out_ind_id
    # elem_mult = grid.get_elemental_multiply_tn(in1_ind_id=in1_tmp_id, in2_ind_id=in2_tmp_id, out_ind_id=out_tmp_id )
    elem_mult.upper_ind_id = out_tmp_id
    elem_mult.lower_ind_id = in1_tmp_id
    elem_mult.reindex_extra_inds(0, in2_tmp_id)

    ## contract MPOs together
    L = elem_mult.L
    out_tensors = []
    prev_tensor = []
    left_inds = []
    for i in range(L):
        xtensors = [elem_mult[i]] + prev_tensor
        if in1_mpo is not None:
            xtensors += [in1_mpo[i]]
        if in2_mpo is not None:
            xtensors += [in2_mpo[i]]
        if out_mpo is not None:
            xtensors += [out_mpo[i]]

        tens = qtn.tensor_contract(*xtensors)

        if i == L -1:
            out_tensors += [tens]
            tens.drop_tags()
            tens.add_tag(site_tag_id.format(i))
        else:
            left_inds += [in1_ind_id.format(i), in2_ind_id.format(i), out_ind_id.format(i)]
            q, r = qtn.tensor_split(tens, left_inds=left_inds, absorb='right', cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE,)
            r.drop_tags()
            q.drop_tags()
            q.add_tag(site_tag_id.format(i))

            left_inds = [next(iter(q.bonds(r)))]
            out_tensors += [q]
            prev_tensor = [r]


    # ### tmp ###
    # out_tensors_tmp = []
    # for i in range(L):
    #     sum_elem = elem_mult[i].sum_reduce(out_tmp_id.format(i))
    #     print('sum elem', sum_elem.data)
    #     xtensors = [sum_elem] + prev_tensor
    #     if in1_mpo is not None:
    #         xtensors += [in1_mpo[i]]
    #     if in2_mpo is not None:
    #         xtensors += [in2_mpo[i]]
    #     if out_mpo is not None:
    #         xtensors += [out_mpo[i]]
    #
    #     print('x tensors', xtensors)
    #
    #     tens = qtn.tensor_contract(*xtensors)
    #
    #     if i == L -1:
    #         out_tensors_tmp += [tens]
    #         tens.drop_tags()
    #         tens.add_tag(site_tag_id.format(i))
    #     else:
    #         left_inds += [in1_ind_id.format(i), in2_ind_id.format(i)]
    #         q, r = qtn.tensor_split(tens, left_inds=left_inds, absorb='right', cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE,)
    #         r.drop_tags()
    #         q.drop_tags()
    #         q.add_tag(site_tag_id.format(i))
    #
    #         left_inds = [next(iter(q.bonds(r)))]
    #         out_tensors_tmp += [q]
    #         prev_tensor = [r]
    #
    # op_tn = qtn.TensorNetwork(out_tensors)
    # op_tn.exponent = exponent
    #
    # op_tn = op_tn.view_as(qtn.MatrixProductOperator, inplace=True, L=L, cyclic=False, site_tag_id=site_tag_id,
    #                                 upper_ind_id=in2_ind_id, lower_ind_id=in1_ind_id,)
    # helper.compress(op_tn, compress_opts={'form': 'right', 'cutoff': CUTOFF}, canonize=False)
    # print('sum red op tn', helper.norm(op_tn), op_tn.exponent)
    #
    # if in1_mpo and in2_mpo is not None:
    #     tmp = helper.apply(helper.mpo_conj_transpose(in2_mpo), in1_mpo)
    #     print('tmp', helper.norm(tmp), tmp.exponent)


    op_tn = qtn.TensorNetwork(out_tensors)
    op_tn.exponent = exponent

    op_tn = op_tn.view_as(MatrixProductTensor, inplace=True, L=L, cyclic=False, site_tag_id=site_tag_id,
                                    upper_ind_id=out_ind_id, lower_ind_id=in1_ind_id, extra_ind_ids=(in2_ind_id,))
    helper.compress(op_tn, compress_opts={'form': 'right', 'cutoff': CUTOFF}, canonize=False)
    return op_tn


def apply_quadratic_op(quadratic_mpo: 'MatrixProductTensor', mps1: 'qtn.MatrixProductState',
                       mps2: 'qtn.MatrixProductState', site_ind_id='i({})', site_tag_id='T({})',
                       compress_opts: Optional[dict]=None) -> Optional['qtn.MatrixProductState']:

    L = mps1.L

    quadratic_mpo: MatrixProductTensor = quadratic_mpo.copy()
    mps1 = mps1.copy() if mps1 is not None else None
    mps2 = mps2.copy() if mps2 is not None else None
    if mps1 is None or mps2 is None:
        ## one of the inputs is zero, so output is zero
        return None

    mps1.mangle_inner_()
    mps2.mangle_inner_()
    quadratic_mpo.mangle_inner_()

    exponent = quadratic_mpo.exponent + mps1.exponent + mps2.exponent

    mps1.site_ind_id = quadratic_mpo.lower_ind_id
    mps2.site_ind_id = quadratic_mpo.extra_ind_ids[0]
    out_ind_id = quadratic_mpo.upper_ind_id

    ## contract MPOs together
    out_tensors = []
    prev_tensor = []
    left_inds = []

    for i in range(L):
        xtensors = prev_tensor + [quadratic_mpo[i], mps1[i], mps2[i]]
        tens = qtn.tensor_contract(*xtensors)
        # print('tens', tens)

        if i == L - 1:
            out_tensors += [tens]
        else:
            left_inds += [out_ind_id.format(i)]
            q, r = qtn.tensor_split(tens, left_inds=left_inds, absorb='right', cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE, )
            r.drop_tags()
            q.drop_tags()
            q.add_tag(site_tag_id.format(i))

            left_inds = [next(iter(q.bonds(r)))]
            out_tensors += [q]
            prev_tensor = [r]


    out = qtn.TensorNetwork(out_tensors)
    out.exponent = exponent

    out = out.view_as(qtn.MatrixProductState, inplace=True, L=L, cyclic=False, site_tag_id=site_tag_id,
                      site_ind_id=out_ind_id,)

    if compress_opts is None:
        compress_opts = {}
    helper.compress(out, compress_opts={'form': 'right', **compress_opts}, canonize=False)
    out.site_ind_id = site_ind_id

    return out


def add_MPT_list(mpts: list[MatrixProductTensor], compress_opts: Optional[dict] = None):

    ref_mpt = mpts[0]
    exponent = ref_mpt.exponent

    ## make exponents across all MPTs the same
    for i in range(1, len(mpts)):
        mpt = mpts[i].copy()
        scale = 10**(mpt.exponent - exponent)
        mpt[0].modify(apply=lambda x: x * scale)    ## scale MPT by constant factor
        mpt.exponent = exponent

        ## relabel indices
        mpt.upper_ind_id = ref_mpt.upper_ind_id
        mpt.lower_ind_id = ref_mpt.lower_ind_id
        for ix, new_ind in enumerate(ref_mpt.extra_ind_ids):
            mpt.reindex_extra_inds(ix, new_ind)
        helper.match_inner_inds(mpt, ref_mpt, inplace=True)

        mpts[i] = mpt

    ## perform addition
    sum_inds_id = [ref_mpt.upper_ind_id, ref_mpt.lower_ind_id, *ref_mpt.extra_ind_ids]
    tensors = []
    for i in range(ref_mpt.L):
        sum_inds = [sum_ind.format(i) for sum_ind in sum_inds_id]

        prev_tens = ref_mpt[i]
        for mpt in mpts[1:]:
            prev_tens = qtn.tensor_direct_product(prev_tens, mpt[i], sum_inds=sum_inds)

        tensors += [prev_tens]

    ## construct new MPT
    op_tn = qtn.TensorNetwork(tensors)
    op_tn.exponent = exponent
    op_tn = op_tn.view_as(MatrixProductTensor, inplace=True, L=ref_mpt.L, cyclic=False, site_tag_id=ref_mpt.site_tag_id,
                          upper_ind_id=ref_mpt.upper_ind_id, lower_ind_id=ref_mpt.lower_ind_id,
                          extra_ind_ids=ref_mpt.extra_ind_ids)
    compress_opts = {} if compress_opts is None else compress_opts
    helper.compress(op_tn, compress_opts={'form': 'right', **compress_opts}, canonize=False)

    return op_tn
