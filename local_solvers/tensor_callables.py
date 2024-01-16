from setup.defaults import *
import helper_quimb as helper
import local_solvers.helper_tn as helper_tn
from helper_dmrg import qtn_conjugate_gradient_descent_1site, qtn_conjugate_gradient_squared_1site


""" func arguments should be 
    (site_tensor, effective_operators, output_to_input_inds)
"""

def get_element_wise_func(func) -> Callable:

    def tens_func(tens: 'qtn.Tensor', eff_ops: Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']],
                  output_to_input_inds: dict[str, str]):
        tens = tens.copy()
        tens.modify(apply=lambda x: func(x))
        return tens

    return tens_func


def apply_effective_op(site_tens: 'qtn.Tensor', eff_ops: Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']],
                       output_to_input_inds: dict[str, str]):

    tens_list = []
    for eff_op in eff_ops:
        out_tens = qtn.tensor_contract(*eff_op.tensors, site_tens)
        out_tens.modify(apply=lambda x: x * 10 ** eff_op.exponent)
        tens_list += [out_tens]
    out = helper_tn.sum_tens(tens_list)
    out.reindex(output_to_input_inds, inplace=True)
    return out


def add_func(site_tens: 'qtn.Tensor', other_tens: Union['qtn.Tensor', list['qtn.Tensor']],
             output_to_input_inds: dict[str, str]) -> 'qtn.Tensor':

    if not isinstance(other_tens, list):
        other_tens = [other_tens]
    out = helper_tn.sum_tens([site_tens.copy(), *other_tens])
    # out.reindex(output_to_input_inds, inplace=True)
    return out


### for solving linear system
def get_cgd_func(is_H = False, **conv_kwargs):

    conv_kwargs.setdefault('max_iter', 500)
    conv_kwargs.setdefault('conv_tol', 1.0e-6)

    def solve_cgd(site_tens: 'qtn.Tensor', eff_ops: Sequence[Union['qtn.TensorNetwork', 'qtn.Tensor']],
                  output_to_input_inds: dict[str, str]):
        """ solve Ax=b
            b_eff: site_tens
            A_eff: sum(eff_ops)
            return x_eff
        """
        init_guess = site_tens.copy()
        if is_H:
            out, err = qtn_conjugate_gradient_squared_1site(eff_ops, site_tens, init_guess,
                                                            output_to_input_inds, **conv_kwargs)
        else:
            out, err = qtn_conjugate_gradient_descent_1site(eff_ops, site_tens, init_guess,
                                                            output_to_input_inds, **conv_kwargs)

        return out, err

    return solve_cgd


def get_lsq_func(**conv_kwargs):

    conv_kwargs.setdefault('max_iter', 500)
    conv_kwargs.setdefault('conv_tol', 1.0e-6)

    def solve_lsq(site_tens: 'qtn.Tensor', eff_ops: Sequence[Union['qtn.TensorNetwork', 'qtn.Tensor']],
                  output_to_input_inds: dict[str, str]):
        """ solve Ax=b
            b_eff: site_tens
            A_eff: sum(eff_ops)
            return x_eff
        """
        shape_ = site_tens.shape
        size_ = int(np.prod(shape_))

        bonds_o = output_to_input_inds.keys()
        bonds_i = [output_to_input_inds[bo] for bo in bonds_o]
        A_eff_tens = helper_tn.sum_eff_TNs(eff_ops, [*bonds_o, *bonds_i])
        explicit_contribution = site_tens.transpose(*bonds_i)
        # print('A eff tens', A_eff_tens, explicit_contribution)

        Amat = A_eff_tens.data.reshape(size_, size_)
        bvec = explicit_contribution.data.reshape(size_)

        out_data = np.linalg.solve(Amat, bvec)

        out = qtn.Tensor(out_data.reshape(shape_), inds=bonds_i)
        return out

    return solve_lsq