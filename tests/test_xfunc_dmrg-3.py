import sys

sys.path.append('../')

from setup_.configs import *
# from gate import Gate
# from mps import MatrixProductState, MatrixProductOperator
import quimb.tensor as qtn
import helper_quimb as helper
from axis import Axis
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from local_solvers.local_dmrg_eval import local_dmrg_evaluator
from local_solvers.terms_3 import Term_DMRG
import local_solvers.tensor_callables as tens_fcts


L = 5
D = 16
direction = 1
nsites = 1

x = np.linspace(0,2*np.pi,2**L, endpoint=False)
dx = 2*np.pi/(2**L)
print('dx', dx, x[1] - x[0])
fx = lambda x: np.cos(x / 2) / (x+0.1) + 1 + np.sin(5*x)*5.0 * np.exp(-x**2/6)
# fx = lambda x:  x**2

x_scale = 0.1

data = fx(x)
data = data / np.linalg.norm(data)
print('data_tens', np.linalg.norm(data))

ax_x = Axis(L)
grid_X = Grid1D('GX', [ax_x])


# data_tens = qtn.Tensor( data.reshape((2,)*L), inds=tuple([f'i{i}' for i in range(L)]) )
# init_mps = MatrixProductState.from_vec(data_tens, L, 'i{}', split_opts={'max_bond': D})
init_mps = GridTN1D.from_dense_state(data, grid_X)
init_mps.data.exponent += np.log10(x_scale)
print('init mps exp', init_mps.exponent)
print('init mps max bond', init_mps.max_bond())

plt.figure()
plt.plot(init_mps.get_data())
plt.plot(data * x_scale)
plt.show()

### test 1:  return self in canonical form
do_test_1 = True
if do_test_1:
    func = lambda x: x * 0.5
    # func = lambda x: x ** 2
    eval_func = tens_fcts.get_element_wise_func(func)
    # init_mps.data.distribute_exponent()
    term1 = Term_DMRG(init_mps.data.copy(), mps_coeff=0.5)
    func_mps = local_dmrg_evaluator((term1,), direction=direction, nsites=nsites)
    func_gtn = grid_X.make_gridTN(func_mps)

    print('func gtn', func_gtn.data)
    func_gtn.check_orthog()

    # func_mps = init_mps.func_cross(func)
    # print('func mps', func_mps.max_bond())
    func_data = func_gtn.get_data()

    print('err', np.linalg.norm(func_data - func(data * x_scale)))

    plt.figure()
    plt.plot(np.real(func_data), label='func dmrg')
    plt.plot(np.real(func(data * x_scale)), ':', label='exact')
    plt.plot(np.imag(func_data), '--')
    plt.plot(np.imag(func(data * x_scale)), ':')
    plt.legend()
    plt.show()


### add init_mps + constant
do_test_2 = True
if do_test_2:
    y_mps = GridTN1D.get_ones_mps(grid_X)
    y_mps.scalar_multiply(0.5, inplace=True)

    term1 = Term_DMRG(init_mps.data.copy())
    term2 = Term_DMRG(y_mps.data.copy())

    func_mps = local_dmrg_evaluator([term1, term2], direction=direction, nsites=nsites)
    func_gtn = grid_X.make_gridTN(func_mps)

    print('func gtn', func_gtn.data)
    func_gtn.check_orthog()

    func_data = func_gtn.get_data()
    print('err', np.linalg.norm(func_data - (data * x_scale + 0.5)))

    # exit()

    plt.figure()
    plt.plot(np.real(func_data), label='dmrg')
    # plt.plot(np.real(func(data * x_scale)), ':', label='exact')
    plt.plot(np.real(data * x_scale + 0.5), ':', label='exact')
    plt.plot(np.imag(func_data), '--')
    plt.plot(np.imag(data * x_scale + 0.5), ':')
    plt.legend()
    plt.show()



### add init_mps + other fct
do_test_3 = True
if do_test_3:
    fx = lambda x:  x**2 * 4e-4
    y_mps = GridTN1D.from_dense_state(fx(x), grid_X)

    term1 = Term_DMRG(init_mps.data.copy())
    term2 = Term_DMRG(y_mps.data.copy())

    func_mps = local_dmrg_evaluator([term1, term2], nsites=nsites, direction=direction)
    func_gtn = grid_X.make_gridTN(func_mps)

    print('func gtn', func_gtn.data)
    func_gtn.check_orthog()

    func_data = func_gtn.get_data()
    print('err', np.linalg.norm(func_data - (data * x_scale + fx(x))))

    # exit()

    plt.figure()
    plt.plot(np.real(func_data), label='dmrg')
    # plt.plot(np.real(func(data * x_scale)), ':', label='exact')
    plt.plot(np.real(data * x_scale + fx(x)), ':', label='exact')
    plt.plot(np.imag(func_data), '--')
    plt.plot(np.imag(data * x_scale + fx(x)), ':')
    plt.legend()
    plt.show()



### operator * init_mps
do_test_4 = True
if do_test_4:
    op = grid_X.get_firstderivative_mpo(ax_x)
    op2 = ax_x.get_shift_mpo(2)
    helper.add_MPO(op.data, op2, inplace=True)
    print('op exponent', op.exponent)
    # op.data.distribute_exponent()
    # init_mps.data.distribute_exponent()

    glob_apply = init_mps.apply(op)
    glob_data = glob_apply.get_data()

    term1 = Term_DMRG(init_mps.data.copy(), operators=[op.data.copy()],
                      num_tiers=2) #, eval_func=tens_fcts.apply_effective_op)
    loc_mps = local_dmrg_evaluator([term1], direction=1, nsites=nsites)
    loc_gtn = grid_X.make_gridTN(loc_mps)

    loc_data = loc_gtn.get_data()

    print('err', np.linalg.norm(glob_data - loc_data)/np.linalg.norm(glob_data))

    plt.figure()
    plt.plot(np.real(loc_data), label='dmrg')
    plt.plot(np.real(glob_data), ':', label='exact')
    plt.plot(np.imag(loc_data), '--')
    plt.plot(np.imag(glob_data), ':')
    plt.legend()
    plt.show()


do_test_5 = True
if do_test_5:
    power = 3

    # func = lambda x: 0.05 * x + 50 * x ** 3
    func = lambda x: x ** power
    print('init mps exp', init_mps.exponent)

    def power_func(site_tens: 'qtn.Tensor', eff_ops: Sequence['qtn.TensorNetwork'],
                   output_to_input_inds: dict[str, str]) -> 'qtn.Tensor':

        out = tens_fcts.apply_effective_op(site_tens, eff_ops, output_to_input_inds)
        for it in range(power - 2):
            print('power func', it)
            out = tens_fcts.apply_effective_op(out, eff_ops, output_to_input_inds)
        return out

    init_mps.data.distribute_exponent()
    init_data = init_mps.data.copy()
    term1 = Term_DMRG(init_data, mps_power=power) #eval_func=power_func)

    # func_gtn = local_dmrg_evaluator(([[0.05, 0, 50]], init_mps.copy()), direction=-1, nsites=nsites)
    func_mps = local_dmrg_evaluator([term1], direction=-1, nsites=nsites)
    func_gtn = grid_X.make_gridTN(func_mps)

    print('func gtn', func_gtn.data)
    func_gtn.check_orthog()

    # func_mps = init_mps.func_cross(func)
    # print('func mps', func_mps.max_bond())
    func_data = func_gtn.get_data()

    print('err', np.linalg.norm(func_data - func(data * x_scale))/np.linalg.norm(func(data * x_scale)))

    plt.figure()
    plt.plot(np.real(func_data), label='func dmrg')
    plt.plot(np.real(func(data * x_scale)), ':', label='exact')
    plt.plot(np.imag(func_data), '--')
    plt.plot(np.imag(func(data * x_scale)), ':')
    plt.legend()
    plt.show()


do_test_6 = True
if do_test_6:
    ### compute 0.5 d/dx f(x)**3

    power = 3
    coeff = 0.5
    # op = grid_X.get_iden_mpo()
    op = grid_X.get_firstderivative_mpo(ax_x)
    op.data.distribute_exponent()

    print('init mps exp', init_mps.exponent, op.exponent)

    init_data = init_mps.get_data()
    init_mps3 = grid_X.make_gridTN(init_data**power)
    glob_apply = init_mps3.apply(op)
    glob_data = glob_apply.get_data() * coeff

    # init_mps.data.distribute_exponent()
    init_data = init_mps.data.copy()
    term1 = Term_DMRG(init_data, operators=[op.data], mps_power=power, mps_coeff=coeff)

    # func_gtn = local_dmrg_evaluator(([[0.05, 0, 50]], init_mps.copy()), direction=-1, nsites=nsites)
    func_mps = local_dmrg_evaluator([term1], direction=-1, nsites=nsites)
    # func_gtn = grid_X.make_gridTN(term1.bra)
    func_gtn = grid_X.make_gridTN(func_mps)
    # print('check orthog bra', helper.check_orthog(term1.bra))
    # print('check orthog ket', helper.check_orthog(term1.ket))

    print('func gtn', func_gtn.data)
    func_gtn.check_orthog()

    # func_mps = init_mps.func_cross(func)
    # print('func mps', func_mps.max_bond())
    func_data = func_gtn.get_data()

    print('err', np.linalg.norm(func_data - glob_data)/np.linalg.norm(glob_data))

    plt.figure()
    # plt.plot(np.real(func_data), label='func dmrg')
    # plt.plot(np.real(glob_data), ':', label='exact')
    plt.plot(np.real(func_data), label='dmrg')
    plt.plot(np.real(glob_data), ':', label='exact')
    plt.plot(np.imag(func_data), '--')
    # plt.plot(np.imag(glob_data), ':')
    plt.legend()
    plt.show()