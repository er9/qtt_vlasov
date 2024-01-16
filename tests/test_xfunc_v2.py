import sys
sys.path.append('../')

from setup.configs import *
# from gate import Gate
# from mps import MatrixProductState, MatrixProductOperator
import quimb.tensor as qtn
import helper_quimb as helper
from axis import Axis
from grid1D import Grid1D
from gridTN_1D import GridTN1D
# from local_solvers_old.local_evaluator import local_dmrg_evaluator
from local_solvers_old.local_dmrg_evaluator_v2 import local_dmrg_evaluator
from local_solvers_old.local_cross_evaluator import local_cross_evaluator


L = 10
Ds = [4, 8, 16, 24, 32, None]
legs = []
direction = 1
nsites = 1

x = np.linspace(0,2*np.pi,2**L, endpoint=False)
dx = 2*np.pi/(2**L)
print('dx', dx, x[1] - x[0])
fx = lambda x: 1 + np.sin(5*x)*5.0 * np.exp(-x**2/6) + np.cos(10*x)

x_scale = 0.1

data = fx(x)
np.random.seed(0)
data = data + np.random.rand(len(data))
data = data / np.linalg.norm(data)
print('data_tens', np.linalg.norm(data))

ax_x = Axis(L)
grid_X = Grid1D('GX', [ax_x])


# data_tens = qtn.Tensor( data.reshape((2,)*L), inds=tuple([f'i{i}' for i in range(L)]) )
# init_mps = MatrixProductState.from_vec(data_tens, L, 'i{}', split_opts={'max_bond': D})
init_mps = GridTN1D.from_dense_state(data, grid_X)
init_mps.data.exponent += np.log10(x_scale)
data = data * x_scale
print('init mps exp', init_mps.exponent)
print('init mps max bond', init_mps.max_bond())

plt.figure()
plt.plot(init_mps.get_data())
plt.plot(data)
plt.show()

dmrg_errs = []
cross_errs = []
comp_errs = []

### test 1:  return self in canonical form
do_test_1 = False
if do_test_1:
    legs += ['scale f']
    errs, errs_2, errs_3 = [], [], []
    func = lambda x: x * 1.0
    for D in Ds:
        func_gtn = local_dmrg_evaluator(([func], init_mps.copy()), direction=direction,
                                        nsites=nsites, max_bond=D)

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - func(data)) / np.linalg.norm(func(data))
        errs += [err]
        print('dmrg err', err)

        ## cross
        func_gtn = local_cross_evaluator(([func], init_mps.copy()), direction=direction,
                                         nsites=nsites, max_bond=D)
        print('cross soln', func_gtn.data)
        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - func(data)) / np.linalg.norm(func(data))
        errs_3 += [err]
        print('cross err', err)

        # comp_mps_2 = helper.compress_rdm(init_mps.data.copy(), compress_opts={'max_bond': D},
        #                                  direction=-1)
        # comp_gtn = grid_X.make_gridTN(comp_mps_2)
        # comp_data = comp_gtn.get_data()
        # err = np.linalg.norm(func(comp_data) - func(data)) / np.linalg.norm(func(data))
        # print('err rdm', err)

        # comp_mps = init_mps.copy()
        comp_mps = GridTN1D.from_dense_state(func(data), grid_X, split_opts={'max_bond': D})
        print('comp_mps', comp_mps.max_bond())
        comp_data = comp_mps.get_data()
        err = np.linalg.norm(comp_data - func(data)) / np.linalg.norm(func(data))
        print('comp err', err)
        errs_2 += [err]

        if D is not None and D < 16:
            plt.figure()
            plt.plot(np.real(func_data), label='func dmrg')
            plt.plot(np.real(func(data)), ':', label='exact')
            plt.plot(np.imag(func_data), '--')
            plt.plot(np.imag(func(data)), ':')
            plt.legend()
            plt.title(f'D = {D}')
            plt.show()

    dmrg_errs += [errs]
    comp_errs += [errs_2]
    cross_errs += [errs_3]


### add init_mps + constant
do_test_2 = False
if do_test_2:
    legs += ['add const.']
    errs, errs_2, errs_3 = [], [], []
    y_mps = GridTN1D.get_ones_mps(grid_X)
    y_mps.scalar_multiply(0.5, inplace=True)

    for D in Ds:
        # func_gtn = local_dmrg_evaluator(([], y_mps.copy()), ([], init_mps.copy()), direction=1)
        func_gtn = local_dmrg_evaluator(([], init_mps.copy()), ([], y_mps.copy()), direction=direction,
                                        max_bond=D, nsites=nsites)
        # func_gtn = local_dmrg_evaluator(([], y_mps.copy()), ([], init_mps.copy()), direction=direction,
        #                                 max_bond=D, nsites=nsites)

        print('func gtn', func_gtn.data)
        func_gtn.check_orthog()

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - (data + 0.5)) / np.linalg.norm(data + 0.5)
        print('err', err)
        errs += [err]

        ## cross
        func_gtn = local_cross_evaluator(([], init_mps.copy()), ([], y_mps.copy()), direction=direction,
                                          max_bond=D, nsites=nsites)
        print('max bond', func_gtn.data)
        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - (data + 0.5)) / np.linalg.norm(data + 0.5)
        print('err', err)
        errs_3 += [err]

        ## svd
        comp_gtn = GridTN1D.from_dense_state(data + 0.5, grid_X, split_opts={'max_bond': D})
        comp_data = comp_gtn.get_data()
        err = np.linalg.norm(comp_data - (data + 0.5)) / np.linalg.norm(data + 0.5)
        print('comp err', err)
        errs_2 += [err]

        if D is not None and D < 16:
            plt.figure()
            plt.plot(np.real(func_data), label='dmrg')
            # plt.plot(np.real(func(data * x_scale)), ':', label='exact')
            plt.plot(np.real(data + 0.5), ':', label='exact')
            plt.plot(np.imag(func_data), '--')
            plt.plot(np.imag(data + 0.5), ':')
            plt.legend()
            plt.title(f'D = {D}')
            plt.show()

    dmrg_errs += [errs]
    comp_errs += [errs_2]
    cross_errs += [errs_3]


### add init_mps + other fct
do_test_3 = False
if do_test_3:
    legs += ['add fct']
    fx = lambda x:  x**2 * 4e-4
    y_mps = GridTN1D.from_dense_state(fx(x), grid_X)

    errs, errs_2, errs_3 = [], [], []
    for D in Ds:
        func_gtn = local_dmrg_evaluator(([], init_mps.copy()), ([], y_mps.copy()), direction=direction,
                                        max_bond=D, nsites=nsites)

        print('func gtn', func_gtn.data)
        func_gtn.check_orthog()

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - (data + fx(x))) / np.linalg.norm(data + fx(x))
        print('err', err)
        errs += [err]

        func_gtn = local_cross_evaluator(([], init_mps.copy()), ([], y_mps.copy()), direction=direction,
                                          max_bond=D, nsites=nsites)
        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - (data + fx(x))) / np.linalg.norm(data + fx(x))
        print('err', err)
        errs_3 += [err]

        comp_gtn = GridTN1D.from_dense_state(data + fx(x), grid_X, split_opts={'max_bond': D})
        comp_data = comp_gtn.get_data()
        err = np.linalg.norm(comp_data - (data + fx(x))) / np.linalg.norm(data + fx(x))
        print('comp err', err)
        errs_2 += [err]

        # exit()

        if D is not None and D < 16:
            plt.figure()
            plt.plot(np.real(func_data), label='dmrg')
            # plt.plot(np.real(func(data * x_scale)), ':', label='exact')
            plt.plot(np.real(data + fx(x)), ':', label='exact')
            plt.plot(np.imag(func_data), '--')
            plt.plot(np.imag(data + fx(x)), ':')
            plt.legend()
            plt.title(f'D = {D}')
            plt.show()

    dmrg_errs += [errs]
    comp_errs += [errs_2]
    cross_errs += [errs_3]

### operator * init_mps
do_test_4 = False
if do_test_4:
    legs += ['mat-vec mult']
    op = grid_X.get_firstderivative_mpo(ax_x)
    op2 = ax_x.get_shift_mpo(2)
    helper.add_MPO(op.data, op2, inplace=True)
    # print('op exponent', op.exponent)
    # op.data.distribute_exponent()

    glob_apply = init_mps.apply(op)
    glob_data = glob_apply.get_data()

    errs, errs_2, errs_3 = [], [], []
    for D in Ds:
        ## dmrg
        loc_apply = local_dmrg_evaluator(([op.copy()], init_mps.copy()), direction=direction,
                                         max_bond=D, nsites=nsites)
        loc_data = loc_apply.get_data()
        err = np.linalg.norm(glob_data - loc_data) / np.linalg.norm(glob_data)
        print('err', err)
        errs += [err]

        ## cross
        loc_apply = local_cross_evaluator(([op.copy()], init_mps.copy()), direction=direction,
                                          max_bond=D, nsites=nsites)
        loc_data = loc_apply.get_data()
        err = np.linalg.norm(glob_data - loc_data) / np.linalg.norm(glob_data)
        print('err', err)
        errs_3 += [err]

        ## svd
        comp_gtn = glob_apply.compress(compress_opts={'max_bond': D}, inplace=False)
        comp_data = comp_gtn.get_data()
        err = np.linalg.norm(comp_data - glob_data) / np.linalg.norm(glob_data)
        print('comp err', err)
        errs_2 += [err]

        if D is not None and D < 16:
            plt.figure()
            plt.plot(np.real(loc_data), label='dmrg')
            plt.plot(np.real(glob_data), ':', label='exact')
            plt.plot(np.imag(loc_data), '--')
            plt.plot(np.imag(glob_data), ':')
            plt.legend()
            plt.title(f'D = {D}')
            plt.show()

    dmrg_errs += [errs]
    comp_errs += [errs_2]
    cross_errs += [errs_3]


### test 5:  return self in canonical form  (but this method can't include 0 power term)
do_test_5 = True
if do_test_5:
    legs += ['x^3 eval']
    func = lambda x: 0.05 * x ** 3  # + 50 * x ** 3

    errs, errs_2, errs_3 = [], [], []
    for D in Ds:
        ## dmrg
        func_gtn = local_dmrg_evaluator(([[0, 0, 0.05]], init_mps.copy()), direction=direction,
                                        max_bond = D, nsites=nsites)

        func_gtn.check_orthog()
        print('max bond', func_gtn.data)

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - func(data)) / np.linalg.norm(func(data))
        print('err', err)
        errs += [err]

        ## cross
        func_gtn = local_cross_evaluator(([func], init_mps.copy()), direction=direction,
                                         max_bond=D, nsites=nsites)
        print('max bond', func_gtn.data)
        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - func(data)) / np.linalg.norm(func(data))
        print('err', err)
        errs_3 += [err]

        ## svd
        comp_gtn = GridTN1D.from_dense_state(func(data), grid_X, split_opts={'max_bond': D})
        comp_data = comp_gtn.get_data()
        err = np.linalg.norm(comp_data - func(data)) / np.linalg.norm(func(data))
        print('comp err', err)
        errs_2 += [err]

        if D is not None and D < 16:
            plt.figure()
            plt.plot(np.real(func_data), label='func dmrg')
            plt.plot(np.real(func(data)), ':', label='exact')
            plt.plot(np.imag(func_data), '--')
            plt.plot(np.imag(func(data)), ':')
            plt.legend()
            plt.title(f'D = {D}')
            plt.show()

    dmrg_errs += [errs]
    comp_errs += [errs_2]
    cross_errs += [errs_3]

Ds[-1] = 100
i = 0
for errs1, errs2, errs3 in zip(dmrg_errs, cross_errs, comp_errs):
    plt.figure()
    plt.semilogy(Ds, errs1, 'o:', label='dmrg')
    plt.semilogy(Ds, errs2, 'x--', label='cross')
    plt.semilogy(Ds, errs3, 'v-', label='svd')
    plt.legend()
    plt.xlabel('D')
    plt.ylabel('err')
    plt.title(legs[i])
    i += 1
plt.show()