import sys

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2.
import matplotlib.pyplot as plt

sys.path.append('../')

from setup_.configs import *
# from gate import Gate
# from mps import MatrixProductState, MatrixProductOperator
import quimb.tensor as qtn
import helper_quimb
from axis import Axis
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from local_solvers.local_dmrg_eval_orthog import local_orthogonal_compress, local_orthogonal_compress_2
from local_solvers.terms_3 import Term_DMRG
import local_solvers.tensor_callables as tens_fcts


L = 6
D = 16
direction = 1
nsites = 1

x = np.linspace(0,2*np.pi,2**L, endpoint=False)
dx = 2*np.pi/(2**L)
print('dx', dx, x[1] - x[0])

fx = lambda x: np.cos(x / 2) / (x+0.1) + 1 + np.sin(5*x) + np.sqrt(5*x)*5.0 * np.exp(-x**2/6)
fx = lambda x: np.cos(x / 2) / (x+0.1) + 0.5 * np.cos(20*x + 1.3) + np.sin(5*x) + np.sqrt(5*x)*5.0 * np.exp(-x**2/6) + 1.j * x
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
# init_mps.data.distribute_exponent()
print('init mps exp', init_mps.exponent)
print('init mps max bond', init_mps.max_bond())
data *= x_scale

plt.figure()
plt.plot(init_mps.get_data())
plt.plot(data)
plt.show()

### test 1:  return self in canonical form
do_test_1 = True
if do_test_1:
    D = 4

    orthog_state = grid_X.get_ones_mps()
    orthog_state.scalar_multiply(1./orthog_state.frobenius_norm(), inplace=True)
    ### projection of init mps onto orthog_state
    proj_val = init_mps.ovlp(orthog_state)
    # proj_val -= 1.e-2 * np.sign(proj_val)
    # # proj_val = np.conj(proj_val)    ## <b|x> -> <x|b>
    print('init proj val', proj_val)


    # ########################
    # ### naive compression
    comp_gtn = init_mps.compress(compress_opts={'max_bond': D}, inplace=False)
    print('compress naive')
    print('ranks', init_mps.max_bond(), comp_gtn.max_bond())
    print('compression err', np.linalg.norm(data - comp_gtn.get_data()))
    print('orthog err', orthog_state.ovlp(comp_gtn) - proj_val)
    print()

    ########################
    ### projection and remainder    (I - proj * |b><b|) x
    ### <b|(I - |b><b|/|b|)|x> = <b|x> - <b|x> = 0
    ### |x> = (I - |b><b|) |x> + proj |b> = |x> - |b> <b|x> + proj |b> so proj = <x|b>
    remainder = init_mps.add(orthog_state.scalar_multiply(-(proj_val)))
    tot_gtn_0 = remainder.add(orthog_state.scalar_multiply(proj_val))

    tmp = remainder.compress(inplace=False)
    print('compressed remainder max rank', remainder.max_bond(), tmp.max_bond())
    print('mod overlap', remainder.ovlp(orthog_state), init_mps.ovlp(orthog_state))

    #######################
    ### naive compression of remainder
    comp_remainder_naive = remainder.compress(compress_opts={'max_bond': D - 1})
    tot_gtn_3 = comp_remainder_naive.add(orthog_state.scalar_multiply(proj_val))
    tot_gtn_3.compress(inplace=True)

    print('naive compress remainder')
    print('ranks', remainder.max_bond(), comp_remainder_naive.max_bond(), tot_gtn_3.max_bond())
    print('compression err', np.linalg.norm(data - tot_gtn_3.get_data()))
    print('orthog err', orthog_state.ovlp(comp_remainder_naive), orthog_state.ovlp(tot_gtn_3) - proj_val)
    print()

    # exit()

    ### compression of remainder part
    comp_remainder = local_orthogonal_compress(remainder.data, [orthog_state.data],
                                               max_bond=D-1, nsites=nsites)
    print('comp remainder', comp_remainder.max_bond())
    comp_remainder_gtn = grid_X.make_gridTN(comp_remainder)
    tot_gtn_1 = comp_remainder_gtn.add( orthog_state.scalar_multiply(proj_val) )

    print('compress remainder')
    print('ranks', init_mps.max_bond(), orthog_state.max_bond(), remainder.max_bond(), comp_remainder.max_bond(),
          tot_gtn_1.max_bond())
    print('compression err', np.linalg.norm(data - tot_gtn_1.get_data()))
    print('(1) orthog err', orthog_state.ovlp(comp_remainder_gtn), orthog_state.ovlp(tot_gtn_1) - proj_val)
    print()

    ### compression with orthogonal constraint
    print('init mps rank', init_mps.max_bond())
    scaled_ostate = orthog_state #.scalar_multiply(2, inplace=True)
    comp_init = local_orthogonal_compress_2(init_mps.data, [scaled_ostate.data],
                                               max_bond=D, nsites=nsites)
    comp_init_gtn = grid_X.make_gridTN(comp_init)
    tot_gtn_2 = comp_init_gtn
    tot_gtn_2.compress(inplace=True)
    # tot_gtn_2 = comp_init_gtn.add(orthog_state.scalar_multiply(proj_val))

    print('compress init_MPS')
    print('ranks', tot_gtn_2.max_bond(), init_mps.max_bond())
    print('compression err', np.linalg.norm(data - tot_gtn_2.get_data()))
    print('(2) orthog err', orthog_state.ovlp(comp_init_gtn), orthog_state.ovlp(tot_gtn_2) - proj_val)
    print()



    plt.figure()
    plt.plot(np.real(comp_gtn.get_data()), label='naive comp')
    plt.plot(np.imag(comp_gtn.get_data()), ':')
    # plt.plot(np.real(remainder.get_data()), label='remainder')
    # plt.plot(np.imag(remainder.get_data()), '--', label='remainder')
    # plt.plot(np.real(remainder.get_data() + proj_val * orthog_state.get_data()), label='remainder')
    # plt.plot(np.imag(remainder.get_data() + proj_val * orthog_state.get_data()), '--', label='remainder')
    plt.plot(np.real(tot_gtn_1.get_data()), label='orthog remainder comp')
    plt.plot(np.imag(tot_gtn_1.get_data()), '--', ) #label='orthog remainder comp (im)')
    plt.plot(np.real(tot_gtn_2.get_data()), '--', label='orthog init comp')
    plt.plot(np.imag(tot_gtn_2.get_data()), ':')
    plt.plot(np.real(tot_gtn_3.get_data()), '--', label='naive remainder comp')
    plt.plot(np.imag(tot_gtn_3.get_data()), ':')
    plt.plot(np.real(data), 'k--', label='exact')
    plt.plot(np.imag(data), 'k:')
    plt.legend()
    plt.show()



### test 2:  return self in canonical form, orthogonal to multiple polynomial bases
do_test_2 = True
if do_test_2:
    D = 3

    num_orthog_states = 3
    orthog_states = []
    for n in range(num_orthog_states):
        poly = np.polynomial.Legendre(((0,)*n + (1,)))
        poly_data = poly.linspace(2**L)
        poly_y = poly_data[1]

        ## orthogonalize
        for m in range(n):
            ovlp = poly_y @ orthog_states[m]
            poly_y -= ovlp * orthog_states[m]

        coeff = 1. / np.linalg.norm(poly_y)
        orthog_states += [coeff * poly_y]

        ## check orthog
        for m in range(n + 1):
            ovlp = orthog_states[-1] @ orthog_states[m]
            print('ovlp', m, n, ovlp)

    orthog_mps_list = [grid_X.map_state_to_mps(ostate) for ostate in orthog_states]

    ### projection of init mps onto orthog_state
    proj_vals = []
    for orthog_state in orthog_mps_list:
        proj_val = init_mps.ovlp(orthog_state)
        # proj_val -= 1.e-2 * np.sign(proj_val)
        # # proj_val = np.conj(proj_val)    ## <b|x> -> <x|b>
        print('init proj val', proj_val)
        proj_vals += [proj_val]


    # ########################
    # ### naive compression
    comp_gtn = init_mps.compress(compress_opts={'max_bond': D}, inplace=False)
    print('compress naive')
    print('ranks', init_mps.max_bond(), comp_gtn.max_bond())
    print('compression err', np.linalg.norm(data - comp_gtn.get_data()))
    for orthog_state, proj_val in zip(orthog_mps_list, proj_vals):
        print('orthog err', orthog_state.ovlp(comp_gtn) - proj_val)
    print()

    ########################
    ### projection and remainder    (I - proj * |b><b|) x
    ### <b|(I - |b><b|/|b|)|x> = <b|x> - <b|x> = 0
    ### |x> = (I - |b><b|) |x> + proj |b> = |x> - |b> <b|x> + proj |b> so proj = <x|b>
    remainder = init_mps.copy()
    for orthog_state, proj_val in zip(orthog_mps_list, proj_vals):
        remainder = remainder.add(orthog_state.scalar_multiply(-(proj_val)))

    tmp = remainder.compress(inplace=False)
    print('compressed remainder max rank', remainder.max_bond(), tmp.max_bond())
    for orthog_state, proj_val in zip(orthog_mps_list, proj_vals):
        print('mod overlap', remainder.ovlp(orthog_state), init_mps.ovlp(orthog_state))

    #######################
    ### naive compression of remainder
    comp_remainder_naive = remainder.compress(compress_opts={'max_bond': D - 1})
    tot_gtn_3 = comp_remainder_naive.copy()
    for orthog_state, proj_val in zip(orthog_mps_list, proj_vals):
        tot_gtn_3 = tot_gtn_3.add(orthog_state.scalar_multiply(proj_val))
    tot_gtn_3.compress(inplace=True)

    print('naive compress remainder')
    print('ranks', remainder.max_bond(), comp_remainder_naive.max_bond(), tot_gtn_3.max_bond())
    print('compression err', np.linalg.norm(data - tot_gtn_3.get_data()))
    for orthog_state, proj_val in zip(orthog_mps_list, proj_vals):
        print('orthog err', orthog_state.ovlp(comp_remainder_naive), orthog_state.ovlp(tot_gtn_3) - proj_val)
    print()

    # exit()

    # ### compression of remainder part
    # comp_remainder = local_orthogonal_compress(remainder.data, [orthog_state.data for orthog_state in orthog_mps_list],
    #                                            max_bond=D-1, nsites=nsites)
    # print('comp remainder', comp_remainder.max_bond())
    # comp_remainder_gtn = grid_X.make_gridTN(comp_remainder)
    # tot_gtn_1 = comp_remainder_gtn.copy()
    # for orthog_state, proj_val in zip(orthog_mps_list, proj_vals):
    #     tot_gtn_1 = tot_gtn_1.add( orthog_state.scalar_multiply(proj_val) )
    #
    # print('compress remainder')
    # print('ranks', init_mps.max_bond(), [orthog_state.max_bond() for orthog_state in orthog_mps_list],
    #       remainder.max_bond(), comp_remainder.max_bond(),
    #       tot_gtn_1.max_bond())
    # print('compression err', np.linalg.norm(data - tot_gtn_1.get_data()))
    # for orthog_state, proj_val in zip(orthog_mps_list, proj_vals):
    #     print('orthog err', orthog_state.ovlp(comp_remainder_gtn), orthog_state.ovlp(tot_gtn_1) - proj_val)
    # print()

    ### compression with orthogonal constraint
    print('init mps rank', init_mps.max_bond())
    comp_init = local_orthogonal_compress_2(init_mps.data, [ostate.data for ostate in orthog_mps_list],
                                               max_bond=D, nsites=nsites)
    comp_init_gtn = grid_X.make_gridTN(comp_init)
    tot_gtn_2 = comp_init_gtn
    tot_gtn_2.compress(inplace=True)
    # tot_gtn_2 = comp_init_gtn.add(orthog_state.scalar_multiply(proj_val))

    print('compress init_MPS')
    print('ranks', tot_gtn_2.max_bond(), init_mps.max_bond())
    print('compression err', np.linalg.norm(data - tot_gtn_2.get_data()))
    for orthog_state, proj_val in zip(orthog_mps_list, proj_vals):
        print('orthog err', orthog_state.ovlp(comp_init_gtn), orthog_state.ovlp(tot_gtn_2) - proj_val)
    print()



    plt.figure()
    plt.plot(np.real(comp_gtn.get_data()), label='naive comp (re)')
    plt.plot(np.imag(comp_gtn.get_data()), ':', label='naive comp (im)')
    # plt.plot(np.real(remainder.get_data()), label='remainder')
    # plt.plot(np.imag(remainder.get_data()), '--', label='remainder')
    # plt.plot(np.real(remainder.get_data() + proj_val * orthog_state.get_data()), label='remainder')
    # plt.plot(np.imag(remainder.get_data() + proj_val * orthog_state.get_data()), '--', label='remainder')
    # plt.plot(np.real(tot_gtn_1.get_data()), label='orthog remainder comp (re)')
    # plt.plot(np.imag(tot_gtn_1.get_data()), '--', label='orthog remainder comp (im)')
    plt.plot(np.real(tot_gtn_2.get_data()), '-', label='orthog init comp (re)')
    plt.plot(np.imag(tot_gtn_2.get_data()), '--', label='orthog remainder comp (im)')
    # plt.plot(np.real(tot_gtn_3.get_data()), '--', label='naive remainder comp')
    # plt.plot(np.imag(tot_gtn_3.get_data()), ':')
    plt.plot(np.real(data), 'k--', label='exact (re)')
    plt.plot(np.imag(data), 'k:', label='exact (im)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()
