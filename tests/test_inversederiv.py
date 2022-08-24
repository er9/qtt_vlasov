import sys
sys.path.append('../')

from setup.configs import *
# from enums import *

import numpy as np
import matplotlib.pyplot as plt

from coord.coord_sys import Coordinate
from axis import Axis
from grid1D import Grid1D

import helper_quimb as helper

TEST_LAYOUT = LayoutType.SEQUENTIAL
TEST_FD = FDType.CENTER
ORDER = 4

do_test1_firstderiv_O = False
do_test2_firstderiv_P = False
do_test3_firstderiv_O = False  # True
do_test3_secondderiv_O = False
do_test3_secondderiv_P = True
do_test4_laplacian = False  # same as test3???
do_test5_2D = False

""" NOTE:  This implementation is kind of jank, since technically one should
    add two rows in the matrix that we're inverting defining the bc's:
        f(0) = #
        f(1)-f(0) = #
    however, we instead add these values into the d^2/dx^2 matrix
    (both in the matrix d^2/dx^2 and the vector d^2 f / dx^2)
    
    This is less of an issue when all we're missing is the offset bc we easily 
    correct things by shifting it
"""

## nonperiodic 1D system ##
## inverse via numpy, via arrays ##
if do_test1_firstderiv_O:
    print('\ntest inverse first derivative OPEN, np')

    L = 5
    q = 2
    K = 1
    npts = q**L

    X = Coordinate('X',CoordinateType.X)
    xdata = np.linspace(0, 5, npts)
    axis1 = Axis(L, q, coordinate=X, xpts=xdata)
    grid1 = Grid1D('G1', (axis1,), layout_type=TEST_LAYOUT)

    ### define initial boundary condition at E(x=0)
    bc_vec = np.array([1] + [0,]*(axis1.q**axis1.L-1))
    bc_mat = np.outer(bc_vec,bc_vec) * 1./axis1.dx
    #### 1/dx factor already included in ddx_mpo so makes bc_mat same order of magnitude
    bc_mpo = grid1.map_operator_to_mpo(bc_mat)
    print('max bond bc', bc_mpo.max_bond())
    print(bc_mpo.data.singular_values(1))
    print(bc_mpo)    ## for some reqson max_bond gives 2 even though it should be one?

    deriv_config = DerivativeConfiguration(left_bc='o', right_bc='o', order=ORDER, fd_type=TEST_FD)
    ddx_mpo = grid1.get_firstderivative_mpo(axis1, deriv_config=deriv_config)
    # print(ddx_mpo.data)
    print('max bond ddx', ddx_mpo.max_bond())
    ddx_mpo.add(bc_mpo, inplace=True, compress=True)
    print('max bond ddx+bc', ddx_mpo.max_bond())
    ddx_tens = grid1.map_mpo_to_operator(ddx_mpo)

    plt.figure()
    plt.imshow(ddx_tens)
    plt.colorbar()
    plt.title('d/dx')

    print(ddx_tens.shape)
    ddx_tens_inv = np.linalg.pinv(ddx_tens)
    out0 = np.dot(ddx_tens, ddx_tens_inv)
    out1 = np.dot(ddx_tens_inv, ddx_tens)
    print(np.linalg.norm(out0-np.eye(len(out0))))
    print(np.linalg.norm(out1-np.eye(len(out1))))

    # print('ddx max bond', ddx_mpo.max_bond())
    # ddx_mpo_inv_2 = inverse.get_mpo_inverse_dmrg(ddx_mpo, ns_opt=2, conv_tol=5.0e-7, **gs.compress_opts)
    #     # max_bond=10, cutoff=1.0e-20)
    #     ## ns_opt = 2 converges faster but is exponentially? slower
    # ddx_inv_2  = grid1.map_mpo_to_operator(ddx_mpo_inv_2)
    # out0_ = helper.apply(ddx_mpo, ddx_mpo_inv_2)
    # out0_ = grid1.map_mpo_to_operator(out0_)
    # out1_ = helper.apply(ddx_mpo_inv_2,ddx_mpo)
    # out1_ = grid1.map_mpo_to_operator(out1_)
    # print(np.linalg.norm(out0_-np.eye(len(out0_))))
    # print(np.linalg.norm(out1_-np.eye(len(out1_))))
    

    func   = lambda x:  2*((x-4)**5) + 10*(x+2)**2.5
    dfunc  = lambda x:  10*(x-4)**4  + 10*2.5*(x+2)**1.5
    dfunc_data = dfunc(xdata)
    func_data  = func(xdata)
    
    integ_dfunc = np.dot(ddx_tens_inv, dfunc_data) + func(0)
    # integ_dfunc2 = np.dot(ddx_inv_2, dfunc_data) + func(0)

    plt.figure()
    plt.plot(xdata, dfunc_data, label='dfunc')
    plt.plot(xdata, np.dot(ddx_tens, func_data), label='tens')
    plt.plot(xdata, np.dot( np.dot(ddx_tens, ddx_tens_inv), dfunc_data), label='tens inv')
    plt.legend()

    plt.figure()
    plt.plot(xdata,integ_dfunc, label='tens')
    # plt.plot(data,integ_dfunc2,label='dmrg')
    plt.plot(xdata,func_data,'--',label='func')
    plt.legend()
    plt.show()


## periodic 1D system ##
## inverse via numpy, test arrays ##
if do_test2_firstderiv_P:
    print('\ntest inverse first derivative PERIODIC, np')

    L = 8
    q = 2
    K = 1
    npts = q**(L*K)

    X = Coordinate('X')
    xdata = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    dx = xdata[1] - xdata[0]
    axis1 = Axis(L, q, coordinate=X, xpts=xdata)
    grid1 = Grid1D('G1', (axis1,), layout_type=TEST_LAYOUT)
    # field = Field('F1', grid1, data={None: vec_data})

    ### define initial boundary condition x=0,, 
    ### two needed if using periodic boundary conditions
    bc_vec1 = np.array([1] + [0,]*(grid1.npts-1))
    bc_mat1 = np.outer(bc_vec1,bc_vec1)
    
    bc_vec2 = np.array([0,1] + [0,]*(grid1.npts-2))
    # bc_vec2 = bc_vec1[::-1].copy()
    bc_mat2 = np.outer(bc_vec2,bc_vec2)

    deriv_config = DerivativeConfiguration(left_bc='p', right_bc='p', order=ORDER, fd_type=TEST_FD)
    ddx_mpo = grid1.get_firstderivative_mpo(axis1, deriv_config=deriv_config)
    ddx_tens = grid1.map_mpo_to_operator(ddx_mpo)

    ## ddx_tens = ddx_tens + bc_mat1 + bc_mat2
    ddx_tens[0,:] += bc_vec1 * 1./axis1.dx
    ddx_tens[1,:] += bc_vec2 * 1./axis1.dx
    u,s,vt = np.linalg.svd(ddx_tens)
    print('sing vals, ...', s[-10:])
    
    ddx_tens_inv = np.linalg.pinv(ddx_tens)
    out0 = np.dot(ddx_tens, ddx_tens_inv)
    out1 = np.dot(ddx_tens_inv, ddx_tens)
    print(np.linalg.norm(out0-np.eye(len(out0))))
    print(np.linalg.norm(out1-np.eye(len(out1))))
    # print(ddx_tens_inv)


    d2dx2_mpo = grid1.get_secondderivative_mpo(axis1, None, deriv_config1=deriv_config)
    d2dx2_tens = grid1.map_mpo_to_operator(d2dx2_mpo)

    ## d2dx2_tens = d2dx2_tens + bc_mat1
    d2dx2_tens[0,:] += bc_vec1 * 1./axis1.dx**2
    d2dx2_tens_inv = np.linalg.pinv(d2dx2_tens)
    out0 = np.dot(d2dx2_tens, d2dx2_tens_inv)
    out1 = np.dot(d2dx2_tens_inv, d2dx2_tens)
    print('d2dx2 inv err', np.linalg.norm(out0 - np.eye(len(out0))))
    print('d2dx2 inv err', np.linalg.norm(out1 - np.eye(len(out1))))

    plt.figure()
    plt.imshow(ddx_tens)
    plt.colorbar()
    plt.title('ddx')

    plt.figure()
    plt.imshow(d2dx2_tens)
    plt.colorbar()
    plt.title('d2dx2')

    func   = lambda x:  np.sin(2*x+3*np.pi/5)
    dfunc  = lambda x:  np.cos(2*x+3*np.pi/5) *  2
    d2func = lambda x: -np.sin(2*x + 3*np.pi / 5) * 4
    func_data   = func(xdata)
    dfunc_data  = dfunc(xdata)
    d2func_data = d2func(xdata)

    ## d/dx inverse
    print(func(xdata[0]), func(xdata[1]))
    bc_data = dfunc_data # + np.array([func(data[0])] + [0]*(npts-1)) \
                         # + np.array([0,func(data[1])] + [0]*(npts-2))
    bc_data[0] += func(xdata[0])
    bc_data[1] += func(xdata[0]) + dx / 2 * (dfunc_data[0] + dfunc_data[1])
                  ## noisy if approx, better with higher order
    # bc_data[-1] += func(data[-1]) ## doesn't work: np.trapz(dfunc_data)*dx
    integ_dfunc = np.dot(ddx_tens_inv, dfunc_data)


    ## d2/dx2 inverse
    bc_data = dfunc_data
    bc_data[0] = func(xdata[0])
    d2inv_d2func = np.dot(d2dx2_tens_inv, d2func_data)
    
    plt.figure()
    plt.plot(xdata, integ_dfunc, label='d integ -> func')
    plt.plot(xdata, func_data, '--', label='func')
    plt.plot(xdata, d2inv_d2func, ':', label='d2 integ -> func')
    plt.legend()
    plt.show()


### operator to MPO (cheating method and TODO: TN optimization)
### inverse via numpy, put into TN form
if do_test3_firstderiv_O:
    print('\ntest inverse first derivative OPEN')

    L = 4
    K = 1
    q = 2

    ORDER = 1

    npts = q**L
    xdata = np.linspace(0, 2*np.pi, npts)

    X = Coordinate('X')
    axis1 = Axis(L, q, coordinate=X, xpts=xdata)
    grid1 = Grid1D('G1', (axis1,), layout_type=TEST_LAYOUT)
    # field = Field('F1', grid1, data={None: vec_data})

    ### define initial boundary condition at f(x=0)
    ### two needed if using periodic boundary conditions
    bc_vec = np.array([1.0] + [0,]*(grid1.npts-1))
    bc_mat = np.outer(bc_vec,bc_vec) * 1./axis1.dx
    ## ^ divide to make same norm (weight) as ddx_mpo below. reduces error in TN dot methods
    bc_mpo = grid1.map_operator_to_mpo(bc_mat)  #, split_opts={'cutoff':1.0E-30})
    print('max bond bc', bc_mpo.max_bond())
    print(bc_mpo.data.singular_values(1))
    print(bc_mpo)    ## for some reqson max_bond gives 2 even though it should be one?

    # deriv_params = {'bc': BCType.OPEN, 'fd_type': TEST_FD, 'order': ORDER}
    deriv_config = DerivativeConfiguration(left_bc='o', right_bc='o', order=ORDER, fd_type=TEST_FD)
    ddx_mpo = grid1.get_firstderivative_mpo(axis1, deriv_config=deriv_config)
    print('max bond ddx', ddx_mpo.max_bonds(), ddx_mpo.data._exponent)
    print('ddx sing vals', ddx_mpo.data.singular_values(L // 2 + 1))
    # ddx_mpo_data = helper.add_MPO(ddx_mpo.data, bc_mpo.data, compress=False)
    ddx_mpo.add(bc_mpo, inplace=True, compress=True, compress_opts={'cutoff': 1.0E-30})
    print('max bond ddx+bc', ddx_mpo.max_bonds())
    print('ddx+bc sing vals',ddx_mpo.data.singular_values(L // 2 + 1))
    # print('ddx_mpo_data sing vals', ddx_mpo_data.singular_values(L//2 + 1))
    ddx_tens = grid1.map_mpo_to_operator(ddx_mpo)

    ### cheating method of getting inverse ###
    ### check inverse mult via numpy
    ddx_tens_inv = np.linalg.pinv(ddx_tens)
    out0 = np.dot(ddx_tens, ddx_tens_inv)
    out1 = np.dot(ddx_tens_inv, ddx_tens)
    print('out0',np.linalg.norm(out0-np.eye(len(out0))))
    print('out1',np.linalg.norm(out1-np.eye(len(out1))))
    # print(ddx_tens_inv)

    ### check inverse mult via TN
    ddx_mpo_inv = grid1.map_operator_to_mpo(ddx_tens_inv)
    # out0_ = helper.apply(ddx_mpo.data, ddx_mpo_inv.data)
    # out0_ = LayoutSequentialF.map_mpo_to_operator((axis1,), out0_)
    # out1_ = helper.apply(ddx_mpo_inv.data, ddx_mpo.data)
    # out1_ = LayoutSequentialF.map_mpo_to_operator((axis1,), out1_)
    out0_ = ddx_mpo.apply(ddx_mpo_inv)  # helper.apply(ddx_mpo, ddx_mpo_inv)
    out0_ = grid1.map_mpo_to_operator(out0_)
    out1_ = ddx_mpo_inv.apply(ddx_mpo)  # helper.apply(ddx_mpo_inv,ddx_mpo)
    out1_ = grid1.map_mpo_to_operator(out1_)
    print('out0',np.linalg.norm(out0_-np.eye(len(out0_))))
    print('out1',np.linalg.norm(out1_-np.eye(len(out1_))))
    print(ddx_mpo_inv.max_bond())
    print(ddx_mpo_inv.data.singular_values(L//2+1))
    
    ddx_mpo_inv.compress(compress_opts={'cutoff':1.0E-30}, inplace=True)
    out0_ = ddx_mpo.apply(ddx_mpo_inv)
    out0_ = grid1.map_mpo_to_operator(out0_)
    out1_ = ddx_mpo_inv.apply(ddx_mpo)
    out1_ = grid1.map_mpo_to_operator(out1_)
    print('inv compress error',np.linalg.norm(out0_-np.eye(len(out0_))))
    print('inv compress error',np.linalg.norm(out1_-np.eye(len(out1_))))
    print('inv compress',ddx_mpo_inv.max_bond())
    print('inv compress',ddx_mpo_inv.data.singular_values(L//2+1))
    print('inv exponent',ddx_mpo_inv.data.exponent)
    print(ddx_mpo_inv)
    
    # iden_mpo = qtn.MPO_identity_like(ddx_mpo.data)
    iden_mpo = grid1.get_iden_mpo()
    print(helper.norm(iden_mpo.data))
    print(helper.norm((iden_mpo.scalar_multiply(2)).data))
    iden_tens = grid1.map_mpo_to_operator(iden_mpo)
    print('id error', np.linalg.norm(iden_tens - np.eye(2**L)))
    
    # ddx_inv_norm = helper.norm(ddx_mpo_inv.data)
    ddx_inv_norm = ddx_mpo_inv.norm()
    print(ddx_inv_norm)
    print(iden_mpo.norm())
    print(helper.norm(ddx_mpo.data))
    print(helper.norm(ddx_mpo.data) * ddx_inv_norm)

    ### TN methods of inversion ###
    # # ddx_mpo_inv_2 = helper.get_mpo_inverse(ddx_mpo, iden_mpo, init_guess=None, # ddx_mpo_inv,
    # #                                        **gs.compress_opts)
    # ddx_mpo_inv_2 = inverse.get_mpo_inverse_dmrg(ddx_mpo)
    # out0_ = ddx_mpo.apply(ddx_mpo, ddx_mpo_inv_2)
    # diff0 = out0_.add(iden_mpo.multiply(-1), inplace=True)
    # out0_ = grid1.map_mpo_to_operator(out0_)
    # out1_ = ddx_mpo_inv_2.apply(ddx_mpo)
    # diff1 = out1_.add(iden_mpo.multiply(-1), inplace=True)
    # out1_ = grid1.map_mpo_to_operator(out1_)
    # print('diff exponent', diff0.data.exponent, diff1.data.exponent)
    # print('diff quimb norm', diff0.data.norm(), diff1.data.norm())
    # print('diff helper norm', helper.norm(diff0.data), helper.norm(diff1.data))
    # diff0.data.distribute_exponent()
    # diff1.data.distribute_exponent()
    # print('diff dist norm', diff0.data.norm(), diff1.data.norm())
    # print('inv mpo compress error',np.linalg.norm(out0_-np.eye(len(out0_))))
    # print('inv mpo compress error',np.linalg.norm(out1_-np.eye(len(out1_))))
    # print('inv mpo compress',ddx_mpo_inv_2.data.max_bond())
    # print('inv mpo compress',ddx_mpo_inv_2.data.singular_values(L//2+1))
    # print('inv mpo exponent',ddx_mpo_inv_2.data.exponent)



### operator of d2dx2 to MPO (cheating method and TODO: TN optimization)
### cheating method: inverse via numpy, put into TN form
if do_test3_secondderiv_O:
    print('\ntest inverse second derivative OPEN')

    """ NOTE:  This implementation is kind of jank, since technically one should
        add two rows in the matrix that we're inverting defining the bc's:
            f(0) = #
            f(1)-f(0) = #
        however, we instead add these values into the d^2/dx^2 matrix
        (both in the matrix d^2/dx^2 and the vector d^2 f / dx^2)           
    """

    L = 4
    K = 1
    q = 2

    npts = q ** L
    xdata = np.linspace(0, 2 * np.pi, npts)
    X = Coordinate('X')
    axis1 = Axis(L, q, coordinate=X, xpts=xdata)
    grid1 = Grid1D('G1', (axis1,), layout_type=TEST_LAYOUT)

    # deriv_params = {'bc': BCType.OPEN, 'fd_type': TEST_FD, 'order': ORDER}
    deriv_config = DerivativeConfiguration(left_bc='o', right_bc='o', order=ORDER, fd_type=TEST_FD)
    d2dx2_mpo = grid1.get_secondderivative_mpo(axis1, None, deriv_config1=deriv_config)
    print('max bond ddx', d2dx2_mpo.max_bonds(), d2dx2_mpo.data._exponent)
    print('d2dx2 sing vals', d2dx2_mpo.data.singular_values(L // 2 + 1))
    d2dx2_tens = grid1.map_mpo_to_operator(d2dx2_mpo)

    ### define initial boundary condition at f(x=0)
    bc_mat = np.zeros((grid1.npts, grid1.npts))
    # bc_mat[0,:] = -1*d2dx2_tens[0,:]
    bc_mat[-1,0] += 1.0
    # bc_vec = np.array([1.0] + [0, ] * (grid1.npts - 1))
    # bc_mat = np.outer(bc_vec, bc_vec) / axis1.dx
    ## ^ divide to make same norm (weight) as ddx_mpo below. reduces error in TN dot methods
    bc_mpo = grid1.map_operator_to_mpo(bc_mat)  # , split_opts={'cutoff':1.0E-30})
    print('max bond bc', bc_mpo.max_bond())
    print(bc_mpo.data.singular_values(1))

    bc_mat_2 = np.zeros_like(bc_mat)
    # bc_mat_2[1,:] = -1*d2dx2_tens[1,:]
    bc_mat_2[-2,:3] += np.array([-1., 1, 0.]) # / axis1.dx
    # bc_mat_2[1,1] = 1.0
    bc_mpo_2 = grid1.map_operator_to_mpo(bc_mat_2)
    print('max bond bc', bc_mpo_2.max_bond())

    d2dx2_tens_bc = d2dx2_tens + bc_mat + bc_mat_2


    # ddx_mpo_data = helper.add_MPO(ddx_mpo.data, bc_mpo.data, compress=False)
    d2dx2_mpo.add(bc_mpo, inplace=True, compress=True, compress_opts={'cutoff': 1.0E-30})
    d2dx2_mpo.add(bc_mpo_2, inplace=True, compress=True, compress_opts={'cutoff': 1.0E-30})
    print('max bond ddx+bc', d2dx2_mpo.max_bonds())
    print('d2dx2+bc sing vals', d2dx2_mpo.data.singular_values(L // 2 + 1))
    # print('ddx_mpo_data sing vals', ddx_mpo_data.singular_values(L//2 + 1))
    d2dx2_tens = grid1.map_mpo_to_operator(d2dx2_mpo)
    print('np d2dx2 + bc tens', np.linalg.norm(d2dx2_tens - d2dx2_tens_bc))

    ### cheating method of getting inverse ###
    ### check inverse mult via numpy
    d2dx2_tens_inv = np.linalg.pinv(d2dx2_tens)
    out0 = np.dot(d2dx2_tens, d2dx2_tens_inv)
    out1 = np.dot(d2dx2_tens_inv, d2dx2_tens)
    print('d2dx2 np out0', np.linalg.norm(out0 - np.eye(len(out0))))
    print('d2dx2 np out1', np.linalg.norm(out1 - np.eye(len(out1))))
    # print(ddx_tens_inv)

    ### check inverse mult via TN
    d2dx2_mpo_inv = grid1.map_operator_to_mpo(d2dx2_tens_inv)
    # out0_ = helper.apply(ddx_mpo.data, ddx_mpo_inv.data)
    # out0_ = LayoutSequentialF.map_mpo_to_operator((axis1,), out0_)
    # out1_ = helper.apply(ddx_mpo_inv.data, ddx_mpo.data)
    # out1_ = LayoutSequentialF.map_mpo_to_operator((axis1,), out1_)
    out0_ = d2dx2_mpo.apply(d2dx2_mpo_inv)  # helper.apply(ddx_mpo, ddx_mpo_inv)
    out0_ = grid1.map_mpo_to_operator(out0_)
    out1_ = d2dx2_mpo_inv.apply(d2dx2_mpo)  # helper.apply(ddx_mpo_inv,ddx_mpo)
    out1_ = grid1.map_mpo_to_operator(out1_)
    print('d2dx2 tn out0', np.linalg.norm(out0_ - np.eye(len(out0_))))
    print('d2dx2 tn out1', np.linalg.norm(out1_ - np.eye(len(out1_))))
    print(d2dx2_mpo_inv.max_bond())
    print(d2dx2_mpo_inv.data.singular_values(L // 2 + 1))

    d2dx2_mpo_inv.compress(compress_opts={'cutoff': 1.0E-30}, inplace=True)
    out0_ = d2dx2_mpo.apply(d2dx2_mpo_inv)
    out0_ = grid1.map_mpo_to_operator(out0_)
    out1_ = d2dx2_mpo_inv.apply(d2dx2_mpo)
    out1_ = grid1.map_mpo_to_operator(out1_)
    print('inv compress error', np.linalg.norm(out0_ - np.eye(len(out0_))))
    print('inv compress error', np.linalg.norm(out1_ - np.eye(len(out1_))))
    print('inv compress', d2dx2_mpo_inv.max_bond())
    print('inv compress', d2dx2_mpo_inv.data.singular_values(L // 2 + 1))
    print('inv exponent', d2dx2_mpo_inv.data.exponent)
    print(d2dx2_mpo_inv)

    # iden_mpo = qtn.MPO_identity_like(ddx_mpo.data)
    iden_mpo = grid1.get_iden_mpo()

    # ddx_inv_norm = helper.norm(ddx_mpo_inv.data)
    d2dx2_inv_norm = d2dx2_mpo_inv.norm()
    print(d2dx2_inv_norm)
    print(iden_mpo.norm())
    print(helper.norm(d2dx2_mpo.data))
    print(helper.norm(d2dx2_mpo.data) * d2dx2_inv_norm)

    ### TN methods of inversion ###
    # # ddx_mpo_inv_2 = helper.get_mpo_inverse(ddx_mpo, iden_mpo, init_guess=None, # ddx_mpo_inv,
    # #                                        **gs.compress_opts)
    # ddx_mpo_inv_2 = inverse.get_mpo_inverse_dmrg(ddx_mpo)
    # out0_ = ddx_mpo.apply(ddx_mpo, ddx_mpo_inv_2)
    # diff0 = out0_.add(iden_mpo.multiply(-1), inplace=True)
    # out0_ = grid1.map_mpo_to_operator(out0_)
    # out1_ = ddx_mpo_inv_2.apply(ddx_mpo)
    # diff1 = out1_.add(iden_mpo.multiply(-1), inplace=True)
    # out1_ = grid1.map_mpo_to_operator(out1_)
    # print('diff exponent', diff0.data.exponent, diff1.data.exponent)
    # print('diff quimb norm', diff0.data.norm(), diff1.data.norm())
    # print('diff helper norm', helper.norm(diff0.data), helper.norm(diff1.data))
    # diff0.data.distribute_exponent()
    # diff1.data.distribute_exponent()
    # print('diff dist norm', diff0.data.norm(), diff1.data.norm())
    # print('inv mpo compress error',np.linalg.norm(out0_-np.eye(len(out0_))))
    # print('inv mpo compress error',np.linalg.norm(out1_-np.eye(len(out1_))))
    # print('inv mpo compress',ddx_mpo_inv_2.data.max_bond())
    # print('inv mpo compress',ddx_mpo_inv_2.data.singular_values(L//2+1))
    # print('inv mpo exponent',ddx_mpo_inv_2.data.exponent)

    func = lambda x: 2 * ((x - 4) ** 5) + 10 * (x + 2) ** 2.5
    dfunc = lambda x: 10 * (x - 4) ** 4 + 10 * 2.5 * (x + 2) ** 1.5
    d2func = lambda x: 40 * (x - 4) ** 3 + 10 * 2.5 * 1.5 * (x + 2) ** 0.5
    func_data = func(xdata)
    dfunc_data = dfunc(xdata)
    d2func_data = d2func(xdata)

    d2func_data[-1] += func_data[0]
    d2func_data[-2] += func_data[1] - func_data[0]
    d2fdata_mps = grid1.map_state_to_mps(d2func_data)
    out_gtn = d2fdata_mps.apply(d2dx2_mpo_inv)
    print(type(d2fdata_mps), type(out_gtn))

    plt.figure()
    plt.plot(xdata, func_data, label='func')
    plt.plot(xdata, np.dot(d2dx2_tens_inv, d2func_data), label='np')
    plt.plot(xdata, out_gtn.get_data(), linestyle=':', label='tn')
    plt.legend()
    plt.show()


### inverse Laplacian
if do_test3_secondderiv_P:
    print('\ntest inverse second derivative PERIODIC')
    ORDER = 1
    TEST_LAYOUT = LayoutType.SEQUENTIAL
    L = 8
    K = 1
    q = 2

    npts = q ** L
    xdata = np.linspace(0, 2 * np.pi, npts)
    X = Coordinate('X')
    axis1 = Axis(L, q, coordinate=X, xpts=xdata)
    grid1 = Grid1D('G1', (axis1,), layout_type=TEST_LAYOUT)

    ### define initial boundary condition at f(x=0)
    ### two needed if using periodic boundary conditions
    bc_vec = np.array([1.0] + [0, ] * (grid1.npts - 1))
    bc_mat = np.outer(bc_vec, bc_vec) * 1. / axis1.dx**2
    ## ^ divide to make same norm (weight) as ddx_mpo below. reduces error in TN dot methods
    bc_mpo = grid1.map_operator_to_mpo(bc_mat)  # , split_opts={'cutoff':1.0E-30})
    print('max bond bc', bc_mpo.max_bond())
    print(bc_mpo.data.singular_values(1))
    print(bc_mpo)  ## for some reqson max_bond gives 2 even though it should be one?

    deriv_config = DerivativeConfiguration(left_bc='p', right_bc='p', order=ORDER, fd_type=TEST_FD)
    lapl_mpo = grid1.get_secondderivative_mpo(axis1, None, deriv_config1=deriv_config)
    print('max bond ddx', lapl_mpo.max_bond(), lapl_mpo.data._exponent)
    print('ddx sing vals', lapl_mpo.data.singular_values(L // 2 + 1))
    lapl_mpo.add(bc_mpo, inplace=True, compress=True, compress_opts={'cutoff': 1.0E-30})
    print('max bond ddx+bc', lapl_mpo.max_bond())
    print('ddx+bc sing vals', lapl_mpo.data.singular_values(L // 2 + 1))
    lapl_tens = grid1.map_mpo_to_operator(lapl_mpo)

    ### check inverse mult via numpy
    lapl_tens_inv = np.linalg.pinv(lapl_tens)
    out0 = np.dot(lapl_tens, lapl_tens_inv)
    out1 = np.dot(lapl_tens_inv, lapl_tens)
    print('out0', np.linalg.norm(out0 - np.eye(len(out0))))
    print('out1', np.linalg.norm(out1 - np.eye(len(out1))))
    # print(ddx_tens_inv)

    ### check inverse mult via TN
    lapl_mpo_inv = grid1.map_operator_to_mpo(lapl_tens_inv)
    # out0_ = helper.apply(ddx_mpo.data, ddx_mpo_inv.data)
    # out0_ = LayoutSequentialF.map_mpo_to_operator((axis1,), out0_)
    # out1_ = helper.apply(ddx_mpo_inv.data, ddx_mpo.data)
    # out1_ = LayoutSequentialF.map_mpo_to_operator((axis1,), out1_)
    out0_ = lapl_mpo.apply(lapl_mpo_inv)  # helper.apply(ddx_mpo, ddx_mpo_inv)
    out0_ = grid1.map_mpo_to_operator(out0_)
    out1_ = lapl_mpo_inv.apply(lapl_mpo)  # helper.apply(ddx_mpo_inv,ddx_mpo)
    out1_ = grid1.map_mpo_to_operator(out1_)
    print('out0', np.linalg.norm(out0_ - np.eye(len(out0_))))
    print('out1', np.linalg.norm(out1_ - np.eye(len(out1_))))
    print(lapl_mpo_inv.max_bond())
    print(lapl_mpo_inv.data.singular_values(L // 2 + 1))

    lapl_mpo_inv.compress(compress_opts={'cutoff': 1.0E-30}, inplace=True)
    out0_ = lapl_mpo.apply(lapl_mpo_inv)
    out0_ = grid1.map_mpo_to_operator(out0_)
    out1_ = lapl_mpo_inv.apply(lapl_mpo)
    out1_ = grid1.map_mpo_to_operator(out1_)
    print('inv compress error', np.linalg.norm(out0_ - np.eye(len(out0_))))
    print('inv compress error', np.linalg.norm(out1_ - np.eye(len(out1_))))
    print('inv compress', lapl_mpo_inv.max_bond())
    print('inv compress svals', lapl_mpo_inv.data.singular_values(L // 2 + 1))
    print('inv exponent', lapl_mpo_inv.data.exponent)
    print(lapl_mpo_inv)

    # iden_mpo = qtn.MPO_identity_like(ddx_mpo.data)
    iden_mpo = grid1.get_iden_mpo()

    # ddx_inv_norm = helper.norm(ddx_mpo_inv.data)
    lapl_inv_norm = lapl_mpo_inv.norm()
    print('lapl inv norm', lapl_inv_norm)
    print('iden mpo norm', iden_mpo.norm())
    print('lapl mpo data norm', helper.norm(lapl_mpo.data))
    print('lapl mpo norm * inv_norm', helper.norm(lapl_mpo.data) * lapl_inv_norm)


## [un]periodic 2D system ##
if do_test5_2D:

    raise NotImplementedError('inversion of 2D derivative operators not done yet... boundary conditions?')

    q = 2
    L = 4
    K = 2

    npts = q ** L
    xdata = np.linspace(0, 5, npts, endpoint=False)
    data_y, data_x = np.meshgrid(xdata, xdata)
    dx = xdata[1] - xdata[0]

    axis1 = Axis('X', L, q, xpts=xdata)
    axis2 = Axis('Y', L, q, xpts=xdata)
    grid1 = Grid1D('G1', (axis1, axis2), layout_type=TEST_LAYOUT)
    # field = Field('F1', grid1, data={None: vec_data})

    func = lambda x, y: 2 * ((x - 4) ** 5) + 10 * (x + 2) ** 2.5 * np.cos(2 * np.pi / 2.5 * y + 1.)
    dfunc_x = lambda x, y: 10 * (x - 4) ** 4 + 10 * 2.5 * (x + 2) ** 1.5 * np.cos(2 * np.pi / 2.5 * y + 1.)
    dfunc_y = lambda x, y: -10 * (x + 2) ** 2.5 * np.sin(2 * np.pi / 2.5 * y + 1.) * (2 * np.pi / 2.5)
    dfunc_xy = lambda x, y: -10 * 2.5 * (x + 2) ** 1.5 * np.sin(2 * np.pi / 2.5 * y + 1.) * (2 * np.pi / 2.5)

    dfunc_xy_data = dfunc_xy(data_x, data_y)
    dfunc_y_data = dfunc_y(data_x, data_y)
    dfunc_x_data = dfunc_x(data_x, data_y)
    func_data = func(data_x, data_y)

    print(4 ** L, npts)
    bcx_vec = np.array([1, ] + [0, ] * (npts - 1))
    bcy_vec = np.array([1, ] * (npts))
    bc_vec = np.outer(bcx_vec, bcy_vec)
    bc_mat = np.outer(bc_vec, bc_vec)
    print(np.sum(bc_vec))
    bc_mpo = grid1.map_operator_to_mpo(bc_mat),
             deriv_config = DerivativeConfiguration(left_bc='o', right_bc='o', fd_type=TEST_FD, order=ORDER)
    ddx_mpo = grid1.get_firstderivative_mpo(axis1, deriv_config=deriv_config)
    # ddx_mpo = helper.add_MPO(ddx_mpo,bc_mpo,compress=True,**gs.compress_opts)
    ddx_tens = grid1.map_mpo_to_operator(ddx_mpo).reshape(npts ** 2, npts ** 2)
    ddx_tens = ddx_tens + bc_mat
    ddx_tens_inv = np.linalg.pinv(ddx_tens)
    out0 = np.dot(ddx_tens, ddx_tens_inv)
    out1 = np.dot(ddx_tens_inv, ddx_tens)
    print(out0.shape, out1.shape)
    print('ddx gs pinv', np.linalg.norms(out0 - np.eye(len(out0))))
    print('ddx gs pinv', np.linalg.norms(out1 - np.eye(len(out1))))

    deriv_config1 = DerivativeConfiguration(left_bc='o', right_bc='o', order=ORDER, fd_type=TEST_FD)
    deriv_config2 = DerivativeConfiguration(left_bc='p', right_bc='p', order=ORDER, fd_type=TEST_FD)
    # d2dxdy_mpo  = grid1.get_secondderivative_mpo(axis1, axis2, deriv_params,'p')
    d2dxdy_mpo = grid1.get_secondderivative_mpo(axis1, axis2, deriv_config1=deriv_config1, deriv_config2=deriv_config2)
    d2dxdy_tens = grid1.map_mpo_to_operator(d2dxdy_mpo)
    print(d2dxdy_tens.shape)
    u, s, vt = np.linalg.svd(d2dxdy_tens.reshape(npts ** 2, npts ** 2))
    print(np.sum(s < 1.0e-10), len(s), npts)  ## need to specify BCs for f(x=0,y) and f(x,y=0)

    ### define initial boundary condition at E(x=0)
    grid_x = Grid1D('GX', (axis1,), layout_type=TEST_LAYOUT)
    grid_y = Grid1D('GY', (axis2,), layout_type=TEST_LAYOUT)

    ### boundary conditions
    bc_vec = np.array([1] + [0, ] * (grid_x.npts - 1))
    bc_mat = np.outer(bc_vec, bc_vec)
    bc_mpo = grid_x.map_operator_to_mpo(bc_mat),

             ddx_mpo = grid_x.get_firstderivative_mpo(axis1, deriv_config=deriv_config1)
    ddx_mpo.add(bc_mpo, inplace=True, compress=True)
    # ddx_mpo = helper.add_MPO(ddx_mpo,bc_mpo,compress=True)
    ddx_tens = grid_x.map_mpo_to_operator(ddx_mpo)
    # ddx_tens = ddx_tens + bc_mat
    ddx_tens_inv = np.linalg.pinv(ddx_tens)
    out0 = np.dot(ddx_tens, ddx_tens_inv)
    out1 = np.dot(ddx_tens_inv, ddx_tens)
    print('ddx pinv', np.linalg.norms(out0 - np.eye(len(out0))))
    print('ddx pinv', np.linalg.norms(out1 - np.eye(len(out1))))

    # ddx_mpo = helper.add_MPO(ddx_mpo,bc_mpo,compress=True,**gs.compress_opts)

    ### boundary conditions
    bc_vec1 = np.array([1] + [0, ] * (grid_y.npts - 1))
    bc_mat1 = np.outer(bc_vec1, bc_vec1)
    bc_mpo1 = grid_y.map_operator_to_mpo(bc_mat1),
              bc_vec2 = np.array([0, 1] + [0, ] * (grid_y.npts - 2))
    bc_mat2 = np.outer(bc_vec1, bc_vec1)
    bc_mpo2 = grid_y.map_operator_to_mpo(bc_mat2),

              ddy_mpo = grid_y.get_firstderivative_mpo(axis1, deriv_config=deriv_config2)
    ddy_mpo.add(bc_mpo1, inplace=True, compress=True)
    # ddy_mpo = helper.add_MPO(ddy_mpo,bc_mpo1,compress=True)
    # ddy_mpo = helper.add_MPO(ddy_mpo,bc_mpo2,compress=True)
    ddy_tens = grid_y.map_mpo_to_operator(ddy_mpo)
    # ddy_tens = ddy_tens + bc_mat1 + bc_mat2
    ddy_tens_inv = np.linalg.pinv(ddy_tens)
    out0 = np.dot(ddy_tens, ddy_tens_inv)
    out1 = np.dot(ddy_tens_inv, ddy_tens)
    print('ddy pinv', np.linalg.norms(out0 - np.eye(len(out0))))
    print('ddy pinv', np.linalg.norms(out1 - np.eye(len(out1))))

    # ddy_mpo = helper.add_MPO(ddy_mpo,bc_mpo1,compress=False)
    # ddy_mpo = helper.add_MPO(ddy_mpo,bc_mpo2,compress=True,**gs.compress_opts)

    ddx_x = grid1.make_mpo_ndim({axis1.coordinate: ddx_mpo.data})
    ddx_mpo2 = grid1.get_firstderivative_mpo(axis1, deriv_config=deriv_config1)
    print('distance', helper.distance(ddx_x.data, ddx_mpo2.data))

    ddy_x = grid1.make_mpo_ndim({axis2.coordinate: ddy_mpo.data})
    ddy_mpo2 = grid1.get_firstderivative_mpo(axis2, deriv_config=deriv_config2)
    print('distance', helper.distance(ddy_x.data, ddy_mpo2.data))

    d2dxdy_x = grid1.make_mpo_ndim({axis1.coordinate: ddx_mpo.data, axis2.coordinate: ddy_mpo.data})
    d2dxdy_x_tens = grid1.map_mpo_to_operator(d2dxdy_x)
    u, s, vt = np.linalg.svd(d2dxdy_x_tens.reshape(npts ** 2, npts ** 2))
    print(np.sum(s < 1.0e-10), len(s), npts)  ## need to specify BCs for f(x=0,y) and f(x,y=0)

    print('distance', helper.distance(d2dxdy_x.data, d2dxdy_mpo.data))
    exit()

    ### take inverses
    ddx_inv_mpo = grid_x.map_mpo_to_operator(ddx_tens_inv)
    ddy_inv_mpo = grid_x.map_mpo_to_operator(ddy_tens_inv)

    integ_dfunc = np.dot(ddx_tens_inv, dfunc_data) * dx + func(0)

    # npts = grid1.npts
    # xdata = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    # dx = xdata[1] - xdata[0]
    #

    ### define initial boundary condition x=0,,
    ### two needed if using periodic boundary conditions
    bc_vec1 = np.array([1] + [0, ] * (grid1.npts - 1))
    bc_mat1 = np.outer(bc_vec1, bc_vec1)

    bc_vec2 = np.array([0, 1] + [0, ] * (grid1.npts - 2))
    # bc_vec2 = bc_vec1[::-1].copy()
    bc_mat2 = np.outer(bc_vec2, bc_vec2)

    ddx_mpo = grid1.get_firstderivative_mpo(axis1, deriv_config=deriv_params1)
    ddx_tens = grid1.map_mpo_to_operator(ddx_mpo) / dx

    # ddx_tens = ddx_tens + bc_mat1 + bc_mat2
    ddx_tens[0, :] += bc_vec1
    ddx_tens[1, :] += bc_vec2
    u, s, vt = np.linalg.svd(ddx_tens)
    print('sing vals, ...', s[-10:])

    ddx_tens_inv = np.linalg.pinv(ddx_tens)
    out0 = np.dot(ddx_tens, ddx_tens_inv)
    out1 = np.dot(ddx_tens_inv, ddx_tens)
    print(np.linalg.norms(out0 - np.eye(len(out0))))
    print(np.linalg.norms(out1 - np.eye(len(out1))))
    # print(ddx_tens_inv)

    func = lambda x: np.sin(2 * x + 3 * np.pi / 5)
    dfunc = lambda x: np.cos(2 * x + 3 * np.pi / 5) * 2
    dfunc_data = dfunc(xdata)
    func_data = func(xdata)

    print(func(xdata[0]), func(xdata[1]))
    bc_data = dfunc_data  # + np.array([func(data[0])] + [0]*(npts-1)) \
    # + np.array([0,func(data[1])] + [0]*(npts-2))
    bc_data[0] += func(xdata[0])
    bc_data[1] += func(xdata[0]) + dx / 2 * (dfunc_data[0] + dfunc_data[1])
    ## noisy if approx, better with higher order
    # bc_data[-1] += func(data[-1]) ## doesn't work: np.trapz(dfunc_data)*dx
    integ_dfunc = np.dot(ddx_tens_inv, dfunc_data)
    plt.figure()
    plt.plot(xdata, integ_dfunc, xdata, func_data, '--')
    plt.show()

    exit()

    # ## not implemented
    # ddx_mpo_inv_2 = inverse.get_mpo_inverse_dmrg(ddx_mpo)
    # ddy_mpo_inv_2 = inverse.get_mpo_inverse_dmrg(ddy_mpo)


