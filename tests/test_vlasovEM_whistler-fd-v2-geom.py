import os, sys, pickle, time, glob

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import scipy.optimize

sys.path.append('../')

from setup_.configs import *
import setup_.helper as helper_test
import helper_quimb as helper
import setup_test as test_setup

from axis import Axis
import axis_map
from basis.basis_spatial import SpatialBasis
from basis.basis_hermite import AWHermiteGaussianBasis
from basis.basis_hermite import HermiteGaussianBasis
from field import Field, ScalarField
from pde_vlasovEM import VlasovMaxwell, Maxwell
from vlasov_tests import Whistler

if TYPE_CHECKING:
    from gridTN import GridTN

"""
Whistler 1D3V
measures rank for desired cutoff error for different geometries
measures error vs specific rank cut-off
looks at spectra of compressed fcts
"""

save_figs = False
save_data = False  # True
load_data = False  # True
restart   = False  # True
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

is_sqrt = False # True
if restart or save_data or save_figs:
    if is_sqrt:
        sdir = '/pool001/erikaye/tns_pde_v2/data15/VM13-whistler-fd-sq-v2/'
        fdir = 'data15/VM13-whistler-fd-sq-v2/'
    else:
        sdir = '/pool001/erikaye/tns_pde_v2/data15/VM13-whistler-fd/'
        fdir = 'data15/VM13-whistler-fd/'

    if not os.path.exists(sdir):
        os.makedirs(sdir)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)

extra_str = ''

#######################
# standard parameters #
#######################

plasma_config = UnitsConfiguration()  # eps0=1.0e-3, mu0=1.0e-3)
charge = plasma_config.e  # unit of charge
eV = plasma_config.eV  # unit of kB*T
mu0 = plasma_config.mu0  # permeability
eps0 = plasma_config.eps0  # permittivity

mass_e = 1.0  # mass of electron
mass_i = 1836 * mass_e  # mass of ion
n0_e = 1.0  # electron number density
n0_i = n0_e  # ion number density
T_e = 1.0e-6  # electron temperature [eV], so that vth_e/c = 1.0e-3
T_i = T_e  # ion temperature [eV]

ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config.vth  # electron thermal speed
wp_e = elc_config.wp  # plasma e' frequency
lamD = elc_config.lamD  # Debye length
wp_i = ion_config.wp  # plasma i+ frequency
vth_i = ion_config.vth  # ion thermal speed
d_e = elc_config.skin_depth

print('vth_e', vth_e, 'vth_i', vth_i, 'c', plasma_config.c)
print('wp_e', wp_e, 'lamD', lamD)
print('d_e', elc_config.skin_depth, 'd_p', ion_config.skin_depth)




def initial_state_k(kx_vals, vy_vals, vz_vals, vth=1.0, density=1.0, flow=0.0):
    """ initial state for w
    """
    dk = kx_vals[1] - kx_vals[0]
    x_vals = np.linspace(-np.pi / dk, np.pi / dk, len(kx_vals), endpoint=False)
    f_real = initial_state_real_vy_vz(x_vals, vy_vals, vz_vals, vth=vth, density=density, flow=flow)
    f_k = np.fft.fftshift(np.fft.fft(f_real, axis=0), axes=[0])

    # plt.figure()
    # plt.imshow(np.real(f_k[:, :, len(vz_vals) // 2]))
    # plt.figure()
    # plt.imshow(np.real(f_k[:, len(vy_vals) // 2, :]))
    # plt.show()
    return f_k


def initial_state_real_vy_vz(x_vals, vy_vals, vz_vals, vth=1.0, density=1.0, flow=0.0):
    """ initial state for w
    """
    x_mesh, vy_mesh, vz_mesh = np.meshgrid(x_vals, vy_vals, vz_vals, indexing='ij')

    dv_y = dvey_func(x_mesh)
    dv_z = dvez_func(x_mesh)
    vy_mesh = vy_mesh - dv_y
    vz_mesh = vz_mesh - dv_z

    out = helper_test.maxwellian_ndim(vy_mesh, vz_mesh,
                                      vth2=vth ** 2, density=density, flow=flow)


    npts = len(vy_vals)

    # plt.figure()
    # plt.imshow(np.real(out[:, npts//2, :]))
    # plt.figure()
    # plt.imshow(np.imag(out[:, npts // 2, :]))
    # plt.colorbar()
    # plt.show()

    return out


def calc_omega(k):
    ## n^2 = (k c / omega)^2 = 1 - wp_e ** 2 / (w + wC_e) / (w + wC_i)
    guess = B0 * k ** 2 / (1 + k ** 2)
    n2 = lambda w: (dk * plasma_config.c / w)**2
    if lr_sign > 0:
        disp = lambda w:  1 - wp_e ** 2 / (w + wC_e) / (w + wC_i)
    else:
        disp = lambda w:  1 - wp_e ** 2 / (w - wC_e) / (w - wC_i)

    out = scipy.optimize.fsolve( lambda w: n2(w) - disp(w),  guess)
    return out.item()



#####################
# define parameters #
#####################

### system parameters ###
A = 1.e-3  # * (vth_e/plasma_config.c)
dk = 10.0 / d_e  # 0.10 / d_e
B0 = 1.0
lr_sign = 1

dvey_coeff = A
dvez_coeff = -A

dvey_func = lambda x: dvey_coeff * np.cos(dk * x)
dvez_func = lambda x: dvez_coeff * np.sin(dk * x)


wC_i = ion_config.cyclotron_frequency(B0)
wC_e = elc_config.cyclotron_frequency(B0)

print('dk / de', dk, d_e)
print('OmegaC_i', ion_config.cyclotron_frequency(B0), 'OmegaC_e', elc_config.cyclotron_frequency(B0))
print('omega', calc_omega(dk), 2 * np.pi / calc_omega(dk))
print('dt max?', 2. * np.pi / wC_e)
print('dt max?', 2. * np.pi / wp_e, wp_e)
# exit()

## runtime parameter
q = 2
KX = 1
KV = 3
# Lx, Lve, Lvi = 6, 7, 6  ## [413] dt = 0.212 - 0.168  [433] similar; EM dt = 0.98
Lx, Lve, Lvi = 5, 5, 5        ## [413] dt = 0.105 - 0.08   [433] similar
# Lx, Lve, Lvi = 7, 7, 7  ## [413] dt = 0.105 - 0.08   [433] similar; EM dt = 0.49
# Lx, Lve, Lvi = 8, 8, 8

order = 1   ## derivative order
if order != 1:
    extra_str = f'_o{order}' + extra_str

def get_layout(idx_: int):

    if idx_ == 0:
        TN_layout = LayoutType.SEQUENTIAL
        x_map_key = 'F'
        v_map_key = 'B'
    elif idx_ == 1:
        TN_layout = LayoutType.SEQUENTIAL
        x_map_key = 'F'
        v_map_key = 'F'
    elif idx_ == 2:
        TN_layout = LayoutType.PARALLEL
        x_map_key = 'F'
        v_map_key = 'F'
    elif idx_ == 3:
        TN_layout = LayoutType.PARALLEL_GROUP
        x_map_key = 'F'
        v_map_key = 'F'
    elif idx_ == 4:
        TN_layout = LayoutType.COMB
        x_map_key = 'F'
        v_map_key = 'F'
    elif idx_ == 5:
        TN_layout = LayoutType.COMB
        x_map_key = 'B'
        v_map_key = 'B'
    else:
        raise NotImplementedError

    return TN_layout, x_map_key, v_map_key


fig_err, ax_err = plt.subplots(1,1)
ax_err.set_title(f'fro. error')
ax_err.set_xlabel('D'), ax_err.set_ylabel('error')

fig_tv, ax_tv = plt.subplots(1,1)
ax_tv.set_title(f'<d2/dx2>')
ax_tv.set_xlabel('D'), ax_tv.set_ylabel('expec')


for idx in range(6):

    grid_idx = 2
    TN_layout, x_map_key, v_map_key = get_layout(idx)

    T = np.round(min(3000, 50 * 2 * np.pi / calc_omega(dk)))  # 50  # 2000 / wp_e
    print('T', T)

    VP_test = Whistler(ks=(dk,), A=A,
                       Lxs=(Lx,), Lves=(Lve, Lve, Lve), Lvis=(Lvi, Lvi, Lvi), KX=KX, KV=KV, q=q,
                       x_map_key=x_map_key, ve_map_key=v_map_key, grid_idx=grid_idx,
                       basis_xs=SpatialBasis(),
                       #
                       ion_config=ion_config, elc_config=elc_config,
                       grid_layout=TN_layout
                       )


    #############################
    ##
    #############################

    ## initial conditions

    ## define axis objects; position -> k-space,
    ax_x, = VP_test.pos_axes
    ax_vx_i, ax_vy_i, ax_vz_i = VP_test.vi_axes
    ax_vx_e, ax_vy_e, ax_vz_e = VP_test.ve_axes
    print('ax_x zero ind', ax_x.zero_ind)

    x_min, x_max = VP_test.x_lims[0]
    print('x range', ax_x.dx, x_min, x_max)
    ve_x_min, ve_x_max = VP_test.ve_lims[0]
    ve_y_min, ve_y_max = VP_test.ve_lims[1]
    ve_z_min, ve_z_max = VP_test.ve_lims[2]
    tot_x = x_max - x_min

    coords_x = VP_test.pos_coordsys
    coords_vi = VP_test.vi_coordsys
    coords_ve = VP_test.ve_coordsys

    X, Y, Z = coords_x.coords
    VX, VY, VZ = coords_ve.coords

    npts_x = ax_x.npts
    npts_vxe = ax_vx_e.npts
    npts_vye = ax_vy_e.npts
    npts_vze = ax_vz_e.npts

    # ## define MPS grids
    grid_i = VP_test.grid_i
    grid_e = VP_test.grid_e
    grid_X = VP_test.grid_X

    x_vals = ax_x.xpts
    vx_e_vals = ax_vx_e.xpts
    vy_e_vals = ax_vy_e.xpts
    vz_e_vals = ax_vz_e.xpts


    ##################################
    ## electron density:  1 + A * np.cos(k * x)

    fe_x_vy_vz = initial_state_real_vy_vz(x_vals, vy_e_vals, vz_e_vals, vth=vth_e, flow=0.0, density=n0_e)
    fe_vx = helper_test.maxwellian(vx_e_vals, vth2=vth_e ** 2, density=1.0, flow=0.0)
    if is_sqrt:
        fe_x_vy_vz = np.sqrt(fe_x_vy_vz)
        fe_vx = np.sqrt(fe_vx)

    ax_inds_list = [(0, 2, 3), (1,)]
    gtn_data_list = [fe_x_vy_vz, fe_vx]
    init_fe_gtn = grid_e.build_mps_from_subgtns(list(zip(ax_inds_list, gtn_data_list)))
    print('init fe max bond', init_fe_gtn.max_bond())

    deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
    deriv_v = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
    init_fe_gtn.ax_deriv_configs.update({ax_x: deriv_x, ax_vx_e: deriv_v, ax_vy_e: deriv_v, ax_vz_e: deriv_v})


    ###
    geom_str = TN_layout.value + x_map_key + v_map_key
    DMAXs = list(np.arange(10, init_fe_gtn.max_bond(), init_fe_gtn.max_bond()//12)) + [init_fe_gtn.max_bond()]
    # DMAXs = list(np.arange(10, init_fe_gtn.max_bond(), 5)) + [init_fe_gtn.max_bond()]
    # DMAXs = [init_fe_gtn.max_bond()]
    cutoffs = [10**x for x in range(-12, -3, 2)]
    print('DMAXs', DMAXs)
    print('cutoffs', cutoffs)

    ddx_0 = grid_e.get_firstderivative_mpo(ax_x)
    ddx_1 = grid_e.get_firstderivative_mpo(ax_vx_e)
    ddx_2 = grid_e.get_firstderivative_mpo(ax_vy_e)
    ddx_3 = grid_e.get_firstderivative_mpo(ax_vz_e)

    d2dx2_0 = ddx_0.apply(ddx_0.transpose(inplace=False).conj())
    d2dx2_1 = ddx_1.apply(ddx_1.transpose(inplace=False).conj())
    d2dx2_2 = ddx_2.apply(ddx_2.transpose(inplace=False).conj())
    d2dx2_3 = ddx_3.apply(ddx_3.transpose(inplace=False).conj())

    # d2dx2_0.data.distribute_exponent()
    # d2dx2_1.data.distribute_exponent()
    # d2dx2_2.data.distribute_exponent()
    # d2dx2_3.data.distribute_exponent()

    # d2dx2_0 = grid_e.get_secondderivative_mpo(ax_x, None)
    # d2dx2_1 = grid_e.get_secondderivative_mpo(ax_vx_e, None)
    # d2dx2_2 = grid_e.get_secondderivative_mpo(ax_vy_e, None)
    # d2dx2_3 = grid_e.get_secondderivative_mpo(ax_vz_e, None)

    vx_mpo = grid_e.get_xmultiply_mpo([ax_vx_e])
    vy_mpo = grid_e.get_xmultiply_mpo([ax_vy_e])
    vz_mpo = grid_e.get_xmultiply_mpo([ax_vz_e])
    v_axes = [ax_vx_e, ax_vy_e, ax_vz_e]

    errs_D, d2dx2_expec_D = [], []
    jxs_D, jys_D, jzs_D = [], [], []
    for DMAX in DMAXs:
        fe_copy = init_fe_gtn.copy()
        # fe_copy.data.distribute_exponent()
        fe_copy.compress(compress_opts={'max_bond': DMAX})

        # if idx < 4:
        #     ### dmrg compress w/o TV
        #     # from local_solvers.local_dmrg_compress_constrained_v3 import local_constrained_compress
        #     # out = local_constrained_compress(fe_copy.data, [], max_bond=DMAX)
        #     # fe_copy.data = out
        #
        #     ### dmrg compress w/ TV
        #     tv0 = fe_copy.meas_expec(d2dx2_0, is_sqrt=True, exclude_weights=True)
        #     tv1 = fe_copy.meas_expec(d2dx2_1, is_sqrt=True, exclude_weights=True)
        #     tv2 = fe_copy.meas_expec(d2dx2_2, is_sqrt=True, exclude_weights=True)
        #     tv3 = fe_copy.meas_expec(d2dx2_3, is_sqrt=True, exclude_weights=True)
        #     # tv0 = helper.expectation_value(fe_copy.data, d2dx2_0.data)
        #     # tv1 = helper.expectation_value(fe_copy.data, d2dx2_1.data)
        #     # tv2 = helper.expectation_value(fe_copy.data, d2dx2_2.data)
        #     # tv3 = helper.expectation_value(fe_copy.data, d2dx2_3.data)
        #     print('tvs', tv0, tv1, tv2, tv3)
        #
        #     from local_solvers.local_dmrg_compress_constrained_v3 import local_constrained_compress
        #
        #     geq_constraints = [(d2dx2_0.data, None, tv0), (d2dx2_1.data, None, tv1),
        #                        (d2dx2_2.data, None, tv2), (d2dx2_3.data, None, tv3)]
        #     out = local_constrained_compress(fe_copy.data, [], geq_constraints=geq_constraints, max_bond=DMAX,)
        #     fe_copy.data = out
        #
        # else:
        #     fe_copy.compress(compress_opts={'max_bond': DMAX})

        errs_D += [init_fe_gtn.distance(fe_copy) / init_fe_gtn.frobenius_norm()]

        # d2dx2_norm = fe_copy.meas_expec(d2dx2_0, is_sqrt=True) + fe_copy.meas_expec(d2dx2_1, is_sqrt=True) + \
        #              fe_copy.meas_expec(d2dx2_2, is_sqrt=True) + fe_copy.meas_expec(d2dx2_3, is_sqrt=True)
        # print('d2dx2', d2dx2_norm)
        # d2dx2_expec_D += [d2dx2_norm]

        fe_ddx0 = fe_copy.take_firstderivative(ax_x)
        fe_ddx1 = fe_copy.take_firstderivative(ax_vx_e)
        fe_ddx2 = fe_copy.take_firstderivative(ax_vy_e)
        fe_ddx3 = fe_copy.take_firstderivative(ax_vz_e)
        d2dx2_norm = fe_ddx1.frobenius_norm() + fe_ddx3.frobenius_norm() + \
                     fe_ddx0.frobenius_norm() + fe_ddx2.frobenius_norm()
        d2dx2_expec_D += [d2dx2_norm]

        ## measure current and measure error in current
        jxe: 'GridTN' = fe_copy.meas_expec(vx_mpo, integ_axes=v_axes, new_grid=grid_X)
        jye: 'GridTN' = fe_copy.meas_expec(vy_mpo, integ_axes=v_axes, new_grid=grid_X)
        jze: 'GridTN' = fe_copy.meas_expec(vz_mpo, integ_axes=v_axes, new_grid=grid_X)

        jxe_data = jxe.get_data()
        jye_data = jye.get_data()
        jze_data = jze.get_data()

        jxs_D += [jxe_data]
        jys_D += [jye_data]
        jzs_D += [jze_data]

    ax_err.semilogy(DMAXs, errs_D, 'o-', label=f'{geom_str}')
    ax_tv.semilogy(DMAXs, np.abs(d2dx2_expec_D), 'x-', label=f'{geom_str}')

    k_vals = np.linspace(-np.pi/tot_x, np.pi/tot_x, 2**Lx)

    # for compID, js in zip([X, Y, Z], [jxs_D, jys_D, jzs_D]):
    #     fig1, ax1 = plt.subplots(1,1)
    #     fig2, ax2s = plt.subplots(2,1, sharex=True)
    #
    #     for DMAX, j_data in zip(DMAXs, js):
    #         j_data_k = np.fft.fftshift(np.fft.fft(j_data))
    #         ax1.plot(x_vals, j_data, label=f'D={DMAX}')
    #         ax2s[0].semilogy(k_vals, np.abs(np.real(j_data_k)), label=f'D={DMAX}')
    #         ax2s[1].semilogy(k_vals, np.abs(np.imag(j_data_k)), label=f'D={DMAX}')
    #
    #     ax1.set_xlabel('x'), ax1.set_ylabel('J')
    #     ax1.legend()
    #     ax2s[0].set_xlabel('x'), ax2s[0].set_ylabel('|re FFT(J)|')
    #     ax2s[1].set_xlabel('x'), ax2s[1].set_ylabel('|im FFT(J)|')
    #     ax1.set_title(f'j {compID}, {geom_str}')
    #     ax2s[0].set_title(f'FFT(j) {compID}, {geom_str}')
    #     # ax2s[1].set_title(f'im FFT(j) {compID}, {geom_str}')
    #     ax2s[0].legend(bbox_to_anchor=(1.05,0.95), loc='upper left')
    #
    #     fig2.subplots_adjust(right=0.50)


ax_err.legend()
ax_tv.legend()

plt.show()

