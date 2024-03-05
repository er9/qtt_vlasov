import os, sys, pickle, time, glob

import numpy as np
import matplotlib

sys.path.append('../')

from setup.configs import *
import setup.helper as helper_test
import helper_quimb as helper
import setup_test as test_setup

from axis import Axis
import axis_map
from basis.basis_k import FourierBasis
from basis.basis_spatial import SpatialBasis
from basis.basis_hermite import AWHermiteGaussianBasis
from basis.basis_hermite import HermiteGaussianBasis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from grid_comb import GridsComb
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_vlasovEM import VlasovMaxwell, Maxwell
from vlasov_tests import OrszagTang

"""
Landau damping (k-space in space, real space in velocity)
"""

if __name__ == '__main__':

    save_figs = False
    save_data = False  # True
    load_data = False  # True
    restart   = False  # True
    restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

    is_sqrt = True

    extra_str = ''
    if is_sqrt:
        extra_str += '_sq'

    if restart or save_data or save_figs:
        if is_sqrt:
            ## v1: do_adapt = True
            # sdir = '/pool001/erikaye/tns_pde_v2/data17/VM23-ot-fd-sq/'
            # fdir = 'data17/VM23-ot-fd-sq/'
            ## v1: do_adapt = False
            # sdir = '/pool001/erikaye/tns_pde_v2/data17/VM23-ot-fd-sq-v2/'
            # fdir = 'data17/VM23-ot-fd-sq-v2/'
            ## don't smooth B
            sdir = '/pool001/erikaye/tns_pde_v2/data17/VM23-ot-fd-sq-v2B/'
            fdir = 'data17/VM23-ot-fd-sq-v2B/'
        else:
            sdir = '/pool001/erikaye/tns_pde_v2/data17/VM23-ot-fd/'
            fdir = 'data17/VM23-ot-fd/'
        sdir_bkp = sdir + 'bkp/'

        # sdir = '/mnt/c/Users/eyeri/wsl/code/tns/psfc_data/tns_pde_v2/data/VM23_ot/'
        if not os.path.exists(sdir):
            os.makedirs(sdir)

        if not os.path.exists(sdir_bkp):
            os.makedirs(sdir_bkp)

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        print('fdir', fdir, sdir)

    #######################
    # standard parameters #
    #######################

    ## Wc_i = Bz0 * Z / m_i = 1 --> Bz0 = 1, m_i = 1 --> m_e = 1/25
    ## vA_i = Bz0 / np.sqrt(mu0 n0 m_i) = 1  --> mu0 = 1
    ## vA_e = Bz0 / np.sqrt(mu0 n0 m_i / 25) = 5 Bz0 / np.sqrt(mu0 n0 m_i) = 0.1 / sqrt(eps0 mu0)
    ##   -->  5 Bz0 sqrt(eps0) / sqrt(n0 m_i) = 0.1
    ## wp_e = Z * sqrt(n0 / eps0 / mass)
    ## Wc_e = Bz0 * Z / m_e = 2 * 625 / 50 = 2 * 125 / 10 = 25
    ## Wc_e / wp_e = Bz0 / m_e * sqrt(eps0 * m_e / n0) = 0.1
    ##    --> Bz0 sqrt(eps0) / sqrt(m_e n0) = 0.1
    ##    --> 5 Bz0 sqrt(eps0) / sqrt(m_i n0) = 0.1  (same as above)
    ## --> 5 sqrt(eps0) = 0.1 --> eps0 = 0.02 ** 2
    ## --> c = 1 / sqrt(mu0 eps0) = 1 / 0.02 = 50

    ## vth_i = 0.2 = sqrt(Z kB T_i / m_i) --> T_i / m_i = 0.04 --> T_i = 0.04
    ## vth_e = 5 = sqrt(Z kB T_e / m_e) --> 25 * 1 / 25**2 = T_e --> T_e = 1 / 25

    Bg = 1.0

    plasma_config = UnitsConfiguration(eps0=1 / 2500, mu0=1)  # eps0 * mu0 = 1/2500
    charge = plasma_config.e  # unit of charge
    eV = plasma_config.eV  # unit of kB*T
    mu0 = plasma_config.mu0  # permeability
    eps0 = plasma_config.eps0  # permittivity

    mass_i = 1  # mass of ion
    mass_e = mass_i / 25.0  # mass of electron
    n0_e = 1.0  # electron number density
    n0_i = n0_e  # ion number density
    T_e = 0.04  # 1.0  # electron temperature [eV]
    T_i = 0.04  # ion temperature [eV], vth_i = 0.2

    ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
    elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
    vth_e = elc_config.vth  # electron thermal speed, sqrt(Z kB T / m)
    wp_e = elc_config.wp  # plasma e' frequency
    lamD = elc_config.lamD  # Debye length
    wp_i = ion_config.wp  # plasma i+ frequency
    vth_i = ion_config.vth  # ion thermal speed, sqrt(Z kB T / m) = 0.2 --> T = 1.0
    d_e = elc_config.skin_depth  # c / wp_e
    d_p = ion_config.skin_depth  # c / wp_i

    print('c', plasma_config.c)
    print('vth_e', vth_e, 'vth_i', vth_i)
    print('wp_e', wp_e, 'Wc_e', elc_config.cyclotron_frequency(Bg))
    print('wp_i', wp_i, 'Wc_i', ion_config.cyclotron_frequency(Bg))


    def initial_state_real(x_vals, y_vals, vx_vals, vy_vals, vz_vals, A, dkx, dky,
                           vth=1.0, density=1.0, flow=0.0):
        """ initial state for w
        """
        # x_mesh, vx_mesh = np.meshgrid(x_vals, vx_vals, indexing='ij')
        # y_mesh, vy_mesh = np.meshgrid(y_vals, vy_vals, indexing='ij')

        x_mesh, y_mesh, vx_mesh, vy_mesh, vz_mesh = \
            np.meshgrid(x_vals, y_vals, vx_vals, vy_vals, vz_vals, indexing='ij')

        dv_x = -A * np.sin(dky * y_mesh)
        dv_y = +A * np.cos(dkx * x_mesh)
        vx_mesh = vx_mesh - dv_x
        vy_mesh = vy_mesh - dv_y

        out = helper_test.maxwellian_ndim(vx_mesh, vy_mesh, vz_mesh,
                                          vth2=vth ** 2, density=density, flow=flow)

        plt.figure()
        plt.imshow(out[:, 0, len(vx_vals) // 2, :, len(vz_vals) // 2], aspect='auto')
        plt.title('f(x,vy)')
        plt.colorbar()

        plt.figure()
        plt.imshow(out[0, :, :, len(vy_vals) // 2, len(vz_vals) // 2], aspect='auto')
        plt.title('f(y,vx)')
        plt.colorbar()
        plt.show()

        return out


    def initial_state_k(kx_vals, ky_vals, vx_vals, vy_vals, vz_vals, A, k, vth=1.0, density=1.0, flow=0.0):
        """ initial state for w
        """
        dkx = kx_vals[1] - kx_vals[0]
        dky = ky_vals[1] - ky_vals[0]
        x_vals = np.linspace(-np.pi / dkx, np.pi / dkx, len(kx_vals), endpoint=False)
        y_vals = np.linspace(-np.pi / dky, np.pi / dky, len(ky_vals), endpoint=False)
        f_real = initial_state_real(x_vals, y_vals, vx_vals, vy_vals, vz_vals, A, k, vth=vth, density=density,
                                    flow=flow)
        f_k = np.fft.fftshift(np.fft.fft(f_real, axis=0), axes=[0])
        f_k = np.fft.fftshift(np.fft.fft(f_k, axis=1), axes=[1])

        plt.figure()
        plt.imshow(np.real(f_k[:, len(vx_vals) // 2, :, len(vz_vals) // 2]))
        plt.figure()
        plt.imshow(np.real(f_k[:, :, len(vy_vals) // 2, len(vz_vals) // 2]))
        plt.show()
        return f_k


    def initial_state_ion_x(x_vals, vy_vals, A, dkx, vth=1.0, density=1.0, flow=0.0):
        """ initial state for w
        """
        x_mesh, vy_mesh = np.meshgrid(x_vals, vy_vals, indexing='ij')

        dv_y = ui_y(x_mesh)
        # dv_y = +A * np.sin(dkx * x_mesh)
        vy_mesh = vy_mesh - dv_y

        out = helper_test.maxwellian_ndim(vy_mesh, vth2=vth ** 2, density=density, flow=flow)
        return out


    def initial_state_ion_y(y_vals, vx_vals, A, dky, vth=1.0, density=1.0, flow=0.0):
        """ initial state for w
        """
        y_mesh, vx_mesh = np.meshgrid(y_vals, vx_vals, indexing='ij')

        # dv_x = -A * np.sin(dky * y_mesh)
        dv_x = ui_x(y_mesh)
        vx_mesh = vx_mesh - dv_x

        out = helper_test.maxwellian_ndim(vx_mesh, vth2=vth ** 2, density=density, flow=flow)
        return out


    def initial_state_elec_z(x_vals, y_vals, vz_vals, vth=1.0, density=1.0):
        """ initial state for w
        """
        x_mesh, y_mesh, vz_mesh = np.meshgrid(x_vals, y_vals, vz_vals, indexing='ij')

        dv_z = ue_z(x_mesh, y_mesh)
        vz_mesh = vz_mesh - dv_z

        out = helper_test.maxwellian_ndim(vz_mesh, vth2=vth ** 2, density=density, flow=0.0)
        return out


    #####################
    # define parameters #
    #####################

    ### system parameters ###
    vA_i = ion_config.alfven_speed(Bg)  ## = 1
    u0 = 0.2 * vA_i
    B0 = 0.2 * Bg
    # u0 = 0.5 * vA_i
    # B0 = 0.5 * Bg
    # dk = 0.10 / lamD
    dkx = 2 * np.pi / 20.48 / d_p
    dky = 2 * np.pi / 20.48 / d_p
    print('d_p', d_p, 'd_e', d_e)

    print('vA_i', vA_i)
    print('vA_e', elc_config.alfven_speed(Bg), plasma_config.c)
    print('beta_i', ion_config.beta(Bg))
    print('beta_e', elc_config.beta(Bg))
    print(ion_config.skin_depth)
    print('debye length', elc_config.lamD, ion_config.lamD)
    print('d_p', d_p, 'd_e', d_e, 'lamD', lamD)
    print('rho_p', ion_config.larmor_radius(Bg, 0.2), 'rho_e', elc_config.larmor_radius(Bg, 0.2))

    print('dt max?', 2. * np.pi / elc_config.cyclotron_frequency(Bg))
    print('dt max?', 2. * np.pi / elc_config.plasma_frequency, elc_config.wp)

    ## runtime parameter
    q = 2
    KX = 2
    KV = 3
    # Lx, Lve, Lvi = 6, 4, 4
    # Lx, Lve, Lvi = 6, 6, 6        ## dt=0.0025, SL MPO 13 / 5
    # Lx, Lve, Lvi = 8, 6, 6          ## dt=0.0025, SL MPO 13 / 5 still
    # Lx, Lve, Lvi = 7, 7, 7
    Lx, Lve, Lvi = 9, 6, 6  # to resolve d_e
    # Lx, Lve, Lvi = 9, 7, 7  # to resolve d_e

    # TN_layout = LayoutType.SEQUENTIAL
    # x_map_key = 'F'
    # v_map_key = 'B'

    # TN_layout = LayoutType.PARALLEL
    # x_map_key = 'F'
    # v_map_key = 'M'

    TN_layout = LayoutType.COMB
    x_map_key = 'B'
    v_map_key = 'B'

    order = 1  # 1  ## derivative order
    if order > 1:
        extra_str = f'_o{order}' + extra_str
    T = 40
    print('T', T)

    do_adapt_dt = False      # False  # True
    dt_frac = 4.0
    te_order = 3164  # 4164   # 4164 # 333
    dt_max = 0.005  # 0.25 * 2. * np.pi / max(np.abs(elc_config.cyclotron_frequency(Bg)), np.abs(elc_config.wp))
    dt = 0.002

    # do_adapt_dt = False  # True
    # te_order = 4164
    # dt = 0.001  # 0.0005  # None  # 0.05 # None # 0.005
    # dt_frac = None
    # dt_max = dt   # 0.25 * 2. * np.pi / elc_config.wp

    te_order_EM = 33 #223
    is_yee = True if te_order_EM == 31 else False

    save_every_nt = 50 if TN_layout == LayoutType.PARALLEL else 100  # 250

    DMAX = 16   # 128 # 64
    DMAX_F = DMAX
    comp_style = 3

    DMAX_EM = DMAX
    if DMAX_EM != DMAX:
       extra_str = f'_DE{DMAX_EM}' + extra_str

    VP_test = OrszagTang(ks=(dkx, dky), B0=B0, Bg=Bg, u0=u0,
                         Lxs=(Lx, Lx), Lves=(Lve, Lve, Lve), Lvis=(Lvi, Lvi, Lvi), KX=KX, KV=KV, q=q,
                         x_map_key=x_map_key, ve_map_key=v_map_key,
                         basis_xs=SpatialBasis(),
                         #
                         ion_config=ion_config, elc_config=elc_config,
                         coll_type=None, do_adapt_coll=True, coll_coeff_e=0.005, coll_rate_e=0.01,
                         DMAX=DMAX, DMAX_F=DMAX_F, compress_style=comp_style, compress_style_mod=0,
                         do_adapt_dt=do_adapt_dt, dt_frac=dt_frac, dt=dt, T=T, te_order=te_order,
                         te_order_EM=te_order_EM,
                         grid_layout=TN_layout
                         )

    te_compress_i = VP_test.get_compression_config(DMAX=DMAX)
    te_compress_e = VP_test.get_compression_config(DMAX=DMAX)
    te_compress_E = VP_test.get_compression_config(DMAX=DMAX_EM)
    te_compress_B = VP_test.get_compression_config(DMAX=DMAX_EM)

    print('te compress fe', te_compress_e)
    print('te compress fi', te_compress_i)
    print('te compress E', te_compress_E)
    print('te compress B', te_compress_B)

    comp_levels = VP_test.get_compress_levels()

    print('compress', DMAX, VP_test.compress_style, VP_test.compress_style_mod)

    #############################
    ## start building PDE system
    #############################

    ## initial conditions

    ## define axis objects; position -> k-space,
    ax_x, ax_y = VP_test.pos_axes
    ax_vx_i, ax_vy_i, ax_vz_i = VP_test.vi_axes
    ax_vx_e, ax_vy_e, ax_vz_e = VP_test.ve_axes
    print('ax_x zero ind', ax_x.zero_ind)

    x_min, x_max = VP_test.x_lims[0]
    y_min, y_max = VP_test.x_lims[1]
    print('x range', ax_x.dx, x_min, x_max)
    print('vxe range', ax_vx_e.dx, vth_e)
    print('vxi range', ax_vx_i.dx, vth_i)
    ve_x_min, ve_x_max = VP_test.ve_lims[0]
    ve_y_min, ve_y_max = VP_test.ve_lims[1]
    ve_z_min, ve_z_max = VP_test.ve_lims[2]
    vi_x_min, vi_x_max = VP_test.vi_lims[0]
    vi_y_min, vi_y_max = VP_test.vi_lims[1]
    vi_z_min, vi_z_max = VP_test.vi_lims[2]
    tot_x = x_max - x_min
    tot_y = y_max - y_min
    tot_V = tot_x * tot_y

    exit()

    coords_x = VP_test.pos_coordsys
    coords_vi = VP_test.vi_coordsys
    coords_ve = VP_test.ve_coordsys

    X, Y, Z = coords_x.coords
    VX, VY, VZ = coords_ve.coords

    npts_x = ax_x.npts
    npts_y = ax_y.npts
    npts_vxe = ax_vx_e.npts
    npts_vye = ax_vy_e.npts
    npts_vze = ax_vz_e.npts
    npts_vxi = ax_vx_i.npts
    npts_vyi = ax_vy_i.npts
    npts_vzi = ax_vz_i.npts

    # ## define MPS grids
    grid_i = VP_test.grid_i
    grid_e = VP_test.grid_e
    grid_X = VP_test.grid_X

    x_vals = ax_x.xpts
    xB_vals = x_vals + ax_x.dx / 2 if is_yee else x_vals # for staggered Yee cell
    y_vals = ax_y.xpts
    yB_vals = y_vals + ax_y.dx / 2 if is_yee else y_vals # for staggered Yee cell

    vx_e_vals = ax_vx_e.xpts
    vy_e_vals = ax_vy_e.xpts
    vz_e_vals = ax_vz_e.xpts
    vx_i_vals = ax_vx_i.xpts
    vy_i_vals = ax_vy_i.xpts
    vz_i_vals = ax_vz_i.xpts

    ########
    ## B = - B0 * sin(2*pi*y/Ly) [x] + B0 * sin(4*pi*x/Lx) [y] + Bg [z]
    ## up = - u0 * sin(2*pi*y/Ly) [x] + u0 * sin(2*pi*x/Lx) [y]
    ## J = 1/mu0 curl B = e * n0 (up - ue) = [z] (B0 * 2*pi/Ly cos(2*pi*y/Ly) + B0 * 4*pi/Lx cos(4*pi*x/Lx))
    ## --> ue = - u0 * sin(2*pi*y/Ly) [x] + u0 * sin(2*pi*x/Lx) [y] - B0/mu0/n0/e (...)
    ## E = -ue x B = [x] u0 * sin(2*pi*x/Lx) * Bg - ue[z] *
    Bx = lambda y: - B0 * np.sin(dky * y)
    By = lambda x: B0 * np.sin(2 * dkx * x)
    ui_x = lambda y: -u0 * np.sin(dky * y)
    ui_y = lambda x: u0 * np.sin(dkx * x)
    ue_x, ue_y = ui_x, ui_y
    ue_z = lambda x, y: - B0 / mu0 / charge / n0_e * (dky * np.cos(dky * y) + 2 * dkx * np.cos(2 * dkx * x))
    Ex = lambda x, y: - ue_y(x) * Bg + ue_z(x, y) * By(x)
    Ey = lambda x, y: - ue_z(x, y) * Bx(y) + ue_x(y) * Bg
    Ez = lambda x, y: - ue_x(y) * By(x) + ue_y(x) * Bx(y)

    ## ion density
    fi_x_vy = initial_state_ion_x(x_vals, vy_i_vals, u0, dkx, vth=vth_i, density=1.0)
    fi_y_vx = initial_state_ion_y(y_vals, vx_i_vals, u0, dky, vth=vth_i, density=1.0)
    fi_vz = helper_test.maxwellian(vz_i_vals, vth2=vth_i ** 2, density=n0_i, flow=0.0)
    if is_sqrt:
        fi_x_vy = np.sqrt(fi_x_vy)
        fi_y_vx = np.sqrt(fi_y_vx)
        fi_vz = np.sqrt(fi_vz)

    ax_inds_list = [(0, 3), (1, 2), (4,)]
    gtn_data_list = [fi_x_vy, fi_y_vx, fi_vz]
    comp_opts_i = te_compress_i.get_compress_opts(1) if TN_layout == LayoutType.PARALLEL else None
    init_fi_gtn = grid_i.build_mps_from_subgtns(list(zip(ax_inds_list, gtn_data_list)),
                                                compress_opts=comp_opts_i)

    ## electron density
    fe_x_vy = initial_state_ion_x(x_vals, vy_e_vals, u0, dkx, vth=vth_e, density=n0_e)
    fe_y_vx = initial_state_ion_y(y_vals, vx_e_vals, u0, dky, vth=vth_e, density=1.0)
    fe_vz = initial_state_elec_z(x_vals, y_vals, vz_e_vals, vth=vth_e, density=1.0)
    if is_sqrt:
        fe_x_vy = np.sqrt(fe_x_vy)
        fe_y_vx = np.sqrt(fe_y_vx)
        fe_vz = np.sqrt(fe_vz)

    ax_inds_list = [(0, 3), (1, 2), (0, 1, 4)]
    gtn_data_list = [fe_x_vy, fe_y_vx, fe_vz]
    comp_opts_e = te_compress_e.get_compress_opts(1) if TN_layout == LayoutType.PARALLEL else None
    init_fe_gtn = grid_e.build_mps_from_subgtns(list(zip(ax_inds_list, gtn_data_list)),
                                                compress_opts=comp_opts_e)

    ## E field:
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing='ij')
    Ex_vals = Ex(x_mesh, y_mesh)
    Ey_vals = Ey(x_mesh, y_mesh)
    Ez_vals = Ez(x_mesh, y_mesh)

    ## B field:
    # Bx = - B0 * np.sin(dky * yB_vals)
    # By = + B0 * np.sin(2 * dkx * xB_vals)
    Bx_vals = Bx(yB_vals)
    By_vals = By(xB_vals)

    ### B0
    Bz0_gtn = ax_x.get_iden_mps()
    helper.scalar_multiply(Bz0_gtn, Bg, inplace=True)

    ## define initial grid1DTNs
    init_Ex_gtn = grid_X.map_state_to_mps(Ex_vals)
    init_Ey_gtn = grid_X.map_state_to_mps(Ey_vals)
    init_Ez_gtn = grid_X.map_state_to_mps(Ez_vals)
    # init_Ez_gtn = grid_X.make_empty_gridTN()
    init_Bz0_gtn = grid_X.make_mps_ndim({ax_x: Bz0_gtn})
    init_Bz0_gtn.is_constant = True
    init_Bz_gtn = grid_X.make_empty_gridTN()
    init_By_gtn = grid_X.make_mps_ndim({ax_x: By_vals})
    init_Bx_gtn = grid_X.make_mps_ndim({ax_y: Bx_vals})
    init_phi_gtn = grid_X.make_empty_gridTN()
    init_psi_gtn = grid_X.make_empty_gridTN()

    deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
    deriv_y = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
    if te_order in [3154, 3150, 3164, 3160, 4154, 4150, 4164, 4160, 64, 60]:
        deriv_v = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
    else:
        deriv_v = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)
    init_fe_gtn.ax_deriv_configs.update(
        {ax_x: deriv_x, ax_y: deriv_y, ax_vx_e: deriv_v, ax_vy_e: deriv_v, ax_vz_e: deriv_v})
    init_fi_gtn.ax_deriv_configs.update(
        {ax_x: deriv_x, ax_y: deriv_y, ax_vx_i: deriv_v, ax_vy_i: deriv_v, ax_vz_i: deriv_v})
    deriv_x_EM = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
    deriv_y_EM = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
    init_Ex_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM, ax_y: deriv_y_EM})
    init_Ey_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM, ax_y: deriv_y_EM})
    init_Ez_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM, ax_y: deriv_y_EM})
    init_Bx_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM, ax_y: deriv_y_EM})
    init_By_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM, ax_y: deriv_y_EM})
    init_Bz_gtn.ax_deriv_configs.update({ax_x: deriv_x_EM, ax_y: deriv_y_EM})

    ### time step info
    if str(te_order_EM)[:2] not in ['21', '22', '33']:
        dt_max = min(dt_max, 0.1 * ax_x.dx / plasma_config.c)
    print('EM dt max', dt_max, ax_x.dx / plasma_config.c, ax_x.dx, plasma_config.c)

    if te_order in [410, 413, 414, 416, 4164]:
        Fmax_e = B0 * np.array([ve_x_max, ve_z_max, ve_y_max]) * charge / mass_e
        Fmax_i = B0 * np.array([vi_x_max, vi_z_max, vi_y_max]) * charge / mass_i
    else:
        Fmax_e = Bg * np.array([0, ve_z_max, ve_y_max]) * charge / mass_e
        Fmax_i = Bg * np.array([0, vi_z_max, vi_y_max]) * charge / mass_i
    do_advec_x_cfl = False if (te_order in [410, 413, 414, 416, 4164, 310, 313, 314, 316, 3164]) else True
    print('B0', B0, Bg, Fmax_e)
    dt_cfl = VP_test.get_dt(Fmax_e, Fmax_i, do_advec_x_cfl=do_advec_x_cfl)

    dt = dt_cfl if dt is None or do_adapt_dt else dt
    dt = min(dt_max, dt)
    num_tsteps = int(T / dt)
    print('dt', dt, num_tsteps)

    ### get file naming string
    fstr = VP_test.get_fstr(extra_str=extra_str)
    print('filename', fstr)

    ########################
    #### run simulation ####
    ########################

    ### define v2 mpo for measurement ###
    # vx2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vx_e], x_power=2)
    # vy2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vy_e], x_power=2)
    # vz2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vz_e], x_power=2)
    # v2e_mpo = vx2e_mpo.add(vy2e_mpo)
    # v2e_mpo = v2e_mpo.add(vz2e_mpo)
    v2e_mpo = test_setup.get_v2_mpo(grid_e, VP_test.ve_axes)

    # vx2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vx_i], x_power=2)
    # vy2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vy_i], x_power=2)
    # vz2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vz_i], x_power=2)
    # v2i_mpo = vx2i_mpo.add(vy2i_mpo)
    # v2i_mpo = v2i_mpo.add(vz2i_mpo)
    v2i_mpo = test_setup.get_v2_mpo(grid_i, VP_test.vi_axes)

    #### define fields
    init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, is_sqrt=is_sqrt, compress_config=te_compress_e)
    init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, is_sqrt=is_sqrt, compress_config=te_compress_i)
    init_E_field = Field('E', grid_X, data={X: init_Ex_gtn, Y: init_Ey_gtn, Z: init_Ez_gtn},
                         is_sqrt=True, compress_config=te_compress_E)
    init_B_field = Field('B', grid_X, data={X: init_Bx_gtn, Y: init_By_gtn, Z: init_Bz_gtn},
                         is_sqrt=True, compress_config=te_compress_B)
    B0_field = Field('B0', grid_X, data={Z: init_Bz0_gtn})
    curlB0_field = Field('B0', grid_X, data={})
    init_phi_field = ScalarField('phi', grid_X, data=init_phi_gtn, is_sqrt=True, compress_config=te_compress_E)
    init_psi_field = ScalarField('psi', grid_X, data=init_psi_gtn, is_sqrt=True, compress_config=te_compress_B)

    if TN_layout == LayoutType.PARALLEL:
        init_fe_field.compress()
        init_fi_field.compress()
        init_E_field.compress()
        init_B_field.compress()
        print('fe max bond', init_fe_field.max_bond())


    ### load data if restarting ###
    try:
        out = test_setup.load_data_EM(VP_test,
                                      init_fe_field, init_fi_field, init_E_field, init_B_field, init_phi_field,
                                      init_psi_field,
                                      fdir=fdir, sdir=sdir, extra_str=extra_str, save_every_nt=save_every_nt,
                                      load_data=load_data, restart=restart, restart_from_T=restart_from_T,
                                      )

        (init_fe_field, init_fi_field, init_E_field, init_B_field, init_phi_field, init_psi_field,
         ts, nrg_fe_ts, nrg_fi_ts, nrg_E_ts, nrg_B_ts,
         max_bond_fe, max_bond_fi, max_bond_E, max_bond_B, max_bond_phi, max_bond_psi) = out

        nt = len(ts) - 1
        print('loaded nt', nt)
        restarted = True

    except(IOError, OSError, NameError, ValueError, FileNotFoundError):

        ts = [0]
        nt = 0
        restarted = False

        nrg_fe = init_fe_gtn.meas_expec(v2e_mpo, is_sqrt=is_sqrt) * mass_e / 2 / tot_V
        nrg_fi = init_fi_gtn.meas_expec(v2i_mpo, is_sqrt=is_sqrt) * mass_i / 2 / tot_V
        nrg_E  = init_E_field.norm() ** 2 * 1. / 2 / tot_V
        nrg_B  = init_B_field.norm() ** 2 * 1. / 2 / tot_V
        # print('nrg E', nrg_E)
        # print('nrg B', nrg_B)

        nrg_E = np.array([np.nan] * 3)
        # nrg_B  = [np.nan] * 3 # A / dk
        nrg_B = np.array(init_B_field.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_V
        print('tot_x', dkx, 2 * np.pi / dkx, tot_x)
        print('tot_y', dky, 2 * np.pi / dky, tot_y)

        ## maybe should initialize system, run EM_sys for half time step to start

        nrg_fe_ts = [nrg_fe]
        nrg_fi_ts = [nrg_fi]
        nrg_E_ts = [nrg_E]
        nrg_B_ts = [nrg_B]
        # nrg_By_ts = [nrg_B]

        print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
        max_bond_fe = [init_fe_gtn.max_bond()]
        max_bond_fi = [init_fi_gtn.max_bond()]
        max_bond_E = [init_E_field.max_bonds(compIDs=coords_x.coords) if init_E_field is not None else np.nan]
        max_bond_B = [init_B_field.max_bonds(compIDs=coords_x.coords) if init_B_field is not None else np.nan]
        max_bond_phi = [init_phi_field.max_bond() if init_phi_field is not None else np.nan]
        max_bond_psi = [init_psi_field.max_bond() if init_psi_field is not None else np.nan]

    print('current nrgs', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1], nrg_B_ts[-1])

    norm_fe = init_fe_gtn.norm(is_sqrt=is_sqrt)
    norm_fi = init_fi_gtn.norm(is_sqrt=is_sqrt)
    if not is_sqrt:
        print('field norm', norm_fe, norm_fi, 'tot V', tot_V)
    else:
        print('field norm', norm_fe ** 2, norm_fi ** 2, 'tot V', tot_V)

    ### BUILD PDE
    em_sys = Maxwell(init_E_field, init_B_field, phi=init_phi_field, psi=init_psi_field,
                     coords_x=coords_x, matl_params=plasma_config, te_order=te_order_EM,
                     background_B0=B0_field, normalize=False,
                     is_electrostatic=False, clean=False, is_yee=is_yee)
    em_sys.curlB0 = curlB0_field

    vm_sys = VlasovMaxwell(init_fe_field, init_fi_field, em_sys,
                           coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                           elc_params=elc_config, ion_params=ion_config, normalize=True, zipup=True,
                           te_order=te_order, compress_levels=comp_levels, )
    print('initialized vm_sys')

    vm_sys.collision = VP_test.collision_config
    print('colls', vm_sys.collision.coll_params)

    vm_sys.compress_F = VP_test.compress_F
    vm_sys.compress_F_opts = VP_test.compress_F_opts
    print('compress E?', vm_sys.compress_F, vm_sys.compress_F_opts)

    print('E nrg', vm_sys.E.norm() ** 2 * 1. / 2 / tot_V)
    print('B nrg', vm_sys.B.norm() ** 2 * 1. / 2 / tot_V)

    print('fe bond dims', vm_sys.fe.component.all_virtual_sizes())

    # ### plot current density
    # j_field = vm_sys.compute_current()
    # for i in range(3):
    #     compID = em_sys.coords_x.type_coords[i]
    #     EC = em_sys.E.get_comp_data(compID)
    #     EC = ax_x.basis.get_realspace_1D(EC, 0)
    #     EC = ax_y.basis.get_realspace_1D(EC, 1)
    #     if EC is not None:
    #         plt.figure()
    #         plt.imshow(np.real(EC))
    #         plt.title(f'E{compID}')
    #         plt.colorbar()
    # for i in range(3):
    #     compID = em_sys.coords_x.type_coords[i]
    #     BC = em_sys.B.get_comp_data(compID)
    #     BC = ax_x.basis.get_realspace_1D(BC, 0)
    #     BC = ax_y.basis.get_realspace_1D(BC, 1)
    #     if BC is not None:
    #         plt.figure()
    #         plt.imshow(np.real(BC))
    #         plt.title(f'B{compID}')
    #         plt.colorbar()
    # plt.show()
    # for C in coords_x.coords:
    #     fig1, ax1 = plt.subplots(1)
    #     current_data = j_field.get_comp_data(C)
    #     if current_data is not None:
    #         if KX == 1:
    #             ax1.plot(current_data)
    #             ax1.xlabel('x')
    #             ax1.ylabel(f'current density {C}')
    #         if KX == 2:
    #             im1 = ax1.imshow(current_data)
    #             fig1.colorbar(im1, ax=ax1)
    #             ax1.set_xlabel('y'), ax1.set_ylabel('x')
    #             ax1.set_title(f'current density {C}')
    # plt.show()

    if nt == 0:
        # fig1, ax1 = plt.subplots(2, 1)
        # time_ = nt * dt
        # fi_data = init_fi_gtn.get_data(
        #     ax_select={ax_y: 0, ax_vx_i: ax_vx_i.npts // 2, ax_vz_i: ax_vz_i.npts // 2})
        # fi_data = ax_x.basis.get_realspace_1D(fi_data, 0)
        # fi_data = ax_vz_i.basis.get_realspace_1D(fi_data, 1)
        # if is_sqrt:
        #     fi_data = np.abs(fi_data) ** 2
        # im1 = ax1[0].imshow(np.real(fi_data).T,
        #                     extent=(-np.pi / dkx, np.pi / dkx, vi_y_min, vi_y_max),
        #                     aspect='auto', origin='lower')
        #
        # fe_data = init_fe_gtn.get_data(
        #     ax_select={ax_y: 0, ax_vx_e: ax_vx_e.npts // 2, ax_vz_e: ax_vz_e.npts // 2})
        # fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
        # fe_data = ax_vz_e.basis.get_realspace_1D(fe_data, 1)
        # if is_sqrt:
        #     fe_data = np.abs(fe_data) ** 2
        # im2 = ax1[1].imshow(np.real(fe_data).T,
        #                     extent=(-np.pi / dkx, np.pi / dkx, ve_x_min, ve_x_max),
        #                     aspect='auto', origin='lower')
        #
        # fig1.colorbar(im1, ax=ax1[0])
        # fig1.colorbar(im2, ax=ax1[1])
        # ax1[0].set_title(f'time {time_:3.2f}')
        # ax1[0].set_xlabel('x'), ax1[0].set_ylabel('vy,i')
        # ax1[1].set_xlabel('x'), ax1[1].set_ylabel('vy,e')
        # # if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')
        # # plt.close(fig1)
        #
        # plt.show()

        ## y, vx
        fig1, ax1 = plt.subplots(2, 1)
        time_ = ts[-1]
        # fi_data = fi_field.get_comp_data(ax_select={ax_vy_i: npts_vyi//2, ax_vz_i: npts_vzi//2})
        fi_data = init_fi_gtn.get_data(ax_select={ax_x: 0, ax_vy_i: npts_vyi // 2, ax_vz_i: npts_vzi // 2})
        fi_data = ax_y.basis.get_realspace_1D(fi_data, 0)
        fi_data = ax_vz_i.basis.get_realspace_1D(fi_data, 1)
        if is_sqrt:
            fi_data = np.abs(fi_data) ** 2
        im1 = ax1[0].imshow(np.real(fi_data).T, extent=(-np.pi / dkx, np.pi / dkx, vi_x_min, vi_x_max),
                            aspect='auto', origin='lower')

        # fe_data = fe_field.get_comp_data(ax_select={ax_vy_e: npts_vye//2, ax_vz_e: npts_vze//2})
        fe_data = init_fe_gtn.get_data(
            ax_select={ax_x: 0, ax_vz_e: ax_vz_e.npts // 2, ax_vy_e: ax_vy_e.npts // 2})
        fe_data = ax_y.basis.get_realspace_1D(fe_data, 0)
        fe_data = ax_vz_e.basis.get_realspace_1D(fe_data, 1)
        if is_sqrt:
            fe_data = np.abs(fe_data) ** 2
        im2 = ax1[1].imshow(np.real(fe_data).T, extent=(-np.pi / dkx, np.pi / dkx, ve_x_min, ve_x_max),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].xaxis.set_visible(False)
        ax1[0].set_xlabel('y'), ax1[0].set_ylabel('vx')
        ax1[1].set_xlabel('y'), ax1[1].set_ylabel('vx')

        if save_figs:
            fig1.savefig(fdir + f'plot_fy0_' + fstr + f'_nt{nt}.png')
        else:
            plt.show()
        plt.close(fig1)

        ### x, vy
        fig1, ax1 = plt.subplots(2, 1)
        time_ = ts[-1]
        fi_data = init_fi_gtn.get_data(ax_select={ax_y: 0, ax_vz_i: npts_vzi // 2, ax_vx_i: npts_vxi // 2})
        fi_data = ax_x.basis.get_realspace_1D(fi_data, 0)
        fi_data = ax_vz_i.basis.get_realspace_1D(fi_data, 1)
        if is_sqrt:
            fi_data = np.abs(fi_data) ** 2
        im1 = ax1[0].imshow(np.real(fi_data).T, extent=(-np.pi / dkx, np.pi / dkx, vi_y_min, vi_y_max),
                            aspect='auto', origin='lower')

        # fe_data = fe_field.get_comp_data(ax_select={ax_vy_e: npts_vye//2, ax_vz_e: npts_vze//2})
        fe_data = init_fe_gtn.get_data(
            ax_select={ax_y: 0, ax_vx_e: ax_vx_e.npts // 2, ax_vz_e: ax_vz_e.npts // 2})
        fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
        fe_data = ax_vz_e.basis.get_realspace_1D(fe_data, 1)
        if is_sqrt:
            fe_data = np.abs(fe_data) ** 2
        im2 = ax1[1].imshow(np.real(fe_data).T, extent=(-np.pi / dkx, np.pi / dkx, ve_y_min, ve_y_max),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].xaxis.set_visible(False)
        ax1[0].set_xlabel('x'), ax1[0].set_ylabel('vy')
        ax1[1].set_xlabel('x'), ax1[1].set_ylabel('vy')

        if save_figs:
            fig1.savefig(fdir + f'plot_fx2_' + fstr + f'_nt{nt}.png')
        else:
            plt.show()
        plt.close(fig1)

        ### x, vz
        fig1, ax1 = plt.subplots(2, 1)
        time_ = ts[-1]
        fi_data = init_fi_gtn.get_data(ax_select={ax_y: 0, ax_vy_i: npts_vyi // 2, ax_vx_i: npts_vxi // 2})
        fi_data = ax_x.basis.get_realspace_1D(fi_data, 0)
        fi_data = ax_vz_i.basis.get_realspace_1D(fi_data, 1)
        if is_sqrt:
            fi_data = np.abs(fi_data) ** 2
        im1 = ax1[0].imshow(np.real(fi_data).T, extent=(-np.pi / dkx, np.pi / dkx, vi_z_min, vi_z_max),
                            aspect='auto', origin='lower')

        # fe_data = fe_field.get_comp_data(ax_select={ax_vy_e: npts_vye//2, ax_vz_e: npts_vze//2})
        fe_data = init_fe_gtn.get_data(
            ax_select={ax_y: 0, ax_vx_e: ax_vx_e.npts // 2, ax_vy_e: ax_vy_e.npts // 2})
        fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
        fe_data = ax_vz_e.basis.get_realspace_1D(fe_data, 1)
        if is_sqrt:
            fe_data = np.abs(fe_data) ** 2
        im2 = ax1[1].imshow(np.real(fe_data).T, extent=(-np.pi / dkx, np.pi / dkx, ve_z_min, ve_z_max),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].xaxis.set_visible(False)
        ax1[0].set_xlabel('x'), ax1[0].set_ylabel('vz')
        ax1[1].set_xlabel('x'), ax1[1].set_ylabel('vz')

        if save_figs:
            fig1.savefig(fdir + f'plot_fx2_' + fstr + f'_nt{nt}.png')
        else:
            plt.show()
        plt.close(fig1)

    if restarted:
        if do_adapt_dt:
            use_bg = False if (te_order in [413, 414, 416, 4164]) else True
            do_advec_x_cfl = False if (te_order in [413, 414, 416, 4164, 313, 314, 316, 3164]) else True
            em_term = vm_sys.compute_force_term(is_ion=False, background_force=use_bg)

            max_em = []
            for coord in [X, Y, Z]:
                try:
                    em_data = em_term.get_comp_data(coord, ax_select={ax_vx_e: 0, ax_vy_e: 0, ax_vz_e: 0})
                    em_max = np.max(np.abs(em_data))
                except KeyError:
                    em_max = 0.
                max_em += [em_max]

            Fe_maxs = np.array(max_em)
            # Fi_maxs = np.array(max_em) * charge / mass_i
            dt_cfl = VP_test.get_dt(Fe_maxs, do_advec_x_cfl=do_advec_x_cfl)
            dt = min(dt_cfl, dt_max)
            print('restarted dt', dt)

    time1 = time.time()

    print('starting time evolution')
    while ts[-1] < T:

        walltime = time.time()
        dt_ = min(dt_cfl, dt_max) if nt == 0 and te_order in [3164, 3154, 3175, 4154, 4164, 4174] else dt
        vm_sys = vm_sys.next_time_step(dt_, compress_level=1, err_tol=1.0e-7, verbose_plot=False,
                                       is_first_time_step=(nt == 0))
        vm_sys.EM_sys.realspace_smoother(inplace=True, B_comps=[])
        print('wall time', time.time() - walltime)

        print('fe max bond', vm_sys.fe.max_bond(), v2e_mpo.max_bond())
        fe_field = vm_sys.fe
        max_bond_fe += [fe_field.component.max_bond()]
        nrg_fe = fe_field.component.meas_expec(v2e_mpo, is_sqrt=is_sqrt) * mass_e / 2 / tot_V

        print('fi max bond', vm_sys.fi.max_bond())
        fi_field = vm_sys.fi
        max_bond_fi += [fi_field.component.max_bond()]
        nrg_fi = fi_field.component.meas_expec(v2i_mpo, is_sqrt=is_sqrt) * mass_i / 2 / tot_V

        print('E max bond', vm_sys.E.max_bonds())
        E_field = vm_sys.E
        max_bond_E += [E_field.max_bonds(compIDs=coords_x.coords)]
        energy_E_t = np.array(E_field.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_V  # E field energy

        print('B max bond', vm_sys.B.max_bonds())
        B_field = vm_sys.B
        max_bond_B += [B_field.max_bonds(compIDs=coords_x.coords)]
        energy_B_t = np.array(B_field.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_V  # B field energy
        # norm_BY = B_field.norms([Y])[0]
        # energy_By_t = norm_BY ** 2 * 1. / 2 / tot_V if norm_BY is not None else np.nan

        gauss_err = vm_sys.EM_sys.check_poisson().component.norm()
        gauss_err = gauss_err / np.sqrt(grid_X.npts) if gauss_err is not None else None
        divB_err = vm_sys.EM_sys.check_divB().component.norm()
        divB_err = divB_err / np.sqrt(grid_X.npts) if divB_err is not None else None
        print('gauss err', gauss_err, 'divB err', divB_err)

        nrg_fe_ts += [nrg_fe]
        nrg_fi_ts += [nrg_fi]
        nrg_E_ts += [energy_E_t]
        nrg_B_ts += [energy_B_t]
        # nrg_By_ts += [energy_By_t]

        ts += [ts[-1] + dt_]
        nt += 1

        if do_adapt_dt:
            use_bg = False if (te_order in [413, 414, 416, 4164]) else True
            do_advec_x_cfl = False if (te_order in [413, 414, 416, 4164, 313, 314, 316, 3164]) else True
            em_term = vm_sys.compute_force_term(is_ion=False, background_force=use_bg)
            max_em = []
            for coord in [X, Y, Z]:
                try:
                    em_data = em_term.get_comp_data(coord, ax_select={ax_vx_e: 0, ax_vy_e: 0, ax_vz_e: 0})
                    em_max = np.max(np.abs(em_data))
                except KeyError:
                    em_max = 0.
                max_em += [em_max]

            print('max em', max_em, Bg * ax_vx_e.xpts[0] * charge / mass_e)
            Fe_maxs = np.array(max_em)
            dt_cfl = VP_test.get_dt(Fe_maxs, do_advec_x_cfl=do_advec_x_cfl)
            dt = min(dt_cfl, dt_max)
            # print('restarted dt', dt)

        norm = vm_sys.fe.norm() - vm_sys.fi.norm()
        print('result norm', nt, ts[-1], dt_, norm,  # nrg_fe, nrg_fi,
              energy_E_t, energy_B_t,
              E_field.max_bond(X), E_field.max_bond(Y), E_field.max_bond(Z),
              B_field.max_bond(X), B_field.max_bond(Y), B_field.max_bond(Z),
              fi_field.max_bond(), fe_field.max_bond(), Lx, Lvi, Lve)

        if nt % (save_every_nt // 10) == 0:

            print('fe bond dims', vm_sys.fe.component.all_virtual_sizes())

            if save_data:
                if not os.path.exists(sdir + 'restart/'):
                    os.makedirs(sdir + 'restart/')

                test_setup.save_data_EM(VP_test, vm_sys, sdir + 'restart/', extra_str=extra_str,
                                        add_fstr=f'-nt{nt}',
                                        ts=ts, nrg_fe_ts=nrg_fe_ts, nrg_fi_ts=nrg_fi_ts, nrg_E_ts=nrg_E_ts,
                                        nrg_B_ts=nrg_B_ts, max_bond_fe=max_bond_fe, max_bond_fi=max_bond_fi,
                                        max_bond_E=max_bond_E, max_bond_B=max_bond_B,
                                        max_bond_phi=max_bond_phi, max_bond_psi=max_bond_psi)

                if nt % (save_every_nt * 5) == 0:
                    test_setup.save_data_EM(VP_test, vm_sys, sdir_bkp, extra_str=extra_str,
                                            add_fstr=f'-nt{nt}',
                                            ts=ts, nrg_fe_ts=nrg_fe_ts, nrg_fi_ts=nrg_fi_ts, nrg_E_ts=nrg_E_ts,
                                            nrg_B_ts=nrg_B_ts, max_bond_fe=max_bond_fe, max_bond_fi=max_bond_fi,
                                            max_bond_E=max_bond_E, max_bond_B=max_bond_B,
                                            max_bond_phi=max_bond_phi, max_bond_psi=max_bond_psi)

            if save_figs or matplotlib.get_backend() != 'agg':

                ## y, vx
                fig1, ax1 = plt.subplots(2, 1)
                time_ = ts[-1]
                # fi_data = fi_field.get_comp_data(ax_select={ax_vy_i: npts_vyi//2, ax_vz_i: npts_vzi//2})
                fi_data = fi_field.get_comp_data(ax_select={ax_x: 0, ax_vy_i: npts_vyi // 2, ax_vz_i: npts_vzi // 2})
                fi_data = ax_y.basis.get_realspace_1D(fi_data, 0)
                fi_data = ax_vz_i.basis.get_realspace_1D(fi_data, 1)
                if is_sqrt:
                    fi_data = np.abs(fi_data) ** 2
                im1 = ax1[0].imshow(np.real(fi_data).T, extent=(-np.pi / dkx, np.pi / dkx, vi_x_min, vi_x_max),
                                    aspect='auto', origin='lower')

                # fe_data = fe_field.get_comp_data(ax_select={ax_vy_e: npts_vye//2, ax_vz_e: npts_vze//2})
                fe_data = fe_field.get_comp_data(
                    ax_select={ax_x: 0, ax_vz_e: ax_vz_e.npts // 2, ax_vy_e: ax_vy_e.npts // 2})
                fe_data = ax_y.basis.get_realspace_1D(fe_data, 0)
                fe_data = ax_vz_e.basis.get_realspace_1D(fe_data, 1)
                if is_sqrt:
                    fe_data = np.abs(fe_data) ** 2
                im2 = ax1[1].imshow(np.real(fe_data).T, extent=(-np.pi / dkx, np.pi / dkx, ve_x_min, ve_x_max),
                                    aspect='auto', origin='lower')

                fig1.colorbar(im1, ax=ax1[0])
                fig1.colorbar(im2, ax=ax1[1])
                ax1[0].set_title(f'time {time_:3.2f}')
                ax1[0].xaxis.set_visible(False)
                ax1[0].set_xlabel('y'), ax1[0].set_ylabel('vx')
                ax1[1].set_xlabel('y'), ax1[1].set_ylabel('vx')

                if save_figs:
                    fig1.savefig(fdir + f'plot_fy0_' + fstr + f'_nt{nt}.png')
                else:
                    plt.show()
                plt.close(fig1)

                ### x, vy
                fig1, ax1 = plt.subplots(2, 1)
                time_ = ts[-1]
                fi_data = fi_field.get_comp_data(ax_select={ax_y: 0, ax_vz_i: npts_vzi // 2, ax_vx_i: npts_vxi // 2})
                fi_data = ax_x.basis.get_realspace_1D(fi_data, 0)
                fi_data = ax_vz_i.basis.get_realspace_1D(fi_data, 1)
                if is_sqrt:
                    fi_data = np.abs(fi_data) ** 2
                im1 = ax1[0].imshow(np.real(fi_data).T, extent=(-np.pi / dkx, np.pi / dkx, vi_y_min, vi_y_max),
                                    aspect='auto', origin='lower')

                # fe_data = fe_field.get_comp_data(ax_select={ax_vy_e: npts_vye//2, ax_vz_e: npts_vze//2})
                fe_data = fe_field.get_comp_data(
                    ax_select={ax_y: 0, ax_vx_e: ax_vx_e.npts // 2, ax_vz_e: ax_vz_e.npts // 2})
                fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
                fe_data = ax_vz_e.basis.get_realspace_1D(fe_data, 1)
                if is_sqrt:
                    fe_data = np.abs(fe_data) ** 2
                im2 = ax1[1].imshow(np.real(fe_data).T, extent=(-np.pi / dkx, np.pi / dkx, ve_y_min, ve_y_max),
                                    aspect='auto', origin='lower')

                fig1.colorbar(im1, ax=ax1[0])
                fig1.colorbar(im2, ax=ax1[1])
                ax1[0].set_title(f'time {time_:3.2f}')
                ax1[0].xaxis.set_visible(False)
                ax1[0].set_xlabel('x'), ax1[0].set_ylabel('vy')
                ax1[1].set_xlabel('x'), ax1[1].set_ylabel('vy')

                if save_figs:
                    fig1.savefig(fdir + f'plot_fx2_' + fstr + f'_nt{nt}.png')
                else:
                    plt.show()
                plt.close(fig1)

                ### x, vz
                fig1, ax1 = plt.subplots(2, 1)
                time_ = ts[-1]
                fi_data = fi_field.get_comp_data(ax_select={ax_y: 0, ax_vy_i: npts_vyi // 2, ax_vx_i: npts_vxi // 2})
                fi_data = ax_x.basis.get_realspace_1D(fi_data, 0)
                fi_data = ax_vz_i.basis.get_realspace_1D(fi_data, 1)
                if is_sqrt:
                    fi_data = np.abs(fi_data) ** 2
                im1 = ax1[0].imshow(np.real(fi_data).T, extent=(-np.pi / dkx, np.pi / dkx, vi_z_min, vi_z_max),
                                    aspect='auto', origin='lower')

                # fe_data = fe_field.get_comp_data(ax_select={ax_vy_e: npts_vye//2, ax_vz_e: npts_vze//2})
                fe_data = fe_field.get_comp_data(
                    ax_select={ax_y: 0, ax_vx_e: ax_vx_e.npts // 2, ax_vy_e: ax_vy_e.npts // 2})
                fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
                fe_data = ax_vz_e.basis.get_realspace_1D(fe_data, 1)
                if is_sqrt:
                    fe_data = np.abs(fe_data) ** 2
                im2 = ax1[1].imshow(np.real(fe_data).T, extent=(-np.pi / dkx, np.pi / dkx, ve_z_min, ve_z_max),
                                    aspect='auto', origin='lower')

                fig1.colorbar(im1, ax=ax1[0])
                fig1.colorbar(im2, ax=ax1[1])
                ax1[0].set_title(f'time {time_:3.2f}')
                ax1[0].xaxis.set_visible(False)
                ax1[0].set_xlabel('x'), ax1[0].set_ylabel('vz')
                ax1[1].set_xlabel('x'), ax1[1].set_ylabel('vz')

                if save_figs:
                    fig1.savefig(fdir + f'plot_fx2_' + fstr + f'_nt{nt}.png')
                else:
                    plt.show()
                plt.close(fig1)

                # plt.figure()
                # E_data = vm_sys.E.get_comp_data(Y)
                # E_data = ax_x.basis.get_realspace_1D(E_data, 0)
                # plt.imshow(np.real(E_data))
                # plt.imshow(np.imag(E_data))
                # E_data = vm_sys.E.get_comp_data(Z)
                # E_data = ax_x.basis.get_realspace_1D(E_data, 0)
                # plt.imshow(np.real(E_data))
                # plt.imshow(np.imag(E_data))
                # plt.colorbar()
                #
                # plt.figure()
                # By_data = vm_sys.B.get_comp_data(Y)
                # By_data = ax_x.basis.get_realspace_1D(By_data, 0)
                # plt.imshow(np.real(By_data))
                # plt.imshow(np.imag(By_data))
                # Bz_data = vm_sys.B.get_comp_data(Z)
                # Bz_data = ax_x.basis.get_realspace_1D(Bz_data, 0)
                # plt.imshow(np.real(Bz_data))
                # plt.imshow(np.imag(Bz_data))
                # plt.colorbar()

                if not save_figs:
                    plt.show()
                plt.close()
                plt.close()

                for C in vm_sys.EM_sys.E.componentIDs:
                    plt.figure()
                    EC = vm_sys.EM_sys.E.get_comp_data(C)
                    if EC is not None:
                        plt.imshow(EC, label=f'E{C}')
                        plt.title(f'E{C}')
                        plt.colorbar()

                for C in vm_sys.EM_sys.B.componentIDs:
                    plt.figure()
                    BC = vm_sys.EM_sys.B.get_comp_data(C)
                    if BC is not None:
                        plt.imshow(BC, label=f'C{C}')
                        plt.title(f'B{C}')
                        plt.colorbar()

                # Z = coords_x.coords[-1]
                # fig1, ax1 = plt.subplots(1)
                # j_field = vm_sys.compute_current()
                # current_data = j_field.get_comp_data(Z)
                # if current_data is not None:
                #     im1 = ax1.imshow(current_data)
                #     fig1.colorbar(im1, ax=ax1)
                #     ax1.set_xlabel('y'), ax1.set_ylabel('x')
                #     ax1.set_title(f'current density {Z}')

                plt.show()

                energy_E_t = np.array(nrg_E_ts)
                energy_B_t = np.array(nrg_B_ts)
                energy_fi_t = np.array(nrg_fi_ts)
                energy_fe_t = np.array(nrg_fe_ts)
                total_nrg_t = energy_fe_t + energy_fi_t + np.sum(energy_E_t, 1) + np.sum(energy_B_t, 1)

                fig7, ax7 = plt.subplots()
                # ts = np.arange(num_tsteps+1)*dt
                ax7.semilogy(ts, np.abs(energy_fe_t), label='elec')
                ax7.semilogy(ts, np.abs(energy_fi_t), label='ion')
                ax7.semilogy(ts, np.abs(energy_E_t), label='E field')
                ax7.semilogy(ts, np.abs(energy_B_t), label='B field')
                ax7.semilogy(ts, np.abs(total_nrg_t), ls='--', label='total')

                ax7.legend()
                ax7.set_xlabel('t')
                ax7.set_ylabel('energy')
                plt.show()
                plt.close()

                fig7, ax7 = plt.subplots()
                ax7.plot(ts, [nrg[1] for nrg in nrg_B_ts], label='By')
                ax7.legend()
                ax7.set_xlabel('t')
                ax7.set_ylabel('energy')
                ax7.legend()
                if save_figs:
                    fig7.savefig(fdir + 'plot_nrg_By_' + fstr + '.png')
                else:
                    plt.show()
                plt.close()

    if save_data:
        test_setup.save_data_EM(VP_test, vm_sys, fdir, extra_str=extra_str,
                                ts=ts, nrg_fe_ts=nrg_fe_ts, nrg_fi_ts=nrg_fi_ts, nrg_E_ts=nrg_E_ts,
                                nrg_B_ts=nrg_B_ts, max_bond_fe=max_bond_fe, max_bond_fi=max_bond_fi,
                                max_bond_E=max_bond_E, max_bond_B=max_bond_B,
                                max_bond_phi=max_bond_phi, max_bond_psi=max_bond_psi)

    print('wall time', time.time() - time1)

    fig1, ax1 = plt.subplots(2, 1)
    time_ = ts[-1]
    fi_data = vm_sys.fi.get_comp_data(ax_select={ax_vy_i: npts_vyi // 2, ax_vz_i: npts_vzi // 2})
    fi_real = ax_x.basis.get_realspace_1D(fi_data, 0)
    fi_real = ax_vz_i.basis.get_realspace_1D(fi_real, 1)
    if is_sqrt:
        fi_real = np.abs(fi_real) ** 2
    im1 = ax1[0].imshow(np.real(fi_real).T, extent=(-np.pi / dkx, np.pi / dkx, vi_x_min, vi_x_max),
                        aspect='auto', origin='lower')

    fe_data = vm_sys.fe.get_comp_data(ax_select={ax_vy_e: npts_vye // 2, ax_vz_e: npts_vze // 2})
    fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
    fe_real = ax_vx_e.basis.get_realspace_1D(fe_real, 1)
    if is_sqrt:
        fe_real = np.abs(fe_real) ** 2
    im2 = ax1[1].imshow(np.real(fe_real).T, extent=(-np.pi / dkx, np.pi / dkx, ve_x_min, ve_x_max),
                        aspect='auto', origin='lower')

    fig1.colorbar(im1, ax=ax1[0])
    fig1.colorbar(im2, ax=ax1[1])
    ax1[0].set_title(f'time {time_:3.2f}')
    ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
    ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
    if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_T{time_:2.1f}.png')

    fig2, ax2 = plt.subplots()
    ax2.plot(np.log(max_bond_fe) / np.log(q), label=r'$f_e$')
    ax2.plot(np.log(max_bond_fi) / np.log(q), label=r'$f_i$')
    max_bond_Ey = [mb[1] if mb[1] is not None else np.nan for mb in max_bond_E]
    ax2.plot(np.log(max_bond_Ey) / np.log(q), label='Ey')
    max_bond_By = [mb[1] if mb[1] is not None else np.nan for mb in max_bond_B]
    ax2.plot(np.log(max_bond_By) / np.log(q), label='By')
    ax2.plot([0, num_tsteps], [(Lx + KV * Lve) // 2] * 2, ':', label='max fe')
    ax2.plot([0, num_tsteps], [(Lx) // 2] * 2, ':', label='max V')
    ax2.legend()
    ax2.set_xlabel('t')
    ax2.set_ylabel('log_{q^K}(max bond dimension)')
    if save_figs:   fig2.savefig(fdir + 'plot_maxbond_' + fstr + '.png')

    # energy_E_t = np.sum(Ex_t*np.conj(Ex_t),axis=1)*dx*1./2   # electric field energy
    energy_E_t = np.array(nrg_E_ts)
    energy_B_t = np.array(nrg_B_ts)
    energy_fi_t = np.array(nrg_fi_ts)
    energy_fe_t = np.array(nrg_fe_ts)
    total_nrg_t = energy_fe_t + energy_fi_t + np.sum(energy_E_t, 1) + np.sum(energy_B_t, 1)
    # delta_nrg_fe_t = energy_fe_t - energy_fe_t[0]
    # delta_nrg_fi_t = energy_fi_t - energy_fi_t[0]
    # delta_nrg_E_t  = energy_E_t  - energy_E_t[0]
    # total_E_t = delta_nrg_fe_t + delta_nrg_fi_t + delta_nrg_E_t

    ##############

    fig7, ax7 = plt.subplots()
    # ts = np.arange(num_tsteps+1)*dt
    ax7.semilogy(ts, np.abs(energy_fe_t), label='elec')
    ax7.semilogy(ts, np.abs(energy_fi_t), label='ion')
    ax7.semilogy(ts, np.abs(energy_E_t), label='E field')
    ax7.semilogy(ts, np.abs(energy_B_t), label='B field')
    ax7.semilogy(ts, np.abs(total_nrg_t), ls='--', label='total')

    ax7.legend()
    ax7.set_xlabel('t')
    ax7.set_ylabel('energy')
    # ax7.set_ylim(top=10**4)
    if save_figs:
        fig7.savefig(fdir + 'plot_nrg_' + fstr + '.png')
        print(fdir + 'plot_nrg_' + fstr + '.png')

    plt.show()

    ######## plot EE vs bond #####
    fig7, ax7 = plt.subplots()
    ax7.semilogy(vm_sys.fe.component.entanglement_entropy_all(), '-x', label='elec')
    ax7.semilogy(vm_sys.fi.component.entanglement_entropy_all(), '-o', label='ion')

    ax7.legend()
    ax7.set_xlabel('bond number')
    ax7.set_ylabel('EE')
    if save_figs:
        fig7.savefig(fdir + 'plot_EE_' + fstr + '.png')
        print(fdir + 'plot_EE_' + fstr + '.png')

    plt.show()

    print('done')
