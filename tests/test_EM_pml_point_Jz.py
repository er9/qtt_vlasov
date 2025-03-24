import os
import sys
import time

sys.path.append('../')

from setup_.configs import *
# print(sys.argv)
# import pdb; pdb.set_trace()

import plot_defaults

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D_sum import GridTNSum
from field import Field
from tests.helper_EM_PML import _calculate_time_derivative_E_part
from tests.helper_EM_PML import _calculate_time_derivative_B
from tests.helper_EM_PML import *
from local_solvers.time_integrator_EM_v2 import evolve_E_part
import helper_compress_v3

"""
2D delta-point sinusoidal current source
"""

plotting = True
save_figs = False
save_data = False  # True
load_data = False  # True
restart = False
restart_from_T = 0  # to restart from an output file (vs a saved restart file)

#######################
# problem  parameters #
#######################

# permittivity = 1. / 2500  # 1.0    # permittivity of free space
# permeability = 1.0    # permeability of free space
permittivity = (1 / (36 * np.pi)) * 1e-9
permeability = 4 * np.pi * 1e-7
c = 1 / np.sqrt(permittivity*permeability)  # speed of light
print('c', c)

matl_config = UnitsConfiguration(eps0=1, mu0=1, is_cgs=False)
# import pdb; pdb.set_trace()

# Spatial grid step length (spatial grid step= 1 micron and can be changed)
delta = 1e-6

# initial conditions (cavity size)
Lx = 256*delta
Ly = 256*delta
T = Lx / c * 2  # total sim time (3.24 for ~0.05)

E0 = 1.             # electric field magnitude

xsource = Lx/2
ysource = Ly/2
sigma = 4e-5

### current source parameters
I0 = 1.0    # current strength
wavelength = Lx / 20
omega = 2 * np.pi * c / wavelength
# wavelength = 2 * np.pi * c / omega
# print('wavelength', wavelength)
sigma_wrt_wavelength = 0     ## width of gaussian with respect to wavelength

print('omega', omega, 'T', T, 'period', 2 * np.pi/omega)
print('wavleenght', wavelength, 'L', Lx)



### smoothing parameters
smooth_type = None  # 'cubic'  # 'avg'
smooth_kwargs = {}   # {'mu': 0.99}

### file saving strings
save_dir = 'data_Jz/'
save_str_0 = f'Jsig{sigma_wrt_wavelength}_omega{omega:1.3e}'
save_str = save_str_0
if smooth_type is not None:
    save_str = save_str + f'_{smooth_type}'
    for k, v in smooth_kwargs.items():
        save_str += f'_{k}{v}'

qtt_lengths = [7]  # Lengths of the tensor train
elapsed_time = []  # List to store the elapsed time for each run

for j in qtt_lengths:

    #######################
    # run-time parameters #
    #######################

    q = 2           # size of physical bond dimension
    L = j           # length of tensor train per dimension
    K = 2           # number of dimension
    npts = q ** L   # total number of grid points per dimension
    TN_layout = LayoutType.SEQUENTIAL

    # Courant stability factor
    S = 1 / (2**0.5)
    print('S', S)

    # Temporal grid step obtained using Courant condition
    delta = Lx / (q ** L)
    dt = S * delta / c
    T = 600 * dt

    print('omega', omega, dt)
    print('period', 2 * np.pi / omega / dt)

    save_every_nt = 10      # save points (in units of time steps)

    # Sequence including None and integers for max bond values, first is reference
    # values = [None, 4, 8, 12]  # max bond values
    # values = [None, 8]  # max bond values for testing
    values = [1.0e-30]   #[1e-30]  # cutoff values
    # values = [1e-10, 1e-5]  # cutoff values

    # Initialize an index variable
    index = 0

    # Initialize  dictionaries
    components_dict = {}  # components with dynamic names
    rel_diff_dict = {}  # relative differences
    data_max_bond_E = {}  # max bond values
    data_max_bond_B = {}
    data_mem_size_E = {}  # memory size values
    data_mem_size_B = {}

    for co in values:
        # define compression parameters #
        # compress_style options
        """ comp1:   compress final state
            comp2:   + compress intermediate rk4 states
            comp3:   + compress sum of derivatives
            comp4:   + compress derivatives
            comp5:   + compress during calculation of derivative

            mod0:  no modifications
            mod1:  comp5 = 0 (don't actively comp in interior of calc deriv)
        """
        DMAX = None  # None   # 100             # maximum rank of TT (None means no maximum)
        if DMAX is not None:
            # save_str += f'_D{DMAX}_dmrg'
            save_str += f'_D{DMAX}'
        compress_style = 2      # see above
        compress_style_mod = 0

        # CompressionConfiguation holds all of the comp info, cutoff=CUTOFF
        te_compress_E = CompressionConfiguration()
        for ci in range(1, compress_style + 1):
            te_compress_E.set_compress_opts(ci, max_bond=DMAX,
                                            cutoff_mode=CUTOFF_MODE, cutoff=co)
        te_compress_B = te_compress_E

        print('te compress E', te_compress_E)

        comp_levels = list(range(1, 6))
        if compress_style_mod == 1:  # don't compress inside the derivative
            comp_levels = [1, 2, 3, 4, 0]
        elif compress_style_mod == 2:  # don't compress inside the derivative
            comp_levels = [1, 2, 3, 0, 0]

        print('compress', DMAX, compress_style, compress_style_mod)

        #############################
        # start building PDE system #
        #############################

        # Record the start time
        start_time = time.time()

        # define coordinates (to label axes and vector components)
        X = Coordinate('X', CoordinateType.X)
        Y = Coordinate('Y', CoordinateType.Y)
        Z = Coordinate('Z', CoordinateType.Z)

        # define axis objects
        x_vals = np.linspace(0, Lx, npts, endpoint=False)
        x1_vals = x_vals + (x_vals[1] - x_vals[0]) / 2
        y_vals = np.linspace(0, Ly, npts, endpoint=False)
        y1_vals = y_vals + (y_vals[1] - y_vals[0]) / 2

        ax_x = Axis(L, q, coordinate=X, dx=delta, xpts=x_vals)
        ax_y = Axis(L, q, coordinate=Y, dx=delta, xpts=y_vals)
        # below: in theory one can build a not-quantized TT, though it's less deved
        # ax_x = Axis(1, q**L, coordinate=X, xpts=x_vals)
        # ax_y = Axis(1, q**L, coordinate=Y, xpts=y_vals)

        # define a Cartesian coordinate system
        coord_sys = CartesianCoordinateSpace('cart', ax_x, ax_y, coords=[X, Y, Z])

        # configure derivatives. DerivativeConfiguration contains info ab derivs
        deriv_zg = DerivativeConfiguration(
            left_bc=BCType.SYMMETRIC, fd_type=FDType.CENTER)
        deriv_zv = DerivativeConfiguration(
            left_bc=BCType.ANTISYMMETRIC, fd_type=FDType.CENTER)
        deriv_zg1 = DerivativeConfiguration(
            left_bc=BCType.SYMMETRIC, fd_type=FDType.CENTER, offset=1, offset_r=-1)
        deriv_zv1 = DerivativeConfiguration(
            left_bc=BCType.ANTISYMMETRIC, fd_type=FDType.CENTER,
            offset=1, offset_r=-1)

        # note: endpoint is False:
        # offset = 0:  0|0, 1 , 2 ..., n-1 | n , (n+1...)
        # offset = 1:  (-1/2) | 1/2, 3/2, ..., n+1/2 | (n+3/2)

        # define Grid object:  Grid1D means a TT geometry
        grid_xy = Grid1D('XY', (ax_x, ax_y), layout_type=TN_layout)

        ## define "delta function" in space for the current source
        if sigma_wrt_wavelength == 0:
            delta_fct_gtn = grid_xy.make_mps_ndim({ax_x: ax_x.get_select_elems_mps(inds=[ax_x.npts // 2]),
                                                   ax_y: ax_y.get_select_elems_mps(inds=[ax_y.npts // 2])})
        else:
            ## instead use gaussian for current source
            x0, y0, sigma = Lx/2, Ly/2, wavelength * sigma_wrt_wavelength
            delta_fct_gtn = grid_xy.make_mps_ndim({ax_x: 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x_vals - x0)**2 / 2 / sigma**2),
                                                   ax_y: 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(y_vals - y0)**2 / 2 / sigma**2)})
            # delta_fct_gtn.compress(compress_opts={'cutoff': co})


        ## current source (uniform current "point" source that varies with respect to time)
        def Jz_gtn(t):
            jz = delta_fct_gtn.scalar_multiply(np.sin(omega * t) * I0 / permittivity, inplace=False)
            print('np.sin(omega * t)', np.sin(omega * t))
            print('t', t)
            print('omega * t', omega * t)
            return jz


        # define initial grid1DTNs
        # data_x, data_y = np.meshgrid(x_vals, y_vals, indexing='ij')

        # map_state_to_mps converts np.ndarray to MPS
        ## NOTE: need two separate fields for Ezx, Ezy
        init_Ezx_gtn = grid_xy.make_empty_gridTN()
        init_Ezy_gtn = grid_xy.make_empty_gridTN()
        init_Bx_gtn = grid_xy.make_empty_gridTN()
        init_By_gtn = grid_xy.make_empty_gridTN()

        # set gridTN boundary conditions
        init_Ezx_gtn.ax_deriv_configs = {ax_x: deriv_zv,
                                         ax_y: deriv_zv}     # i, j, k+1/2
        init_Ezy_gtn.ax_deriv_configs = {ax_x: deriv_zv,
                                         ax_y: deriv_zv}     # i, j, k+1/2
        init_Bx_gtn.ax_deriv_configs = {ax_x: deriv_zv,
                                        ax_y: deriv_zg1}     # i, j+1/2, k+1/2
        init_By_gtn.ax_deriv_configs = {ax_x: deriv_zg1,
                                        ax_y: deriv_zv}      # i+1/2, j, k+1/2

        print('init E', init_Ezx_gtn.ax_deriv_configs[ax_x].bc)

        # define Field object: contains comps for x, y, z comps of the field.
        Ezx_field = Field('E', grid_xy, data={Z: init_Ezx_gtn},
                          compress_config=te_compress_E)
        Ezy_field = Field('E', grid_xy, data={Z: init_Ezy_gtn},
                          compress_config=te_compress_E)
        B_field = Field('B', grid_xy, data={X: init_Bx_gtn, Y: init_By_gtn},
                        compress_config=te_compress_B)

        dx, dy = ax_x.dx, ax_y.dx
        num_tsteps = int(T / dt)
        print('dt', dt, num_tsteps)
        print('dx', dx, 'dy', dy)


        try:
            if not load_data:
                raise IOError
            nt = num_tsteps - 1
            Ezx_field.reload_data(save_dir + 'Ezx_' + save_str + f'_co{co:1.2e}_nt{nt}', safe_pass=False)
            Ezy_field.reload_data(save_dir + 'Ezy_' + save_str + f'_co{co:1.2e}_nt{nt}', safe_pass=False)
            B_field.reload_data(save_dir + 'B_' + save_str + f'_co{co:1.2e}_nt{nt}', safe_pass=False)

            # Ezx_field.reload_data(f'data_Jz/deltaJ_Ezx_w{omega}_co{co}_mu0.99', safe_pass=False)
            # Ezy_field.reload_data(f'data_Jz/deltaJ_Ezy_w{omega}_co{co}_mu0.99', safe_pass=False)
            # B_field.reload_data(f'data_Jz/deltaJ_B_w{omega}_co{co}_mu0.99', safe_pass=False)

        except IOError:

            ######
            nt = 0
            ts = [0]
            restarted = False

            nrg_E_ts = [0.0]
            nrg_B_ts = [0.0]

            max_bond_Ezx = [init_Ezx_gtn.max_bond()]
            max_bond_Ezy = [init_Ezy_gtn.max_bond()]
            max_bond_B = [(np.nan, np.nan)]

            mem_size_Ezx = []
            mem_size_Ezy = []
            mem_size_B = []

            print('start')
            time1 = time.time()
            ## NOTE: don't save the entire array. that's expensive.
            ## also, Ez = Ezx + Ezy
            Ez_t = [init_Ezx_gtn.add(init_Ezy_gtn)]
            t0 = ts[-1]
            nt0 = nt

            # B_field = em_sys.B
            # divB = B_field.divergence(coord_sys=coord_sys)
            # divB_data = divB.get_comp_data()
            # print('divB_data', divB_data)
            # print('norm divB', np.linalg.norm(divB_data))

            #######################
            #    mult matrices    #
            #######################

            max_bonds = []
            mem_sizes = []

            # Initialization of permittivity and permeability matrices
            epsilon = permittivity * np.ones((npts, npts))
            mu = permeability * np.ones((npts, npts))

            # Initializing electric conductivity matrices in x and y directions
            sigmax = np.zeros((npts, npts))
            sigmay = np.zeros((npts, npts))

            # Perfectly matched layer boundary design
            # Reference: Adapted from a 2-D FDTD TE code example

            # Boundary width of PML in all directions
            # Typically 8-10, low end 5-6, high end 15-20
            bound_width = [8]
            # bound_width = [6, 8, 10, 16, 18, 20]

            # Order of polynomial on which sigma is modeled
            # try 2, 4, 6
            # gradingorder = [1, 2, 3, 4, 5, 6]
            gradingorder = [2]

            # Required reflection coefficient
            refl_coeff = 1e-6      # = 1 for no PML

            for bw in bound_width:
                for grdg_ord in gradingorder:
                    # Polynomial model for sigma
                    sigmamax = (-np.log10(refl_coeff) * (grdg_ord + 1)
                                * permittivity * c) / (2 * bw * dx)
                    boundfact1 = ((epsilon[int(npts / 2), bw] / permittivity)
                                  * sigmamax) / ((bw ** grdg_ord)
                                                 * (grdg_ord + 1))
                    boundfact2 = (((epsilon[int(npts / 2), npts - bw]
                                    / permittivity) * sigmamax)
                                  / ((bw ** grdg_ord) * (grdg_ord + 1)))
                    boundfact3 = ((epsilon[bw, int(npts / 2)] / permittivity)
                                  * sigmamax) / ((bw ** grdg_ord)
                                                 * (grdg_ord + 1))
                    boundfact4 = (((epsilon[npts - bw, int(npts / 2)]
                                    / permittivity) * sigmamax)
                                  / ((bw ** grdg_ord) * (grdg_ord + 1)))
                    x = np.arange(0, bw + 1)
                    for i_ in range(npts):
                        sigmax[i_, bw::-1] = boundfact1 * (
                            (x + 0.5 * np.ones(bw + 1)) ** (grdg_ord + 1)
                            - (x - 0.5 * np.array([0] + [1] * bw))
                            ** (grdg_ord + 1))
                        sigmax[i_, npts - bw - 1:npts] = boundfact2 * (
                            (x + 0.5 * np.ones(bw + 1)) ** (grdg_ord + 1)
                            - (x - 0.5 * np.array([0] + [1] * bw))
                            ** (grdg_ord + 1))
                    for i_ in range(npts):
                        sigmay[bw::-1, i_] = boundfact3 * (
                            (x + 0.5 * np.ones(bw + 1)) ** (grdg_ord + 1)
                            - (x - 0.5 * np.array([0] + [1] * bw))
                            ** (grdg_ord + 1)).T
                        sigmay[npts - bw - 1:npts, i_] = boundfact4 * (
                            (x + 0.5 * np.ones(bw + 1)) ** (grdg_ord + 1)
                            - (x - 0.5 * np.array([0] + [1] * bw))
                            ** (grdg_ord + 1)).T

            # Magnetic conductivity matrix obtained by PML condition
            # This is also split into x and y dirs in Berenger's model
            sigma_starx = (sigmax * mu) / epsilon
            sigma_stary = (sigmay * mu) / epsilon

            # Multiplication factor matrices for H matrix update to...
            # avoid being calculated many times
            # in the time update loop so as to increase comp speed
            G = ((mu - 0.5 * dt * sigma_starx)
                    / (mu + 0.5 * dt * sigma_starx))
            H = 1. / (mu + 0.5 * dt * sigma_starx)
            A = ((mu - 0.5 * dt * sigma_stary)
                    / (mu + 0.5 * dt * sigma_stary))
            B = 1. / (mu + 0.5 * dt * sigma_stary)

            # Multiplication factor matrices for E matrix update...
            # to avoid being calculated many times...
            # in the time update loop so as to increase comp speed
            C = ((epsilon - 0.5 * dt * sigmax)
                    / (epsilon + 0.5 * dt * sigmax))
            D = 1. / (epsilon + 0.5 * dt * sigmax)
            E = ((epsilon - 0.5 * dt * sigmay)
                    / (epsilon + 0.5 * dt * sigmay))
            F = 1. / (epsilon + 0.5 * dt * sigmay)

            # plt.figure()
            # plt.imshow(sigma_starx)
            # plt.colorbar()
            # plt.show()

            G_gtn = grid_xy.map_state_to_mps(G)
            H_gtn = grid_xy.map_state_to_mps(H)
            A_gtn = grid_xy.map_state_to_mps(A)
            B_gtn = grid_xy.map_state_to_mps(B)
            C_gtn = grid_xy.map_state_to_mps(C)
            D_gtn = grid_xy.map_state_to_mps(D)
            E_gtn = grid_xy.map_state_to_mps(E)
            F_gtn = grid_xy.map_state_to_mps(F)

            ## these max bonds do not change over time.
            print('corrector max bond', [G_gtn.max_bond(), H_gtn.max_bond(),
                            A_gtn.max_bond(), B_gtn.max_bond(),
                            C_gtn.max_bond(), D_gtn.max_bond(),
                            E_gtn.max_bond(), F_gtn.max_bond()])

            print('corrector mem size', [G_gtn.mem_size(), H_gtn.mem_size(),
                            A_gtn.mem_size(), B_gtn.mem_size(),
                            C_gtn.mem_size(), D_gtn.mem_size(),
                            E_gtn.mem_size(), F_gtn.mem_size()])

            # if plotting and (len(bound_width) > 1 or len(gradingorder) > 1):
            #     gtn_list = ['G', 'H', 'A', 'B', 'C', 'D', 'E', 'F']
            #     # length = len(bound_width)
            #     length = len(gradingorder)
            #     max_bonds = np.array(max_bonds)
            #     max_bonds = max_bonds.reshape(length, len(gtn_list))
            #     mem_sizes = np.array(mem_sizes)
            #     mem_sizes = mem_sizes.reshape(length, len(gtn_list))
            #
            #     fig7, ax7 = plt.subplots()
            #     for ind in range(len(gtn_list)):
            #         # ax7.plot(bound_width, max_bonds[:, ind],
            #         #          label=gtn_list[ind])
            #         ax7.plot(gradingorder, max_bonds[:, ind],
            #                  label=gtn_list[ind])
            #     # ax7.set_title(f'Max Bond grading_order={gradingorder} L={j}')
            #     # ax7.set_xlabel('bound_width')
            #     # ax7.set_ylabel('Max Bond')
            #     ax7.set_title(f'Max Bond bound_width={bound_width} L={j}')
            #     ax7.set_xlabel('gradingorder')
            #     ax7.set_ylabel('Max Bond (kb)')
            #     ax7.legend()
            #
            #     plt.show()
            #
            #     fig8, ax8 = plt.subplots()
            #     for ind in range(len(gtn_list)):
            #         # ax8.plot(bound_width, mem_sizes[:, ind],
            #         #          label=gtn_list[ind])
            #         ax8.plot(gradingorder, mem_sizes[:, ind],
            #                  label=gtn_list[ind])
            #     # ax8.set_title(f'Mem Size grading_order={gradingorder} L={j}')
            #     # ax8.set_xlabel('bound_width')
            #     # ax8.set_ylabel('Mem Size (kb)')
            #     ax8.set_title(f'Mem Size bound_width={bound_width} L={j}')
            #     ax8.set_xlabel('gradingorder')
            #     ax8.set_ylabel('Mem Size (kb)')
            #     ax8.legend()
            #
            #     plt.show()

            compress_opts_E = te_compress_E.get_compress_opts(1)
            compress_opts_B = te_compress_B.get_compress_opts(1)

            while ts[-1] < T:

                # walltime = time.time()
                # em_sys = em_sys.next_time_step(dt, compress_level=1,
                #                             is_first_time_step=(nt == 0))
                # print('wall time', time.time() - walltime)

                print('ts[-1]', ts[-1], dt)
                current_density = Field('J', grid_xy, data={Z: Jz_gtn(ts[-1] + dt / 2)})
                Ez_field = Ezx_field.add(Ezy_field) # , compress_level=1)

                dt_B = _calculate_time_derivative_B(
                    Ez_field, coord_sys, matl_config, compress=0,
                    pml_correction_dict={X: H_gtn, Y: B_gtn})

                print('evolve B')

                evolve_B(dt=dt, B_comp=B_field[Y], deriv1=dt_B[Y], pml_correction_mps=A_gtn,
                         compID=Y, do_upwind=False, permittivity=permittivity, permeability=permeability,
                         inplace=True, compress_opts=compress_opts_B) #, compress_type=CompressType.DMRG,)
                evolve_B(dt=dt, B_comp=B_field[X], deriv1=dt_B[X], pml_correction_mps=G_gtn,
                         compID=X, do_upwind=False, permittivity=permittivity, permeability=permeability,
                         inplace=True, compress_opts=compress_opts_B) #, compress_type=CompressType.DMRG,)

                ### smoother
                if smooth_type == 'avg':
                    print('smoothing B naive')
                    B_field[X].average_fine_scale(inplace=True, **smooth_kwargs, compress_opts=compress_opts_B)
                    B_field[Y].average_fine_scale(inplace=True, **smooth_kwargs, compress_opts=compress_opts_B)
                elif smooth_type == 'nlin_avg':
                    print('smoothing B nonlinear avg')
                    smoothed_BX = helper_compress_v3.smooth_nonlinear_diffusion_gtn(B_field[X], filter_weight=0.1, cutoff=co)
                    smoothed_BY = helper_compress_v3.smooth_nonlinear_diffusion_gtn(B_field[Y], filter_weight=0.1, cutoff=co)
                    B_field[X].data = smoothed_BX.data
                    B_field[Y].data = smoothed_BY.data
                elif smooth_type == 'cubic':
                    if nt > 20:  # (nt + 1) % 1 == 0:
                        print('smoothing B cubic')
                        # Bx_data = B_field[X].get_data()
                        # By_data = B_field[Y].get_data()
                        # plt.figure()
                        # plt.plot(Bx_data[npts//2,:], label='Bx')
                        # plt.plot(By_data[npts // 2, :], label='By')

                        smoothed_BX = helper_compress_v3.compress_cubic_interpolation(B_field[X], depth=1, cutoff=co)
                        smoothed_BY = helper_compress_v3.compress_cubic_interpolation(B_field[Y], depth=1, cutoff=co)
                        B_field[X].data = smoothed_BX.data
                        B_field[Y].data = smoothed_BY.data

                        # Bx_data = B_field[X].get_data()
                        # By_data = B_field[Y].get_data()
                        # plt.plot(Bx_data[npts // 2, :], '--', label='Bx c')
                        # plt.plot(By_data[npts // 2, :], '--', label='By c')
                        # plt.show()

                dt_E_zx = _calculate_time_derivative_E_part(
                    'zx', B_field, coord_sys, matl_config, current_density=current_density,
                    pml_correction_mps=D_gtn, compress=0)
                dt_E_zy = _calculate_time_derivative_E_part(
                    'zy', B_field, coord_sys, matl_config, current_density=current_density,
                    pml_correction_mps=F_gtn, compress=0)

                print('evolve E')

                ## evolve the two different Ez fields
                evolve_E(dt=dt, E_comp=Ezx_field[Z], deriv=dt_E_zx, pml_correction_mps=C_gtn,
                         do_upwind=False, permittivity=permittivity, permeability=permeability,
                         inplace=True, compress_opts=compress_opts_E) #, compress_type=CompressType.DMRG,)
                evolve_E(dt=dt, E_comp=Ezy_field[Z], deriv=dt_E_zy, pml_correction_mps=E_gtn,
                         do_upwind=False, permittivity=permittivity, permeability=permeability,
                         inplace=True, compress_opts=compress_opts_E) #, compress_type=CompressType.DMRG,)

                # evolve_E_part('zx', dt, Ezx_field[Z], B_field, coords_x=coord_sys, matl_params=matl_config,
                #               current_density=current_density,
                #               deriv_pml_correction_gtn=D_gtn, E_pml_correction_gtn=C_gtn,
                #               max_bond=DMAX, cutoff=co)
                # evolve_E_part('zy', dt, Ezy_field[Z], B_field, coords_x=coord_sys, matl_params=matl_config,
                #               current_density=current_density,
                #               deriv_pml_correction_gtn=F_gtn, E_pml_correction_gtn=E_gtn,
                #               max_bond=DMAX, cutoff=co)

                # exit()


                # ### smoother
                if smooth_type == 'avg':
                    print('smoothing E naive', smooth_kwargs)
                    Ezx_field[Z].average_fine_scale(inplace=True, **smooth_kwargs, compress_opts=compress_opts_E)
                    Ezy_field[Z].average_fine_scale(inplace=True, **smooth_kwargs, compress_opts=compress_opts_E)
                elif smooth_type == 'nlin_avg':
                    print('smoothing nonlinear averaging', smooth_kwargs)
                    smoothed_EZX = helper_compress_v3.smooth_nonlinear_diffusion_gtn(Ezx_field[Z], filter_weight=0.1, cutoff=co)
                    smoothed_EZY = helper_compress_v3.smooth_nonlinear_diffusion_gtn(Ezy_field[Z], filter_weight=0.1, cutoff=co)
                    Ezx_field[Z].data = smoothed_EZX.data
                    Ezy_field[Z].data = smoothed_EZY.data
                elif smooth_type == 'cubic':
                    if nt > 20:  # (nt + 1) % 1 == 0:
                        print('smoothing E cubic')

                        # E_data = Ezx_field[Z].get_data() + Ezy_field[Z].get_data()
                        # plt.figure()
                        # plt.plot(E_data[npts//2, :], label='E')

                        smoothed_EZX = helper_compress_v3.compress_cubic_interpolation(Ezx_field[Z], depth=1, cutoff=co)
                        smoothed_EZY = helper_compress_v3.compress_cubic_interpolation(Ezy_field[Z], depth=1, cutoff=co)
                        Ezx_field[Z].data = smoothed_EZX.data
                        Ezy_field[Z].data = smoothed_EZY.data

                        # E_data = Ezx_field[Z].get_data() + Ezy_field[Z].get_data()
                        # plt.plot(E_data[npts // 2, :], '--', label='E c')
                        # plt.show()


                ### NOTE: measure this after updating to the next time step
                max_bond_Ezx += [init_Ezx_gtn.max_bond()]  # find maximum rank of QTT
                mem_size_Ezx += [init_Ezx_gtn.mem_size()]

                max_bond_Ezy += [init_Ezy_gtn.max_bond()]  # find maximum rank of QTT
                mem_size_Ezy += [init_Ezy_gtn.mem_size()]

                # B_field = em_sys.B
                max_bond_B += [(B_field[X].max_bond(), B_field[Y].max_bond())]
                mem_size_B += [(B_field[X].mem_size(), B_field[Y].mem_size())]

                if plotting:
                    Ez_t += [init_Ezx_gtn.add(init_Ezy_gtn)]

                    Bx_data = B_field.get_comp_data(X)
                    By_data = B_field.get_comp_data(Y)

                    # measure energy of electric field
                    # nrg_E = np.sum(E_data * np.conj(E_data)) * dx * dy * 1. / 2
                    nrg_E = (init_Ezy_gtn.frobenius_norm() ** 2 + init_Ezx_gtn.frobenius_norm() ** 2) * dx * dy / 2
                    nrg_E_ts += [nrg_E]

                    # measure energy of magnetic field
                    Bx_comp = B_field[X]
                    By_comp = B_field[Y]
                    nrg_Bx = (Bx_comp.frobenius_norm() ** 2)
                    nrg_By = (By_comp.frobenius_norm() ** 2)
                    nrg_B_ts += [(nrg_Bx + nrg_By) * dx * dy * 1. / 2]

                    if index == 0:
                        E_non_comp = init_Ezx_gtn.add(init_Ezy_gtn)

                        # Constructing the dynamic key with the index
                        key_name = f"E0_{ts[-1]}"

                        # Adding the relative diff to the dict with the dynamic key
                        components_dict[key_name] = E_non_comp
                    else:
                        E_comp = init_Ezx_gtn.add(init_Ezy_gtn)
                        E_non_comp = components_dict[key_name]

                        rel_diff = E_non_comp.distance(E_comp)

                        if index not in rel_diff_dict:
                            rel_diff_dict[index] = []

                        # Append rel_diff to the list for the current index
                        rel_diff_dict[index].append(rel_diff)

                    # total energy
                    nrg_norm = nrg_E_ts[-1] + nrg_B_ts[-1]

                    # print out results
                    print('result norm', nt, ts[-1], dt, nrg_E, nrg_norm,
                          Ezx_field[Z].max_bond(), Ezy_field[Z].max_bond(), B_field[X].max_bond(),
                          B_field[Y].max_bond(), 2 ** L)

                    divB = get_divB(coord_sys, B_field)
                    print('norm divB', divB.norm())

                ts += [ts[-1] + dt]
                nt += 1

                # routine for every few time steps (for plotting, etc)
                if nt % save_every_nt == 0 and plotting:
                    if save_data:
                        Ezx_field.save_data(save_dir + 'Ezx_' + save_str + f'_co{co:1.2e}_nt{nt}')
                        Ezy_field.save_data(save_dir + 'Ezy_' + save_str + f'_co{co:1.2e}_nt{nt}')
                        B_field.save_data(save_dir + 'B_' + save_str + f'_co{co:1.2e}_nt{nt}')

                    if True:  # smooth_type is not None or DMAX is not None or co > 1.1e-30:
                        try:
                            ref_Ezx_field = Ezx_field.copy()
                            ref_Ezy_field = Ezy_field.copy()
                            ref_B_field = B_field.copy()
                            ref_Ezx_field.reload_data(save_dir + 'Ezx_' + save_str_0 + f'_co{1e-30:1.2e}_nt{nt}', safe_pass=False)
                            ref_Ezy_field.reload_data(save_dir + 'Ezy_' + save_str_0 + f'_co{1e-30:1.2e}_nt{nt}', safe_pass=False)
                            ref_B_field.reload_data(save_dir + 'B_' + save_str_0 + f'_co{1e-30:1.2e}_nt{nt}', safe_pass=False)
                            print('diff', ref_Ezx_field[Z].distance(Ezx_field[Z]) / ref_Ezx_field[Z].frobenius_norm())
                            print('diff', ref_Ezy_field[Z].distance(Ezy_field[Z]) / ref_Ezy_field[Z].frobenius_norm())
                            print('diff', ref_B_field[X].distance(B_field[X]) / ref_B_field[X].frobenius_norm())
                            print('diff', ref_B_field[Y].distance(B_field[Y]) / ref_B_field[Y].frobenius_norm())

                            plt.figure()
                            plt.imshow((ref_Ezx_field.get_comp_data(Z) - Ezx_field.get_comp_data(Z))
                                            /ref_Ezx_field[Z].frobenius_norm())
                            plt.colorbar()
                            plt.title('Ezx error')

                            plt.figure()
                            plt.imshow((ref_Ezy_field.get_comp_data(Z) - Ezy_field.get_comp_data(Z))
                                       / ref_Ezy_field[Z].frobenius_norm())
                            plt.colorbar()
                            plt.title('Ezy error')

                            plt.figure()
                            plt.imshow((ref_B_field.get_comp_data(X) - B_field.get_comp_data(X))
                                       / ref_B_field[X].frobenius_norm())
                            plt.colorbar()
                            plt.title('Bx error')

                            plt.figure()
                            plt.imshow((ref_B_field.get_comp_data(Y) - B_field.get_comp_data(Y))
                                       / ref_B_field[Y].frobenius_norm())
                            plt.colorbar()
                            plt.title('By error')
                            plt.show()

                        except (OSError):
                            pass

                    E_data = Ezx_field[Z].get_data() + Ezy_field[Z].get_data()
                    Bx_data = B_field[X].get_data()
                    By_data = B_field[Y].get_data()

                    k = omega / c
                    Ez_theory = np.real(np.exp(1.j * (k * np.abs(x_vals-0.5) - omega * ts[-1]))) \
                                * np.log(np.abs(x_vals-Ly/2)) * permittivity * 10

                    plt.figure()
                    plt.plot(E_data[npts // 2, :])
                    plt.plot(Ez_theory, '--')
                    # plt.plot(-np.log( np.abs(x_vals-Lx/2) / (Lx/2) ) * 0.5 * 1e7, ':')
                    plt.plot(-np.log(np.abs(x_vals - Lx / 2)) * delta_fct_gtn.norm() / permittivity * 1e-5, ':')
                    plt.title('Ez')

                    print('delta norm', Jz_gtn(omega/4).norm(is_sqrt=False))

                    fig1, ax1 = plt.subplots()
                    time_ = ts[-1]
                    im1 = (ax1.imshow(E_data.T,
                                    extent=(x_vals[0], x_vals[-1],
                                            y_vals[0], y_vals[-1]),
                                    aspect='auto', origin='lower'))

                    fig1.colorbar(im1, ax=ax1)
                    ax1.set_title(f'time {time_:3.4f} CUTOFF {values[index]}')
                    ax1.set_xlabel('x'), ax1.set_ylabel('y')

                    fig1, ax1 = plt.subplots()
                    time_ = ts[-1]
                    im1 = (ax1.imshow(Bx_data.T,
                                      extent=(x_vals[0], x_vals[-1],
                                              y_vals[0], y_vals[-1]),
                                      aspect='auto', origin='lower'))

                    fig1.colorbar(im1, ax=ax1)
                    ax1.set_title(f'Bx time {time_:3.4f} CUTOFF {values[index]}')
                    ax1.set_xlabel('x'), ax1.set_ylabel('y')

                    divB_data = divB.get_comp_data()
                    if divB_data is not None:
                        plt.figure()
                        plt.imshow(divB_data)
                        # plt.imshow(np.rot90(divB_data))
                        plt.title(f'div(B) time {time_:3.4f} CUTOFF {values[index]}')
                        plt.colorbar()
                        plt.show()

                    fig10, ax10 = plt.subplots()
                    index = 0
                    ax10.plot(ts, max_bond_Ezx, label='Ezx')  # CUTOFF/DMAX
                    ax10.plot(ts, max_bond_Ezy, label='Ezy')  # CUTOFF/DMAX
                    ax10.plot(ts, [mb[0] for mb in max_bond_B], label=f'Bx')
                    ax10.plot(ts, [mb[1] for mb in max_bond_B], label=f'By')
                    ax10.set_title(f'Max Bond, CUTOFF {values[index]}')
                    ax10.set_xlabel('Time')
                    ax10.set_ylabel('Max Bond')
                    ax10.legend()
                    plt.show()

                    if index > 0:
                        E_comp = init_Ezx_gtn.add(init_Ezy_gtn)
                        E_non_comp = components_dict[key_name]
                        norm_diff = E_comp.get_data() - E_non_comp.get_data()   ## this is not correctly named.

                        plt.figure()
                        plt.imshow(norm_diff)
                        plt.title(
                            f'norm diff time {time_:3.4f} CUTOFF {values[index]}')
                        plt.colorbar()
                        plt.show()

            # Record the end time
            end_time = time.time()

            # Calculate the elapsed time
            elapsed_time += [(end_time - start_time)]

            print('wall time', time.time() - time1)

            if plotting:
                max_bond_B = np.array(max_bond_B)
                max_bond_Ezx = np.array(max_bond_Ezx)
                max_bond_Ezy = np.array(max_bond_Ezy)
                mem_size_Ezx = np.array(mem_size_Ezx)
                mem_size_Ezy = np.array(mem_size_Ezy)
                mem_size_B = np.array(mem_size_B)

                key_name_Ezx = f"max_bond_Ezx_{index}"
                key_name_Ezy = f"max_bond_Ezy_{index}"
                key_name_B = f"max_bond_B_{index}"
                data_max_bond_E[index] = np.array([x for x in zip(max_bond_Ezx, max_bond_Ezy)])
                data_max_bond_B[index] = max_bond_B

                key_name_mem_Ezx = f"mem_size_Ezx_{index}"
                key_name_mem_Ezy = f"mem_size_Ezy_{index}"
                key_name_mem_B = f"mem_size_B_{index}"
                data_mem_size_E[index] = mem_size_Ezx
                data_mem_size_E[index] = mem_size_Ezx
                data_mem_size_B[index] = mem_size_B

            # Increment the index at the end of each loop iteration
            index += 1


            ts_ = ts[nt0:]

            if plotting:

                # Iterate over all indices in rel_diff_dict and plot
                fig9, ax9 = plt.subplots()
                for index in rel_diff_dict:
                    if index == 0:
                        continue
                    ax9.plot(ts, rel_diff_dict[index],
                            label=f'CUTOFF {values[index]}')  # CUTOFF/DMAX
                ax9.set_title('Relative Difference')
                ax9.set_xlabel('Time')
                ax9.set_ylabel('Relative Difference')
                ax9.set_yscale('log')  # Set y-axis to logarithmic scale
                ax9.legend()

                plt.show()

                fig10, ax10 = plt.subplots()
                index = 0
                ax10.plot(ts_, data_max_bond_E[index][:,0], label='Ezx')  # CUTOFF/DMAX
                ax10.plot(ts_, data_max_bond_E[index][:,1], label='Ezy')  # CUTOFF/DMAX
                ax10.set_title(f'Max Bond E, CUTOFF {values[index]}')
                ax10.set_xlabel('Time')
                ax10.set_ylabel('Max Bond')
                ax10.legend()

                plt.show()

                fig11, ax11 = plt.subplots()
                index = 0
                ax11.plot(ts_, data_max_bond_B[index][:, 0], label=f'Bx')
                ax11.plot(ts_, data_max_bond_B[index][:, 1], label=f'By')
                ax11.set_title(f'Max Bond B, CUTOFF {values[index]}')
                ax11.set_xlabel('Time')
                ax11.set_ylabel('Max Bond')
                ax11.legend()

                plt.show()

                fig12, ax12 = plt.subplots()
                co = 0
                for index in data_mem_size_E:
                    if co == 0:
                        pass
                    else:
                        ax12.plot(ts_[:nt], data_mem_size_E[index],
                                  label=f'CUTOFF {values[co]}')  # CUTOFF/DMAX
                    co += 1
                ax12.set_title('Memory Size E')
                ax12.set_xlabel('Time')
                ax12.set_ylabel('Memory (kB)')
                ax12.legend()

                plt.show()

                fig13, ax13 = plt.subplots()
                co = 0
                for index in data_mem_size_B:
                    if co == 0:
                        pass
                    else:
                        ax13.plot(ts_[:nt], data_mem_size_B[index][:, 0],
                                  label=f'Bx, CUTOFF {values[co]}')  # CUTOFF/DMAX
                        # ax13.plot(ts_[:nt], data_mem_size_B[index][:, 1],
                        #           label=f'By, DMAX {values[i]}')
                    co += 1
                ax13.set_title('Memory Size B')
                ax13.set_xlabel('Time')
                ax13.set_ylabel('Memory (kB)')
                ax13.legend()

                plt.show()

            elapsed_time = np.array(elapsed_time)
            print('elapsed time', elapsed_time)

    ### compress final results
    fig1, ax1 = plt.subplots(1,1)
    Ezx_gtn = Ezx_field[Z]
    Ezy_gtn = Ezy_field[Z]
    Bx_gtn = B_field[X]
    By_gtn = B_field[Y]

    ref_Ezx_field = Ezx_field.copy()
    ref_Ezy_field = Ezy_field.copy()
    ref_B_field = B_field.copy()
    ref_Ezx_field.reload_data(f'data_Jz/deltaJ_Ezx_w{omega}_co{1e-30}')
    ref_Ezy_field.reload_data(f'data_Jz/deltaJ_Ezy_w{omega}_co{1e-30}')
    ref_B_field.reload_data(f'data_Jz/deltaJ_B_w{omega}_co{1e-30}')
    print('diff', ref_Ezx_field[Z].distance(Ezx_field[Z]) / ref_Ezx_field[Z].frobenius_norm())
    print('diff', ref_Ezy_field[Z].distance(Ezy_field[Z]) / ref_Ezy_field[Z].frobenius_norm())
    print('diff', ref_B_field[X].distance(B_field[X]) / ref_B_field[X].frobenius_norm())
    print('diff', ref_B_field[Y].distance(B_field[Y]) / ref_B_field[Y].frobenius_norm())

    cutoffs = [1e-4, 1e-6, 1e-8, 1e-10, 1e-16, 1e-20, 1e-25, 1e-30]
    errs_Ezx, errs_Ezy, errs_Bx, errs_By = [], [], [], []
    rank_Ezx, rank_Ezy, rank_Bx, rank_By = [], [], [], []
    for cutoff in cutoffs:
        tmp = Ezx_gtn.copy()
        tmp.compress(compress_opts={'cutoff': cutoff})
        errs_Ezx += [Ezx_gtn.distance(tmp) / Ezx_gtn.frobenius_norm()]
        rank_Ezx += [tmp.max_bond()]

        tmp = Ezy_gtn.copy()
        tmp.compress(compress_opts={'cutoff': cutoff})
        errs_Ezy += [Ezy_gtn.distance(tmp) / Ezy_gtn.frobenius_norm()]
        rank_Ezy += [tmp.max_bond()]

        tmp = Bx_gtn.copy()
        tmp.compress(compress_opts={'cutoff': cutoff})
        errs_Bx += [Bx_gtn.distance(tmp) / Bx_gtn.frobenius_norm()]
        rank_Bx += [tmp.max_bond()]

        tmp = By_gtn.copy()
        tmp.compress(compress_opts={'cutoff': cutoff})
        errs_By += [By_gtn.distance(tmp) / By_gtn.frobenius_norm()]
        rank_By += [tmp.max_bond()]

    log_cutoffs = [np.log10(cutoff)/2 for cutoff in cutoffs]

    # plt.figure()
    # plt.semilogy(log_cutoffs, errs_log, 'x-', label='log')
    # plt.semilogy(log_cutoffs, errs_sqrt, 'x-', label='sqrt')
    # plt.semilogy(log_cutoffs, errs_osc, 'x-', label='osc')
    # plt.semilogy(log_cutoffs, errs_log_osc, 'x-', label='log osc')
    # plt.legend()
    # plt.title('compression error')
    #
    # plt.figure()
    # plt.plot(log_cutoffs, rank_log, 'x-', label='log')
    # plt.plot(log_cutoffs, rank_sqrt, 'x-', label='sqrt')
    # plt.plot(log_cutoffs, rank_osc, 'x-', label='osc')
    # plt.plot(log_cutoffs, rank_log_osc, 'x-', label='log osc')
    # plt.legend()
    # plt.title(f'QTT rank {TN_layout}')

    ax1.plot(log_cutoffs, rank_Ezx, 'x-', label=r'$E_{z,x}$')
    ax1.plot(log_cutoffs, rank_Ezy, 'x-', label=r'$E_{z,y}$')
    ax1.plot(log_cutoffs, rank_Bx, 'x-', label=r'$B_x$')
    ax1.plot(log_cutoffs, rank_By, 'x-', label=r'$B_y$')

    # ax1.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99))
    ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 0.99))
    ax1.set_xlabel(r'$\log_{10}(\epsilon)$'), ax1.set_ylabel('rank')
    ax1.set_ylim(top=256, bottom=20)



    plt.show()

    print('done')
