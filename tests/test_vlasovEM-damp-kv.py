import os, sys, pickle, time, glob

sys.path.append('../')

from setup.configs import *
import setup.helper as helper_test
import helper_quimb as helper

from axis import Axis
import axis_map
from basis.basis_k import FourierBasis
from basis.basis_hermite import AWHermiteGaussianBasis
from basis.basis_hermite import HermiteGaussianBasis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from grid_comb import GridsComb
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_vlasovEM import VlasovMaxwell
from vlasov_tests import LandauDamping

"""
Landau damping (k-space in space, real space in velocity)
"""

save_figs = False
save_data = False
load_data = False
restart   = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_data_v2/VP1-damp-kv/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data/VP1-damp-kv/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)

#######################
# standard parameters #
#######################

plasma_config = UnitsConfiguration()
charge = plasma_config.e  # unit of charge
eV = plasma_config.eV     # unit of kB*T

mass_e = 1.0  # mass of electron
mass_i = 1836 * mass_e  # mass of ion
n0_e = 1.0  # electron number density
n0_i = n0_e  # ion number density
T_e = 1.0  # electron temperature [eV]
T_i = T_e  # ion temperature [eV]

ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config.vth    # electron thermal speed
wp_e  = elc_config.wp     # plasma e' frequency
lamD  = elc_config.lamD   # Debye length
wp_i  = ion_config.wp     # plasma i+ frequency
vth_i = ion_config.vth    # ion thermal speed


def maxwellian(*vs, density=1.0, flow=0.0, mass=1.0, temp=1.0):
    """ vs: velocity grid for each dimension
            summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
        flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    vth2 = eV * temp / mass
    return helper_test.maxwellian(*vs, vth2=vth2, density=density, flow=flow)



#####################
# define parameters #
#####################

### system parameters ###
A = 0.5
k = 0.10 / lamD

## runtime parameter
q = 2
K = 1
Lx, Lve, Lvi = 6, 6, 6

TN_layout = LayoutType.SEQUENTIAL
ax_map_key = 'FB'

order = 1   # derivative order
T = 5.0
do_adapt_dt = False  # True
dt_frac = 0.9
te_order = 4
dt = 0.005

DMAX = None
DMAX_F = None

VP_test = LandauDamping(k=k, A=A,
                        Lxs=(Lx,), Lves=(Lve,), Lvis=(Lvi,), KX=1, KV=1, q=2,
                        x_map_key = 'F', ve_map_key = 'B',
                        basis_xs = FourierBasis(),
                        ion_config = ion_config, elc_config = elc_config,
                        #
                        coll_type = None, do_adapt_coll = True, coll_coeff_e = 0.005, coll_rate_e = 0.01,
                        DMAX = DMAX, DMAX_F = DMAX_F, compress_style = 1, compress_style_mod = 0,
                        do_adapt_dt=do_adapt_dt, dt_frac=dt_frac, dt=dt, T=T, te_order=te_order,
                        grid_layout=TN_layout
                        )

collision_config = VP_test.collision_config

te_compress_i = VP_test.get_compression_config(DMAX=DMAX)
te_compress_e = VP_test.get_compression_config(DMAX=DMAX)
te_compress_E = VP_test.get_compression_config(DMAX=None)
te_compress_B = VP_test.get_compression_config(DMAX=None)

print('te compress fe', te_compress_e)
print('te compress fi', te_compress_i)
print('te compress E', te_compress_E)
print('te compress B', te_compress_B)

comp_levels = VP_test.get_compress_levels()

print('compress', DMAX, VP_test.compress_style, VP_test.compress_style_mod)

#############################
## start building PDE system
#############################

# ## specify k values, hermite moments used
# # dk = 0.10 -- lowest frequency mode
# npts_x, npts_ve, npts_vi = q ** Lx, q ** Lve, q ** Lvi
# x_lims, ve_lims, vi_lims = get_grid_params(grid_idx)
# vi_min, vi_max = vi_lims
# ve_min, ve_max = ve_lims
# kmax = k * npts_x / 2
# k_vals = np.linspace(-kmax, kmax, npts_x, endpoint=False)
# ve_vals = np.linspace(ve_min, ve_max, npts_ve,endpoint=False)
# vi_vals = np.linspace(vi_min, vi_max, npts_vi,endpoint=False)

# npts_x, npts_ve, npts_vi = VP_test.npts_x[0], VP_test.npts_ve[0], VP_test.npts_vi[0]
# v_max_e, v_max_i = np.max(np.abs(VP_test.ve_lims)), np.max(np.abs(VP_test.vi_lims))


## initial conditions
x_min, x_max = VP_test.x_lims[0]
ve_min, ve_max = VP_test.ve_lims[0]
vi_min, vi_max = VP_test.vi_lims[0]
tot_x = 2*np.pi/k

## electron density:  1 + A * np.cos(k * x)
npts_x = VP_test.get_npts_x(0)
fe_x = np.zeros(npts_x, dtype=complex)
fe_x[npts_x // 2] = 1.0 * n0_e
fe_x[npts_x // 2 - 1] = A / 2
fe_x[npts_x // 2 + 1] = A / 2
ve_vals = VP_test.ve_axes[0].xpts
fe_v = helper_test.maxwellian(ve_vals, vth2=vth_e**2, density=n0_e, flow=0.0)

## ion density
fi_x = np.zeros(npts_x, dtype=complex)
print('VP_test', VP_test.ion_config.n0)
fi_x[npts_x // 2] = 1.0 * n0_i
vi_vals = VP_test.vi_axes[0].xpts
fi_v = helper_test.maxwellian(vi_vals, vth2=vth_i**2, density=n0_i, flow=0.0)

# ## V field: -A / k ** 2 * np.cos(k * x)
# V_x = np.zeros(npts_x, dtype=complex)
# V_x[npts_x // 2 - 1] = -A / k ** 2 / 2
# V_x[npts_x // 2 + 1] = -A / k ** 2 / 2

## E field: -A / k * np.sin(k * x)
E_x = np.zeros(npts_x, dtype=complex)
E_x[npts_x // 2 - 1] = -1.j * A / k / 2
E_x[npts_x // 2 + 1] = 1.j * A / k / 2


# ## define axis objects; position -> k-space,
ax_x  = VP_test.pos_axes[0]
ax_vi = VP_test.vi_axes[0]
ax_ve = VP_test.ve_axes[0]
print('ax_x zero ind', ax_x.zero_ind)

X = ax_x.coordinate
V = ax_ve.coordinate

# ## define MPS grids
grid_i = VP_test.grid_i
grid_e = VP_test.grid_e
grid_X = VP_test.grid_X


## define initial grid1DTNs
fe_x = ax_x.map_state_to_mps(fe_x)
fi_x = ax_x.map_state_to_mps(fi_x)
fe_v = ax_ve.map_state_to_mps(fe_v)
fi_v = ax_vi.map_state_to_mps(fi_v)

init_fe_gtn = grid_e.make_mps_ndim({ax_x: fe_x, ax_ve: fe_v})
init_fi_gtn = grid_i.make_mps_ndim({ax_x: fi_x, ax_vi: fi_v})
init_E_gtn  = grid_X.map_state_to_mps(E_x)
init_B_gtn   = grid_X.make_empty_gridTN()
init_phi_gtn = grid_X.make_empty_gridTN()
init_psi_gtn = grid_X.make_empty_gridTN()

deriv_v = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)
init_fe_gtn.ax_deriv_configs.update({ax_ve: deriv_v})
init_fi_gtn.ax_deriv_configs.update({ax_vi: deriv_v})


### time step info
if dt is None:
    dt = VP_test.get_dt([A / k * charge / mass_e])
num_tsteps = int(T / dt)
print('dt', dt, num_tsteps)

### get file naming string
fstr = VP_test.get_fstr()
print('filename', fstr)

try:
    if not load_data:  raise IOError
    fe_t = optional_load(open(fdir + 'fe_' + fstr + '.pkl', 'rb'))
    fi_t = optional_load(open(fdir + 'fi_' + fstr + '.pkl', 'rb'))
    E_t = optional_load(fdir + 'E_' + fstr + '.pkl')
    B_t = optional_load(fdir + 'B_' + fstr + '.pkl')
    phi_t = optional_load(fdir + 'E_' + fstr + '.pkl')
    psi_t = optional_load(fdir + 'E_' + fstr + '.pkl')
    max_bond_fe = pickle.load(open(fdir + 'maxD_fe_' + fstr + '.pkl', 'rb'))
    max_bond_fi = pickle.load(open(fdir + 'maxD_fi_' + fstr + '.pkl', 'rb'))
    max_bond_E = pickle.load(open(fdir + 'maxD_E_' + fstr + '.pkl', 'rb'))
    # max_bond_B = pickle.load(open(fdir + 'maxD_B_' + fstr + '.pkl', 'rb'))
    # max_bond_phi = pickle.load(open(fdir + 'maxD_phi_' + fstr + '.pkl', 'rb'))
    # max_bond_psi = pickle.load(open(fdir + 'maxD_psi_' + fstr + '.pkl', 'rb'))
    nrg_fe_ts = pickle.load(open(fdir + 'nrg_e_' + fstr + '.pkl', 'rb'))
    nrg_fi_ts = pickle.load(open(fdir + 'nrg_i_' + fstr + '.pkl', 'rb'))
    nrg_E_ts = pickle.load(open(fdir + 'nrg_E_' + fstr + '.pkl', 'rb'))
    ts = pickle.load(open(fdir + 'ts_' + fstr + '.pkl', 'rb'))

    nt = num_tsteps
    print('loaded data')

    fe_data = grid_e.map_mps_to_state(fe_t) if fe_t is not None else None
    fi_data = grid_i.map_mps_to_state(fi_t) if fi_t is not None else None
    E_data = grid_X.map_mps_to_state(E_t) if E_t is not None else None

except(IOError, EOFError):

    try:
        if restart_from_T > 0:
            restart_fstr = fstr
            fe_t = pickle.load(open(fdir + 'fe_' + restart_fstr + '.pkl', 'rb'))
            fi_t = pickle.load(open(fdir + 'fi_' + restart_fstr + '.pkl', 'rb'))
            E_t = optional_load(fdir + 'E_' + restart_fstr + '.pkl')
            B_t = optional_load(fdir + 'B_' + restart_fstr + '.pkl')
            phi_t = optional_load(fdir + 'phi_' + restart_fstr + '.pkl')
            psi_t = optional_load(fdir + 'psi_' + restart_fstr + '.pkl')
            max_bond_fe = pickle.load(open(fdir + 'maxD_fe_' + restart_fstr + '.pkl', 'rb'))
            max_bond_fi = pickle.load(open(fdir + 'maxD_fi_' + restart_fstr + '.pkl', 'rb'))
            max_bond_E = pickle.load(open(fdir + 'maxD_E_' + restart_fstr + '.pkl', 'rb'))
            # max_bond_B = pickle.load(open(fdir + 'maxD_B_' + restart_fstr + '.pkl', 'rb'))
            # max_bond_psi = pickle.load(open(fdir + 'maxD_psi_' + restart_fstr + '.pkl', 'rb'))
            # max_bond_phi = pickle.load(open(fdir + 'maxD_phi_' + restart_fstr + '.pkl', 'rb'))
            nrg_fe_ts = pickle.load(open(fdir + 'nrg_e_' + fstr + '.pkl', 'rb'))
            nrg_fi_ts = pickle.load(open(fdir + 'nrg_i_' + fstr + '.pkl', 'rb'))
            nrg_E_ts = pickle.load(open(fdir + 'nrg_E_' + fstr + '.pkl', 'rb'))
            ts = pickle.load(open(fdir + 'ts_' + fstr + '.pkl', 'rb'))
            print('restarting from ', restart_fstr)
            ts = list(ts)
            nt = len(ts)
        else:
            if not restart:  raise ValueError

            re_fstr = fstr
            print(re_fstr)

            restart_fstrs = glob.glob(sdir + 'restart/fe_' + re_fstr + '*.pkl')
            print(sdir + 'restart/')
            print('restart?', len(restart_fstrs))
            nts = []
            for f in restart_fstrs:
                nt_ind = f.find('nt')
                nt_str = f[nt_ind + 2:-4]
                nts += [int(nt_str)]
            nt = max(nts)
            print('nt', nt)
            nts.remove(nt)

            try:
                raise NameError
                for nt_ in nts:  ## remove old restart files
                    if nt_ > nt:
                        print('removing restart', nt_)
                        restart_files = glob.glob(sdir + 'restart/*' + re_fstr + f'-nt{nt_}.pkl')
                        for f in restart_files:  os.remove(f)
            except NameError:
                pass

            fe_t = optional_load(sdir + 'restart/fe_' + re_fstr + f'-nt{nt}.pkl')
            fi_t = optional_load(sdir + 'restart/fi_' + re_fstr + f'-nt{nt}.pkl')
            E_t = optional_load(sdir + 'restart/E_' + re_fstr + f'-nt{nt}.pkl')
            B_t = optional_load(sdir + 'restart/B_' + re_fstr + f'-nt{nt}.pkl')
            phi_t = optional_load(sdir + 'restart/phi_' + re_fstr + f'-nt{nt}.pkl')
            psi_t = optional_load(sdir + 'restart/psi_' + re_fstr + f'-nt{nt}.pkl')
            max_bond_fe = pickle.load(open(sdir + 'restart/maxD_fe_' + re_fstr + f'.pkl', 'rb'))
            max_bond_fi = pickle.load(open(sdir + 'restart/maxD_fi_' + re_fstr + f'.pkl', 'rb'))
            max_bond_E = pickle.load(open(sdir + 'restart/maxD_E_' + re_fstr + f'.pkl', 'rb'))
            # max_bond_B = pickle.load(open(sdir + 'restart/maxD_B_' + re_fstr + f'.pkl', 'rb'))
            # max_bond_phi = pickle.load(open(sdir + 'restart/maxD_phi_' + re_fstr + f'.pkl', 'rb'))
            # max_bond_psi = pickle.load(open(sdir + 'restart/maxD_psi_' + re_fstr + f'.pkl', 'rb'))
            print('loaded', nt, len(max_bond_E))
            nrg_fe_ts = pickle.load(open(sdir + 'restart/nrg_e_' + re_fstr + f'.pkl', 'rb'))
            nrg_fi_ts = pickle.load(open(sdir + 'restart/nrg_i_' + re_fstr + f'.pkl', 'rb'))
            nrg_E_ts = pickle.load(open(sdir + 'restart/nrg_E_' + re_fstr + f'.pkl', 'rb'))
            ts = pickle.load(open(sdir + 'restart/ts_' + re_fstr + f'.pkl', 'rb'))
            ts = list(ts)
            print('loaded', nt, len(nrg_fe_ts))

        max_bond_fe = max_bond_fe[:nt + 1]
        max_bond_fi = max_bond_fi[:nt + 1]
        max_bond_E = max_bond_E[:nt + 1]
        # max_bond_B = max_bond_B[:nt + 1]
        # max_bond_phi = max_bond_phi[:nt + 1]
        # max_bond_psi = max_bond_psi[:nt + 1]
        nrg_fe_ts = nrg_fe_ts[:nt + 1]
        nrg_fi_ts = nrg_fi_ts[:nt + 1]
        nrg_E_ts = nrg_E_ts[:nt + 1]
        ts = ts[:nt + 1]
        total_E0 = nrg_E_ts[0] + nrg_fi_ts[0] + nrg_fe_ts[0]

        fig7, ax7 = plt.subplots()
        ax7.semilogy(ts, np.abs(nrg_fe_ts), label='elec')
        ax7.semilogy(ts, np.abs(nrg_fi_ts), label='ion')
        ax7.semilogy(ts, np.abs(nrg_E_ts), label='E field')
        total_nrg_t = np.array(nrg_fe_ts) + np.array(nrg_fi_ts) + np.array(nrg_E_ts)
        ax7.semilogy(ts, np.abs(total_nrg_t), label='total')

        # try:
        #     ref_fstr = f'/home/erikaye/gkeyll_tests/vm-damp/L{L:02d}_A{Astr}/' + \
        #                f'Ex_t_m{int(mass_i / mass_e)}_k{int(100 * k):03d}.pkl'
        #     print(ref_fstr)
        #     ref_data = pickle.load(open(ref_fstr, 'rb'))
        #     ax7.semilogy(ref_data[0], ref_data[1] * 0.5, 'k--', label='ref')
        # except IOError:
        #     pass

        ax7.legend()
        ax7.set_xlabel('t')
        ax7.set_ylabel('energy')
        if save_figs: fig7.savefig(fdir + 'plot_nrg_' + fstr + '.png')
        print(fdir + 'plot_nrg_' + fstr + '.png')
        # exit()

        init_fe_gtn.data = fe_t
        init_fi_gtn.data = fi_t
        init_E_gtn.data = E_t
        init_B_gtn.data = B_t
        init_phi_gtn.data = phi_t
        init_psi_gtn.data = psi_t

        restarted = True
        print('loaded restart', nt)

    except(ValueError, FileNotFoundError):
        nt = 0
        ts = [0]
        restarted = False

        init_fe_data = init_fe_gtn.get_data()
        init_fi_data = init_fi_gtn.get_data()
        init_E = E_x
        print('mapped mps to vector')

        v2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_ve], x_power=2)
        v2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vi], x_power=2)

        nrg_fe = init_fe_gtn.meas_expec(v2e_mpo) * mass_e / 2 / tot_x
        nrg_fi = init_fi_gtn.meas_expec(v2i_mpo) * mass_i / 2 / tot_x
        nrg_E  = (A/k/2)**2  * 2*np.pi/k / tot_x
        print('tot_x', k, 2*np.pi/k, tot_x)

        nrg_fe_ts = [nrg_fe]
        nrg_fi_ts = [nrg_fi]
        nrg_E_ts  = [nrg_E]
        nrg_B_ts = [0.0]
        print('nrg', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1], nrg_B_ts[-1])
        # exit()

        norm_fe = init_fe_gtn.norm()
        norm_fi = init_fi_gtn.norm()
        print('norm', norm_fe, norm_fi)

        print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
        max_bond_fe = [init_fe_gtn.max_bond()]
        max_bond_fi = [init_fi_gtn.max_bond()]
        max_bond_E = [init_E_gtn.max_bond()]
        max_bond_B = [np.nan]
        max_bond_phi = [np.nan]
        max_bond_psi = [np.nan]

        err_e_ts = [0.]
        err_i_ts = [0.]

    init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, compress_config=te_compress_e)
    init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, compress_config=te_compress_i)
    init_E_field = Field('E', grid_X, data={ax_x.coordinate: init_E_gtn}, is_sqrt=True, compress_config=te_compress_E)
    init_B_field = Field('B', grid_X, data={ax_x.coordinate: init_B_gtn}, is_sqrt=True, compress_config=te_compress_B)
    init_phi_field = ScalarField('phi', grid_X, data=init_phi_gtn, is_sqrt=True, compress_config=te_compress_E)
    init_psi_field = ScalarField('psi', grid_X, data=init_psi_gtn, is_sqrt=True, compress_config=te_compress_B)

    print('current nrgs', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1])
    print('field norm',init_fe_field.norm(), init_fi_field.norm())


    ### BUILD PDE
    vm_sys = VlasovMaxwell(init_fe_field, init_fi_field, init_E_field, init_B_field, init_phi_field, init_psi_field,
                           coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                           elc_params=elc_config, ion_params=ion_config, normalize=True, zipup=False, clean=False,
                           is_electrostatic=True,
                           te_order=te_order, compress_levels=comp_levels, )
    vm_sys.collision = collision_config
    print('colls', collision_config.coll_params)

    vm_sys.compress_F = VP_test.compress_F
    vm_sys.compress_F_opts = VP_test.compress_F_opts
    print('compress E?',  vm_sys.compress_F, vm_sys.compress_F_opts)

    print('E nrg', vm_sys.E.norm()**2 * 1./ 2 / tot_x)


    if nt == 0:
        fig1, ax1 = plt.subplots(2, 1)
        time_ = nt * dt
        fi_real = ax_x.basis.get_realspace_1D(init_fi_data, 0)
        fi_real = ax_vi.basis.get_realspace_1D(fi_real, 1)
        im1 = ax1[0].imshow(np.real(fi_real).T, extent=(-np.pi/k, np.pi/k, vi_min, vi_max),
                            aspect='auto', origin='lower')

        fe_real = ax_x.basis.get_realspace_1D(init_fe_data, 0)
        fe_real = ax_ve.basis.get_realspace_1D(fe_real, 1)
        im2 = ax1[1].imshow(np.real(fe_real).T, extent=(-np.pi/k, np.pi/k, ve_min, ve_max),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
        ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
        # if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')
        # plt.close(fig1)

        # V_data = vp_sys.V.get_field_data(comps=0)
        # plt.figure()
        # plt.plot(x_vals, V_data, label='V field')
        # plt.legend()

        plt.show()


    if restarted:
        if do_adapt_dt:
            em_term = vm_sys.compute_force_term(is_ion=False)
            em_data = em_term.get_comp_data(ax_x.coordinate)

            F_max = np.max(np.abs(em_data))
            dt = VP_test.get_dt(F_maxs=[F_max])
            print('restarted dt', dt)

    time1 = time.time()

    ## for measuring energy
    v2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_ve], x_power=2)
    v2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vi], x_power=2)

    while ts[-1] < T:

        walltime = time.time()
        vm_sys = vm_sys.next_time_step(dt, compress_level=1, verbose_plot=False)
        print('wall time', time.time() - walltime)

        fe_field = vm_sys.fe
        max_bond_fe += [fe_field.component.max_bond()]
        fe_data = fe_field.get_comp_data()
        nrg_fe = fe_field.component.meas_expec(v2e_mpo) * mass_e / 2 / tot_x

        fi_field = vm_sys.fi
        max_bond_fi += [fi_field.component.max_bond()]
        fi_data = fi_field.get_comp_data()
        nrg_fi = fi_field.component.meas_expec(v2i_mpo) * mass_i / 2 / tot_x

        E_field = vm_sys.E
        max_bond_E += [E_field[ax_x.coordinate].max_bond()]
        E_data = E_field.get_comp_data(ax_x.coordinate)
        energy_E_t = E_field.norm() ** 2 * 1. / 2 / tot_x  # electric field energy

        # magnetic field energy
        if vm_sys.B is None or vm_sys.B[ax_x.coordinate].data is None:
            energy_B_t = np.nan
            max_bond_B += [np.nan]
        else:
            B_data = vm_sys.B.get_comp_data(ax_x.coordinate)
            energy_B_t = vm_sys.B.norm() ** 2 * 1. / 2 / tot_x
            max_bond_B += [vp_sys.B[ax_x.coordinate].max_bond()]

        # plt.figure()
        # plt.plot(k_vals, np.real(E_data), label='E', marker='x')
        # plt.plot(k_vals, np.imag(E_data), label='E', marker='.')
        # plt.plot(k_vals, np.real(pot_field.get_comp_data()), label='V', marker='x')
        # plt.plot(k_vals, np.imag(pot_field.get_comp_data()), label='V', marker='.')
        # plt.legend()
        #
        # plt.figure()
        # plt.imshow(np.real(vp_sys.fe.get_comp_data()))
        # plt.title('fe real')
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(np.imag(vp_sys.fe.get_comp_data()))
        # plt.title('fe imag')
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(np.real(vp_sys.fi.get_comp_data()))
        # plt.title('fi real')
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(np.imag(vp_sys.fi.get_comp_data()))
        # plt.title('fi imag')
        # plt.colorbar()
        # plt.show()

        nrg_fe_ts += [nrg_fe]
        nrg_fi_ts += [nrg_fi]
        nrg_E_ts += [energy_E_t]
        nrg_B_ts += [energy_B_t]

        ts += [ts[-1] + dt]
        nt += 1

        if do_adapt_dt:
            E_max = np.max(np.abs(E_data))
            dt = VP_test.get_dt([E_max * charge / mass_e,])

        norm = vm_sys.fe.norm() - vm_sys.fi.norm()
        print('result norm', nt, ts[-1], dt, norm, # nrg_fe, nrg_fi,
              energy_E_t,
              E_field.max_bond(X), fi_field.max_bond(), fe_field.max_bond(), Lx, Lvi, Lve)

        if nt % 250 == 0:
            fig1, ax1 = plt.subplots(2, 1)
            time_ = ts[-1]
            im1 = ax1[0].imshow(np.log10(np.abs(fi_data)).T, extent=(x_min, x_max, vi_min, vi_max),
                                aspect='auto', origin='lower')
            im2 = ax1[1].imshow(np.log10(np.abs(fe_data)).T, extent=(x_min, x_max, ve_min, ve_max),
                                aspect='auto', origin='lower')

            fig1.colorbar(im1, ax=ax1[0])
            fig1.colorbar(im2, ax=ax1[1])
            ax1[0].set_title(f'time {time_:3.2f}')
            ax1[0].xaxis.set_visible(False)
            ax1[0].set_xlabel('kx'), ax1[0].set_ylabel('v')
            ax1[1].set_xlabel('kx'), ax1[1].set_ylabel('v')
            if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')

            E_data = vm_sys.E.get_comp_data(ax_x.coordinate)
            plt.figure()
            plt.plot(np.arange(npts_x) * k, np.real(E_data), label='E field real')
            plt.plot(np.arange(npts_x) * k, np.imag(E_data), label='E field imag')
            # plt.plot(np.arange(npts_x) * k, np.real(V_data), label='V field real')
            # plt.plot(np.arange(npts_x) * k, np.imag(V_data), label='V field imag')
            plt.legend()

            if not save_figs:
                plt.show()
            plt.close()
            plt.close(fig1)

            if save_data:
                if not os.path.exists(sdir + 'restart/'):
                    os.makedirs(sdir + 'restart/')
                else:
                    try:
                        raise NameError
                        restart_files = glob.glob(sdir + 'restart/*' + fstr + f'-nt{save_nt}.pkl')
                        for f in restart_files:  os.remove(f)
                    except NameError:
                        pass

                save_nt = nt
                optional_dump(vm_sys.fe.component.data, open(sdir + 'restart/' + 'fe_' + fstr + f'-nt{nt}.pkl', 'wb'))
                optional_dump(vm_sys.fi.component.data, open(sdir + 'restart/' + 'fi_' + fstr + f'-nt{nt}.pkl', 'wb'))
                optional_dump(vm_sys.E[X], sdir + 'restart/' + 'E_' + fstr + f'-nt{nt}.pkl')
                optional_dump(vm_sys.B[X], sdir + 'restart/' + 'B_' + fstr + f'-nt{nt}.pkl')
                optional_dump(vm_sys.phi.component.data, sdir + 'restart/' + 'phi_' + fstr + f'-nt{nt}.pkl')
                optional_dump(vm_sys.psi.component.data, sdir + 'restart/' + 'psi_' + fstr + f'-nt{nt}.pkl')
                pickle.dump(max_bond_fe, open(sdir + 'restart/' + 'maxD_fe_' + fstr + '.pkl', 'wb'))
                pickle.dump(max_bond_fi, open(sdir + 'restart/' + 'maxD_fi_' + fstr + '.pkl', 'wb'))
                pickle.dump(max_bond_E, open(sdir + 'restart/' + 'maxD_E_' + fstr + '.pkl', 'wb'))
                # pickle.dump(max_bond_B, open(sdir + 'restart/' + 'maxD_B_' + fstr + '.pkl', 'wb'))
                # pickle.dump(max_bond_phi, open(sdir + 'restart/' + 'maxD_phi_' + fstr + '.pkl', 'wb'))
                # pickle.dump(max_bond_psi, open(sdir + 'restart/' + 'maxD_psi_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_fe_ts, open(sdir + 'restart/nrg_e_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_fi_ts, open(sdir + 'restart/nrg_i_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_E_ts, open(sdir + 'restart/nrg_E_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_B_ts, open(sdir + 'restart/nrg_B_' + fstr + '.pkl', 'wb'))
                pickle.dump(ts, open(sdir + 'restart/ts_' + fstr + '.pkl', 'wb'))

    if save_data:
        optional_dump(vm_sys.fe.component.data, fdir + 'fe_' + fstr + '.pkl')
        optional_dump(vm_sys.fi.component.data, fdir + 'fi_' + fstr + '.pkl')
        optional_dump(vm_sys.E[ax_x.coordinate], fdir + 'E_' + fstr + '.pkl')
        optional_dump(vm_sys.B[ax_x.coordinate], fdir + 'B_' + fstr + '.pkl')
        optional_dump(vm_sys.phi.component.data, fdir + 'phi_' + fstr + '.pkl')
        optional_dump(vm_sys.psi.component.data, fdir + 'psi_' + fstr + '.pkl')
        pickle.dump(max_bond_fe, open(fdir + 'maxD_fe_' + fstr + '.pkl', 'wb'))
        pickle.dump(max_bond_fi, open(fdir + 'maxD_fi_' + fstr + '.pkl', 'wb'))
        pickle.dump(max_bond_E, open(fdir + 'maxD_E_' + fstr + '.pkl', 'wb'))
        # pickle.dump(max_bond_B, open(fdir + 'maxD_B_' + fstr + '.pkl', 'wb'))
        # pickle.dump(max_bond_phi, open(fdir + 'maxD_phi_' + fstr + '.pkl', 'wb'))
        # pickle.dump(max_bond_psi, open(fdir + 'maxD_psi_' + fstr + '.pkl', 'wb'))
        pickle.dump(nrg_fe_ts, open(fdir + 'nrg_e_' + fstr + '.pkl', 'wb'))
        pickle.dump(nrg_fi_ts, open(fdir + 'nrg_i_' + fstr + '.pkl', 'wb'))
        pickle.dump(nrg_E_ts, open(fdir + 'nrg_E_' + fstr + '.pkl', 'wb'))
        pickle.dump(nrg_B_ts, open(fdir + 'nrg_B_' + fstr + '.pkl', 'wb'))
        pickle.dump(ts, open(fdir + 'ts_' + fstr + '.pkl', 'wb'))

    print('wall time', time.time() - time1)

fig1, ax1 = plt.subplots(2, 1)
time_ = ts[-1]
fi_real = ax_x.basis.get_realspace_1D(fi_data, 0)
fi_real = ax_vi.basis.get_realspace_1D(fi_real, 1)
im1 = ax1[0].imshow(np.real(fi_real).T, extent=(-np.pi/k, np.pi/k, vi_min, vi_max),
                    aspect='auto', origin='lower')

fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
fe_real = ax_ve.basis.get_realspace_1D(fe_real, 1)
im2 = ax1[1].imshow(np.real(fe_real).T, extent=(-np.pi/k, np.pi/k, ve_min, ve_max),
                    aspect='auto', origin='lower')

fig1.colorbar(im1, ax=ax1[0])
fig1.colorbar(im2, ax=ax1[1])
ax1[0].set_title(f'time {time_:3.2f}')
ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_T{time_:2.1f}.png')

fig2, ax2 = plt.subplots()
# ax2.plot(max_bond_fe,label=r'$\displaystyle f_e$')
# ax2.plot(max_bond_fi,label=r'$\displaystyle f_i$')
ax2.plot(np.log(max_bond_fe) / np.log(q ** K), label=r'$f_e$')
ax2.plot(np.log(max_bond_fi) / np.log(q ** K), label=r'$f_i$')
ax2.plot(np.log(max_bond_E) / np.log(q ** K), label='E')
ax2.plot([0, num_tsteps], [(Lx+Lve) // 2] * 2, ':', label='max f')
ax2.plot([0, num_tsteps], [(Lx+Lve) // 2] * 2, ':', label='max V')
ax2.legend()
ax2.set_xlabel('t')
ax2.set_ylabel('log_{q^K}(max bond dimension)')
if save_figs:   fig2.savefig(fdir + 'plot_maxbond_' + fstr + '.png')

# energy_E_t = np.sum(Ex_t*np.conj(Ex_t),axis=1)*dx*1./2   # electric field energy
energy_E_t = np.array(nrg_E_ts)
energy_fi_t = np.array(nrg_fi_ts)
energy_fe_t = np.array(nrg_fe_ts)
total_nrg_t = energy_fe_t + energy_fi_t + energy_E_t
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
ax7.semilogy(ts, np.abs(total_nrg_t), ls='--',label='total')

try:
    # ref_fstr = f'/home/erikaye/gkeyll_tests/vm-damp/L{L:02d}_A{Astr}/' + \
    #                  f'Ex_t_m{int(mass_i/mass_e)}_k{int(100*k):03d}.pkl'
    # ref_data = pickle.load(open(ref_fstr,'rb'))
    # ax7.semilogy(ref_data[0],ref_data[1]*0.5,'k--',label='ref')

    ref_fstr1 = f'testU1t_TN{TN_layout.value}_cut16_k{k:1.2f}_A{Astr}'
    ref_fstr = ref_fstr1 + \
               f'_L{L:02d}_cfl{cfl}_T{T}_te{te_order}_m{mass_i}'
    nrg_E_ts = pickle.load(open(fdir + 'nrg_E_' + ref_fstr + '.pkl', 'rb'))
    ts = pickle.load(open(fdir + 'ts_' + ref_fstr + '.pkl', 'rb'))
    ax7.semilogy(ts, nrg_E_ts, 'k--', label='ref (cutoff 1.0e-8)')
except(NameError, IOError):
    pass

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
ax7.semilogy( helper.entanglement_entropy_all(fe_field.component.data), '-x', label='elec')
ax7.semilogy( helper.entanglement_entropy_all(fi_field.component.data), '-o', label='ion')

ax7.legend()
ax7.set_xlabel('bond number')
ax7.set_ylabel('EE')
if save_figs:
    fig7.savefig(fdir + 'plot_EE_' + fstr + '.png')
    print(fdir + 'plot_EE_' + fstr + '.png')

plt.show()


print('done')

