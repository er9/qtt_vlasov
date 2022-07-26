import os, sys, pickle, time, glob

import numpy as np

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
from vlasov_tests import Whistler

"""
Landau damping (k-space in space, real space in velocity)
"""

save_figs = False
save_data = False
load_data = False
restart   = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_data_v2/VM1-whistler-kv/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data/VM1-whistler-kv/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)

#######################
# standard parameters #
#######################

plasma_config = UnitsConfiguration()
charge = plasma_config.e  # unit of charge
eV = plasma_config.eV     # unit of kB*T
mu0 = plasma_config.mu0   # permeability
eps0 = plasma_config.eps0 # permittivity

mass_e = 1.0            # mass of electron
mass_i = 1836 * mass_e  # mass of ion
n0_e = 1.0              # electron number density
n0_i = n0_e             # ion number density
T_e = 1.0e-6            # electron temperature [eV], so that vth_e/c = 1.0e-3
T_i = T_e               # ion temperature [eV]

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


def initial_state_k(kx_vals, vx_vals, vy_vals, vz_vals, A, k, vth=1.0, density=1.0, flow=0.0):
    """ initial state for w
    """
    dk = kx_vals[1] - kx_vals[0]
    x_vals = np.linspace(-np.pi/dk, np.pi/dk, len(kx_vals))
    f_real = initial_state_real(x_vals, vx_vals, vy_vals, vz_vals, A, k, vth=vth, density=density, flow=flow)
    f_k = np.fft.fft(f_real, axis=0)
    return f_k


def initial_state_real(x_vals, vx_vals, vy_vals, vz_vals, A, k,
                       vth=1.0, density=1.0, flow=0.0):
    """ initial state for w
    """
    x_mesh, vx_mesh, vy_mesh, vz_mesh = np.meshgrid(x_vals, vx_vals, vy_vals, vz_vals, indexing='ij')

    dv_y = A*np.sin(k*x_mesh)
    dv_z = A*np.sin(k*x_mesh)
    vy_mesh = vy_mesh + dv_y
    vz_mesh = vz_mesh + dv_z

    return helper_test.maxwellian( vx_mesh, vy_mesh, vz_mesh,
                                   vth2=vth**2, density=density, flow=flow )



#####################
# define parameters #
#####################

### system parameters ###
A = 0.5
k = 0.10 / lamD
B0 = 1.0

## runtime parameter
q = 2
KX = 1
KV = 3
Lx, Lve, Lvi = 7, 6, 6

TN_layout = LayoutType.SEQUENTIAL
ax_map_key = 'FB'

order = 1   ## derivative order
T = 5.0
do_adapt_dt = False  # True
dt_frac = 0.9
te_order = 4
dt = 0.005

DMAX = None
DMAX_F = None

VP_test = Whistler(k=k, A=A,
                   Lxs=(Lx,), Lves=(Lve,Lve,Lve), Lvis=(Lvi,Lvi,Lvi), KX=KX, KV=KV, q=q,
                   x_map_key = 'F', ve_map_key = 'B',
                   #
                   ion_config = ion_config, elc_config = elc_config,
                   coll_type=None, do_adapt_coll = True, coll_coeff_e = 0.005, coll_rate_e = 0.01,
                   DMAX = DMAX, DMAX_F = DMAX_F, compress_style = 1, compress_style_mod = 0,
                   do_adapt_dt=do_adapt_dt, dt_frac=dt_frac, dt=dt, T=T, te_order=te_order,
                   grid_layout=TN_layout
                   )

collision_config = VP_test._collision_config

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

## initial conditions

## define axis objects; position -> k-space,
ax_x = VP_test.pos_axes[0]
ax_vx_i = VP_test.vi_axes[0]
ax_vy_i = VP_test.vi_axes[1]
ax_vz_i = VP_test.vi_axes[2]
ax_vx_e = VP_test.ve_axes[0]
ax_vy_e = VP_test.ve_axes[1]
ax_vz_e = VP_test.ve_axes[2]
print('ax_x zero ind', ax_x.zero_ind)

k_min, k_max = VP_test.x_lims[0]
ve_x_min, ve_x_max = VP_test.ve_lims[0]
vi_x_min, vi_x_max = VP_test.vi_lims[0]
tot_x = 2*np.pi/k

coords_x  = VP_test.pos_coordsys
coords_vi = VP_test.vi_coordsys
coords_ve = VP_test.ve_coordsys

X, Y, Z = coords_x.coords
VX, VY, VZ = coords_ve.coords


# ## define MPS grids
grid_i = VP_test.grid_i
grid_e = VP_test.grid_e
grid_X = VP_test.grid_X

kx_vals = ax_x.xpts
vx_e_vals = ax_vx_e.xpts
vy_e_vals = ax_vy_e.xpts
vz_e_vals = ax_vz_e.xpts
vx_i_vals = ax_vx_i.xpts
vy_i_vals = ax_vy_i.xpts
vz_i_vals = ax_vz_i.xpts


## electron density:  1 + A * np.cos(k * x)
fe_data = initial_state_k(kx_vals, vx_e_vals, vy_e_vals, vz_e_vals, vth=vth_e, flow=0.0, density=n0_e)

## ion density
npts_x = ax_x.npts
fi_x = np.zeros(npts_x, dtype=complex)
fi_x[npts_x // 2] = 1.0 * n0_i
fi_vx = helper_test.maxwellian(vx_i_vals, vth2=vth_i**2, density=n0_i, flow=0.0)
fi_vy = helper_test.maxwellian(vx_i_vals, vth2=vth_i**2, density=n0_i, flow=0.0)
fi_vz = helper_test.maxwellian(vx_i_vals, vth2=vth_i**2, density=n0_i, flow=0.0)

## E field: 0

## Bz field: - d/dx Bz = mu_0 (Jz + eps_0 dEz/dt) = mu_0 * q * n0 * A * cos(kx)
idx_km, idx_kp = npts_x // 2 - 1, npts_x // 2 + 1
Bz = np.zeros(npts_x, dtype=complex)
Bz[idx_kp] = mu0 * charge * n0_e * A * k * -1.j
Bz[idx_km] = mu0 * charge * n0_e * A * k * 1.j

## By field:   d/dx By = mu_0 (Jy + eps_0 dEy/dt) = mu_0 * q * n0 * A * sin(kx)
By = np.zeros(npts_x, dtype=complex)
By[idx_kp] = mu0 * charge * n0_e * A * k * -1
By[idx_km] = mu0 * charge * n0_e * A * k * -1


## define initial grid1DTNs
init_fe_gtn = grid_e.map_state_to_mps(fe_data)

fi_x = ax_x.map_state_to_mps(fi_x)
fi_vx = ax_vx_i.map_state_to_mps(fi_vx)
fi_vy = ax_vy_i.map_state_to_mps(fi_vy)
fi_vz = ax_vz_i.map_state_to_mps(fi_vz)
init_fi_gtn = grid_i.make_mps_ndim({ax_x: fi_x, ax_vx_i: fi_vx, ax_vy_i: fi_vy, ax_vz_i: fi_vz})
init_E_gtn   = grid_X.make_empty_gridTN()
init_By_gtn  = grid_X.make_mps_ndim({ax_x: By})
init_Bz_gtn  = grid_X.make_mps_ndim({ax_x: Bz})
init_phi_gtn = grid_X.make_empty_gridTN()
init_psi_gtn = grid_X.make_empty_gridTN()

deriv_v = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)
init_fe_gtn.ax_deriv_configs.update({ax_vx_e: deriv_v, ax_vy_e: deriv_v, ax_vz_e: deriv_v})
init_fi_gtn.ax_deriv_configs.update({ax_vx_i: deriv_v, ax_vy_i: deriv_v, ax_vz_i: deriv_v})


### time step info
if dt is None:
    dt = VP_test.get_dt([(B0 + A/k) * charge / mass_e])
num_tsteps = int(T / dt)
print('dt', dt, num_tsteps)

### get file naming string
fstr = VP_test.get_fstr()
print('filename', fstr)

########################
#### run simulation ####
########################

### define v2 mpo for measurement ###
vx2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vx_e], x_power=2)
vy2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vy_e], x_power=2)
vz2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vz_e], x_power=2)
v2e_mpo = vx2e_mpo.add(vy2e_mpo)
v2e_mpo = v2e_mpo.add(vz2e_mpo)

vx2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vx_i], x_power=2)
vy2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vy_i], x_power=2)
vz2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vz_i], x_power=2)
v2i_mpo = vx2i_mpo.add(vy2i_mpo)
v2i_mpo = v2i_mpo.add(vz2i_mpo)

#### define fields
init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, compress_config=te_compress_e)
init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, compress_config=te_compress_i)
init_E_field = Field('E', grid_X, data={X: None, Y: None, Z: None},
                     is_sqrt=True, compress_config=te_compress_E)
init_B_field = Field('B', grid_X, data={X: None, Y: init_By_gtn, Z: init_Bz_gtn},
                     is_sqrt=True, compress_config=te_compress_B)
init_phi_field = ScalarField('phi', grid_X, data=init_phi_gtn, is_sqrt=True, compress_config=te_compress_E)
init_psi_field = ScalarField('psi', grid_X, data=init_psi_gtn, is_sqrt=True, compress_config=te_compress_B)



### load data ###
try:
    if not load_data:  raise IOError
    fe_t = init_fe_field.reload_data(fdir + 'fe_' + fstr)
    fi_t = init_fe_field.reload_data(fdir + 'fi_' + fstr)
    E_t = init_E_field.reload_data(fdir + 'E_' + fstr)
    B_t = init_B_field.reload_data(fdir + 'B_' + fstr)
    phi_t = init_phi_field.reload_data(fdir + 'phi_' + fstr)
    psi_t = init_psi_field.reload_data(fdir + 'psi_' + fstr)
    max_bond_fe = pickle.load(open(fdir + 'maxD_fe_' + fstr + '.pkl', 'rb'))
    max_bond_fi = pickle.load(open(fdir + 'maxD_fi_' + fstr + '.pkl', 'rb'))
    max_bond_E = pickle.load(open(fdir + 'maxD_E_' + fstr + '.pkl', 'rb'))
    max_bond_B = pickle.load(open(fdir + 'maxD_B_' + fstr + '.pkl', 'rb'))
    # max_bond_phi = pickle.load(open(fdir + 'maxD_phi_' + fstr + '.pkl', 'rb'))
    # max_bond_psi = pickle.load(open(fdir + 'maxD_psi_' + fstr + '.pkl', 'rb'))
    nrg_fe_ts = pickle.load(open(fdir + 'nrg_e_' + fstr + '.pkl', 'rb'))
    nrg_fi_ts = pickle.load(open(fdir + 'nrg_i_' + fstr + '.pkl', 'rb'))
    nrg_E_ts = pickle.load(open(fdir + 'nrg_E_' + fstr + '.pkl', 'rb'))
    ts = pickle.load(open(fdir + 'ts_' + fstr + '.pkl', 'rb'))

    nt = num_tsteps
    print('loaded data')

    fe_data = init_fe_field.get_data()
    fi_data = init_fi_field.get_data()
    Ex_data = init_E_field.get_comp_data(X)
    By_data = init_B_field.get_comp_data(Y)

except(IOError, EOFError):

    try:
        if restart_from_T > 0:
            restart_fstr = fstr
            fe_t = init_fe_field.reload_data(fdir + 'fe_' + restart_fstr)
            fi_t = init_fe_field.reload_data(fdir + 'fi_' + restart_fstr)
            E_t = init_E_field.reload_data(fdir + 'E_' + restart_fstr)
            B_t = init_B_field.reload_data(fdir + 'B_' + restart_fstr)
            phi_t = init_phi_field.reload_data(fdir + 'phi_' + restart_fstr)
            psi_t = init_psi_field.reload_data(fdir + 'psi_' + restart_fstr)
            # fe_t = pickle.load(open(fdir + 'fe_' + restart_fstr + '.pkl', 'rb'))
            # fi_t = pickle.load(open(fdir + 'fi_' + restart_fstr + '.pkl', 'rb'))
            # E_t = optional_load(fdir + 'E_' + restart_fstr + '.pkl')
            # B_t = optional_load(fdir + 'B_' + restart_fstr + '.pkl')
            # phi_t = optional_load(fdir + 'phi_' + restart_fstr + '.pkl')
            # psi_t = optional_load(fdir + 'psi_' + restart_fstr + '.pkl')
            max_bond_fe = pickle.load(open(fdir + 'maxD_fe_' + restart_fstr + '.pkl', 'rb'))
            max_bond_fi = pickle.load(open(fdir + 'maxD_fi_' + restart_fstr + '.pkl', 'rb'))
            max_bond_E = pickle.load(open(fdir + 'maxD_E_' + restart_fstr + '.pkl', 'rb'))
            max_bond_B = pickle.load(open(fdir + 'maxD_B_' + restart_fstr + '.pkl', 'rb'))
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

            fe_t = init_fe_field.reload_data(sdir + 'restart/fe_' + re_fstr + f'-nt{nt}')
            fi_t = init_fe_field.reload_data(sdir + 'restart/fi_' + re_fstr + f'-nt{nt}')
            E_t = init_E_field.reload_data(sdir + 'restart/E_' + re_fstr + f'-nt{nt}')
            B_t = init_B_field.reload_data(sdir + 'restart/B_' + re_fstr + f'-nt{nt}')
            phi_t = init_phi_field.reload_data(fdir + 'restart/phi_' + re_fstr + f'-nt{nt}')
            psi_t = init_psi_field.reload_data(fdir + 'restart/psi_' + re_fstr + f'-nt{nt}')
            # fe_t = grid_e.load_gtn_data(sdir + 'restart/fe_' + re_fstr + f'-nt{nt}.pkl')
            # fi_t = grid_i.load_gtn_data(sdir + 'restart/fi_' + re_fstr + f'-nt{nt}.pkl')
            # E_t = grid_X.load_field_data(sdir + 'restart/E_' + re_fstr + f'-nt{nt}.pkl')
            # B_t = grid_X.load_field_data(sdir + 'restart/B_' + re_fstr + f'-nt{nt}.pkl')
            # phi_t = grid_X.load_gtn_data(sdir + 'restart/phi_' + re_fstr + f'-nt{nt}.pkl')
            # psi_t = grid_X.load_gtn_data(sdir + 'restart/psi_' + re_fstr + f'-nt{nt}.pkl')
            max_bond_fe = pickle.load(open(sdir + 'restart/maxD_fe_' + re_fstr + f'.pkl', 'rb'))
            max_bond_fi = pickle.load(open(sdir + 'restart/maxD_fi_' + re_fstr + f'.pkl', 'rb'))
            max_bond_E = pickle.load(open(sdir + 'restart/maxD_E_' + re_fstr + f'.pkl', 'rb'))
            max_bond_B = pickle.load(open(sdir + 'restart/maxD_B_' + re_fstr + f'.pkl', 'rb'))
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
        max_bond_B = max_bond_B[:nt + 1]
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

        ax7.legend()
        ax7.set_xlabel('t')
        ax7.set_ylabel('energy')
        if save_figs: fig7.savefig(fdir + 'plot_nrg_' + fstr + '.png')
        print(fdir + 'plot_nrg_' + fstr + '.png')
        # exit()

        # init_fe_gtn.data = fe_t
        # init_fi_gtn.data = fi_t
        # init_E_gtn.data = E_t
        # init_B_gtn.data = B_t
        # init_phi_gtn.data = phi_t
        # init_psi_gtn.data = psi_t

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

        nrg_fe = init_fe_gtn.meas_expec(v2e_mpo) * mass_e / 2 / tot_x
        nrg_fi = init_fi_gtn.meas_expec(v2i_mpo) * mass_i / 2 / tot_x
        nrg_E  = 0.0
        nrg_B  = A/k
        print('tot_x', k, 2*np.pi/k, tot_x)

        nrg_fe_ts = [nrg_fe]
        nrg_fi_ts = [nrg_fi]
        nrg_E_ts  = [nrg_E]
        nrg_B_ts  = [nrg_B]
        print('nrg', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1], nrg_B_ts[-1])


        norm_fe = init_fe_gtn.norm()
        norm_fi = init_fi_gtn.norm()
        print('norm', norm_fe, norm_fi)

        print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
        max_bond_fe = [init_fe_gtn.max_bond()]
        max_bond_fi = [init_fi_gtn.max_bond()]
        max_bond_E = [np.nan]
        max_bond_B = [init_E_gtn.max_bond()]
        max_bond_phi = [np.nan]
        max_bond_psi = [np.nan]

        err_e_ts = [0.]
        err_i_ts = [0.]

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
        fi_real = ax_vi.basis.get_realspace_1D(fi_real, 2)
        fi_real = ax_vi.basis.get_realspace_1D(fi_real, 3)
        im1 = ax1[0].imshow(np.real(fi_real[:,:,ax_vy_i.npts//2, ax_vz_i.npts//2]).T,
                            extent=(-np.pi / k, np.pi / k, vi_x_min, vi_x_max),
                            aspect='auto', origin='lower')

        fe_real = ax_x.basis.get_realspace_1D(init_fe_data, 0)
        fe_real = ax_ve.basis.get_realspace_1D(fe_real, 1)
        fe_real = ax_ve.basis.get_realspace_1D(fe_real, 2)
        fe_real = ax_ve.basis.get_realspace_1D(fe_real, 3)
        im2 = ax1[1].imshow(np.real(fe_real[:,:,ax_vy_e.npts//2, ax_vz_e.npts//2]).T,
                            extent=(-np.pi / k, np.pi / k, ve_x_min, ve_x_max),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
        ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
        # if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')
        # plt.close(fig1)

        plt.show()


    if restarted:
        if do_adapt_dt:
            em_term = vm_sys.compute_force_term(is_ion=False)
            em_data = em_term.get_comp_data(ax_x.coordinate)

            F_max = np.max(np.abs(em_data)) * charge / mass_e
            VP_test.get_dt(F_maxs=[F_max])
            print('restarted dt', dt)

    time1 = time.time()

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
        max_bond_E += [E_field.max_bond()]
        Ex_data = E_field.get_comp_data(X)
        energy_E_t = E_field.norm() ** 2 * 1. / 2 / tot_x  # electric field energy

        B_field = vm_sys.B
        max_bond_E += [B_field.max_bond()]
        By_data = B_field.get_comp_data(Y)
        energy_B_t = B_field.norm() ** 2 * 1. / 2 / tot_x  # electric field energy

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
            em_term = vm_sys.compute_force_term(is_ion=False)
            em_data = em_term.get_comp_data(X)

            F_max = np.max(np.abs(em_data)) * charge / mass_e
            VP_test.get_dt(F_maxs=[F_max])
            # print('restarted dt', dt)

        norm = vm_sys.fe.norm() - vm_sys.fi.norm()
        print('result norm', nt, ts[-1], dt, norm, # nrg_fe, nrg_fi,
              energy_E_t, energy_B_t,
              E_field.max_bond(X), fi_field.max_bond(), fe_field.max_bond(), Lx, Lvi, Lve)

        if nt % 250 == 0:
            fig1, ax1 = plt.subplots(2, 1)
            time_ = ts[-1]
            im1 = ax1[0].imshow(np.log10(np.abs(fi_data)).T, extent=(x_min, x_max, vi_x_min, vi_x_max),
                                aspect='auto', origin='lower')
            im2 = ax1[1].imshow(np.log10(np.abs(fe_data)).T, extent=(x_min, x_max, ve_x_min, ve_x_max),
                                aspect='auto', origin='lower')

            fig1.colorbar(im1, ax=ax1[0])
            fig1.colorbar(im2, ax=ax1[1])
            ax1[0].set_title(f'time {time_:3.2f}')
            ax1[0].xaxis.set_visible(False)
            ax1[0].set_xlabel('kx'), ax1[0].set_ylabel('v')
            ax1[1].set_xlabel('kx'), ax1[1].set_ylabel('v')
            if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')

            Ex_data = vm_sys.E.get_comp_data(X)
            plt.figure()
            plt.plot(np.arange(npts_x) * k, np.real(Ex_data), label='E field real')
            plt.plot(np.arange(npts_x) * k, np.imag(Ex_data), label='E field imag')
            plt.legend()

            By_data = vm_sys.B.get_comp_data(Y)
            plt.figure()
            plt.plot(np.arange(npts_x) * k, np.real(By_data), label='B field real')
            plt.plot(np.arange(npts_x) * k, np.imag(By_data), label='B field imag')
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
                vm_sys.fe.save_data(sdir + 'restart/' + 'fe_' + fstr + f'-nt{nt}')
                vm_sys.fi.save_data(sdir + 'restart/' + 'fi_' + fstr + f'-nt{nt}')
                vm_sys.E.save_data(sdir + 'restart/' + 'E_' + fstr + f'-nt{nt}')
                vm_sys.B.save_data(sdir + 'restart/' + 'B_' + fstr + f'-nt{nt}')
                vm_sys.phi.save_data(sdir + 'restart/' + 'phi_' + fstr + f'-nt{nt}')
                vm_sys.psi.save_data(sdir + 'restart/' + 'psi_' + fstr + f'-nt{nt}')
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
        vm_sys.fe.save_data(fdir + 'fe_' + fstr)
        vm_sys.fi.save_data(fdir + 'fi_' + fstr)
        vm_sys.E.save_data(fdir + 'E_' + fstr)
        vm_sys.B.save_data(fdir + 'B_' + fstr)
        vm_sys.phi.save_data(fdir + 'phi_' + fstr)
        vm_sys.psi.save_data(fdir + 'psi_' + fstr)
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
im1 = ax1[0].imshow(np.real(fi_real).T, extent=(-np.pi / k, np.pi / k, vi_x_min, vi_x_max),
                    aspect='auto', origin='lower')

fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
fe_real = ax_ve.basis.get_realspace_1D(fe_real, 1)
im2 = ax1[1].imshow(np.real(fe_real).T, extent=(-np.pi / k, np.pi / k, ve_x_min, ve_x_max),
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

