import os, sys, pickle, time, glob

import matplotlib.pyplot as plt

sys.path.append('../')

from setup.configs import *

from axis import Axis
import axis_map
from basis.basis_k import FourierBasis
from basis.basis_hermite import AWHermiteGaussianBasis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from grid_comb import GridsComb
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_vlasovES import VlasovPoisson
from vlasov_tests import LandauDamping

"""
Landau damping (k-space, Hermite expansion)
"""

save_figs = True
save_data = True
load_data = True
restart   = True
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

zipup = True
print('zipup', zipup)

if restart or save_data or save_figs:
    # sdir = '/pool001/erikaye/tns_pde_v2/data2/VP2-damp-kv/'
    # sdir = '/pool001/erikaye/tns_pde_v2/data2/VP2-damp-kv-v2/'
    sdir = '/pool001/erikaye/tns_pde_v2/data2/VP2-damp-kv-v21/'
    # sdir = '/pool001/erikaye/tns_pde_v2/data2/VP2-damp-kv-v1/'
    # sdir = '/pool001/erikaye/tns_pde_v2/data2/VP2-damp-kv-v11/'
    # sdir = '/pool001/erikaye/tns_pde_v2/data2/VP2-damp-kv-v11nz/'
    # sdir = '/pool001/erikaye/tns_pde_v2/data2/VP2-damp-kv-nzv1/'
    # sdir = '/pool001/erikaye/tns_pde_v2/data2/VP2-damp-kv-nozip/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    # fdir = 'data2/VP2-damp-kv/'
    # fdir = 'data2/VP2-damp-kv-v2/'
    fdir = 'data2/VP2-damp-kv-v21/'
    # fdir = 'data2/VP2-damp-kv-v1/'
    # fdir = 'data2/VP2-damp-kv-v11/'
    # fdir = 'data2/VP2-damp-kv-v11nz/'
    # fdir = 'data2/VP2-damp-kv-nzv1/'
    # fdir = 'data2/VP2-damp-kv-nozip/'
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


# def get_v_grid_params(idx):
#     if idx == 0:
#         ve_bounds = -6 * vth_e, 6 * vth_e
#         vi_bounds = -6 * vth_i, 6 * vth_i
#     elif idx == 1:
#         ve_bounds = -6 * vth_e, 6 * vth_e
#         vi_bounds = -30 * vth_i, 30 * vth_i
#     else:
#         raise IndexError(f'grid parameter for index {idx} not defined')
#     return ve_bounds, vi_bounds



#####################
# define parameters #
#####################

### system parameters ###
A = 0.5  # 0.001
k = 0.75 / lamD
kx = 0.5 / lamD
ky = np.sqrt(k**2 - kx**2)
ks = (kx, ky)  # x,y,...
print('k', k, np.linalg.norm(ks), ks)


## runtime parameter
q = 2
K = 2
Lx, Ly = 6, 6
Lxe, Lye = 8, 8
Lxi, Lyi = 6, 6
order = 1

TN_layout = LayoutType.COMB
x_ax_map_key = 'BB'
v_ax_map_key = 'BB'

# TN_layout = LayoutType.SEQUENTIAL
# x_ax_map_key = 'FF'
# v_ax_map_key = 'BB'

T = 60.0
te_order = 4
do_adapt_dt = False # True
dt = 0.01
dt_frac = 0.9

DMAX = 32
DMAX_F = DMAX

VP_test = LandauDamping(ks=(kx, ky), A=A,
                        Lxs=(Lx, Ly), Lves=(Lxe, Lye), Lvis=(Lxi, Lyi),
                        KX=K, KV=K, q=2,
                        x_map_key = x_ax_map_key, ve_map_key = v_ax_map_key,
                        basis_xs = FourierBasis(),
                        ion_config = ion_config, elc_config = elc_config,
                        #
                        coll_type = None, do_adapt_coll = True, coll_coeff_e = 0.005, coll_rate_e = 0.01,
                        DMAX = DMAX, DMAX_F = DMAX_F, compress_style = 2, compress_style_mod = 0,
                        do_adapt_dt=do_adapt_dt, dt_frac=dt_frac, dt=dt, T=T, te_order=te_order,
                        grid_layout=TN_layout
                        )



#### compression options
te_compress_i = VP_test.get_compression_config(DMAX=DMAX)
te_compress_e = VP_test.get_compression_config(DMAX=DMAX)
te_compress_V = VP_test.get_compression_config(DMAX=None)

print('te compress fe', te_compress_e)
print('te compress fi', te_compress_i)
print('te compress V', te_compress_V)

comp_levels = VP_test.get_compress_levels()

print('compress', DMAX, VP_test.compress_style, VP_test.compress_style_mod)


#############################
## start building PDE system
#############################

## specify k values, hermite moments used
# # dk = k -- lowest frequency mode
# npts_x, npts_ve, npts_vi = q ** Lx, q ** Lve, q ** Lvi
# kx_max = kx * npts_x / 2
# ky_max = ky * npts_x / 2
# kz_max = kz * npts_x / 2
# kx_vals = np.linspace(-kx_max, kx_max, npts_x, endpoint=False)
# ky_vals = np.linspace(-ky_max, ky_max, npts_x, endpoint=False)
# # k_vals = k_vals.reshape((q,)*Lx).transpose().reshape(-1)
# (ve_min, ve_max), (vi_min, vi_max) = get_v_grid_params(grid_idx)
# vi_vals = np.linspace(vi_min, vi_max, npts_vi, endpoint=False)
# ve_vals = np.linspace(ve_min, ve_max, npts_ve, endpoint=False)
# print('ve', ve_min, ve_max, npts_ve)

## initial conditions
x_min, x_max = VP_test.x_lims[0]
y_min, y_max = VP_test.x_lims[1]
vxe_min, vxe_max = VP_test.ve_lims[0]
vye_min, vye_max = VP_test.ve_lims[1]
vxi_min, vxi_max = VP_test.vi_lims[0]
vyi_min, vyi_max = VP_test.vi_lims[1]
tot_x = (2*np.pi / kx) * (2*np.pi / ky)

### axis objects
ax_x, ax_y = VP_test.pos_axes
ax_vxi, ax_vyi = VP_test.vi_axes
ax_vxe, ax_vye = VP_test.ve_axes
print('ax_x zero ind', ax_x.zero_ind)
print('dvi, dve', ax_vxi.dx, ax_vyi.dx, ax_vxe.dx, ax_vye.dx)

X, Y, Z = VP_test.pos_coordsys.coords
VX, VY, VZ = VP_test.ve_coordsys.coords

npts_x = ax_x.npts
npts_y = ax_y.npts
npts_vxe = ax_vxe.npts
npts_vye = ax_vye.npts
npts_vxi = ax_vxi.npts
npts_vyi = ax_vyi.npts

# ## define MPS grids
grid_i = VP_test.grid_i
grid_e = VP_test.grid_e
grid_X = VP_test.grid_X


#### initial conditions ####
## electron density:  1 + A * np.cos(k * x)
f0x = np.zeros(npts_x, dtype=complex)
f0x[npts_x // 2] = 1.0
fpx = np.zeros(npts_x, dtype=complex)
fpx[npts_x // 2 + 1] = 1.0   # A / 2
fmx = np.zeros(npts_x, dtype=complex)
fmx[npts_x // 2 - 1] = 1.0   # A / 2

f0y = np.zeros(npts_y, dtype=complex)
f0y[npts_y // 2] = 1.0
fpy = np.zeros(npts_y, dtype=complex)
fpy[npts_y // 2 + 1] = 1.0   # A / 2
fmy = np.zeros(npts_y, dtype=complex)
fmy[npts_y // 2 - 1] = 1.0   # A / 2

fe_vx = maxwellian(ax_vxe.xpts, vth2=vth_e**2, density=n0_e, flow=0.0,)
fe_vy = maxwellian(ax_vye.xpts, vth2=vth_e**2, density=n0_e, flow=0.0,)
print('fe_v integ', np.sum(np.outer(fe_vx, fe_vy)) * ax_vxe.dx * ax_vye.dx)
init_fe_comps = [{ax_x: f0x, ax_y: f0y, ax_vxe: fe_vx, ax_vye: fe_vy},
                 {ax_x: fpx * A / 2, ax_y: fpy, ax_vxe: fe_vx, ax_vye: fe_vy},
                 {ax_x: fmx * A / 2, ax_y: fmy, ax_vxe: fe_vx, ax_vye: fe_vy},
                 ]
print('fe max', np.max(fe_vx), np.max(fe_vy))

## ion density
fi_vx = maxwellian(ax_vxi.xpts, vth2=vth_i**2, density=n0_i, flow=0.0,)
fi_vy = maxwellian(ax_vyi.xpts, vth2=vth_i**2, density=n0_i, flow=0.0,)
print('fi_v integ', np.sum(np.outer(fi_vx, fi_vy)) * ax_vxi.dx * ax_vyi.dx)
# v2_vals = np.outer(vi_vals**2, np.ones(len(vi_vals))) + np.outer(np.ones(len(vi_vals)), vi_vals**2)
# print('fi_v nrg', 0.5 * mass_i * np.sum(np.outer(fi_v, fi_v) * v2_vals) *ax_vxi.dx * ax_vyi.dx)
init_fi_comps = [{ax_x: f0x, ax_y: f0y, ax_vxi: fi_vx, ax_vyi: fi_vy}]
print('fi max', np.max(fi_vx), np.max(fi_vy))

# init_fe_v_comps = [{ax_vxe: maxwellian(ve_vals, vth2=vth_e**2, density=n0_e, flow=0.0,),
#                     ax_vye: maxwellian(ve_vals, vth2=vth_e**2, density=n0_e, flow=0.0,)}]
# init_fi_v_comps = [{ax_vxi: maxwellian(vi_vals, vth2=vth_i**2, density=n0_i, flow=0.0,),
#                     ax_vyi: maxwellian(vi_vals, vth2=vth_i**2, density=n0_i, flow=0.0,)}]


## V field: -A / k ** 2 * np.cos(k * x)
# V_x = np.zeros(npts_x, dtype=complex)
# V_x[npts_x // 2 - 1] = -A / k ** 2 / 2
# V_x[npts_x // 2 + 1] = -A / k ** 2 / 2
init_V_comps = [{ax_x: fpx * -A / (k ** 2) / 2, ax_y: fpx},
                {ax_x: fmx * -A / (k ** 2) / 2, ax_y: fmx}]

## E field: -A / k * np.sin(k * x)
# E_x = np.zeros(npts_x, dtype=complex)
# E_x[npts_x // 2 - 1] = -1.j * A / kx / 2
# E_x[npts_x // 2 + 1] = 1.j * A / kx / 2
# init_Ex_xy_comps = [{ax_x: fp * -1.j * A/ kx /2, ax_y: fp, ax_z: fp}, {ax_x: fm * 1.j * A/ kx /2, ax_y: fm, ax_z: fm}]

## define initial grid1DTNs
init_fe_gtn = grid_e.make_gtn_from_dicts(init_fe_comps, data_type=DataType.MPS)
init_fi_gtn = grid_i.make_gtn_from_dicts(init_fi_comps, data_type=DataType.MPS)
init_V_gtn  = grid_X.make_gtn_from_dicts(init_V_comps, data_type=DataType.MPS)


### time step info
if dt is None:
    dt = VP_test.get_dt(np.array([A*kx/k**2, A*ky/k**2]) * charge / mass_e)
num_tsteps = int(T / dt)
print('dt', dt, num_tsteps)

fstr = VP_test.get_fstr()
print('filename', fstr)


#######################
## start solving PDE ##
#######################

#### define v**2 for measurement ####
vx2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vxe], x_power=2)
vy2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vye], x_power=2)
v2e_mpo = vx2e_mpo.add(vy2e_mpo)

# v2e_mps =  grid_e.get_ones_mps().apply( v2e_mpo )
# v2e_data = v2e_mps.get_data(ax_select={ax_x: 0, ax_y: 0, ax_z: 0})
# plt.imshow(v2e_data[:,npts_vye//2,:])
# plt.colorbar()
# plt.show()

vx2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vxi], x_power=2)
vy2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vyi], x_power=2)
v2i_mpo = vx2i_mpo.add(vy2i_mpo)
print('v2 mpo max bond', v2e_mpo.max_bond(), v2i_mpo.max_bond())



#### load/initialize data ####
try:
    if not load_data:
        raise IOError
    fe_t = grid_e.load_gtn_data(fdir+'fe_'+fstr, gtn=init_fe_gtn)
    fi_t = grid_i.load_gtn_data(fdir+'fi_'+fstr, gtn=init_fi_gtn)
    # fe_t = pickle.load(open(fdir + 'fe_' + fstr + '.pkl', 'rb'))
    # fi_t = pickle.load(open(fdir + 'fi_' + fstr + '.pkl', 'rb'))
    max_bond_V = pickle.load(open(fdir + 'maxD_V_' + fstr + '.pkl', 'rb'))
    max_bond_fe = pickle.load(open(fdir + 'maxD_fe_' + fstr + '.pkl', 'rb'))
    max_bond_fi = pickle.load(open(fdir + 'maxD_fi_' + fstr + '.pkl', 'rb'))
    nrg_fe_ts = pickle.load(open(fdir + 'nrg_e_' + fstr + '.pkl', 'rb'))
    nrg_fi_ts = pickle.load(open(fdir + 'nrg_i_' + fstr + '.pkl', 'rb'))
    nrg_E_ts = pickle.load(open(fdir + 'nrg_E_' + fstr + '.pkl', 'rb'))
    ts = pickle.load(open(fdir + 'ts_' + fstr + '.pkl', 'rb'))

    nt = num_tsteps
    print('loaded data')

    fe_field = ScalarField('fe', grid_e, data=fe_t, compress_config=te_compress_e)
    fi_field = ScalarField('fi', grid_i, data=fi_t, compress_config=te_compress_i)

    # fe_data = grid_e.map_mps_to_state(fe_t)
    # fi_data = grid_i.map_mps_to_state(fi_t)

except(IOError, EOFError):

    try:
        if restart_from_T > 0:
            restart_fstr = fstr

            fe_t = grid_e.load_gtn_data(fdir+'fe_'+restart_fstr, gtn=init_fe_gtn)
            fi_t = grid_i.load_gtn_data(fdir+'fi_'+restart_fstr, gtn=init_fi_gtn)
            # fe_t = pickle.load(open(fdir + 'fe_' + restart_fstr + '.pkl', 'rb'))
            # fi_t = pickle.load(open(fdir + 'fi_' + restart_fstr + '.pkl', 'rb'))
            # Ex_t = pickle.load(open(fdir+'Ext_' +restart_fstr+'.pkl','rb'))
            max_bond_V = pickle.load(open(fdir + 'maxD_V_' + restart_fstr + '.pkl', 'rb'))
            max_bond_fe = pickle.load(open(fdir + 'maxD_fe_' + restart_fstr + '.pkl', 'rb'))
            max_bond_fi = pickle.load(open(fdir + 'maxD_fi_' + restart_fstr + '.pkl', 'rb'))
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
            # nt = 3000
            print('nt', nt)
            nts.remove(nt)

            try:
                # raise NameError
                for nt_ in nts:
                    if nt_ > nt or nt_%250 != 0:    ## remove old restart files
                        print('removing restart', nt_)
                        restart_files = glob.glob(sdir + 'restart/*' + re_fstr + f'-nt{nt_}.pkl')
                        for f in restart_files:  os.remove(f)
            except NameError:
                pass

            fe_t = grid_e.load_gtn_data(sdir+'restart/fe_'+re_fstr+f'-nt{nt}', gtn=init_fe_gtn)
            fi_t = grid_i.load_gtn_data(sdir+'restart/fi_'+re_fstr+f'-nt{nt}', gtn=init_fi_gtn)
            # fe_t = pickle.load(open(sdir + 'restart/fe_' + re_fstr + f'-nt{nt}.pkl', 'rb'))
            # fi_t = pickle.load(open(sdir + 'restart/fi_' + re_fstr + f'-nt{nt}.pkl', 'rb'))
            max_bond_V = pickle.load(open(sdir + 'restart/maxD_V_' + re_fstr + f'.pkl', 'rb'))
            max_bond_fe = pickle.load(open(sdir + 'restart/maxD_fe_' + re_fstr + f'.pkl', 'rb'))
            max_bond_fi = pickle.load(open(sdir + 'restart/maxD_fi_' + re_fstr + f'.pkl', 'rb'))
            print('loaded', nt, len(max_bond_V))
            nrg_fe_ts = pickle.load(open(sdir + 'restart/nrg_e_' + re_fstr + f'.pkl', 'rb'))
            nrg_fi_ts = pickle.load(open(sdir + 'restart/nrg_i_' + re_fstr + f'.pkl', 'rb'))
            nrg_E_ts = pickle.load(open(sdir + 'restart/nrg_E_' + re_fstr + f'.pkl', 'rb'))
            ts = pickle.load(open(sdir + 'restart/ts_' + re_fstr + f'.pkl', 'rb'))
            ts = list(ts)
            print('loaded', nt, len(nrg_fe_ts))

        max_bond_V = max_bond_V[:nt + 1]
        max_bond_fe = max_bond_fe[:nt + 1]
        max_bond_fi = max_bond_fi[:nt + 1]
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

        try:
            raise IOError
            ref_fstr = f'/home/erikaye/gkeyll_tests/vm-damp/L{L:02d}_A{Astr}/' + \
                       f'Ex_t_m{int(mass_i / mass_e)}_k{int(100 * k):03d}.pkl'
            print(ref_fstr)
            ref_data = pickle.load(open(ref_fstr, 'rb'))
            ax7.semilogy(ref_data[0], ref_data[1] * 0.5, 'k--', label='ref')
        except IOError:
            pass

        ax7.legend()
        ax7.set_xlabel('t')
        ax7.set_ylabel('energy')
        if save_figs: fig7.savefig(fdir + 'plot_nrg_' + fstr + '.png')
        print(fdir + 'plot_nrg_' + fstr + '.png')
        # exit()

        # if not is_comb:
        #     init_fe_gtn.data = fe_t
        #     init_fi_gtn.data = fi_t
        # else:
        #     pass
        #     # init_fe_gtn.data = grid_e.load_state(fe_t)
        #     # init_fi_gtn.data = grid_i.load_state(fi_t)

        # init_V_gtn.data = None

        # init_fe_gtn = GridTN1D(grid_e, data=fe_t, ax_deriv_configs={ax_x.axID: deriv_xe_up, ax_ve.axID: deriv_v})
        # init_fi_gtn = GridTN1D(grid_i, data=fi_t, ax_deriv_configs={ax_x.axID: deriv_xi_up, ax_ve.axID: deriv_v})
        # init_V_gtn  = GridTN1D(grid_V, data=None, ax_deriv_configs={ax_x.axID: deriv_x})

        restarted = True
        print('loaded restart', nt)

    except(ValueError, FileNotFoundError):
        nt = 0
        ts = [0]
        restarted = False

        # init_fe_data = init_fe_gtn.get_data()
        # init_fi_data = init_fi_gtn.get_data()
        # print('mapped mps to vector')

        nrg_fe = init_fe_gtn.meas_expec(v2e_mpo) * mass_e / 2 / tot_x
        nrg_fi = init_fi_gtn.meas_expec(v2i_mpo) * mass_i / 2 / tot_x
        nrg_E  = 1./2 * (1./2) * (A/k)**2  # * 2*np.pi/k  * (k/2/np.pi)
        # extra 1/2 --> 1/2/np.pi integ cos(a*x)^2 dx from 0 to 2*npi/a = 1/2

        nrg_fe_ts = [nrg_fe]
        nrg_fi_ts = [nrg_fi]
        nrg_E_ts  = [nrg_E]
        print('nrg', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1])

        norm_fe = init_fe_gtn.norm() / tot_x
        norm_fi = init_fi_gtn.norm() / tot_x
        print('norm', norm_fe, norm_fi)

        print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
        max_bond_V  = [init_V_gtn.max_bond()]
        max_bond_fe = [init_fe_gtn.max_bond()]
        max_bond_fi = [init_fi_gtn.max_bond()]

        err_e_ts = [0.]
        err_i_ts = [0.]

    init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, compress_config=te_compress_e)
    init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, compress_config=te_compress_i)
    init_V_field  = ScalarField('V', grid_X, data=init_V_gtn, compress_config=te_compress_V, is_sqrt=True)

    print('norm', init_fe_field.norm() / tot_x,
                  init_fi_field.norm() / tot_x)

    if init_V_field.component is not None and init_V_field.component.data is not None:
        elec_field = init_V_field.gradient()
        energy_E_t = elec_field.norm()**2 * 1. / 2 / tot_x     # electric field energy
        theory_Ex = A ** 2 * kx ** 2 / k ** 4 * 0.5
        theory_Ey = A ** 2 * ky ** 2 / k ** 4 * 0.5
        print('nrg theoretical', theory_Ex, theory_Ey, (theory_Ex + theory_Ey) / 2)
        print('E field nrg before', energy_E_t )


    ### BUILD PDE
    vp_sys = VlasovPoisson(init_fe_field, init_fi_field, init_V_field,
                           coords_x=VP_test.pos_coordsys, coords_ve=VP_test.ve_coordsys, coords_vi=VP_test.vi_coordsys,
                           elc_params=elc_config, ion_params=ion_config, normalize=True,
                           zipup=zipup, te_order=te_order, compress_levels=comp_levels)
    vp_sys.collision = VP_test._collision_config
    print('colls', VP_test._collision_config.coll_params)

    vp_sys.compress_F = VP_test.compress_F
    vp_sys.compress_F_opts = VP_test.compress_F_opts
    print('compress E?', vp_sys.compress_F, vp_sys.compress_F_opts)

    print('before update', vp_sys.V.component.ax_deriv_configs[ax_x].bc, ax_x)
    if init_V_gtn.data is not None:
        elec_field = vp_sys.V.gradient()
        energy_E_t = elec_field.norm() ** 2 * 1. / 2 / tot_x  # electric field energy
        print('E field nrg', energy_E_t)

    if True:  # restarted:
        vp_sys.update_V(inplace=True)
    print('after update', vp_sys.V.component.ax_deriv_configs[ax_x].bc)
    elec_field = vp_sys.V.gradient()
    energy_E_t = elec_field.norm() ** 2 * 1. / 2 / tot_x    # electric field energy
    print('E field nrg', energy_E_t)


    if nt == 0:
        fig1, ax1 = plt.subplots(2, 1)
        time_ = nt * dt

        fi_data = init_fi_gtn.get_data(ax_select={ax_vxi: npts_vxi//2, ax_vyi: npts_vyi//2})
        fi_real = ax_x.basis.get_realspace_1D(fi_data, 0)
        fi_real = ax_y.basis.get_realspace_1D(fi_real, 1)
        im1 = ax1[0].imshow(np.real(fi_real).T,
                            extent=(-np.pi/kx, np.pi/kx, -np.pi/ky, np.pi/ky),
                            aspect='auto', origin='lower')

        fi_data = init_fi_gtn.get_data(ax_select={ax_x: npts_x//2, ax_y: npts_y//2})
        fi_real = ax_vxi.basis.get_realspace_1D(fi_data, 0)
        fi_real = ax_vyi.basis.get_realspace_1D(fi_real, 1)
        im2 = ax1[1].imshow(np.real(fi_real).T,
                            extent=(vxi_min, vxi_max, vyi_min, vyi_max),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].set_xlabel('x'), ax1[0].set_ylabel('y')
        ax1[1].set_xlabel('vx'), ax1[1].set_ylabel('vy')

        fig2, ax2 = plt.subplots(2, 1)
        time_ = nt * dt
        fe_data = init_fe_gtn.get_data(ax_select={ax_vxe: npts_vxe // 2, ax_vye: npts_vye // 2})
        fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
        fe_real = ax_y.basis.get_realspace_1D(fe_real, 1)
        im1 = ax2[0].imshow(np.abs(fe_real).T,
                            extent=(-np.pi / kx, np.pi / kx, -np.pi / ky, np.pi / ky),
                            aspect='auto', origin='lower')

        fe_data = init_fe_gtn.get_data(ax_select={ax_x: npts_x // 2, ax_y: npts_y // 2})
        fe_real = ax_vxe.basis.get_realspace_1D(fe_data, 0)
        fe_real = ax_vye.basis.get_realspace_1D(fe_real, 1)
        im2 = ax2[1].imshow(np.real(fe_real).T,
                            extent=(vxe_min, vxe_max, vye_min, vye_max),
                            aspect='auto', origin='lower')

        fig2.colorbar(im1, ax=ax2[0])
        fig2.colorbar(im2, ax=ax2[1])
        ax2[0].set_title(f'time {time_:3.2f}')
        ax2[0].set_xlabel('x'), ax2[0].set_ylabel('y')
        ax2[1].set_xlabel('vx'), ax2[1].set_ylabel('vy')
        # if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')
        # plt.close(fig1)

        # V_data = vp_sys.V.get_field_data(comps=0)
        # plt.figure()
        # plt.plot(x_vals, V_data, label='V field')
        # plt.legend()

        plt.show()


    if restarted:
        if do_adapt_dt:
            V_field = vp_sys.V
            E_field = V_field.gradient()

            Ex_data = E_field.get_comp_data(X)
            Ex_max = np.max(np.abs(Ex_data))
            Ey_data = E_field.get_comp_data(Y)
            Ey_max = np.max(np.abs(Ey_data))

            # tot_npts = npts_x * npts_y
            # Ex_max = elec_field[X].norm(is_sqrt=True) * np.sqrt(tot_npts)  # np.max(np.abs(Ex_data))
            # Ey_max = elec_field[Y].norm(is_sqrt=True) * np.sqrt(tot_npts)  # np.max(np.abs(Ey_data))

            dt = VP_test.get_dt(F_maxs=np.array([Ex_max, Ey_max])*charge/mass_e)
            print('restarted dt', dt)

    time1 = time.time()

    while ts[-1] < T:

        walltime = time.time()
        vp_sys = vp_sys.next_time_step(dt, compress_level=1)  #, verbose_plot=(nt >= 40))
        print('wall time', time.time() - walltime)

        pot_field = vp_sys.V
        max_bond_V += [pot_field.max_bond()]
        elec_field = pot_field.gradient()
        elec_field.scalar_multiply(-1, inplace=True)
        # elec_field.is_sqrt = True
        # Ex_data = elec_field.get_comp_data(X)
        # Ey_data = elec_field.get_comp_data(Y)

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

        fe_field = vp_sys.fe
        max_bond_fe += [fe_field.max_bond()]
        # fe_data = fe_field.get_comp_data()

        fi_field = vp_sys.fi
        max_bond_fi += [fi_field.max_bond()]
        # fi_data = fi_field.get_comp_data()

        nrg_fe = fe_field.component.meas_expec(v2e_mpo) * mass_e / 2 / tot_x
        nrg_fi = fi_field.component.meas_expec(v2i_mpo) * mass_i / 2 / tot_x

        nrg_fe_ts += [nrg_fe]
        nrg_fi_ts += [nrg_fi]

        energy_E_t = elec_field.norm()**2 * 1. / 2 / tot_x     # electric field energy
        nrg_E_ts += [energy_E_t]

        ts += [ts[-1] + dt]
        nt += 1

        if do_adapt_dt:
            Ex_data = E_field.get_comp_data(X)
            Ex_max = np.max(np.abs(Ex_data))
            Ey_data = E_field.get_comp_data(Y)
            Ey_max = np.max(np.abs(Ey_data))

            # tot_npts = npts_x * npts_y
            # Ex_max = elec_field[X].norm(is_sqrt=True) * np.sqrt(tot_npts)  # np.max(np.abs(Ex_data))
            # Ey_max = elec_field[Y].norm(is_sqrt=True) * np.sqrt(tot_npts)  # np.max(np.abs(Ey_data))

            dt = VP_test.get_dt(F_maxs=np.array([Ex_max, Ey_max]) * charge / mass_e)
            # print('restarted dt', dt)

        norm = vp_sys.fe.norm() - vp_sys.fi.norm()
        print('result norm', nt, ts[-1], dt, norm, energy_E_t,
              pot_field.max_bond(), fe_field.max_bond(), Lx, Lxi, Lxe)
        # if nt >= 5:   exit()

        if nt % 10 == 0:
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
                vp_sys.fe.component.save_data(sdir+'restart/'+'fe_'+fstr+f'-nt{nt}')
                vp_sys.fi.component.save_data(sdir+'restart/'+'fi_'+fstr+f'-nt{nt}')
                pickle.dump(max_bond_V, open(sdir + 'restart/' + 'maxD_V_' + fstr + '.pkl', 'wb'))
                pickle.dump(max_bond_fe, open(sdir + 'restart/' + 'maxD_fe_' + fstr + '.pkl', 'wb'))
                pickle.dump(max_bond_fi, open(sdir + 'restart/' + 'maxD_fi_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_fe_ts, open(sdir + 'restart/nrg_e_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_fi_ts, open(sdir + 'restart/nrg_i_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_E_ts, open(sdir + 'restart/nrg_E_' + fstr + '.pkl', 'wb'))
                # pickle.dump(err_e_ts ,open(sdir+'restart/err_e_'+fstr+'.pkl','wb'))
                # pickle.dump(err_i_ts ,open(sdir+'restart/err_i_'+fstr+'.pkl','wb'))
                pickle.dump(ts, open(sdir + 'restart/ts_' + fstr + '.pkl', 'wb'))


        if nt % 250 == 0:
            if save_figs:
                fig1, ax1 = plt.subplots(2, 1)
                time_ = ts[-1]
                fe_data = vp_sys.fe.get_comp_data(ax_select={ax_vye: npts_vye // 2})
                fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
                fe_real = ax_y.basis.get_realspace_1D(fe_real, 1)
                fe_real = ax_vxe.basis.get_realspace_1D(fe_real, 2)

                im1 = ax1[0].imshow(np.log10(np.abs(fe_real[:,:, npts_vxe//2])).T,
                                    extent=(-np.pi / kx, np.pi / kx, -np.pi / ky, np.pi / ky),
                                    aspect='auto', origin='lower')

                im2 = ax1[1].imshow(np.real(fe_real[:, npts_y//2, :]).T,
                                    extent=(-np.pi / kx, np.pi / kx, vxe_min, vxe_max),
                                    aspect='auto', origin='lower')

                fig1.colorbar(im1, ax=ax1[0])
                fig1.colorbar(im2, ax=ax1[1])
                ax1[0].set_title(f'time {time_:3.2f}')
                ax1[0].xaxis.set_visible(False)
                ax1[0].set_xlabel('x'), ax1[0].set_ylabel('y')
                ax1[1].set_xlabel('x'), ax1[1].set_ylabel('vx')
                if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')

                # V_data = pot_field.get_comp_data()
                # plt.figure()
                # plt.imshow(np.log10(np.real(V_data)), label='V field real')
                # plt.xlabel('kx'), plt.ylabel('ky')
                # plt.title('real(V)')
                # plt.colorbar()
                #
                # plt.figure()
                # plt.imshow(np.log(np.imag(V_data)), label='V field imag')
                # plt.xlabel('kx'), plt.ylabel('ky')
                # plt.title('imag(V)')
                # plt.colorbar()

                fig7, ax7 = plt.subplots()
                ax7.semilogy(ts, np.abs(nrg_fe_ts), label='elec')
                ax7.semilogy(ts, np.abs(nrg_fi_ts), label='ion')
                ax7.semilogy(ts, np.abs(nrg_E_ts), label='E field')

                if not save_figs:
                    plt.show()
                plt.close()
                plt.close(fig1)

    if save_data:
        vp_sys.fe.component.save_data(fdir+'fe_'+fstr)
        vp_sys.fi.component.save_data(fdir+'fi_'+fstr)
        # pickle.dump(vp_sys.fe.component.data, open(fdir + 'fe_' + fstr + '.pkl', 'wb'))
        # pickle.dump(vp_sys.fi.component.data, open(fdir + 'fi_' + fstr + '.pkl', 'wb'))
        # pickle.dump(Ex_t ,open(fdir+'Ext_' +fstr+'.pkl','wb'))
        pickle.dump(max_bond_V, open(fdir + 'maxD_V_' + fstr + '.pkl', 'wb'))
        pickle.dump(max_bond_fe, open(fdir + 'maxD_fe_' + fstr + '.pkl', 'wb'))
        pickle.dump(max_bond_fi, open(fdir + 'maxD_fi_' + fstr + '.pkl', 'wb'))
        pickle.dump(nrg_fe_ts, open(fdir + 'nrg_e_' + fstr + '.pkl', 'wb'))
        pickle.dump(nrg_fi_ts, open(fdir + 'nrg_i_' + fstr + '.pkl', 'wb'))
        pickle.dump(nrg_E_ts, open(fdir + 'nrg_E_' + fstr + '.pkl', 'wb'))
        # pickle.dump(err_e_ts ,open(fdir+'err_e_'+fstr+'.pkl','wb'))
        # pickle.dump(err_i_ts ,open(fdir+'err_i_'+fstr+'.pkl','wb'))
        pickle.dump(ts, open(fdir + 'ts_' + fstr + '.pkl', 'wb'))

    print('wall time', time.time() - time1)

fig1, ax1 = plt.subplots(2, 1)
time_ = ts[-1]
fi_data = vp_sys.fi.component.get_data(ax_select={ax_vyi: npts_vyi // 2})
fi_real = ax_x.basis.get_realspace_1D(fi_data, 0)
fi_real = ax_y.basis.get_realspace_1D(fi_real, 1)
fi_real = ax_vxi.basis.get_realspace_1D(fi_real, 2)
im1 = ax1[0].imshow(np.real(fi_real[:,npts_y//2,:]).T,
                    extent=(-np.pi / kx, np.pi / kx, vxi_min, vxi_max),
                    aspect='auto', origin='lower')

fe_data = vp_sys.fe.component.get_data(ax_select={ax_vye: npts_vye // 2})
fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
fe_real = ax_y.basis.get_realspace_1D(fe_real, 1)
fe_real = ax_vxe.basis.get_realspace_1D(fe_real, 2)
im2 = ax1[1].imshow(np.real(fe_real[:,npts_y//2,:]).T,
                    extent=(-np.pi / kx, np.pi / kx, vxe_min, vxe_max),
                    aspect='auto', origin='lower')

fig1.colorbar(im1, ax=ax1[0])
fig1.colorbar(im2, ax=ax1[1])
ax1[0].set_title(f'time {time_:3.2f}')
ax1[0].set_xlabel('x'), ax1[0].set_ylabel('vx')
ax1[1].set_xlabel('x'), ax1[1].set_ylabel('vx')
if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_T{time_:2.1f}.png')

fig2, ax2 = plt.subplots()
# ax2.plot(max_bond_fe,label=r'$\displaystyle f_e$')
# ax2.plot(max_bond_fi,label=r'$\displaystyle f_i$')
ax2.plot(np.log(max_bond_fe) / np.log(q), label=r'$f_e$')
ax2.plot(np.log(max_bond_fi) / np.log(q), label=r'$f_i$')
ax2.plot(np.log(max_bond_V) / np.log(q), label='V')
ax2.plot([0, num_tsteps], [(Lx+Ly+Lxe+Lye) // 2] * 2, ':', label='max e')
ax2.plot([0, num_tsteps], [(Lx+Ly+Lxi+Lyi) // 2] * 2, ':', label='max i')
ax2.plot([0, num_tsteps], [(Lx+Ly) // 2] * 2, ':', label='max V')
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
# ax7.semilogy( helper.entanglement_entropy_all(fe_field.component.data), '-x', label='elec')
# ax7.semilogy( helper.entanglement_entropy_all(fi_field.component.data), '-o', label='ion')
ax7.semilogy( fe_field.component.entanglement_entropy_all(), '-x', label='elec')
ax7.semilogy( fi_field.component.entanglement_entropy_all(), '-o', label='ion')


ax7.legend()
ax7.set_xlabel('bond number')
ax7.set_ylabel('EE')
if save_figs:
    fig7.savefig(fdir + 'plot_EE_' + fstr + '.png')
    print(fdir + 'plot_EE_' + fstr + '.png')

plt.show()


print('done')

if True:
    print('cleaning files')
    
    nts = []
    for f in restart_fstrs:
        nt_ind = f.find('nt')
        nt_str = f[nt_ind + 2:-4]
        nts += [int(nt_str)]
    nt = max(nts)
    # nt = 400
    print('nt', nt)
    nts.remove(nt)
    
    try:
        # raise NameError
        for nt_ in nts:
            if nt_ > nt or nt_%250 != 0:    ## remove old restart files
                print('removing restart', nt_)
                restart_files = glob.glob(sdir + 'restart/*' + re_fstr + f'-nt{nt_}.pkl')
                for f in restart_files:  os.remove(f)
    except NameError:
        pass
    
