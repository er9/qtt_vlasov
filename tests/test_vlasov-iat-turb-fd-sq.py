import os, sys, pickle, time, glob
sys.path.append('../')

from setup.configs import *
import setup.helper as helper_test
import helper_quimb as helper

from axis import Axis
import axis_map
from basis.basis_spatial import SpatialBasis
from basis.basis_k import FourierBasis
from basis.basis_hermite import AWHermiteGaussianBasis
from basis.basis_hermite import HermiteGaussianBasis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from grid_comb import GridsComb
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_vlasovES import VlasovPoisson
from vlasov_tests import IonAcoustic

"""
Ion-acoustic instability (k-space in space, real space in velocity)
"""

save_figs = False
save_data = True
load_data = True
restart   = True
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

extra_str = '_sq'
if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_pde_v2/data4/VP1-iat-fd-turb/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data4/VP1-iat-fd-turb/'
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
mass_i = 25 * mass_e  # mass of ion
n0_e = 1.0  # electron number density
n0_i = n0_e  # ion number density
T_e = 0.0004  # electron temperature [eV]
T_i = 1./50 * T_e  # ion temperature [eV]

ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config.vth    # electron thermal speed, sqrt(eV * T / mass)
wp_e  = elc_config.wp     # plasma e' frequency     sqrt(e ** 2 * n0 / eps0 / mass)
lamD  = elc_config.lamD   # Debye length            vth / wp
wp_i  = ion_config.wp     # plasma i+ frequency
vth_i = ion_config.vth    # ion thermal speed



#####################
# define parameters #
#####################

### system parameters ###
A = 1.0e-4
dk = 1.0 * 2 * np.pi
vdrift_e = 0.
E0 = -5.0e-5
accel_ion = E0/mass_i * ion_config.Z * ion_config.e     # -2.0e-6
noise_ion = 1.0e-6

do_adapt_coll = False
coll_ee = 0.0003
coll_ii = 2.0e-5    # coll_ee/math.sqrt(mass_i/mass_e)*((T_e/T_i)**1.5)

## runtime parameter
q = 2
K = 1
Lx, Lve, Lvi = 8, 8, 8
order = 1

TN_layout = LayoutType.SEQUENTIAL
x_ax_map_key = 'F'
v_ax_map_key = 'B'

T = 1000.0
do_adapt_dt = True
dt = None
dt_frac = 0.9
te_order = 333 # 4  # 311

DMAX = 32 # None
DMAX_F = DMAX
comp_style = 2

def ion_spontaneous_emission_sqrt(t):
    """ previously do cosine perturbation to f.  difficult to do translate that into sqrt scheme.
    instead do cosine pert to E
    """
    xs = ax_x.xpts
    num_k = 8
    rng = np.random.default_rng(int(10e6*t))
    df_ion = np.zeros_like(xs)
    # for nx in range(1, num_k):
    for nx in range(-num_k, num_k):
        if nx == 0:   continue
        df_ion += np.cos(2*np.pi*(nx*xs/tot_x + rng.random()))
    return df_ion


VP_test = IonAcoustic(dks=(dk,), A=A, grid_idx=2,
                      Lxs=(Lx,), Lves=(Lve,), Lvis=(Lvi,),
                      KX=K, KV=K, q=2,
                      x_map_key = x_ax_map_key, ve_map_key = v_ax_map_key,
                      basis_xs = SpatialBasis(),
                      ion_config = ion_config, elc_config = elc_config,
                      #
                      # coll_type = CollisionType.LB, do_adapt_coll = do_adapt_coll, coll_coeff_e = 0.0,
                      coll_type = None, do_adapt_coll = do_adapt_coll, coll_coeff_e = 0.0,
                      coll_rate_e = coll_ee, coll_rate_i = coll_ii,
                      #
                      DMAX = DMAX, DMAX_F = DMAX_F, compress_style = comp_style, compress_style_mod = 0,
                      do_adapt_dt=do_adapt_dt, dt_frac=dt_frac, dt=dt, T=T, te_order=te_order,
                      grid_layout=TN_layout
                      )


#### compression options
te_compress_i = VP_test.get_compression_config(DMAX=DMAX)
te_compress_e = VP_test.get_compression_config(DMAX=DMAX)
te_compress_V = VP_test.get_compression_config(DMAX=DMAX)

print('te compress fe', te_compress_e)
print('te compress fi', te_compress_i)
print('te compress V', te_compress_V)

comp_levels = VP_test.get_compress_levels()

print('compress', DMAX, VP_test.compress_style, VP_test.compress_style_mod)


#### compression of E before dot
compress_E = True
DMAX_E = DMAX
E_compress_opts = {'max_bond': DMAX_E, 'cutoff': CUTOFF, 'cutoff_mode': CUTOFF_MODE}


#############################
## start building PDE system
#############################

## initial conditions
x_min, x_max = VP_test.x_lims[0]
vxe_min, vxe_max = VP_test.ve_lims[0]
vxi_min, vxi_max = VP_test.vi_lims[0]
tot_x = x_max - x_min
print('tot_x is 1?', tot_x)

### axis objects
ax_x, = VP_test.pos_axes
ax_vxi, = VP_test.vi_axes
ax_vxe, = VP_test.ve_axes
print('ax_x zero ind', ax_x.zero_ind)
print('dvi, dve', ax_vxi.dx, ax_vxe.dx)
print('vth', vth_i, vth_e)

X, Y, Z = VP_test.pos_coordsys.coords
VX, VY, VZ = VP_test.ve_coordsys.coords

deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
deriv_v = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)

npts_x = ax_x.npts
npts_vxe = ax_vxe.npts
npts_vxi = ax_vxi.npts

# ## define MPS grids
grid_i = VP_test.grid_i
grid_e = VP_test.grid_e
grid_V = VP_test.grid_X

# km_idx = npts_x // 2 - k_ratio
# kp_idx = npts_x // 2 + k_ratio
# print('k idx', npts_x//2, km_idx, kp_idx)


## initial conditions
## electron density:  1 + A * np.cos(k * x)
fe_x = np.ones(npts_x, dtype=complex)
fe_v = helper_test.maxwellian(ax_vxe.xpts, vth2=vth_e**2, density=n0_e, flow=vdrift_e)
fe_v = np.sqrt(fe_v)

## ion density
fi_x = np.ones(npts_x, dtype=complex)
fi_v = helper_test.maxwellian(ax_vxi.xpts, vth2=vth_i**2, density=n0_i, flow=0.0)
fi_v = np.sqrt(fi_v)

## V field: A / k ** 2 * np.cos(k * x)
V_x = np.zeros(npts_x, dtype=complex)

## E field: -A / k * np.sin(k * x)
E_x = np.zeros(npts_x, dtype=complex)


## define initial grid1DTNs
init_fe_gtn = grid_e.make_mps_ndim({ax_x: fe_x, ax_vxe: fe_v})
init_fi_gtn = grid_i.make_mps_ndim({ax_x: fi_x, ax_vxi: fi_v})
init_V_gtn = grid_V.map_state_to_mps(V_x)

init_fe_gtn.ax_deriv_configs = {ax_x: deriv_x, ax_vxe: deriv_v}
init_fi_gtn.ax_deriv_configs = {ax_x: deriv_x, ax_vxi: deriv_v}
init_V_gtn.ax_deriv_configs  = {ax_x: deriv_x}


dfi = grid_i.make_mps_ndim({ax_x: ion_spontaneous_emission_sqrt(0.),
                            ax_vxi: np.sqrt(maxwellian(ax_vxi.xpts, vth2=vth_i**2, density=n0_i, flow=0.0)) })
init_fi_gtn.add(dfi.scalar_multiply(A), inplace=True)

init_norm_e = init_fe_gtn.norm(is_sqrt=True) / tot_x
init_norm_i = init_fe_gtn.norm(is_sqrt=True) / tot_x
print('init norm', init_norm_e, init_norm_i)


### time step info
if dt is None:
    dt = VP_test.get_dt([0.])
num_tsteps = int(T / dt)
print('dt', dt, num_tsteps)

## filename str
fstr = VP_test.get_fstr(extra_str=extra_str)
print('filename', fstr)

#######################
## start solving PDE ##
#######################

#### define v**2 for measurement ####
vx2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vxe], x_power=2)
v2e_mpo = vx2e_mpo

vx2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vxi], x_power=2)
v2i_mpo = vx2i_mpo
print('v2 mpo max bond', v2e_mpo.max_bond(), v2i_mpo.max_bond())


#### load/initialize data ####
try:
    if not load_data:  raise IOError
    fe_t = grid_e.load_gtn_data(fdir + 'fe_' + fstr, gtn=init_fe_gtn)
    fi_t = grid_i.load_gtn_data(fdir + 'fi_' + fstr, gtn=init_fi_gtn)
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
            fe_t = grid_e.load_gtn_data(fdir + 'fe_' + restart_fstr, gtn=init_fe_gtn)
            fi_t = grid_i.load_gtn_data(fdir + 'fi_' + restart_fstr, gtn=init_fi_gtn)
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
            # nt = 40250
            print('nt', nt)
            nts.remove(nt)

            max_bond_V = pickle.load(open(sdir + 'restart/maxD_V_' + re_fstr + f'.pkl', 'rb'))
            max_bond_fe = pickle.load(open(sdir + 'restart/maxD_fe_' + re_fstr + f'.pkl', 'rb'))
            max_bond_fi = pickle.load(open(sdir + 'restart/maxD_fi_' + re_fstr + f'.pkl', 'rb'))
            print('loaded', nt, len(max_bond_V))
            nrg_fe_ts = pickle.load(open(sdir + 'restart/nrg_e_' + re_fstr + f'.pkl', 'rb'))
            nrg_fi_ts = pickle.load(open(sdir + 'restart/nrg_i_' + re_fstr + f'.pkl', 'rb'))
            nrg_E_ts = pickle.load(open(sdir + 'restart/nrg_E_' + re_fstr + f'.pkl', 'rb'))
            ts = pickle.load(open(sdir + 'restart/ts_' + re_fstr + f'.pkl', 'rb'))
            ts = list(ts)

            try:
                # raise NameError
                for nt_ in nts:  ## remove old restart files
                    # if nt_ > nt:
                    dt = ts[nt_] - ts[nt_-1]
                    if nt_ > nt or not np.abs(ts[nt_]-int(ts[nt_])) < 0.01:
                        print('removing restart', nt_)
                        restart_files = glob.glob(sdir + 'restart/*' + re_fstr + f'-nt{nt_}.pkl')
                        for f in restart_files:  os.remove(f)
            except NameError:
                pass

            fe_t = grid_e.load_gtn_data(sdir + 'restart/fe_' + re_fstr + f'-nt{nt}', gtn=init_fe_gtn)
            fi_t = grid_i.load_gtn_data(sdir + 'restart/fi_' + re_fstr + f'-nt{nt}', gtn=init_fi_gtn)
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

        init_V_gtn.data = None

        restarted = True
        print('loaded restart', nt)

    except(ValueError, FileNotFoundError):
        nt = 0
        ts = [0]
        restarted = False

        v2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vxe], x_power=2)
        v2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vxi], x_power=2)

        nrg_fe = init_fe_gtn.meas_expec(v2e_mpo) * mass_e / 2 / tot_x
        nrg_fi = init_fi_gtn.meas_expec(v2i_mpo) * mass_i / 2 / tot_x
        nrg_E  = 0.

        nrg_fe_ts = [nrg_fe]
        nrg_fi_ts = [nrg_fi]
        nrg_E_ts  = [nrg_E]
        print('nrg', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1])
        # exit()

        norm_fe = init_fe_gtn.norm() / tot_x
        norm_fi = init_fi_gtn.norm() / tot_x
        print('norm', norm_fe, norm_fi)

        print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
        max_bond_V  = [init_V_gtn.max_bond()]
        max_bond_fe = [init_fe_gtn.max_bond()]
        max_bond_fi = [init_fi_gtn.max_bond()]

        err_e_ts = [0.]
        err_i_ts = [0.]

    init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, compress_config=te_compress_e, is_sqrt=True)
    init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, compress_config=te_compress_i, is_sqrt=True)
    init_V_field  = ScalarField('V',  grid_V, data=init_V_gtn, compress_config=te_compress_V, is_sqrt=True)

    print('norm', init_fe_field.norm(), init_fi_field.norm())

    if init_V_field.component is not None and init_V_field.component.data is not None:
        elec_field = init_V_field.gradient() * -1
        energy_E_t = elec_field.norm()**2 * 1./2 / tot_x   # electric field energy density
        print('E field nrg before', energy_E_t )

    background_E0 = Field('E0',grid_V, data={X: grid_V.get_ones_mps().scalar_multiply(E0)})

    ### BUILD PDE
    vp_sys = VlasovPoisson(init_fe_field, init_fi_field, init_V_field,
                           coords_x=VP_test.pos_coordsys, coords_ve=VP_test.ve_coordsys, coords_vi=VP_test.vi_coordsys,
                           elc_params=elc_config, ion_params=ion_config, normalize=True,
                           background_E0=background_E0,
                           te_order=te_order, compress_levels=comp_levels)
    vp_sys.collision = VP_test.collision_config
    print('colls', vp_sys.collision.coll_params)

    vp_sys.compress_F = VP_test.compress_F
    vp_sys.compress_F_opts = VP_test.compress_F_opts
    print('compress E?', vp_sys.compress_F, vp_sys.compress_F_opts)

    # vp_sys.init_norm_e = init_norm_e
    # vp_sys.init_norm_i = init_norm_i

    # vp_sys.v_from_v0_e = grid_e.map_state_to_mps(ve_vals-0.0)
    # vp_sys.v_from_v0_i = grid_i.map_state_to_mps(vi_vals-0.0)

    print('before update', vp_sys.V.component.ax_deriv_configs[ax_x].bc, ax_x)
    if True:
        vp_sys.update_V(inplace=True)
    print('after update', vp_sys.V.component.ax_deriv_configs[ax_x].bc)
    elec_field = vp_sys.V.gradient()
    energy_E_t = elec_field.norm()**2 * 1. / 2 / tot_x # electric field energy
    print('E field nrg', energy_E_t)

    print('vth i', vth_i, 50*vth_i)

    if nt == 0:

        init_fi_data = init_fi_gtn.get_data()
        fig1, ax1 = plt.subplots(2, 1)
        time_ = nt * dt
        fi_real = ax_x.basis.get_realspace_1D(init_fi_data, 0, npts=512)
        fi_real = ax_vxi.basis.get_realspace_1D(fi_real, 1)
        im1 = ax1[0].imshow(np.abs(fi_real.T)**2, extent=(x_min, x_max, vxi_min, vxi_max),
                            aspect='auto', origin='lower')

        init_fe_data = init_fe_gtn.get_data()
        fe_real = ax_x.basis.get_realspace_1D(init_fe_data, 0, npts=512)
        fe_real = ax_vxe.basis.get_realspace_1D(fe_real, 1)
        im2 = ax1[1].imshow(np.abs(fe_real.T)**2, extent=(x_min, x_max, vxe_min, vxe_max),
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
            V_field = vp_sys.V
            E_field = V_field.gradient()

            Ex_data = E_field.get_comp_data(X)
            Ex_max = np.max(np.abs(Ex_data))

            dt = VP_test.get_dt(Fe_maxs=np.array([Ex_max,]) * charge / mass_e)

        if do_adapt_coll:
            VP_test.update_collision_config()

            print('restarted dt', dt)

    time1 = time.time()

    ## for measuring energy
    v2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_vxe], x_power=2)
    v2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vxi], x_power=2)

    while ts[-1] < T:

        walltime = time.time()
        vp_sys = vp_sys.next_time_step(dt, compress_level=1, verbose_plot=False,
                                       is_first_time_step= (nt==0),
                                       is_last_time_step= (T-ts[-1] < dt))
        print('wall time', time.time() - walltime)

        ts += [ts[-1] + dt]
        nt += 1

        # spontaneous emission (add perturbations to potential V. previously added pert to fi)
        if noise_ion > 0:
            V = vp_sys.V
            pert = grid_V.make_mps_ndim({ax_x: ion_spontaneous_emission_sqrt(ts[-1])})
            V.component.add(pert.scalar_multiply(noise_ion * dt), inplace=True, compress=True)
            # fi = vp_sys.fi
            # dfi = grid_i.make_mps_ndim({ax_x: ion_spontaneous_emission_sqrt(ts[-1]),
            #                             ax_vxi: np.sqrt(maxwellian(ax_vxi.xpts, vth_i**2, n0_i, accel_ion*ts[-1]))})
            # fi.component.add(dfi.scalar_multiply(noise_ion * dt), inplace=True)

        pot_field = vp_sys.V
        max_bond_V += [pot_field.max_bond()]
        elec_field = pot_field.gradient()
        elec_field.scalar_multiply(-1, inplace=True)

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

        # print('meas expec', v2e_mpo.max_bond(), v2i_mpo.max_bond())
        # print('fe', fe_field.component)
        nrg_fe = fe_field.component.meas_expec(v2e_mpo, is_sqrt=True) * mass_e / 2 / tot_x
        nrg_fi = fi_field.component.meas_expec(v2i_mpo, is_sqrt=True) * mass_i / 2 / tot_x

        nrg_fe_ts += [nrg_fe]
        nrg_fi_ts += [nrg_fi]

        energy_E_t = elec_field.norm()**2 * 1. / 2 / tot_x    # electric field energy
        nrg_E_ts += [energy_E_t]

        if do_adapt_dt:
            Ex_data = elec_field.get_comp_data(X)
            Ex_max = np.max(np.abs(Ex_data))

            dt = VP_test.get_dt(np.array([Ex_max + abs(E0),]) * charge / mass_e)

        if do_adapt_coll:
            VP_test.update_collision_config()

        norm = vp_sys.fe.norm() - vp_sys.fi.norm()
        print('result norm', nt, ts[-1], dt, norm, energy_E_t,
              pot_field.max_bond(), fi_field.max_bond(), fe_field.max_bond(), Lx, Lvi, Lve)


        if nt % 50 == 0:
            fig1, ax1 = plt.subplots(2, 1)
            time_ = ts[-1]
            fi_data = fi_field.get_comp_data()
            fi_data = ax_x.basis.get_realspace_1D(fi_data, 0)
            fi_data = ax_vxi.basis.get_realspace_1D(fi_data, 1)
            im1 = ax1[0].imshow(np.abs(fi_data.T)**2, extent=(x_min, x_max, vxi_min, vxi_max),
                                aspect='auto', origin='lower')
            fe_data = fe_field.get_comp_data()
            fe_data = ax_x.basis.get_realspace_1D(fe_data, 0)
            fe_data = ax_vxe.basis.get_realspace_1D(fe_data, 1)
            im2 = ax1[1].imshow(np.abs(fe_data.T)**2, extent=(x_min, x_max, vxe_min, vxe_max),
                                aspect='auto', origin='lower')

            fig1.colorbar(im1, ax=ax1[0])
            fig1.colorbar(im2, ax=ax1[1])
            ax1[0].set_title(f'time {time_:3.2f}')
            ax1[0].xaxis.set_visible(False)
            ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
            ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
            if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')

            V_data = pot_field.get_comp_data()
            plt.figure()
            plt.plot(ax_x.xpts, np.real(Ex_data), label='E field real')
            plt.plot(ax_x.xpts, np.imag(Ex_data), label='E field imag')
            plt.plot(ax_x.xpts, np.real(V_data), label='V field real')
            plt.plot(ax_x.xpts, np.imag(V_data), label='V field imag')
            plt.legend()

            if not save_figs:
                plt.show()

            plt.close(fig1)
            plt.close()

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
                # pickle.dump(vp_sys.fe.component.data, open(sdir + 'restart/' + 'fe_' + fstr + f'-nt{nt}.pkl', 'wb'))
                # pickle.dump(vp_sys.fi.component.data, open(sdir + 'restart/' + 'fi_' + fstr + f'-nt{nt}.pkl', 'wb'))
                vp_sys.fe.component.save_data(sdir + 'restart/' + 'fe_' + fstr + f'-nt{nt}')
                vp_sys.fi.component.save_data(sdir + 'restart/' + 'fi_' + fstr + f'-nt{nt}')
                pickle.dump(max_bond_V, open(sdir + 'restart/' + 'maxD_V_' + fstr + '.pkl', 'wb'))
                pickle.dump(max_bond_fe, open(sdir + 'restart/' + 'maxD_fe_' + fstr + '.pkl', 'wb'))
                pickle.dump(max_bond_fi, open(sdir + 'restart/' + 'maxD_fi_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_fe_ts, open(sdir + 'restart/nrg_e_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_fi_ts, open(sdir + 'restart/nrg_i_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_E_ts, open(sdir + 'restart/nrg_E_' + fstr + '.pkl', 'wb'))
                # pickle.dump(err_e_ts ,open(sdir+'restart/err_e_'+fstr+'.pkl','wb'))
                # pickle.dump(err_i_ts ,open(sdir+'restart/err_i_'+fstr+'.pkl','wb'))
                pickle.dump(ts, open(sdir + 'restart/ts_' + fstr + '.pkl', 'wb'))

    if save_data:
        # pickle.dump(vp_sys.fe.component.data, open(fdir + 'fe_' + fstr + '.pkl', 'wb'))
        # pickle.dump(vp_sys.fi.component.data, open(fdir + 'fi_' + fstr + '.pkl', 'wb'))
        vp_sys.fe.component.save_data(fdir + 'fe_' + fstr)
        vp_sys.fi.component.save_data(fdir + 'fi_' + fstr)
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
fi_real = ax_x.basis.get_realspace_1D(fi_data, 0)
fi_real = ax_vxi.basis.get_realspace_1D(fi_real, 1)
im1 = ax1[0].imshow(np.abs(fi_real.T)**2, extent=(-tot_x/2, tot_x/2, vxi_min, vxi_max),
                    aspect='auto', origin='lower')

fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
fe_real = ax_vxe.basis.get_realspace_1D(fe_real, 1)
im2 = ax1[1].imshow(np.abs(fe_real.T)**2, extent=(-tot_x/2, tot_x/2, vxe_min, vxe_max),
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
ax2.plot(np.log(max_bond_V) / np.log(q ** K), label='V')
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


if True:
    print('cleaning files')
    restart_fstrs = glob.glob(sdir + 'restart/fe_' + re_fstr + '*.pkl')
    
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
            # if nt_ > nt or nt_%250 != 0:    ## remove old restart files
            dt = ts[nt_] - ts[nt_-1]
            if nt_ > nt or not np.abs(ts[nt_]-int(ts[nt_])) < 0.01:
                print('removing restart', nt_)
                restart_files = glob.glob(sdir + 'restart/*' + re_fstr + f'-nt{nt_}.pkl')
                for f in restart_files:  os.remove(f)
    except NameError:
        pass
    
