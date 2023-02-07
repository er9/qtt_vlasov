import os, sys, pickle, time, glob

sys.path.append('../')

from setup.configs import *
import helper_quimb as helper

from axis import Axis
import axis_map
# from axis_map.map_binary import BinaryMap
# from axis_map.map_flipbinary import FlipBinaryMap
# from axis_map.map_mirror import MirrorMap
from basis.basis_k import FourierBasis
from basis.basis_hermite import AWHermiteGaussianBasis
from basis.basis_hermite import HermiteGaussianBasis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_vlasovES import VlasovPoisson

"""
Landau damping (k-space, Hermite expansion)
"""

save_figs = True
save_data = True
load_data = True
restart   = True
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_data_v2/VP1-damp-k/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data/VP1-damp-k/'
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
    v2 = 0.0
    for i in range(len(vs)):
        try:
            v2 += (vs[i] - flow[i]) * (vs[i] - flow[i])
        except TypeError:
            v2 += (vs[i] - flow) * (vs[i] - flow)
    vth2 = eV * temp / mass
    return density * np.sqrt(1. / 2 / np.pi / vth2)**len(vs) * np.exp(-v2 / 2 / vth2)


def rk4_limit_FourierHermite(m_max, k_max, E_max=0):
    """ limit dt (estimate) given by rk4 stability constraints:
        1 + z + z^2/2 + z^3/6 + z^4/4 < 1
        for z real, --> z = -2.7853
        for z imag, --> z ~ 1 to ~ 2.75

        calculated assuming in fourier/hermite basis
        df/dt + v * grad_x(f) + q(E)*grad_v(f) = 0
        --> approximate max eigenvalue as sqrt(m_max) * (k_max + E_max)
    """
    dt_max = 2.75 / (np.sqrt(2 * m_max) * (k_max + E_max))
    return dt_max


def get_maps(mkey: str):
    ax_maps = []
    for mk in mkey:
        if mk == 'F':   # forward
            ax_map = axis_map.BinaryMap()
        elif mk == 'B':     # back
            ax_map = axis_map.FlipBinaryMap()
        elif mk == 'M':
            ax_map = axis_map.MirrorMap()
        else:
            raise KeyError(f'{mk} not a valid map key')
        ax_maps += [ax_map]
    print('ax maps', ax_maps)
    return ax_maps


def get_fstr():
    """ get filename
    """
    ## parameters string
    fstr_params = f'k{k}_mr{mass_i}'

    ## collision string
    if coll_type is None:
        fstr_coll = ''
    else:
        if do_adapt_coll:
            fstr_coll = f'_{coll_type.value}{coll_coeff_e}'
        else:
            fstr_coll = f'_{coll_type.value}e{coll_rate_e}i{coll_rate_i}'

    ## compression string
    fstr_comp = f'_C{compress_style}{compress_style_mod}_mb{DMAX}'

    ## E compression string
    if compress_E and DMAX_E is not None:
        fstr_compE = f'_E{DMAX_E}'
    else:
        fstr_compE = ''

    ## grid string
    fstr_grid = f'test_TN{TN_layout.value}{ax_map_key}' + \
                f'_Ls{Lx:02d},{Lve:02d},{Lvi:02d}_'

    fstr_t = f'_dt{dt:1.4f}_T{int(T):03d}_te{te_order}'

    fstr1 = fstr_params + fstr_comp + fstr_compE + fstr_coll + fstr_grid + fstr_t
    return fstr1


#####################
# define parameters #
#####################

### system parameters ###
A = 0.5  # 0.001
Astr = '5e-1'

## runtime parameter
q = 2
K = 1

k = 0.75 / lamD
# Lx, Lve, Lvi = 6, 10, 6
Lx, Lve, Lvi = 5, 10, 5

# k = 0.10 / lamD
# Lx, Lve, Lvi = 9, 10, 9

# TN_layout = LayoutType.PARALLEL_GROUP  ## PG, PF don't work with Fourier
TN_layout = LayoutType.SEQUENTIAL  ## PG, PF don't work with Fourier
ax_map_key = 'MF'

T = 60.0
do_adapt_dt = True
frac_dt = 0.5 
dt = rk4_limit_FourierHermite(q ** Lve - 1, k * (q ** Lx - 1)/2, A / k) * frac_dt
# do_adapt_dt = True
# frac_dt = 0.5
# dt = rk4_limit_FourierHermite(q ** Lve - 1, k * (q ** Lx - 1)/2, A / k) * frac_dt
print('rk4 limit?', rk4_limit_FourierHermite(q ** Lve - 1, k * (q ** Lx - 1)/2, A / k))
## *1.5:  the upper limit for k=0.75, T=15.0
print('dt rk4', dt)
te_order = 4

num_tsteps = int(T / dt)
print('dt', dt, num_tsteps)


#############################
#### define collisions
do_adapt_coll = True
coll_type = CollisionType.LB
coll_coeff_e = 1.0e-4  ## to scale collision frequency if using adaptive collisions
coll_coeff_i = 1.0e-4 / np.sqrt(mass_i / mass_e)
coll_rate_e = 0.01 * wp_e if coll_type is not None else 0.0
coll_rate_i = 0.01 * wp_i if coll_type is not None else 0.0
collision_config = CollisionConfiguration(coll_type,
                                          coeff_e=coll_coeff_e, coeff_i=coll_coeff_i,
                                          rate_e=coll_rate_e, rate_i=coll_rate_i)

#### compression options
"""
    comp1:   compress final state
    comp2:   + compress intermediate rk4 states
    comp3:   + compress sum of derivatives
    comp4:   + compress derivatives
    comp5:   + compress during calculation of derivative

    mod0:  no modifications 
    mod1:  comp5 = 0 (don't actively compress in interior of calculating derivative)
"""
DMAX = None
compress_style = 1
compress_style_mod = 0

te_compress_i = CompressionConfiguration()
te_compress_e = CompressionConfiguration()
te_compress_V = CompressionConfiguration()

for ci in range(1, compress_style + 1):
    te_compress_i.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_e.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_V.set_compress_opts(ci, max_bond=None, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)

print('te compress fe', te_compress_e)
print('te compress fi', te_compress_i)
print('te compress V', te_compress_V)

comp_levels = list(range(1, 6))
if compress_style_mod == 1:  # don't compress inside the derivative
    comp_levels = [1, 2, 3, 4, 0]
elif compress_style_mod == 2:  # don't compress inside the derivative
    comp_levels = [1, 2, 3, 0, 0]

print('compress', DMAX, compress_style, compress_style_mod)

#### compression of E before dot
compress_E = True
DMAX_E = None
E_compress_opts = {'max_bond': DMAX_E, 'cutoff': CUTOFF, 'cutoff_mode': CUTOFF_MODE}


#############################
## start building PDE system
#############################

## specify k values, hermite moments used
# dk = 0.10 -- lowest frequency mode
npts_x, npts_ve, npts_vi = q ** Lx, q ** Lve, q ** Lvi
kmax = k * npts_x / 2
k_vals = np.linspace(-kmax, kmax, npts_x, endpoint=False)
# k_vals = k_vals.reshape((q,)*Lx).transpose().reshape(-1)
me_vals = np.arange(0,npts_ve)
mi_vals = np.arange(0,npts_vi)

## initial conditions
## electron density:  1 + A * np.cos(k * x)
fe_x = np.zeros(npts_x, dtype=complex)
fe_x[npts_x // 2] = 1.0 * n0_e
fe_x[npts_x // 2 - 1] = A / 2
fe_x[npts_x // 2 + 1] = A / 2
fe_v = np.array([1.0/np.sqrt(2)/vth_e] + [0.] * (npts_ve - 1))

## ion density
fi_x = np.zeros(npts_x, dtype=complex)
fi_x[npts_x // 2] = 1.0 * n0_i
fi_v = np.array([1.0/np.sqrt(2)/vth_i] + [0.] * (npts_vi - 1))

## V field: -A / k ** 2 * np.cos(k * x)
V_x = np.zeros(npts_x, dtype=complex)
V_x[npts_x // 2 - 1] = -A / k ** 2 / 2
V_x[npts_x // 2 + 1] = -A / k ** 2 / 2

## E field: -A / k * np.sin(k * x)
E_x = np.zeros(npts_x, dtype=complex)
E_x[npts_x // 2 - 1] = -1.j * A / k / 2
E_x[npts_x // 2 + 1] = 1.j * A / k / 2


## define axis objects; position -> k-space,
X = Coordinate('KX', CoordinateType.X)
V = Coordinate('MV', CoordinateType.X)

map_x, map_v = get_maps(ax_map_key)
ax_x  = Axis(Lx, q, coordinate=X, dx=k, xpts=k_vals, basis=FourierBasis(), ax_map=map_x)
ax_vi = Axis(Lvi, q, coordinate=V, xpts=mi_vals, basis=AWHermiteGaussianBasis(x_scale=np.sqrt(2) * vth_i), ax_map=map_v)
ax_ve = Axis(Lve, q, coordinate=V, xpts=me_vals, basis=AWHermiteGaussianBasis(x_scale=np.sqrt(2) * vth_e), ax_map=map_v)

print('ax_x zero ind', ax_x.zero_ind)

coords_x  = CartesianCoordinateSpace('X', ax_x)
coords_vi = CartesianCoordinateSpace('Vi', ax_vi)
coords_ve = CartesianCoordinateSpace('Ve', ax_ve)

## define grids
grid_i = Grid1D('XVi', (ax_x, ax_vi), layout_type=TN_layout)
grid_e = Grid1D('XVe', (ax_x, ax_ve), layout_type=TN_layout)
grid_V = Grid1D('X', (ax_x,), layout_type=TN_layout)


## define initial grid1DTNs
fe_x = ax_x.map_state_to_mps(fe_x)
fi_x = ax_x.map_state_to_mps(fi_x)
fe_v = ax_ve.map_state_to_mps(fe_v)
fi_v = ax_vi.map_state_to_mps(fi_v)

init_fe_gtn = grid_e.make_mps_ndim({ax_x: fe_x, ax_ve: fe_v})
init_fi_gtn = grid_i.make_mps_ndim({ax_x: fi_x, ax_vi: fi_v})
init_V_gtn = grid_V.map_state_to_mps(V_x)


fstr = get_fstr()
print('filename', fstr)

try:
    if not load_data:  raise IOError
    fe_t = pickle.load(open(fdir + 'fe_' + fstr + '.pkl', 'rb'))
    fi_t = pickle.load(open(fdir + 'fi_' + fstr + '.pkl', 'rb'))
    max_bond_V = pickle.load(open(fdir + 'maxD_V_' + fstr + '.pkl', 'rb'))
    max_bond_fe = pickle.load(open(fdir + 'maxD_fe_' + fstr + '.pkl', 'rb'))
    max_bond_fi = pickle.load(open(fdir + 'maxD_fi_' + fstr + '.pkl', 'rb'))
    nrg_fe_ts = pickle.load(open(fdir + 'nrg_e_' + fstr + '.pkl', 'rb'))
    nrg_fi_ts = pickle.load(open(fdir + 'nrg_i_' + fstr + '.pkl', 'rb'))
    nrg_E_ts = pickle.load(open(fdir + 'nrg_E_' + fstr + '.pkl', 'rb'))
    ts = pickle.load(open(fdir + 'ts_' + fstr + '.pkl', 'rb'))

    nt = num_tsteps
    print('loaded data')

    fe_data = grid_e.map_mps_to_state(fe_t)
    fi_data = grid_i.map_mps_to_state(fi_t)

except(IOError, EOFError):

    try:
        if restart_from_T > 0:
            restart_fstr = fstr
            fe_t = pickle.load(open(fdir + 'fe_' + restart_fstr + '.pkl', 'rb'))
            fi_t = pickle.load(open(fdir + 'fi_' + restart_fstr + '.pkl', 'rb'))
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

            restart_fstrs = glob.glob(sdir + 'restart/fe_' + re_fstr + '-nt*.pkl')
            print(sdir + 'restart/')
            print('restart?', len(restart_fstrs))
            nts = []
            for f in restart_fstrs:
                nt_ind = f.find('nt')
                nt_str = f[nt_ind + 2:-4]
                nts += [int(nt_str)]
            nt = max(nts)
            # nt = 5750
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

            fe_t = pickle.load(open(sdir + 'restart/fe_' + re_fstr + f'-nt{nt}.pkl', 'rb'))
            fi_t = pickle.load(open(sdir + 'restart/fi_' + re_fstr + f'-nt{nt}.pkl', 'rb'))
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
            ref_fstr = f'/home/erikaye/gkeyll_tests/vm-damp/L08_A{Astr}/' + \
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

        init_fe_gtn.data = fe_t
        init_fi_gtn.data = fi_t
        init_V_gtn.data = None

        # init_fe_gtn = GridTN1D(grid_e, data=fe_t, ax_deriv_configs={ax_x.axID: deriv_xe_up, ax_ve.axID: deriv_v})
        # init_fi_gtn = GridTN1D(grid_i, data=fi_t, ax_deriv_configs={ax_x.axID: deriv_xi_up, ax_ve.axID: deriv_v})
        # init_V_gtn  = GridTN1D(grid_V, data=None, ax_deriv_configs={ax_x.axID: deriv_x})

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

        nrg_fe = init_fe_gtn.meas_expec(v2e_mpo) * mass_e / 2 * (k / 2 / np.pi)
        nrg_fi = init_fi_gtn.meas_expec(v2i_mpo) * mass_i / 2 * (k / 2 / np.pi)
        nrg_E  = (A/k/2)**2  * 2*np.pi/k  * (k/2/np.pi)

        nrg_fe_ts = [nrg_fe]
        nrg_fi_ts = [nrg_fi]
        nrg_E_ts  = [nrg_E]
        print('nrg', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1])

        norm_fe = init_fe_gtn.norm()
        norm_fi = init_fi_gtn.norm()
        print('norm', norm_fe, norm_fi)

        print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
        max_bond_V  = [init_V_gtn.max_bond()]
        max_bond_fe = [init_fe_gtn.max_bond()]
        max_bond_fi = [init_fi_gtn.max_bond()]

        err_e_ts = [0.]
        err_i_ts = [0.]

    init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, compress_config=te_compress_e)
    init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, compress_config=te_compress_i)
    init_V_field  = ScalarField('V', grid_V, data=init_V_gtn, compress_config=te_compress_V, is_sqrt=True)

    print('norm', init_fe_field.norm(), init_fi_field.norm())

    elec_field = init_V_field.gradient()
    energy_E_t = elec_field.norm()
    if energy_E_t is not None:   energy_E_t = energy_E_t**2 * 1. / 2     # electric field energy
    print('E field nrg before', energy_E_t )

    ### BUILD PDE
    vp_sys = VlasovPoisson(init_fe_field, init_fi_field, init_V_field, coords_x=coords_x, coords_ve=coords_ve,
                           coords_vi=coords_vi, elc_params=elc_config, ion_params=ion_config, normalize=True,
                           te_order=te_order, compress_levels=comp_levels)
    vp_sys.collision = collision_config
    print('colls', collision_config.coll_params)

    vp_sys.compress_F = compress_E
    vp_sys.compress_F_opts = E_compress_opts
    print('compress E?', compress_E, E_compress_opts)

    # vp_sys.v_from_v0_e = grid_e.map_state_to_mps(ve_vals-0.0)
    # vp_sys.v_from_v0_i = grid_i.map_state_to_mps(vi_vals-0.0)

    print('before update', vp_sys.V.component.ax_deriv_configs[ax_x].bc, ax_x)
    if True:  # restarted:
        vp_sys.update_V(inplace=True)
    print('after update', vp_sys.V.component.ax_deriv_configs[ax_x].bc)
    elec_field = vp_sys.V.gradient()
    energy_E_t = elec_field.norm()**2 * 1. / 2  # electric field energy
    print('E field nrg', energy_E_t)


    if nt == 0:
        fig1, ax1 = plt.subplots(2, 1)
        time_ = nt * dt
        fi_real = ax_x.basis.get_realspace_1D(init_fi_data, 0)
        fi_real = ax_vi.basis.get_realspace_1D(fi_real, 1, -6 * vth_i, 6 * vth_i)
        im1 = ax1[0].imshow(np.real(fi_real).T, extent=(-np.pi/k, np.pi/k, -6 * vth_i, 6 * vth_i),
                            aspect='auto', origin='lower')

        fe_real = ax_x.basis.get_realspace_1D(init_fe_data, 0)
        fe_real = ax_ve.basis.get_realspace_1D(fe_real, 1, -6 * vth_e, 6 * vth_e)
        im2 = ax1[1].imshow(np.real(fe_real).T, extent=(-np.pi/k, np.pi/k, -6 * vth_e, 6 * vth_e),
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
        # vp_sys.update_V(inplace=True)
        if do_adapt_dt:
            V_field = vp_sys.V
            E_field = V_field.gradient()
            E_data = E_field.get_comp_data(X)

            E_max = np.max(np.abs(E_data))
            dt = rk4_limit_FourierHermite(q ** Lve - 1, k * (q ** Lx - 1)/2, E_max / mass_e) * frac_dt
            print('restarted dt', dt)

    time1 = time.time()

    ## for measuring energy
    v2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_ve], x_power=2)
    v2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vi], x_power=2)

    while ts[-1] < T:

        walltime = time.time()
        vp_sys = vp_sys.next_time_step(dt, compress_level=1)  #, verbose_plot=(nt >= 40))
        print('wall time', time.time() - walltime)

        pot_field = vp_sys.V
        max_bond_V += [pot_field.max_bond()]
        elec_field = pot_field.gradient()
        elec_field.scalar_multiply(-1, inplace=True)
        elec_field.is_sqrt = True
        E_data = elec_field.get_comp_data(ax_x.coordinate)

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
        fe_data = fe_field.get_comp_data()

        fi_field = vp_sys.fi
        max_bond_fi += [fi_field.max_bond()]
        fi_data = fi_field.get_comp_data()

        nrg_fe = fe_field.component.meas_expec(v2e_mpo) * mass_e / 2 * (k / 2 / np.pi)
        nrg_fi = fi_field.component.meas_expec(v2i_mpo) * mass_i / 2 * (k / 2 / np.pi)

        nrg_fe_ts += [nrg_fe]
        nrg_fi_ts += [nrg_fi]


        energy_E_t = elec_field.norm()**2 * 1. / 2     # electric field energy
        nrg_E_ts += [energy_E_t]

        ts += [ts[-1] + dt]
        nt += 1

        if do_adapt_dt:
            E_max = np.max(np.abs(E_data))
            dt = rk4_limit_FourierHermite(q ** Lve - 1, k * (q ** Lx - 1)/2, E_max/mass_e) * frac_dt

        norm = vp_sys.fe.norm() - vp_sys.fi.norm()
        print('result norm', nt, ts[-1], dt, norm, energy_E_t,
              pot_field.max_bond(), fe_field.max_bond(), Lx, Lvi, Lve)
        # if nt >= 5:   exit()

        if nt % 100 == 0:
            fig1, ax1 = plt.subplots(2, 1)
            time_ = ts[-1]
            im1 = ax1[0].imshow(np.real(fi_data).T, extent=(-np.pi/k, np.pi/k, -6*vth_i, 6*vth_i),
                                aspect='auto', origin='lower')
            im2 = ax1[1].imshow(np.real(fe_data).T, extent=(-np.pi/k, np.pi/k, -6*vth_e, 6*vth_e),
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
            plt.plot(np.arange(npts_x) * k, np.real(E_data), label='E field real')
            plt.plot(np.arange(npts_x) * k, np.imag(E_data), label='E field imag')
            plt.plot(np.arange(npts_x) * k, np.real(V_data), label='V field real')
            plt.plot(np.arange(npts_x) * k, np.imag(V_data), label='V field imag')
            plt.legend()

            if not save_figs:
                plt.show()
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
                pickle.dump(vp_sys.fe.component.data, open(sdir + 'restart/' + 'fe_' + fstr + f'-nt{nt}.pkl', 'wb'))
                pickle.dump(vp_sys.fi.component.data, open(sdir + 'restart/' + 'fi_' + fstr + f'-nt{nt}.pkl', 'wb'))
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
        pickle.dump(vp_sys.fe.component.data, open(fdir + 'fe_' + fstr + '.pkl', 'wb'))
        pickle.dump(vp_sys.fi.component.data, open(fdir + 'fi_' + fstr + '.pkl', 'wb'))
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
fi_real = ax_vi.basis.get_realspace_1D(fi_real, 1, -6 * vth_i, 6 * vth_i)
im1 = ax1[0].imshow(np.real(fi_real).T, extent=(-np.pi/k, np.pi/k, -6 * vth_i, 6 * vth_i),
                    aspect='auto', origin='lower')

fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
fe_real = ax_ve.basis.get_realspace_1D(fe_real, 1, -6 * vth_e, 6 * vth_e)
im2 = ax1[1].imshow(np.real(fe_real).T, extent=(-np.pi/k, np.pi/k, -6 * vth_e, 6 * vth_e),
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

