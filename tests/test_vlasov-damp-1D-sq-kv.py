import os, sys, pickle, time, glob, scipy
sys.path.append('../')

from setup.configs import *
import setup.helper as helper_test

from axis import Axis
from basis.basis_k import FourierBasis
import axis_map
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from grid_comb import GridsComb
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_vlasovES import VlasovPoisson

"""
Landau damping
"""

save_figs = False
save_data = False  # True
load_data = False  # True
restart   = False  # True
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

is_sqrt = True

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_pde_v2/data3/VP1-damp-kv-sq/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data3/VP1-damp-kv-sq/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir',fdir)

#######################
# standard parameters #
#######################

plasma_config = UnitsConfiguration()
charge = plasma_config.e  # unit of charge
eV = plasma_config.eV     # unit of kB*T

mass_e = 1.0           # mass of electron
mass_i = 1836*mass_e   # mass of ion
n0_e = 1.0             # electron number density
n0_i = n0_e            # ion number density
T_e  = 1.0             # electron temperature [eV]
T_i  = T_e             # ion temperature [eV]

ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config.vth    # electron thermal speed
wp_e  = elc_config.wp     # plasma e' frequency
lamD  = elc_config.lamD   # Debye length
wp_i  = ion_config.wp     # plasma i+ frequency
vth_i = ion_config.vth    # ion thermal speed


######################################
# Landau damping dispersion relation #
######################################

def freq(w_, k):  ## note: we already normalized k -> k/lamD
    w_re, w_im = w_
    w = w_re + 1.j * w_im
    zeta = w / np.sqrt(2) / vth_e / k

    integ_func = lambda x: np.exp(x ** 2)
    _xs = np.linspace(0, zeta, 3001)
    _dx = _xs[1] - _xs[0]
    Z = np.exp(-zeta ** 2) * (1.j * np.sqrt(np.pi) - 2 * np.trapz(integ_func(_xs)) * _dx)

    Zprime = -2 * (1 + zeta * Z)
    result = 1 - 1. / 2 / k ** 2 / lamD ** 2 * Zprime
    return np.abs(result)


#### compute dispersion relation vs k*lambda ####
get_dispersion = False
if get_dispersion:
    fstr = f'landau-damping-rates_Te{T_e:1.1f}.pkl'
    try:
        kLams, freqs, damps = pickle.load(open(fstr, 'rb'))
    except IOError:
        kLams = np.linspace(0.1, 2.1, 51)  # k*lamD
        freqs = []
        damps = []

        for kLam in kLams:
            kLam = kLam / lamD

            x0 = np.array([1.0, -0.1])
            w_result = scipy.optimize.minimize(freq, x0, args=(kLam,), tol=1.0e-15, method='Nelder-Mead')
            w_opt = w_result.x

            print('w?', kLam, w_opt, freq(w_opt, kLam))
            freqs += [w_opt[0]]
            damps += [-1 * w_opt[1]]

        freqs = np.array(freqs)
        damps = np.array(damps)

#     pickle.dump(np.array([kLams, freqs, damps]), open(fstr, 'wb'))

    plt.figure()
    plt.plot(kLams, freqs, label='omega')
    plt.legend()

    plt.figure()
    plt.semilogy(kLams, damps, label='damping')
    plt.legend()
    plt.show()

##########################


def maxwellian(*vs, density=1.0, flow=0.0, mass=1.0, temp=1.0):
    """ vs: velocity grid for each dimension
            summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
        flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    v2 = 0.0
    for i in range(len(vs)):
        try:                v2 += (vs[i]-flow[i])*(vs[i]-flow[i])
        except TypeError:   v2 += (vs[i] - flow) * (vs[i] - flow)
    vth2 = eV * temp / mass
    return density*np.sqrt(1./2/np.pi/vth2) * np.exp(-v2/2/vth2)


def get_grid_params(idx):
    if idx == 0:
        x_bounds = -np.pi/k, np.pi/k
        ve_bounds = -6 * vth_e, 6 * vth_e
        vi_bounds = -6 * vth_i, 6 * vth_i
    elif idx == 1:
        x_bounds = -np.pi/k, np.pi/k
        ve_bounds = -30 * vth_e, 30 * vth_e
        vi_bounds = -6 * vth_i, 6 * vth_i
    else:
        raise IndexError(f'grid parameter for index {idx} not defined')
    return x_bounds, ve_bounds, vi_bounds


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
    return ax_maps


def get_fstr(add_str=''):
    """ get filename
    """
    ## parameters string
    fstr_params = f'k{k:1.2f}_mr{mass_i}'

    ## collision string
    if coll_type is None:
        fstr_coll = ''
    else:
        if do_adapt_coll:
            fstr_coll = f'_{coll_type}{coll_coeff_e}'
        else:
            fstr_coll = f'_{coll_type}e{coll_rate_e}i{coll_rate_i}'

    ## compression string
    fstr_comp = f'_C{compress_style}{compress_style_mod}_mb{DMAX}'

    ## E compression string
    if compress_E and DMAX_E is not None:
        fstr_compE = f'_E{DMAX_E}'
    else:
        fstr_compE = ''

    ## grid string
    fstr_grid = f'_G{grid_idx}_TN{TN_layout.value}{ax_map_key}' + \
                f'_Ls{Lx:02d},{Lve:02d},{Lvi:02d}_o{order}'

    if do_adapt_dt:
        fstr_t = f'_cfl{cfl}_T{int(T):03d}_te{te_order}'
    else:
        fstr_t = f'_dt{dt:1.3f}_T{int(T):03d}_te{te_order}'

    fstr1 = fstr_params + '_xv' + fstr_comp + fstr_compE + fstr_coll + fstr_grid + fstr_t
    return fstr1 + add_str


#########################
### define parameters ###
#########################

### system parameters ###
A = 0.5  # 1.0E-3
# k = 0.75/lamD
k = 0.10/lamD

### run-time parameters ###
q = 2
K = 1

# TN_layout = LayoutType.SEQUENTIAL
# ax_map_key = 'FB'

# TN_layout = LayoutType.PARALLEL
# ax_map_key = 'FF'

TN_layout = LayoutType.COMB
ax_map_key = 'BB'

if k==0.10 and A > 0.01:
    Lx, Lve, Lvi = 8, 10, 8
    # Lx, Lve, Lvi = 8, 11, 8
    # Lx, Lve, Lvi = 8, 12, 8
    grid_idx = 1
else:
    Lx, Lve, Lvi = 6, 8, 6
    grid_idx = 0

T = 60.0
cfl = 3.6  # 0.9
order = 1
te_order = 60  # 4
do_adapt_dt = True
frac_dt = 0.9
dt = None  # 0.01
upwind = False # True

add_str = 'sq' if is_sqrt else ''

#################

### define collisions
do_adapt_coll = True
coll_type = None  # CollisionType.LB
coll_coeff_e = 0.005  ## to scale collision frequency if using adaptive collisions
coll_coeff_i = 0.005/np.sqrt(mass_i/mass_e)
coll_rate_e = 0.01*wp_e if coll_type is not None else 0.0
coll_rate_i = 0.01*wp_i if coll_type is not None else 0.0
# collision_config = CollisionConfiguration(coll_type,
#                                           coeff_e = coll_coeff_e, coeff_i = coll_coeff_i,
#                                           rate_e = coll_rate_e, rate_i=coll_rate_i)
if do_adapt_coll:
    collision_config = CollisionConfiguration(coll_type,
                                              coeff_e = coll_coeff_e, coeff_i = coll_coeff_i,
                                              rate_e=1.0, rate_i=1.0,)
else:
    collision_config = CollisionConfiguration(coll_type,
                                              coeff_e=1.0, coeff_i=1.0,
                                              rate_e=coll_rate_e, rate_i=coll_rate_i,)

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
DMAX = 16  # 32 # None
compress_style = 2 # 2
compress_style_mod = 0

te_compress_i = CompressionConfiguration()
te_compress_e = CompressionConfiguration()
te_compress_V = CompressionConfiguration()

for ci in range(1,compress_style+1):
    te_compress_i.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_e.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_V.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=1.0E-20)

print('te compress fe', te_compress_e)
print('te compress fi', te_compress_i)
print('te compress V',  te_compress_V)

comp_levels = list(range(1,6))
if compress_style_mod == 1:     # don't compress inside the derivative
    comp_levels = [1,2,3,4,0]
elif compress_style_mod == 2:   # don't compress inside the derivative
    comp_levels = [1,2,3,0,0]

print('compress', DMAX, compress_style, compress_style_mod)

#### compression of E before dot
compress_E = True
DMAX_E = DMAX  # None
E_compress_opts = {'max_bond':DMAX_E,'cutoff':CUTOFF,'cutoff_mode':CUTOFF_MODE}


#############################
## start building PDE system
#############################

## define grid
npts_x, npts_ve, npts_vi = q ** Lx, q ** Lve, q ** Lvi
x_lims, ve_lims, vi_lims = get_grid_params(grid_idx)
kmax = k * npts_x / 2
k_vals = np.linspace(-kmax, kmax, npts_x, endpoint=False)
ve_vals = np.linspace(*ve_lims,npts_ve,endpoint=True)
vi_vals = np.linspace(*vi_lims,npts_vi,endpoint=True)
ve_max, vi_max = np.max(np.abs(ve_lims)), np.max(np.abs(vi_lims))
tot_x = 2 * np.pi / k


## initial conditions
# fe = lambda x:  np.sqrt(1 + A*np.cos(k*x))      # (a - b)
# fi = lambda x:  1 + 0*x
# V_field  = lambda x: -A/k**2 * np.cos(k*x)
# E_field  = lambda x: -A/k * np.sin(k*x)

## electron density:  1 + A * np.cos(k * x)
### (1 + A cos(kx) = (1 - i sqrt(A cos(kx)) * (1 + i sqrt(A cos(kx))
### n0e * (1 + A/2 (exp(ikx) + c.c.)) = n0e * (a + b exp(ikx)) * c.c.
### where a^2 + b^2 = 1, a*b = A/2
const_a = np.roots([-1,0,1,0,-A**2/4])[0]
const_b = A / 2 / const_a
print('consts', const_a, const_b)

fe_x = np.zeros(npts_x, dtype=complex)
if is_sqrt:
    fe_x[npts_x // 2] = const_a
    fe_x[npts_x // 2 + 1] = const_b
else:
    fe_x[npts_x // 2] = 1.0
    fe_x[npts_x // 2 - 1] = A / 2
    fe_x[npts_x // 2 + 1] = A / 2
fe_v = helper_test.maxwellian(ve_vals, vth2=vth_e**2, density=n0_e, flow=0.0, is_sqrt=is_sqrt)

## ion density
fi_x = np.zeros(npts_x, dtype=complex)
fi_x[npts_x // 2] = 1.0
fi_v = helper_test.maxwellian(vi_vals, vth2=vth_i**2, density=n0_i, flow=0.0, is_sqrt=is_sqrt)

## V field: -A / k ** 2 * np.cos(k * x)
V_x = np.zeros(npts_x, dtype=complex)
V_x[npts_x // 2 - 1] = -A / k ** 2 / 2
V_x[npts_x // 2 + 1] = -A / k ** 2 / 2

## E field: -A / k * np.sin(k * x)
E_x = np.zeros(npts_x, dtype=complex)
E_x[npts_x // 2 - 1] = -1.j * A / k / 2
E_x[npts_x // 2 + 1] = 1.j * A / k / 2


## define axis objects
X = Coordinate('KX', CoordinateType.X)
V = Coordinate('V', CoordinateType.X)

map_x, map_v = get_maps(ax_map_key)
ax_x  = Axis(Lx, q, coordinate=X, dx=k, xpts=k_vals, basis=FourierBasis(), ax_map=map_x)
ax_vi = Axis(Lvi, q, coordinate=V, xpts=vi_vals, ax_map=map_v)
ax_ve = Axis(Lve, q, coordinate=V, xpts=ve_vals, ax_map=map_v)
print('ax_x zero ind', ax_x.zero_ind)

coords_x  = CartesianCoordinateSpace('X',ax_x)
coords_vi = CartesianCoordinateSpace('Vi',ax_vi)
coords_ve = CartesianCoordinateSpace('Ve',ax_ve)

## configure derivatives
deriv_x = DerivativeConfiguration()
deriv_v = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)

## define grids
if TN_layout == LayoutType.COMB:
    grid_X = Grid1D('X1', (ax_x,), layout_type=LayoutType.SEQUENTIAL)
    grid_Ve = Grid1D('Ve', (ax_ve,), layout_type=LayoutType.SEQUENTIAL)
    grid_Vi = Grid1D('Vi', (ax_vi,), layout_type=LayoutType.SEQUENTIAL)

    grid_i = GridsComb('XVi',(grid_X, grid_Vi))
    grid_e = GridsComb('XVe',(grid_X, grid_Ve))
    grid_V = GridsComb('X',(grid_X,))
else:
    grid_i = Grid1D('XVi',(ax_x,ax_vi), layout_type=TN_layout)
    grid_e = Grid1D('XVe',(ax_x,ax_ve), layout_type=TN_layout)
    grid_V = Grid1D('X',(ax_x,), layout_type=TN_layout)

## define initial grid1DTNs
fe_x = ax_x.map_state_to_mps(fe_x)
fi_x = ax_x.map_state_to_mps(fi_x)
fe_v = ax_ve.map_state_to_mps(fe_v)
fi_v = ax_vi.map_state_to_mps(fi_v)

init_fe_gtn = grid_e.make_mps_ndim({ax_x: fe_x, ax_ve: fe_v})
init_fi_gtn = grid_i.make_mps_ndim({ax_x: fi_x, ax_vi: fi_v})
init_V_gtn  = grid_V.map_state_to_mps(V_x)

init_fe_gtn.ax_deriv_configs = {ax_x: deriv_x, ax_ve: deriv_v}
init_fi_gtn.ax_deriv_configs = {ax_x: deriv_x, ax_vi: deriv_v}
init_V_gtn.ax_deriv_configs  = {ax_x: deriv_x}
print('initV',init_V_gtn.ax_deriv_configs[ax_x].bc)

init_norm_e = init_fe_gtn.norm(is_sqrt=is_sqrt)
init_norm_i = init_fe_gtn.norm(is_sqrt=is_sqrt)
print('init norm', init_norm_e, init_norm_i)


### define time step ###
dx, dv_e, dv_i = ax_x.dx, ax_ve.dx, ax_vi.dx
kx_inds = get_zero_bounds(init_fe_gtn.get_data())[0]
kmax_x = np.max([np.abs(ax_x.xpts[ind]) for ind in kx_inds])

dt_rk4 = rk4_limit_Fourier(kmax_x, max(ve_max,vi_max))
e_params = ((dv_e,), (A/k*eV/mass_e,))
i_params = ((dv_i,), (A/k*eV/mass_i,))

if dt is None:
    if not do_adapt_coll or coll_type is None:
        Ce_params = ((dv_e, dv_e**2/2), (collision_config.coll_rate_e*ve_max,
                                         (vth_e**2)*collision_config.coll_rate_e))
        Ci_params = ((dv_i, dv_i**2/2), (collision_config.coll_rate_i*vi_max,
                                         (vth_i**2)*collision_config.coll_rate_i))
        dt_cfl = cfl_limit( e_params, i_params, Ce_params, Ci_params )
        dt = min(dt_rk4, dt_cfl) * frac_dt
    else:
        dt_cfl = frac_dt * cfl_limit( e_params, i_params) #, Ce_params, Ci_params )
        dt = min(dt_rk4, dt_cfl) * frac_dt
        coll_rate_e = LB_collision_freq_cfl_limit(dt, dv_e, vth_e, ve_max, cfl)
        coll_rate_i = LB_collision_freq_cfl_limit(dt, dv_i, vth_i, vi_max, cfl)
        print('cfl limit collision rates', coll_rate_e, coll_rate_i)
        collision_config.base_rate_e = coll_rate_e
        collision_config.base_rate_i = coll_rate_i
        print('cfl limit scaled collision rates', collision_config.coll_rate_e, collision_config.coll_rate_i)

num_tsteps = int(T/dt)
print('dt',dt,num_tsteps)

fstr = get_fstr(add_str=add_str)
print('filename', fstr)

##################3

v2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_ve], x_power=2)
v2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vi], x_power=2)

try:
    if not load_data:  raise IOError
    fe_t = pickle.load(open(fdir+'fe_'+fstr+'.pkl','rb'))
    fi_t = pickle.load(open(fdir+'fi_'+fstr+'.pkl','rb'))
    max_bond_V  = pickle.load(open(fdir+'maxD_V_'+fstr+'.pkl','rb'))
    max_bond_fe = pickle.load(open(fdir+'maxD_fe_'+fstr+'.pkl','rb'))
    max_bond_fi = pickle.load(open(fdir+'maxD_fi_'+fstr+'.pkl','rb'))
    nrg_fe_ts = pickle.load(open(fdir+'nrg_e_'+fstr+'.pkl','rb'))
    nrg_fi_ts = pickle.load(open(fdir+'nrg_i_'+fstr+'.pkl','rb'))
    nrg_E_ts  = pickle.load(open(fdir+'nrg_E_'+fstr+'.pkl','rb'))
    # err_e_ts  = pickle.load(open(fdir+'err_e_'+fstr+'.pkl','rb'))
    # err_i_ts  = pickle.load(open(fdir+'err_i_'+fstr+'.pkl','rb'))
    ts  = pickle.load(open(fdir+'ts_'+fstr+'.pkl','rb'))

    nt = num_tsteps
    print('loaded data')

    fe_data = grid_e.map_mps_to_state(fe_t)
    fi_data = grid_i.map_mps_to_state(fi_t)

except(IOError,EOFError):

    try:
        if restart_from_T > 0:
            restart_fstr = fstr
            fe_t = pickle.load(open(fdir+'fe_'+restart_fstr+'.pkl','rb'))
            fi_t = pickle.load(open(fdir+'fi_'+restart_fstr+'.pkl','rb'))
            # Ex_t = pickle.load(open(fdir+'Ext_' +restart_fstr+'.pkl','rb'))
            max_bond_V  = pickle.load(open(fdir+'maxD_V_'+restart_fstr+'.pkl','rb'))
            max_bond_fe = pickle.load(open(fdir+'maxD_fe_'+restart_fstr+'.pkl','rb'))
            max_bond_fi = pickle.load(open(fdir+'maxD_fi_'+restart_fstr+'.pkl','rb'))
            nrg_fe_ts = pickle.load(open(fdir+'nrg_e_'+fstr+'.pkl','rb'))
            nrg_fi_ts = pickle.load(open(fdir+'nrg_i_'+fstr+'.pkl','rb'))
            nrg_E_ts  = pickle.load(open(fdir+'nrg_E_'+fstr+'.pkl','rb'))
            # err_e_ts  = pickle.load(open(fdir+'err_e_'+fstr+'.pkl','rb'))
            # err_i_ts  = pickle.load(open(fdir+'err_i_'+fstr+'.pkl','rb'))
            ts  = pickle.load(open(fdir+'ts_'+fstr+'.pkl','rb'))
            print('restarting from ',restart_fstr)
            ts = list(ts)
            nt = len(ts)
        else:
            if not restart:  raise ValueError

            re_fstr = fstr
            print(re_fstr)

            restart_fstrs = glob.glob(sdir+'restart/fe_'+re_fstr+'*.pkl')
            print(sdir+'restart/')
            print('restart?', len(restart_fstrs))
            nts = []
            for f in restart_fstrs:
                nt_ind = f.find('nt')
                nt_str = f[nt_ind+2:-4]
                nts += [int(nt_str)]
            nt = max(nts)
            # nt = 5750
            print('nt',nt)
            nts.remove(nt)

            try: 
                raise NameError
                for nt_ in nts:   ## remove old restart files
                    if nt_ > nt:
                        print('removing restart', nt_)
                        restart_files = glob.glob(sdir+'restart/*'+re_fstr+f'-nt{nt_}.pkl')
                        for f in restart_files:  os.remove(f)
            except NameError:   pass

            fe_t = pickle.load(open(sdir+'restart/fe_'+re_fstr+f'-nt{nt}.pkl','rb'))
            fi_t = pickle.load(open(sdir+'restart/fi_'+re_fstr+f'-nt{nt}.pkl','rb'))
            max_bond_V  = pickle.load(open(sdir+'restart/maxD_V_'+re_fstr+f'.pkl','rb'))
            max_bond_fe = pickle.load(open(sdir+'restart/maxD_fe_'+re_fstr+f'.pkl','rb'))
            max_bond_fi = pickle.load(open(sdir+'restart/maxD_fi_'+re_fstr+f'.pkl','rb'))
            print('loaded',nt,len(max_bond_V))
            nrg_fe_ts = pickle.load(open(sdir+'restart/nrg_e_'+re_fstr+f'.pkl','rb'))
            nrg_fi_ts = pickle.load(open(sdir+'restart/nrg_i_'+re_fstr+f'.pkl','rb'))
            nrg_E_ts  = pickle.load(open(sdir+'restart/nrg_E_'+re_fstr+f'.pkl','rb'))
            # err_e_ts  = pickle.load(open(sdir+'restart/err_e_'+fstr+'.pkl','rb'))
            # err_i_ts  = pickle.load(open(sdir+'restart/err_i_'+fstr+'.pkl','rb'))
            ts = pickle.load(open(sdir+'restart/ts_'+re_fstr+f'.pkl','rb'))
            ts = list(ts)
            print('loaded',nt,len(nrg_fe_ts))

        max_bond_V  = max_bond_V [:nt+1]
        max_bond_fe = max_bond_fe[:nt+1]
        max_bond_fi = max_bond_fi[:nt+1]
        nrg_fe_ts = nrg_fe_ts[:nt+1]
        nrg_fi_ts = nrg_fi_ts[:nt+1]
        nrg_E_ts = nrg_E_ts[:nt+1]
        # err_e_ts = err_e_ts[:nt+1]
        # err_i_ts = err_i_ts[:nt+1]
        ts = ts[:nt+1]
        total_E0 = nrg_E_ts[0] + nrg_fi_ts[0] + nrg_fe_ts[0]


        fig7,ax7 = plt.subplots()
        ax7.semilogy(ts,np.abs(nrg_fe_ts),label='elec')
        ax7.semilogy(ts,np.abs(nrg_fi_ts),label='ion')
        ax7.semilogy(ts,np.abs(nrg_E_ts ),label='E field')
        total_nrg_t = np.array(nrg_fe_ts) + np.array(nrg_fi_ts) + np.array(nrg_E_ts)
        ax7.semilogy(ts,np.abs(total_nrg_t),label='total')
        
        try:
            raise IOError
            ref_fstr = f'/home/erikaye/gkeyll_tests/vm-damp/L{L:02d}_A{Astr}/' + \
                             f'Ex_t_m{int(mass_i/mass_e)}_k{int(100*k):03d}.pkl' 
            print(ref_fstr)
            ref_data = pickle.load(open(ref_fstr,'rb'))
            ax7.semilogy(ref_data[0],ref_data[1]*0.5,'k--',label='ref')
        except IOError:    pass
        
        ax7.legend()
        ax7.set_xlabel('t')
        ax7.set_ylabel('energy')
        if save_figs: fig7.savefig(fdir+'plot_nrg_'+fstr+'.png')
        print(fdir+'plot_nrg_'+fstr+'.png')
        # exit()


        init_fe_gtn.data = fe_t
        init_fi_gtn.data = fi_t
        init_V_gtn.data  = None

        # init_fe_gtn = GridTN1D(grid_e, data=fe_t, ax_deriv_configs={ax_x.axID: deriv_xe_up, ax_ve.axID: deriv_v})
        # init_fi_gtn = GridTN1D(grid_i, data=fi_t, ax_deriv_configs={ax_x.axID: deriv_xi_up, ax_ve.axID: deriv_v})
        # init_V_gtn  = GridTN1D(grid_V, data=None, ax_deriv_configs={ax_x.axID: deriv_x})

        restarted = True
        print('loaded restart',nt)

    except(ValueError,FileNotFoundError):
        nt = 0
        ts = [0]
        restarted = False

        init_fe_data = init_fe_gtn.get_data()
        init_fi_data = init_fi_gtn.get_data()
        init_E = E_x

        nrg_fe = init_fe_gtn.meas_expec(v2e_mpo, is_sqrt=is_sqrt) * mass_e / 2 * (k / 2 / np.pi)
        nrg_fi = init_fi_gtn.meas_expec(v2i_mpo, is_sqrt=is_sqrt) * mass_i / 2 * (k / 2 / np.pi)
        nrg_fe_ts = [nrg_fe]
        nrg_fi_ts = [nrg_fi]
        nrg_E_ts  = [(A/k/2)**2  * 2*np.pi/k  * (k/2/np.pi)]

        norm_fe = init_fe_gtn.norm(is_sqrt=is_sqrt)
        norm_fi = init_fi_gtn.norm(is_sqrt=is_sqrt)
        if is_sqrt:
            norm_fe = norm_fe ** 2
            norm_fi = norm_fi ** 2
        print('norm', norm_fe, norm_fi)

        # fe_v = np.sqrt(maxwellian(ve_vals, density=n0_e, flow=0.0, mass=mass_e, temp=T_e))
        # fi_v = np.sqrt(maxwellian(vi_vals, density=n0_i, flow=0.0, mass=mass_i, temp=T_i))
        # print(np.sum(fe_v**2 * dv_e), np.sum(fe_v**2 * ve_vals**2 * dv_e) * mass_e / 2, charge * T_e)
        # print(np.sum(fi_v**2 * dv_i), np.sum(fi_v**2 * vi_vals**2 * dv_i) * mass_i / 2, charge * T_i)

        print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
        max_bond_V  = [init_V_gtn.max_bond()]
        max_bond_fe = [init_fe_gtn.max_bond()]
        max_bond_fi = [init_fi_gtn.max_bond()]

        err_e_ts = [0.]
        err_i_ts = [0.]

    init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, compress_config=te_compress_e, is_sqrt=is_sqrt)
    init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, compress_config=te_compress_i, is_sqrt=is_sqrt)
    init_V_field  = ScalarField('V' , grid_V, data=init_V_gtn , compress_config=te_compress_V, is_sqrt=True)

    print('current nrgs', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1])
    print('norm',init_fe_field.norm(), init_fi_field.norm())

    ### BUILD PDE
    vp_sys = VlasovPoisson(init_fe_field, init_fi_field, init_V_field, coords_x=coords_x, coords_ve=coords_ve,
                           coords_vi=coords_vi, elc_params=elc_config, ion_params=ion_config, normalize=True,
                           te_order=te_order, compress_levels=comp_levels)
    vp_sys.collision = collision_config
    print('colls', collision_config.coll_params)

    vp_sys.compress_F = compress_E
    vp_sys.compress_F_opts = E_compress_opts
    print('compress E?', compress_E, E_compress_opts)

    vp_sys.init_norm_e = init_norm_e
    vp_sys.init_norm_i = init_norm_i

    print('before update', vp_sys.V.component.ax_deriv_configs[ax_x].bc, ax_x)
    E_field = vp_sys.V.gradient()
    E_data = E_field.get_comp_data(X)

    plt.figure()
    if E_data is not None:
        print('V nrg', E_field.norm()**2 * 1. / 2 / tot_x)
        plt.plot(vp_sys.V.get_comp_data(), label='before')

    if True:  # restarted:
        vp_sys.update_V(inplace=True)
    print('after update', vp_sys.V.component.ax_deriv_configs[ax_x].bc)
    E_field = vp_sys.V.gradient()
    E_data = E_field.get_comp_data(X)
    print('V nrg', np.sum(E_data * np.conj(E_data)) * dx * 1. / 2 / tot_x)
    print('E nrg', E_field.integrate(compIDs=[X])[X] * 1. / 2 / tot_x )
    energy_E_t = E_field.norm()**2 * 1./ 2 / tot_x

    plt.plot(vp_sys.V.get_comp_data(), '--', label='after')
    plt.legend()
    plt.show()

    # exit()

    if nt == 0:
        fig1,ax1 = plt.subplots(2,1)
        time_ = nt*dt
        fi_real = ax_x.basis.get_realspace_1D(init_fi_data, 0)
        fi_real = ax_vi.basis.get_realspace_1D(fi_real, 1)
        fi_real = np.abs(fi_real) ** 2 if is_sqrt else fi_real
        im1 = ax1[0].imshow(np.abs(init_fi_data.T)**2, extent=(-np.pi/k, np.pi/k, vi_vals[0], vi_vals[-1]),
                            aspect='auto', origin='lower')

        fe_real = ax_x.basis.get_realspace_1D(init_fe_data, 0)
        fe_real = ax_ve.basis.get_realspace_1D(fe_real, 1)
        fe_real = np.abs(fe_real) ** 2 if is_sqrt else fe_real
        im2 = ax1[1].imshow(np.abs(init_fe_data.T)**2, extent=(-np.pi/k, np.pi/k, ve_vals[0], ve_vals[-1]),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
        ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
        # if save_figs:   fig1.savefig(fdir+f'plot_f_'+fstr+f'_nt{nt}.png')
        plt.show()
        plt.close(fig1)

        # V_data = vp_sys.V.get_field_data(comps=0)
        # plt.figure()
        # plt.plot(x_vals, V_data, label='V field')
        # plt.legend()

        plt.show()

    if restarted:
        # vp_sys.update_V(inplace=True)
        if do_adapt_dt:
            E_field = vp_sys.V.gradient()
            E_data = E_field.get_comp_data(X)

            E_max = np.max(np.abs(E_data))
            kx_inds = get_zero_bounds(init_fe_gtn.get_data())[0]
            kmax_x = np.max([np.abs(ax_x.xpts[ind]) for ind in kx_inds])
            dt_rk4 = rk4_limit_Fourier(kmax_x, max(ve_max, vi_max))
            e_params = ((dx, dv_e), (ve_max, np.max(np.abs(E_data)) * charge / mass_e))  ## can use sqrt(sum(E**2))
            i_params = ((dx, dv_i), (vi_max, np.max(np.abs(E_data)) * charge / mass_i))

            if not do_adapt_coll or coll_type is None:
                Ce_params = ((dv_e, dv_e**2/2), (collision_config.coll_rate_e*ve_max,
                                                 (vth_e**2)*collision_config.coll_rate_e))
                Ci_params = ((dv_i, dv_i**2/2), (collision_config.coll_rate_i*vi_max,
                                                 (vth_i**2)*collision_config.coll_rate_i))
                dt = cfl * helper_test.cfl_limit( e_params, i_params, Ce_params, Ci_params )
                dt = min(dt_rk4, dt_cfl) * frac_dt
            else:
                dt_cfl = frac_dt * helper_test.cfl_limit( e_params, i_params) #, Ce_params, Ci_params )
                dt = min(dt_rk4, dt_cfl) * frac_dt
                coll_rate_e = helper_test.LB_collision_freq_cfl_limit(dt, dv_e, vth_e, ve_max, cfl)
                coll_rate_i = helper_test.LB_collision_freq_cfl_limit(dt, dv_i, vth_i, vi_max, cfl)
                print('cfl limit collision rates', coll_rate_e, coll_rate_i)
                collision_config.base_rate_e = coll_rate_e
                collision_config.base_rate_i = coll_rate_i
                print('cfl limit scaled collision rates', collision_config.coll_rate_e, collision_config.coll_rate_i)

            print('restarted dt', dt)


    time1 = time.time()
    
    while ts[-1] < T:

        walltime = time.time()
        vp_sys = vp_sys.next_time_step(dt, compress_level=1, is_first_time_step=nt==0)  #, verbose_plot=(nt > 18 or nt<=5))
        print('wall time', time.time() - walltime)

        pot_field = vp_sys.V
        max_bond_V += [pot_field.max_bond()]
        elec_field = pot_field.gradient()
        E_data = elec_field.get_comp_data(ax_x.coordinate)

        # plt.figure()
        # plt.plot(x_vals, E_data, label='E', marker='x')
        # plt.plot(x_vals, pot_field.get_comp_data(), label='V', marker='x')
        # plt.show()

        fe_field : 'ScalarField' = vp_sys.fe
        max_bond_fe += [fe_field.max_bond()]
        # fe_data = fe_field.get_comp_data()
    
        fi_field : 'ScalarField' = vp_sys.fi
        max_bond_fi += [fi_field.max_bond()]
        # fi_data = fi_field.get_comp_data()

        nrg_fe = fe_field.meas_expec(v2e_mpo).component * mass_e / 2 / tot_x
        nrg_fi = fi_field.meas_expec(v2i_mpo).component * mass_i / 2 / tot_x
        nrg_fe_ts += [nrg_fe]
        nrg_fi_ts += [nrg_fi]
        energy_E_t = elec_field.norm() ** 2 * 1. / 2 / tot_x  # electric field energy
        nrg_E_ts  += [energy_E_t]
   
        ts += [ts[-1]+dt]
        nt += 1

        if do_adapt_dt:
            E_max = np.max(np.abs(E_data))
            kx_inds = get_zero_bounds(fe_field.get_comp_data())[0]
            kmax_x = np.max([np.abs(ax_x.xpts[ind]) for ind in kx_inds])
            dt_rk4 = rk4_limit_Fourier(kmax_x, max(ve_max, vi_max))
            e_params = ((dx, dv_e), (ve_max, np.max(np.abs(E_data))*charge/mass_e))  ## can use sqrt(sum(E**2)) too
            i_params = ((dx, dv_i), (vi_max, np.max(np.abs(E_data))*charge/mass_i))

            if not do_adapt_coll or coll_type is None:
                Ce_params = ((dv_e, dv_e**2/2), (collision_config.coll_rate_e*ve_max,
                                                 (vth_e**2)*collision_config.coll_rate_e))
                Ci_params = ((dv_i, dv_i**2/2), (collision_config.coll_rate_i*vi_max,
                                                 (vth_i**2)*collision_config.coll_rate_i))
                dt_cfl = cfl_limit(e_params, i_params, Ce_params, Ci_params)
                dt = min(dt_rk4, dt_cfl) * frac_dt
            else:
                dt_cfl = frac_dt * cfl_limit(e_params, i_params)
                dt = min(dt_rk4, dt_cfl) * frac_dt
                coll_rate_e = helper_test.LB_collision_freq_cfl_limit(dt, dv_e, vth_e, ve_max, cfl)
                coll_rate_i = helper_test.LB_collision_freq_cfl_limit(dt, dv_i, vth_i, vi_max, cfl)
                collision_config.base_rate_e = coll_rate_e
                collision_config.base_rate_i = coll_rate_i

        norm_diff = vp_sys.fe.norm() - vp_sys.fi.norm()
        print('result norm', nt, ts[-1], dt, norm_diff, energy_E_t,
              pot_field.max_bond(), fe_field.max_bond(), 2 ** ((Lx+Lve)/2))

        if nt%10 == 0:
            fig1,ax1 = plt.subplots(2,1)
            time_ = ts[-1]
            fi_data = fi_field.get_comp_data()
            fe_data = fe_field.get_comp_data()
            im1 = ax1[0].imshow(np.log10(np.abs(fi_data)).T, extent=(-kmax, kmax, vi_lims[0], vi_lims[1]),
                                aspect='auto', origin='lower')
            im2 = ax1[1].imshow(np.log10(np.abs(fe_data)).T, extent=(-kmax, kmax, ve_lims[0], ve_lims[1]),
                                aspect='auto', origin='lower')

            fig1.colorbar(im1, ax=ax1[0])
            fig1.colorbar(im2, ax=ax1[1])
            ax1[0].set_title(f'time {time_:3.2f}')
            ax1[0].xaxis.set_visible(False)
            ax1[0].set_xlabel('kx'), ax1[0].set_ylabel('v')
            ax1[1].set_xlabel('kx'), ax1[1].set_ylabel('v')
            if save_figs:   fig1.savefig(fdir+f'plot_f_'+fstr+f'_nt{nt}.png')

            V_data = pot_field.get_comp_data()
            plt.figure()
            plt.plot(np.arange(npts_x) * k, np.real(E_data), label='E field real')
            plt.plot(np.arange(npts_x) * k, np.imag(E_data), label='E field imag')
            plt.plot(np.arange(npts_x) * k, np.real(V_data), label='V field real')
            plt.plot(np.arange(npts_x) * k, np.imag(V_data), label='V field imag')
            plt.legend()

            fig7, ax7 = plt.subplots()
            energy_E_t = np.array(nrg_E_ts)
            energy_fi_t = np.array(nrg_fi_ts)
            energy_fe_t = np.array(nrg_fe_ts)
            total_nrg_t = energy_fe_t + energy_fi_t + energy_E_t
            ax7.semilogy(ts, np.abs(energy_fe_t), label='elec')
            ax7.semilogy(ts, np.abs(energy_fi_t), label='ion')
            ax7.semilogy(ts, np.abs(energy_E_t), label='E field')
            ax7.semilogy(ts, np.abs(total_nrg_t), label='total')
            ax7.legend()

            if not save_figs:
                plt.show()
            plt.close()
            plt.close()
            plt.close()


            if save_data:
                if not os.path.exists(sdir+'restart/'):
                    os.makedirs(sdir+'restart/')
                else:
                    try: 
                        raise NameError
                        restart_files = glob.glob(sdir+'restart/*'+fstr+f'-nt{save_nt}.pkl')
                        for f in restart_files:  os.remove(f)
                    except NameError:   pass

                save_nt = nt
                pickle.dump(vp_sys.fe.component.data,open(sdir+'restart/'+'fe_'+fstr+f'-nt{nt}.pkl','wb'))
                pickle.dump(vp_sys.fi.component.data,open(sdir+'restart/'+'fi_'+fstr+f'-nt{nt}.pkl','wb'))
                pickle.dump(max_bond_V ,open(sdir+'restart/'+'maxD_V_'+fstr+'.pkl','wb'))
                pickle.dump(max_bond_fe,open(sdir+'restart/'+'maxD_fe_'+fstr+'.pkl','wb'))
                pickle.dump(max_bond_fi,open(sdir+'restart/'+'maxD_fi_'+fstr+'.pkl','wb'))
                pickle.dump(nrg_fe_ts,open(sdir+'restart/nrg_e_'+fstr+'.pkl','wb'))
                pickle.dump(nrg_fi_ts,open(sdir+'restart/nrg_i_'+fstr+'.pkl','wb'))
                pickle.dump(nrg_E_ts ,open(sdir+'restart/nrg_E_'+fstr+'.pkl','wb'))
                # pickle.dump(err_e_ts ,open(sdir+'restart/err_e_'+fstr+'.pkl','wb'))
                # pickle.dump(err_i_ts ,open(sdir+'restart/err_i_'+fstr+'.pkl','wb'))
                pickle.dump(ts ,open(sdir+'restart/ts_'+fstr+'.pkl','wb'))

    

    if save_data:
        pickle.dump(vp_sys.fe.component.data,open(fdir+'fe_'+fstr+'.pkl','wb'))
        pickle.dump(vp_sys.fi.component.data,open(fdir+'fi_'+fstr+'.pkl','wb'))
        # pickle.dump(Ex_t ,open(fdir+'Ext_' +fstr+'.pkl','wb'))
        pickle.dump(max_bond_V ,open(fdir+'maxD_V_'+fstr+'.pkl','wb'))
        pickle.dump(max_bond_fe,open(fdir+'maxD_fe_'+fstr+'.pkl','wb'))
        pickle.dump(max_bond_fi,open(fdir+'maxD_fi_'+fstr+'.pkl','wb'))
        pickle.dump(nrg_fe_ts,open(fdir+'nrg_e_'+fstr+'.pkl','wb'))
        pickle.dump(nrg_fi_ts,open(fdir+'nrg_i_'+fstr+'.pkl','wb'))
        pickle.dump(nrg_E_ts ,open(fdir+'nrg_E_'+fstr+'.pkl','wb'))
        # pickle.dump(err_e_ts ,open(fdir+'err_e_'+fstr+'.pkl','wb'))
        # pickle.dump(err_i_ts ,open(fdir+'err_i_'+fstr+'.pkl','wb'))
        pickle.dump(ts ,open(fdir+'ts_'+fstr+'.pkl','wb'))

    print('wall time', time.time()-time1)



fig1,ax1 = plt.subplots(2,1)
time_ = ts[-1]
fi_real = ax_x.basis.get_realspace_1D(fi_data, 0)
fi_real = ax_vi.basis.get_realspace_1D(fi_real, 1)
fi_real = np.abs(fi_real) ** 2 if is_sqrt else fi_real
vmax = max(np.max(fe_data),np.max(fi_real))
vmin = min(np.min(fe_data),np.min(fi_rel))
im1 = ax1[0].imshow(fi_real, extent=(-np.pi/k, np.pi/k, vi_vals[0], vi_vals[-1]),
                    aspect='auto', origin='lower')
                    # vmin=vmin,vmax=vmax,aspect='auto')

fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
fe_real = ax_ve.basis.get_realspace_1D(fe_real, 1)
fe_real = np.abs(fe_real) ** 2 if is_sqrt else fe_real
im2 = ax1[1].imshow(fe_real, extent=(-np.pi/k, np.pi/k, ve_vals[0], ve_vals[-1]),
                    aspect='auto', origin='lower')
                    # vmin=vmin,vmax=vmax,aspect='auto')

# fig1.subplots_adjust(right=0.8)
# cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
# fig1.colorbar(im1, cax=cbar_ax)
fig1.colorbar(im1, ax=ax1[0])
fig1.colorbar(im2, ax=ax1[1])
ax1[0].set_title(f'time {time_:2.1f}')
ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
if save_figs:   fig1.savefig(fdir+f'plot_f_'+fstr+f'_T{time_:2.1f}.png')


fig2,ax2 = plt.subplots()
# ax2.plot(max_bond_fe,label=r'$\displaystyle f_e$')
# ax2.plot(max_bond_fi,label=r'$\displaystyle f_i$')
ax2.plot(np.log(max_bond_fe)/np.log(q**K),label=r'$f_e$')
ax2.plot(np.log(max_bond_fi)/np.log(q**K),label=r'$f_i$')
ax2.plot(np.log(max_bond_V )/np.log(q**K),label='V')
ax2.plot([0,num_tsteps],[(Lx+Lve)//2]*2, ':', label='max fe')
ax2.plot([0,num_tsteps],[(Lx+Lvi)//2]*2, ':', label='max fi')
ax2.plot([0,num_tsteps],[Lx//2]*2, ':', label='max V')
ax2.legend()
ax2.set_xlabel('t')
ax2.set_ylabel('log_{q^K}(max bond dimension)')
if save_figs:   fig2.savefig(fdir+'plot_maxbond_'+fstr+'.png')

energy_E_t  = np.array(nrg_E_ts)
energy_fi_t = np.array(nrg_fi_ts)
energy_fe_t = np.array(nrg_fe_ts)
total_nrg_t = energy_fe_t + energy_fi_t + energy_E_t

######################################################
### real space analysis to get damping rate, frequency
######################################################

## theoretical damping rate
x0 = np.array([1.0,0.1])
w_result = scipy.optimize.minimize(freq, x0, args=(k,), tol=1.0e-15, method='Nelder-Mead')
w_opt = w_result.x
print('w?',k, w_opt,freq(w_opt,k))
theory_w, theory_damping = w_opt
print('theory w', theory_w, 'theory damping', theory_damping)

# ## observed damping rate
# from scipy import signal
# log_energy = np.log(np.abs(energy_E_t))
# if k > 0.75:  log_energy = log_energy[:num_tsteps // 2]
# peak_inds, peak_data = signal.find_peaks(log_energy, height=np.max(log_energy) - 11)
# peak_heights = peak_data['peak_heights']
# w_obs = 2 * np.pi / (np.average(np.diff(peak_inds)) * dt) / 2  ## because max and min of cosine are found
# p_coeffs = np.polyfit(ts[peak_inds], peak_heights, 1)
# damping_rate_obs = p_coeffs[0] / 2
# damping_line = np.exp(np.polyval(p_coeffs, ts))
# print('w obs', w_obs, 'theory w', theory_w, 'damping obs', damping_rate_obs, 'damping', theory_damping)
##############

fig7,ax7 = plt.subplots()
# ts = np.arange(num_tsteps+1)*dt
ax7.semilogy(ts,np.abs(energy_fe_t),label='elec')
ax7.semilogy(ts,np.abs(energy_fi_t),label='ion')
ax7.semilogy(ts,np.abs(energy_E_t ),label='E field')
ax7.semilogy(ts,np.abs(total_nrg_t),label='total')

decay = np.exp(2*theory_damping*np.array(ts)) * energy_E_t[0]   # x 2 bc it's ~|Ex|^2
ax7.semilogy(ts,decay,'k:',linewidth=2,label='theor. damping')
# ax7.semilogy(ts,damping_line,'b:',label='obs. damping')
# ax7.semilogy(ts[peak_inds],np.exp(peak_heights),'bx')

try:
    # ref_fstr = f'/home/erikaye/gkeyll_tests/vm-damp/L{L:02d}_A{Astr}/' + \
    #                  f'Ex_t_m{int(mass_i/mass_e)}_k{int(100*k):03d}.pkl'
    # ref_data = pickle.load(open(ref_fstr,'rb'))
    # ax7.semilogy(ref_data[0],ref_data[1]*0.5,'k--',label='ref')

    ref_fstr1 = f'testU1t_TN{TN_layout.value}_cut16_k{k:1.2f}_A{Astr}'
    ref_fstr = ref_fstr1 + \
           f'_L{L:02d}_cfl{cfl}_T{T}_te{te_order}_m{mass_i}'
    nrg_E_ts  = pickle.load(open(fdir+'nrg_E_'+ref_fstr+'.pkl','rb'))
    ts  = pickle.load(open(fdir+'ts_'+ref_fstr+'.pkl','rb'))
    ax7.semilogy(ts,nrg_E_ts,'k--',label='ref (cutoff 1.0e-8)')
except(NameError, IOError):
    pass

ax7.legend()
ax7.set_xlabel('t')
ax7.set_ylabel('energy')
if save_figs:
    fig7.savefig(fdir+'plot_nrg_'+fstr+'.png')
    print(fdir+'plot_nrg_'+fstr+'.png')

plt.show()
   
   
print('done')

if True:
    print('cleaning files')
    restart_fstrs = glob.glob(sdir + 'restart/fe_' + fstr + '-nt*.pkl')
    
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
                restart_files = glob.glob(sdir + 'restart/*' + fstr + f'-nt{nt_}.pkl')
                for f in restart_files:  os.remove(f)
    except NameError:
        pass
