import os, sys, pickle, time, glob, scipy

sys.path.append('../')

from setup.configs import *
import setup.helper as helper_test

from axis import Axis
import axis_map
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from grid_comb import GridsComb
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from vlasov_tests import LandauDamping
from pde_vlasovES import VlasovPoisson

"""
Landau damping in K-dimensions
isotropic system but arbitrary orientation of cosine perturbation A*cos(kx*x + ky*y + kz*z)
"""

save_figs = False
save_data = True
load_data = True
restart   = True
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

tt_comp = -1  #-1 # 0 # 1, 2, 3

if restart or save_data or save_figs:
    if tt_comp < 0:   # default
        sdir = '/pool001/erikaye/tns_pde_v2/data4/VP2-damp-fd/'
        fdir = 'data4/VP2-damp-fd/'

        # sdir = '/pool001/erikaye/tns_pde_v2/data4/VP2-damp-fd-fullinv/'
        # fdir = 'data4/VP2-damp-fd-fullinv/'
    else:
        sdir = f'/pool001/erikaye/tns_pde_v2/data4/VP2-damp-fd-tt{tt_comp}/'
        fdir = f'data4/VP2-damp-fd-tt{tt_comp}/'

    if not os.path.exists(sdir):
        os.makedirs(sdir)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)

#######################
# standard parameters #
#######################

plasma_config = UnitsConfiguration()
charge = plasma_config.e  # unit of charge
eV = plasma_config.eV  # unit of kB*T

mass_e = 1.0  # mass of electron
mass_i = 1836 * mass_e  # mass of ion
n0_e = 1.0  # electron number density
n0_i = n0_e  # ion number density
T_e = 1.0  # electron temperature [eV]
T_i = T_e  # ion temperature [eV]

ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config.vth  # electron thermal speed
wp_e = elc_config.wp  # plasma e' frequency
lamD = elc_config.lamD  # Debye length
wp_i = ion_config.wp  # plasma i+ frequency
vth_i = ion_config.vth  # ion thermal speed



#########################
### define parameters ###
#########################

### system parameters ###
A = 0.5   # 1.0E-3
k = 0.75/lamD
kx = 0.5/lamD
ky = np.sqrt(k**2 - kx**2)
ks = (kx, ky)  # x,y,...
print('k', k, np.linalg.norm(ks), ks)

### run-time parameters ###
q = 2
KX = KV = 2
Lx, Ly = 6, 6
Lxe, Lye = 8, 8
Lxi, Lyi = 6, 6
order = 1

grid_idx = None

# TN_layout = LayoutType.SEQUENTIAL
# x_ax_map_key = 'FF'
# v_ax_map_key = 'BB'

TN_layout = LayoutType.COMB
x_ax_map_key = 'BB'
v_ax_map_key = 'BB'


T = 60.0
dt = None  # 0.002  # None
dt_frac = 0.9
te_order = 333
do_adapt_dt = True
upwind = False

save_every_nt = 100 # 250

DMAX = 32
DMAX_F = DMAX
comp_style = 2

VP_test = LandauDamping(ks=(kx, ky), A=A, grid_idx=grid_idx,
                        Lxs=(Lx, Ly), Lves=(Lxe, Lye), Lvis=(Lxi, Lyi,),
                        KX=KX, KV=KV, q=q,
                        x_map_key = x_ax_map_key, ve_map_key = v_ax_map_key,
                        ion_config = ion_config, elc_config = elc_config,
                        #
                        coll_type = None, do_adapt_coll = True, coll_coeff_e = 0.005, coll_rate_e = 0.01,
                        DMAX = DMAX, DMAX_F = DMAX_F, compress_style=comp_style, compress_style_mod = 0,
                        do_adapt_dt=do_adapt_dt, dt_frac=dt_frac, dt=dt, T=T, te_order=te_order,
                        grid_layout=TN_layout, # axes_order='alt',
                        )


#################

#### compression options
te_compress_i = VP_test.get_compression_config(DMAX=DMAX)
te_compress_e = VP_test.get_compression_config(DMAX=DMAX)
te_compress_V = VP_test.get_compression_config(DMAX=DMAX)

if tt_comp > -1:    # do default otherwise
    if tt_comp == 0:
        tt_max_bond = None
    elif tt_comp == 1 or tt_comp == 2:
        tt_max_bond = DMAX # * ((KX+KV)//2)
        te_compress_i = VP_test.get_compression_config(DMAX=None)
        te_compress_e = VP_test.get_compression_config(DMAX=None)
        te_compress_V = VP_test.get_compression_config(DMAX=None)
    elif tt_comp == 3:
        tt_max_bond = DMAX # * ((KX+KV)//2)
    else:
        raise ValueError

    TT_key = SubCompressConfigType.TT
    for comp_level in range(1, VP_test.compress_style + 1):
        te_compress_i.set_sub_compress_opts(TT_key, comp_level, max_bond=tt_max_bond)
        te_compress_e.set_sub_compress_opts(TT_key, comp_level, max_bond=tt_max_bond)
        te_compress_V.set_sub_compress_opts(TT_key, comp_level, max_bond=tt_max_bond)

print('te compress fe', te_compress_e)
print('te compress fi', te_compress_i)
print('te compress V', te_compress_V)

comp_levels = VP_test.get_compress_levels()

print('compress', DMAX, VP_test.compress_style, VP_test.compress_style_mod)


#############################
## start building PDE system
#############################

## initial conditions
def fe_x(*mesh_xs):
    wavevector = 0.
    for i in range(KX):
        wavevector += ks[i] * mesh_xs[i]
    return 1 + np.cos(wavevector)

def fi_x(*mesh_xs):
    return np.ones(next(iter(mesh_xs)).shape)

def V_field(*mesh_xs):
    wavevector = 0.
    for i in range(KX):
        wavevector += ks[i] * mesh_xs[i]
    potential = - A/k**2 * np.cos(wavevector)
    return potential

def E_field(*mesh_xs):
    wavevector = 0.
    for i in range(KX):
        wavevector += ks[i] * mesh_xs[i]

    e_fields = []
    for i in range(KX):
        e_fields += [ A * ks[i] / k**2 * np.sin(wavevector) ]
    return e_fields


####################

## initial conditions
x_min, x_max = VP_test.x_lims[0]
y_min, y_max = VP_test.x_lims[1]
vxe_min, vxe_max = VP_test.ve_lims[0]
vye_min, vye_max = VP_test.ve_lims[1]
vxi_min, vxi_max = VP_test.vi_lims[0]
vyi_min, vyi_max = VP_test.vi_lims[1]
tot_x = (x_max-x_min) * (y_max-y_min)

### axis objects
ax_x, ax_y = VP_test.pos_axes
ax_vxi, ax_vyi = VP_test.vi_axes
ax_vxe, ax_vye = VP_test.ve_axes
print('ax_x zero ind', ax_x.zero_ind)

X, Y, Z = VP_test.pos_coordsys.coords
VX, VY, VZ = VP_test.ve_coordsys.coords

## configure derivatives
deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
deriv_v = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)

# ## define MPS grids
grid_i = VP_test.grid_i
grid_e = VP_test.grid_e
grid_X = VP_test.grid_X

###
dx = ax_x.dx
dy = ax_y.dx
dvxe = ax_vxe.dx
dvye = ax_vye.dx
dvxi = ax_vxi.dx
dvyi = ax_vyi.dx
print('dxs', dx, dy, dvxe, dvye, dvxi, dvyi)

npts_x = ax_x.npts
npts_y = ax_y.npts
npts_vxe = ax_vxe.npts
npts_vye = ax_vye.npts
npts_vxi = ax_vxi.npts
npts_vyi = ax_vyi.npts

x_vals = ax_x.xpts
y_vals = ax_y.xpts
vxe_vals = ax_vxe.xpts
vye_vals = ax_vye.xpts
vxi_vals = ax_vxi.xpts
vyi_vals = ax_vyi.xpts


## define initial grid1DTNs
fe_vx = maxwellian(vxe_vals, vth2=vth_e**2, density=n0_e, flow=0.0)
fe_vy = maxwellian(vye_vals, vth2=vth_e**2, density=n0_e, flow=0.0)
fi_vx = maxwellian(vxi_vals, vth2=vth_i**2, density=n0_i, flow=0.0)
fi_vy = maxwellian(vyi_vals, vth2=vth_i**2, density=n0_i, flow=0.0)

init_fi_comps = [{ax_vxi: fi_vx, ax_vyi: fi_vy}]

# init_fe_comps = [{ax_vxe: fe_vx, ax_vye: fe_vy},
#                  {ax_x: A/2*np.exp(-1.j*kx*x_vals), ax_y: np.exp(-1.j*ky*y_vals), ax_vxe: fe_vx, ax_vye: fe_vy},
#                  {ax_x: A/2*np.exp(+1.j*kx*x_vals), ax_y: np.exp(+1.j*ky*y_vals), ax_vxe: fe_vx, ax_vye: fe_vy}]
init_fe_comps = [{ax_vxe: fe_vx, ax_vye: fe_vy},
                 {ax_x:  A*np.cos(kx*x_vals), ax_y: np.cos(ky*y_vals), ax_vxe: fe_vx, ax_vye: fe_vy},
                 {ax_x: -A*np.sin(kx*x_vals), ax_y: np.sin(ky*y_vals), ax_vxe: fe_vx, ax_vye: fe_vy}]

# init_V_comps = [{ax_x: -A/2/k**2 * np.exp(-1.j*kx*x_vals), ax_y: np.exp(-1.j*ky*y_vals)},
#                 {ax_x: -A/2/k**2 * np.exp(+1.j*kx*x_vals), ax_y: np.exp(+1.j*ky*y_vals)}]
init_V_comps = [{ax_x: -A/k**2 * np.cos(kx*x_vals), ax_y: np.cos(ky*y_vals)},
                {ax_x: +A/k**2 * np.sin(kx*x_vals), ax_y: np.sin(ky*y_vals)},]


init_fi_gtn = grid_i.make_gtn_from_dicts(init_fi_comps, data_type=DataType.MPS)
init_fe_gtn = grid_e.make_gtn_from_dicts(init_fe_comps, data_type=DataType.MPS)
init_V_gtn  = grid_X.make_gtn_from_dicts(init_V_comps, data_type=DataType.MPS)

plt.figure()
plt.imshow(np.real(init_fe_gtn.get_data(ax_select={ax_vxe:npts_vxe//2, ax_vye:npts_vye//2})))
plt.colorbar()
plt.show()
# exit()

init_fe_gtn.ax_deriv_configs = {ax_x: deriv_x, ax_y: deriv_x, ax_vxe: deriv_v, ax_vye: deriv_v}
init_fi_gtn.ax_deriv_configs = {ax_x: deriv_x, ax_y: deriv_x, ax_vxi: deriv_v, ax_vyi: deriv_v}
init_V_gtn.ax_deriv_configs  = {ax_x: deriv_x, ax_y: deriv_x}

### define time step ###
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
# v2e_data = v2e_mps.get_data(ax_select={ax_x: 0, ax_y: 0})
# plt.imshow(v2e_data[:,:])
# plt.colorbar()
# plt.show()

vx2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vxi], x_power=2)
vy2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vyi], x_power=2)
v2i_mpo = vx2i_mpo.add(vy2i_mpo)
print('v2 mpo max bond', v2e_mpo.max_bond(), v2i_mpo.max_bond())

##################3

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
    # err_e_ts  = pickle.load(open(fdir+'err_e_'+fstr+'.pkl','rb'))
    # err_i_ts  = pickle.load(open(fdir+'err_i_'+fstr+'.pkl','rb'))
    ts = pickle.load(open(fdir + 'ts_' + fstr + '.pkl', 'rb'))

    nt = num_tsteps
    print('loaded data')

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
            # err_e_ts  = pickle.load(open(fdir+'err_e_'+fstr+'.pkl','rb'))
            # err_i_ts  = pickle.load(open(fdir+'err_i_'+fstr+'.pkl','rb'))
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
            # nt = 5750
            print('nt', nt)
            nts.remove(nt)

            try:
                # raise NameError
                for nt_ in nts:  ## remove old restart files
                    if nt_ > nt or nt_%250 != 0:
                        print('removing restart', nt_)
                        restart_files = glob.glob(sdir + 'restart/*' + re_fstr + f'-nt{nt_}.pkl')
                        for f in restart_files:  os.remove(f)
            except NameError:
                pass

            fe_t = grid_e.load_gtn_data(sdir + 'restart/fe_' + re_fstr + f'-nt{nt}', gtn=init_fe_gtn)
            fi_t = grid_i.load_gtn_data(sdir + 'restart/fi_' + re_fstr + f'-nt{nt}', gtn=init_fi_gtn)
            max_bond_V = pickle.load(open(sdir + 'restart/maxD_V_' + re_fstr + f'.pkl', 'rb'))
            max_bond_fe = pickle.load(open(sdir + 'restart/maxD_fe_' + re_fstr + f'.pkl', 'rb'))
            max_bond_fi = pickle.load(open(sdir + 'restart/maxD_fi_' + re_fstr + f'.pkl', 'rb'))
            print('loaded', nt, len(max_bond_V))
            nrg_fe_ts = pickle.load(open(sdir + 'restart/nrg_e_' + re_fstr + f'.pkl', 'rb'))
            nrg_fi_ts = pickle.load(open(sdir + 'restart/nrg_i_' + re_fstr + f'.pkl', 'rb'))
            nrg_E_ts = pickle.load(open(sdir + 'restart/nrg_E_' + re_fstr + f'.pkl', 'rb'))
            # err_e_ts  = pickle.load(open(sdir+'restart/err_e_'+fstr+'.pkl','rb'))
            # err_i_ts  = pickle.load(open(sdir+'restart/err_i_'+fstr+'.pkl','rb'))
            ts = pickle.load(open(sdir + 'restart/ts_' + re_fstr + f'.pkl', 'rb'))
            ts = list(ts)
            print('loaded', nt, len(nrg_fe_ts))

        max_bond_V = max_bond_V[:nt + 1]
        max_bond_fe = max_bond_fe[:nt + 1]
        max_bond_fi = max_bond_fi[:nt + 1]
        nrg_fe_ts = nrg_fe_ts[:nt + 1]
        nrg_fi_ts = nrg_fi_ts[:nt + 1]
        nrg_E_ts = nrg_E_ts[:nt + 1]
        # err_e_ts = err_e_ts[:nt+1]
        # err_i_ts = err_i_ts[:nt+1]
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
        print('init_V_gtn', init_V_gtn.data)

        restarted = True
        print('loaded restart', nt)

    except(ValueError, FileNotFoundError):
        nt = 0
        ts = [0]
        restarted = False

        # init_fe_data = init_fe_gtn.get_data(ax_order=[ax_x, ax_y, ax_vxe, ax_vye])
        # init_fi_data = init_fi_gtn.get_data(ax_order=[ax_x, ax_y, ax_vxi, ax_vyi])
        # x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing='ij')
        # vxe_mesh, vye_mesh = np.meshgrid(vxe_vals, vye_vals, indexing='ij')
        # vxi_mesh, vyi_mesh = np.meshgrid(vxi_vals, vyi_vals, indexing='ij')
        # v2e_vals = vxe_mesh**2 + vye_mesh**2
        # v2i_vals = vxi_mesh**2 + vyi_mesh**2
        # init_Ex, init_Ey = E_field(x_mesh, y_mesh)
        #
        # nrg_fe = np.sum(init_fe_data * v2e_vals) * dx * dy * dvxe * dvye * mass_e / 2 / tot_x
        # nrg_fi = np.sum(init_fi_data * v2i_vals) * dx * dy * dvxi * dvyi * mass_i / 2 / tot_x
        # nrg_E = (np.sum(init_Ex * np.conj(init_Ex))  + np.sum(init_Ey * np.conj(init_Ey))) * 1. / 2 * dx * dy / tot_x
        #
        # print('init nrgs', nrg_fe, nrg_fi, nrg_E)

        nrg_fe = init_fe_gtn.meas_expec(v2e_mpo) * mass_e / 2 / tot_x
        nrg_fi = init_fi_gtn.meas_expec(v2i_mpo) * mass_i / 2 / tot_x
        nrg_E = 1. / 2 * (1. / 2) * (A / k) ** 2  # * 2*np.pi/k  * (k/2/np.pi)
        # extra 1/2 --> 1/2/np.pi integ cos(a*x)^2 dx from 0 to 2*npi/a = 1/2

        nrg_fe_ts = [nrg_fe]
        nrg_fi_ts = [nrg_fi]
        nrg_E_ts = [nrg_E]
        print('nrg', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1])

        # norm_fe = np.sum(init_fe_data) * dx * dy * dvxe * dvye
        # norm_fi = np.sum(init_fi_data) * dx * dy * dvxi * dvyi
        # print('norms', norm_fe, norm_fi)
        norm_fe = init_fe_gtn.norm()
        norm_fi = init_fi_gtn.norm()
        print('norms', norm_fe, norm_fi)

        print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
        max_bond_V = [init_V_gtn.max_bond()]
        max_bond_fe = [init_fe_gtn.max_bond()]
        max_bond_fi = [init_fi_gtn.max_bond()]

        err_e_ts = [0.]
        err_i_ts = [0.]

    init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, compress_config=te_compress_e)
    init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, compress_config=te_compress_i)
    init_V_field = ScalarField('V', grid_X, data=init_V_gtn, compress_config=te_compress_V, is_sqrt=True)

    print('gtn norm', init_fe_field.norm(), init_fi_field.norm())

    ## boundary conditions on V:
    ## sum of all points = 1
    bc00 = np.ones(ax_x.npts * ax_y.npts) * 1. / ax_x.dx / ax_y.dx

    ### BUILD PDE
    vp_sys = VlasovPoisson(init_fe_field, init_fi_field, init_V_field,
                           coords_x=VP_test.pos_coordsys, coords_ve=VP_test.ve_coordsys,coords_vi=VP_test.vi_coordsys,
                           elc_params=elc_config, ion_params=ion_config, upwind=upwind, normalize=True,
                           potential_boundary_conditions=(bc00, None),
                           zipup=True,
                           te_order=te_order, compress_levels=comp_levels)
    vp_sys.collision = VP_test._collision_config
    print('colls', VP_test._collision_config.coll_params)

    vp_sys.compress_F = VP_test.compress_F
    vp_sys.compress_F_opts = VP_test.compress_F_opts
    print('compress E?', vp_sys.compress_F, vp_sys.compress_F_opts)

    # vp_sys.v_from_v0_e = grid_e.map_state_to_mps(ve_vals-0.0)
    # vp_sys.v_from_v0_i = grid_i.map_state_to_mps(vi_vals-0.0)

    print('before update', vp_sys.V.component.ax_deriv_configs[ax_x].bc, ax_x)

    if TN_layout == LayoutType.COMB:
        E_field = vp_sys.V.gradient()
        E_field.components[X].collect_exponents()
        E_field.components[Y].collect_exponents()
    else:
        E_field = vp_sys.V.gradient()

    print('init_V_gtn', init_V_gtn.data)
    if init_V_gtn.data is not None:
        print('init_V_gtn.data is not None')
        E_nrg = E_field.norm()**2 * 1./2 / tot_x
        print('got E norm')
        nrg_E_ts[-1] = E_nrg
        Ex_data = E_field.get_comp_data(X)
        Ey_data = E_field.get_comp_data(Y)
        print('V nrg', E_nrg, np.sum(Ex_data * np.conj(Ex_data) + Ey_data * np.conj(Ey_data)) * dx * dy * 1. / 2 / tot_x )
    # print('Ex nrg', np.sum(Ex_data * np.conj(Ex_data)) * dx * dy, np.sum(Ey_data * np.conj(Ey_data)))
    # print('Ex nrg', np.sum(Ex_data * np.conj(Ex_data)) * dx * dy * 1./2/tot_x)

    if True:  # restarted:
        vp_sys.update_V(inplace=True)

    print('after update', vp_sys.V.component.ax_deriv_configs[ax_x].bc)
    E_field = vp_sys.V.gradient()
    E_nrg = E_field.norm() ** 2 * 1. / 2 / tot_x
    Ex_data = E_field.get_comp_data(X)
    Ey_data = E_field.get_comp_data(Y)
    print('V nrg', nrg_E_ts[-1], E_nrg, 
           np.sum(Ex_data * np.conj(Ex_data) + Ey_data * np.conj(Ey_data)) * dx * dy * 1. / 2 / tot_x )


    if nt == 0:
        fig1, ax1 = plt.subplots(2, 1)
        time_ = nt * dt
        init_fe_data = init_fe_gtn.get_data(ax_select={ax_y:0, ax_vye:0})
        init_fi_data = init_fi_gtn.get_data(ax_select={ax_y:0, ax_vyi:0})
        im1 = ax1[0].imshow(np.real(init_fi_data[:,:]).T, extent=(x_min, x_max, vxi_min, vxi_max),
                            aspect='auto', origin='lower')
        im2 = ax1[1].imshow(np.real(init_fe_data[:,:]).T, extent=(x_min, x_max, vxe_min, vxe_max),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
        ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
        if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')
        plt.close(fig1)

        # V_data = vp_sys.V.get_field_data(comps=0)
        # plt.figure()
        # plt.plot(x_vals, V_data, label='V field')
        # plt.legend()

        # plt.show()

    if restarted:
        # vp_sys.update_V(inplace=True)
        if do_adapt_dt:
            E_field = vp_sys.V.gradient()
            Ex_data = E_field.get_comp_data(X)
            Ey_data = E_field.get_comp_data(Y)
            Ex_max = np.max(np.abs(Ex_data))
            Ey_max = np.max(np.abs(Ey_data))

            dt = VP_test.get_dt([Ex_max * charge/mass_e, Ey_max * charge/mass_e])
            print('restarted dt', dt)

    time1 = time.time()

    while ts[-1] < T:

        walltime = time.time()
        vp_sys = vp_sys.next_time_step(dt, compress_level=1)  # , verbose_plot=(nt > 18 or nt<=5))
        print('wall time', time.time() - walltime)

        pot_field = vp_sys.V
        max_bond_V += [pot_field.max_bond()]
        elec_field = pot_field.gradient()
        E_data = elec_field.get_comp_data(ax_x.coordinate)

        # plt.figure()
        # plt.plot(x_vals, E_data, label='E', marker='x')
        # plt.plot(x_vals, pot_field.get_field_data([0])[0], label='V', marker='x')
        # plt.show()

        fe_field = vp_sys.fe
        max_bond_fe += [fe_field.max_bond()]
        # fe_data = fe_field.get_comp_data()

        fi_field = vp_sys.fi
        max_bond_fi += [fi_field.max_bond()]
        # fi_data = fi_field.get_comp_data()

        # nrg_fe_ts += [np.sum(fe_data * (ve_vals ** 2)) * dx * dv_e * mass_e / 2 * k / 2 / np.pi]
        # nrg_fi_ts += [np.sum(fi_data * (vi_vals ** 2)) * dx * dv_i * mass_i / 2 * k / 2 / np.pi]
        # energy_E_t = np.sum(E_data * np.conj(E_data)) * dx * 1. / 2 * k / 2 / np.pi  # electric field energy density
        # nrg_E_ts += [energy_E_t]

        nrg_fe = fe_field.component.meas_expec(v2e_mpo) * mass_e / 2 / tot_x
        nrg_fi = fi_field.component.meas_expec(v2i_mpo) * mass_i / 2 / tot_x
        energy_E_t = elec_field.norm() ** 2 * 1. / 2 / tot_x  # electric field energy

        nrg_fe_ts += [nrg_fe]
        nrg_fi_ts += [nrg_fi]
        nrg_E_ts += [energy_E_t]

        ts += [ts[-1] + dt]
        nt += 1

        if do_adapt_dt:
            E_field = vp_sys.V.gradient()
            Ex_data = E_field.get_comp_data(X)
            Ey_data = E_field.get_comp_data(Y)
            Ex_max = np.max(np.abs(Ex_data))
            Ey_max = np.max(np.abs(Ey_data))

            dt = VP_test.get_dt([Ex_max * charge / mass_e, Ey_max * charge / mass_e])
            print('restarted dt', dt)

        norm_diff = vp_sys.fe.norm() - vp_sys.fi.norm()
        print('result norm', nt, ts[-1], dt, norm_diff, energy_E_t,
              pot_field.max_bond(), fe_field.max_bond(), 2 ** ((Lx + Ly + Lxe + Lye) / 2))
        # print('V spine', [pot_field.component.spine.bond_size(i,i+1) for i in range(KX - 1)])
        # print('fe spine', [fe_field.component.spine.bond_size(i,i+1) for i in range(KX + KV - 1)])
        # print('fi spine', [fe_field.component.spine.bond_size(i,i+1) for i in range(KX + KV - 1)])


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


        if nt % save_every_nt == 0:
            if True: # save_figs:
                fig1, ax1 = plt.subplots(2, 1)
                time_ = ts[-1]
                fi_data = fi_field.get_comp_data(ax_select={ax_x: npts_x // 2, ax_vxi: npts_vxi // 2})
                im1 = ax1[0].imshow(np.real(fi_data).T, extent=(y_min, y_max, vyi_min, vyi_max),
                                    aspect='auto', origin='lower')

                fe_data = fe_field.get_comp_data(ax_select={ax_x: npts_x // 2, ax_vxe: npts_vxe // 2})
                im2 = ax1[1].imshow(np.real(fe_data).T, extent=(y_min, y_max, vye_min, vye_max),
                                    aspect='auto', origin='lower')

                fig1.colorbar(im1, ax=ax1[0])
                fig1.colorbar(im2, ax=ax1[1])
                ax1[0].set_title(f'time {time_:3.2f}')
                ax1[0].xaxis.set_visible(False)
                ax1[0].set_xlabel('y'), ax1[0].set_ylabel('vy')
                ax1[1].set_xlabel('y'), ax1[1].set_ylabel('vy')
                if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')

                fig1, ax1 = plt.subplots(2, 1)
                time_ = ts[-1]
                fi_data = fi_field.get_comp_data(ax_select={ax_y: npts_y // 2, ax_vyi: npts_vyi // 2})
                im1 = ax1[0].imshow(np.real(fi_data).T, extent=(x_min, x_max, vxi_min, vxi_max),
                                    aspect='auto', origin='lower')

                fe_data = fe_field.get_comp_data(ax_select={ax_y: npts_y // 2, ax_vye: npts_vye // 2})
                im2 = ax1[1].imshow(np.real(fe_data).T, extent=(x_min, x_max, vxe_min, vxe_max),
                                    aspect='auto', origin='lower')

                fig1.colorbar(im1, ax=ax1[0])
                fig1.colorbar(im2, ax=ax1[1])
                ax1[0].set_title(f'time {time_:3.2f}')
                ax1[0].xaxis.set_visible(False)
                ax1[0].set_xlabel('x'), ax1[0].set_ylabel('vx')
                ax1[1].set_xlabel('x'), ax1[1].set_ylabel('vx')
                if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')

            # V_data = pot_field.get_comp_data()
            # plt.figure()
            # plt.imshow(np.real(V_data))
            # plt.legend()

            fig7, ax7 = plt.subplots()
            ax7.semilogy(ts, np.abs(nrg_fe_ts), label='elec')
            ax7.semilogy(ts, np.abs(nrg_fi_ts), label='ion')
            ax7.semilogy(ts, np.abs(nrg_E_ts), label='E field')
            # ax7.semilogy(ts, np.abs(total_nrg_t), label='total')

            if not save_figs:
                plt.show()
            plt.close()

    if save_data:
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
# vmax = max(np.max(fe_data), np.max(fi_data))
# vmin = min(np.min(fe_data), np.min(fi_data))
fi_data = vp_sys.fi.get_comp_data(ax_select={ax_y: npts_y // 2, ax_vyi: npts_vyi // 2})
im1 = ax1[0].imshow(np.real(fi_data).T, extent=(x_min, x_max, vxi_min, vxi_max),
                    aspect='auto', origin='lower')

fe_data = vp_sys.fe.get_comp_data(ax_select={ax_y: npts_y // 2, ax_vye: npts_vye // 2})
im2 = ax1[1].imshow(np.real(fe_data).T, extent=(x_min, x_max, vxe_min, vxe_max),
                    aspect='auto', origin='lower')
                    # vmin=vmin,vmax=vmax)

# fig1.subplots_adjust(right=0.8)
# cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
# fig1.colorbar(im1, cax=cbar_ax)
fig1.colorbar(im1, ax=ax1[0])
fig1.colorbar(im2, ax=ax1[1])
ax1[0].set_title(f'time {time_:2.1f}')
ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_T{time_:2.1f}.png')

fig2, ax2 = plt.subplots()
# ax2.plot(max_bond_fe,label=r'$\displaystyle f_e$')
# ax2.plot(max_bond_fi,label=r'$\displaystyle f_i$')
ax2.plot(np.log(max_bond_fe) / np.log(q), label=r'$f_e$')
ax2.plot(np.log(max_bond_fi) / np.log(q), label=r'$f_i$')
ax2.plot(np.log(max_bond_V) / np.log(q), label='V')
ax2.plot([0, num_tsteps], [(Lx + Ly + Lxe + Lye) // 2] * 2, ':', label='max fe')
ax2.plot([0, num_tsteps], [(Lx + Ly + Lxi + Lyi) // 2] * 2, ':', label='max fi')
ax2.plot([0, num_tsteps], [(Lx + Ly) // 2] * 2, ':', label='max V')
ax2.legend()
ax2.set_xlabel('t')
ax2.set_ylabel('log_{q^K}(max bond dimension)')
if save_figs:   fig2.savefig(fdir + 'plot_maxbond_' + fstr + '.png')

energy_E_t = np.array(nrg_E_ts)
energy_fi_t = np.array(nrg_fi_ts)
energy_fe_t = np.array(nrg_fe_ts)
total_nrg_t = energy_fe_t + energy_fi_t + energy_E_t

######################################################
### real space analysis to get damping rate, frequency
######################################################

fig7, ax7 = plt.subplots()
# ts = np.arange(num_tsteps+1)*dt
ax7.semilogy(ts, np.abs(energy_fe_t), label='elec')
ax7.semilogy(ts, np.abs(energy_fi_t), label='ion')
ax7.semilogy(ts, np.abs(energy_E_t), label='E field')
ax7.semilogy(ts, np.abs(total_nrg_t), label='total')

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
if save_figs:
    fig7.savefig(fdir + 'plot_nrg_' + fstr + '.png')
    print(fdir + 'plot_nrg_' + fstr + '.png')

plt.show()

print('done')

if True:
    print('cleaning files')
    restart_fstrs = glob.glob(sdir + 'restart/fe_' + fstr + '*.pkl')
    
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
    
