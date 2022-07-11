import os, sys, pickle, time, glob, scipy
sys.path.append('../')

from setup.configs import *
import setup.helper as test_helper

from axis import Axis
import axis_map
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_vlasovES import VlasovPoisson

"""
two-stream instability -> Buneman instability
ions, electrons have different flow velocities
see http://ammar-hakim.org/sj/je/je33/je33-buneman.html
    Chen 6.6 (fluid picture)
"""

save_figs = True
save_data = True
load_data = True
restart   = True
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_data_v2/VP1-iat-fd/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data/VP1-iat-fd/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)


#####################
# plasma parameters #
#####################

plasma_config = UnitsConfiguration()
charge = plasma_config.e  # unit of charge
eV = plasma_config.eV     # unit of kB*T
permittivity = plasma_config.eps0

n0_e = 1.0             # electron number density
n0_i = n0_e            # ion number density
T_e  = 0.0004          # electron temperature [eV]
T_i  = 1./50*T_e       # ion temperature [eV]

mass_ratio = 25.
mass_e = 1.0          # mass of electron
mass_i = mass_ratio*mass_e    # mass of ion

ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config.vth    # electron thermal speed
wp_e  = elc_config.wp     # plasma e' frequency
lamD  = elc_config.lamD   # Debye length
wp_i  = ion_config.wp     # plasma i+ frequency
vth_i = ion_config.vth    # ion thermal speed



#######################
## system parameters ##
#######################

A = 1.0e-3
k = 0.10/lamD
v0 = wp_e/k     ## avg flow of electrons with respect to ions, k set s.t. v0 >> vth_e


##########################
# useful functions #
##########################

def maxwellian(*vs, density=1.0, flow=0.0, mass=1.0, temp=1.0):
    """ vs: velocity grid for each dimension
            summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
        flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    vth2 = eV*temp/mass
    return test_helper.maxwellian(*vs, vth2=vth2, density=density, flow=flow)


def dv_maxwellian(*vs, density=1.0, flow=0.0, mass=1.0, temp=1.0, eV=1.0):
    """ vs: velocity grid for each dimension
            summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
        flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    v2 = 0.0
    for i in range(len(vs)):
        try:                            v2 += (vs[i]-flow[i])*(vs[i]-flow[i])
        except(IndexError,TypeError):   v2 += (vs[i]-flow)   *(vs[i]-flow)
    vth2  = eV*temp/mass
    grads = [ density*np.sqrt(1./2/np.pi/vth2) * np.exp(-v2/2/vth2) * (-v/vth2) for v in vs]
    return grads


def get_grid_params(idx):
    if idx == 0:
        x_bounds = -np.pi/k, np.pi/k  # 0, 1
        ve_bounds = -4 * vth_e, 6 * vth_e
        vi_bounds = -24 * vth_i, 36 * vth_i
    elif idx == 1:
        x_bounds = 0, 1
        ve_bounds = -4 * vth_e, 6 * vth_e
        vi_bounds = -24 * vth_i, 36 * vth_i
    else:
        raise IndexError(f'grid parameter for index {idx} not defined')
    return x_bounds, ve_bounds, vi_bounds


# def get_maps(mkey: str):
#     ax_maps = []
#     for mk in mkey:
#         if mk == 'F':   # forward
#             ax_map = axis_map.BinaryMap()
#         elif mk == 'B':     # back
#             ax_map = axis_map.FlipBinaryMap()
#         elif mk == 'M':
#             ax_map = axis_map.MirrorMap()
#         else:
#             raise KeyError(f'{mk} not a valid map key')
#         ax_maps += [ax_map]
#     return ax_maps


def get_fstr():
    """ get filename
    """
    ## parameters string
    fstr_params = f'k{k:1.2f}_mr{mass_i}_Te{T_e:1.1e}_TeTi{T_e/T_i:2.1f}'

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
    fstr_grid = f'_G{grid_idx}_TN{TN_layout.value}{ax_map_key}' + \
                f'_Ls{Lx:02d},{Lve:02d},{Lvi:02d}'

    if do_adapt_dt:
        fstr_t = f'_cfl{cfl}_T{int(T):03d}_te{te_order}'
    else:
        fstr_t = f'_dt{dt:1.3f}_T{int(T):03d}_te{te_order}'

    fstr1 = fstr_params + fstr_comp + fstr_compE + fstr_coll + fstr_grid + fstr_t
    return fstr1



#####################################

#########################
### define parameters ###
#########################

### system parameters ###
A = 1.0e-4
k = 6.0 * 2 * np.pi
vdrift_e = 0.01

### run-time parameters ###
q = 2
K = 1
# Lx, Lve, Lvi = 5, 8, 6
Lx, Lve, Lvi = 8, 8, 6

upwind = True
TN_layout = LayoutType.SEQUENTIAL
ax_map_key = 'FB'
grid_idx = 1

T = 1000.0     # total/final T (including load_T)
order = 1
te_order = 4
cfl = 0.9
do_adapt_dt = True

print('lambda D', lamD, 'k', k)


#################################
### define collisions ###
do_adapt_coll = True
coll_type = None  # CollisionType.LB
coll_coeff_e = 0.005  ## to scale collision frequency if using adaptive collisions
coll_coeff_i = 0.005/np.sqrt(mass_i/mass_e)
coll_rate_e = 0.01*wp_e if coll_type is not None else 0.0   ## rates if not using adaptive collisions
coll_rate_i = 0.01*wp_i if coll_type is not None else 0.0
# collision_config = CollisionConfiguration(coll_type,
#                                           coeff_e = coll_coeff_e, coeff_i = coll_coeff_i,
#                                           rate_e = coll_rate_e, rate_i=coll_rate_i, v0_e=v0)
if do_adapt_coll:
    collision_config = CollisionConfiguration(coll_type, v0_e=vdrift_e,
                                              coeff_e = coll_coeff_e, coeff_i = coll_coeff_i,
                                              rate_e=1.0, rate_i=1.0,)
else:
    collision_config = CollisionConfiguration(coll_type, v0_e=vdrift_e,
                                              coeff_e=1.0, coeff_i=1.0,
                                              rate_e=coll_rate_e, rate_i=coll_rate_i,)

print('collision config', collision_config)
if coll_type is None:
    fstr_coll = ''
else:
    if do_adapt_coll:
        fstr_coll = f'_{coll_type}{coll_coeff_e}'
    else:
        fstr_coll = f'_{coll_type}e{coll_rate_e}i{coll_rate_i}'

### compression options ###
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

for ci in range(1,compress_style+1):
    te_compress_i.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_e.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_V.set_compress_opts(ci, max_bond=None, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)

if TN_layout == LayoutType.SEQUENTIAL:
    te_compress_i.do_midpt = True
    te_compress_e.do_midpt = True
    te_compress_i.midpt = Lx
    te_compress_e.midpt = Lx

print('te compress fe', te_compress_e)
print('te compress fi', te_compress_i)
print('te compress V',  te_compress_V)

comp_levels = list(range(1,6))
if compress_style_mod == 1:     # don't compress inside the derivative
    comp_levels = [1,2,3,4,0]
elif compress_style_mod == 2:   # don't compress inside the derivative
    comp_levels = [1,2,3,0,0]

print('compress', DMAX, compress_style, compress_style_mod)

### compression of E before dot  ###
compress_E = True
DMAX_E = None
E_compress_opts = {'max_bond':DMAX_E,'cutoff':CUTOFF,'cutoff_mode':CUTOFF_MODE}


###############################
## start building PDE system ##
###############################

## initial conditions
fe = lambda x:  1 + 0*x
fi = lambda x:  1 + A*np.cos(k*x)
V_field  = lambda x: A/k**2 * np.cos(k*x)
E_field  = lambda x: A/k * np.sin(k*x)

## define grid
x_npts, ve_npts, vi_npts = q**Lx, q**Lve, q**Lvi
x_lims, ve_lims, vi_lims = get_grid_params(grid_idx)
tot_x = x_lims[1]-x_lims[0]
x_vals  = np.linspace(*x_lims,x_npts,endpoint=False)
ve_vals = np.linspace(*ve_lims,ve_npts,endpoint=False)
vi_vals = np.linspace(*vi_lims,vi_npts,endpoint=False)

## define axis objects
X = Coordinate('X', CoordinateType.X)
V = Coordinate('V', CoordinateType.X)

map_x, map_v = get_maps(ax_map_key)
print('maps', map_x, map_v)
ax_x  = Axis(Lx, q, coordinate=X, xpts=x_vals, ax_map=map_x)
ax_vi = Axis(Lvi, q, coordinate=V, xpts=vi_vals, ax_map=map_v)
ax_ve = Axis(Lve, q, coordinate=V, xpts=ve_vals, ax_map=map_v)
print('dx',ax_x.dx, 'dv e', ax_ve.dx, 'dv i', ax_vi.dx)


coords_x  = CartesianCoordinateSpace('X',ax_x)
coords_vi = CartesianCoordinateSpace('Vi',ax_vi)
coords_ve = CartesianCoordinateSpace('Ve',ax_ve)

## configure derivatives
deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
deriv_v = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)

## define grids
grid_i = Grid1D('XVi',(ax_x,ax_vi), layout_type=TN_layout)
grid_e = Grid1D('XVe',(ax_x,ax_ve), layout_type=TN_layout)
grid_V = Grid1D('X',(ax_x,), layout_type=TN_layout)


#### intiialize fields
fe_x = ax_x.map_state_to_mps(fe(x_vals))
fi_x = ax_x.map_state_to_mps(fi(x_vals))
fe_v = ax_ve.map_state_to_mps(maxwellian(ve_vals, density=n0_e, flow=vdrift_e,  mass=mass_e, temp=T_e))
fi_v = ax_vi.map_state_to_mps(maxwellian(vi_vals, density=n0_i, flow=0.0, mass=mass_i, temp=T_i))

init_fe_gtn = grid_e.make_mps_ndim({ax_x: fe_x, ax_ve: fe_v})
init_fi_gtn = grid_i.make_mps_ndim({ax_x: fi_x, ax_vi: fi_v})
init_V_gtn  = grid_V.map_state_to_mps(V_field(x_vals))

init_fe_gtn.ax_deriv_configs = {ax_x: deriv_x, ax_ve: deriv_v}
init_fi_gtn.ax_deriv_configs = {ax_x: deriv_x, ax_vi: deriv_v}
init_V_gtn.ax_deriv_configs  = {ax_x: deriv_x}


f0_e = maxwellian(ve_vals,density=n0_e,flow=vdrift_e,mass=mass_e,temp=T_e)
f0_i = maxwellian(vi_vals,density=n0_i,flow=0.,mass=mass_i,temp=T_i)

vdrift_x_e = grid_e.make_mps_ndim({ax_ve: ax_ve.map_state_to_mps(ve_vals - vdrift_e)})
vdrift_x_i = grid_i.make_mps_ndim({ax_vi: ax_vi.map_state_to_mps(vi_vals)})


### define dt ###
dx, dv_e, dv_i = ax_x.dx, ax_ve.dx, ax_vi.dx
max_v_e = np.max(np.abs(ve_vals))
max_v_i = np.max(np.abs(vi_vals))
e_params = ((dx, dv_e), (max_v_e, (A/k)*charge/mass_e))
i_params = ((dx, dv_i), (max_v_i, (A/k)*charge/mass_i))

if not do_adapt_coll or coll_type is None:
    Ce_params = ((dv_e, dv_e**2/2), ((max_v_e+np.abs(vdrift_e))*collision_config.coll_rate_e,
                                     (vth_e**2)*collision_config.coll_rate_e))
    Ci_params = ((dv_i, dv_i**2/2), (max_v_i*collision_config.coll_rate_i,
                                     (vth_i**2)*collision_config.coll_rate_i))
    dt = cfl * cfl_limit( e_params, i_params, Ce_params, Ci_params )
else:
    dt = cfl * cfl_limit( e_params, i_params )
    coll_rate_e = test_helper.LB_collision_freq_cfl_limit(dt, dv_e, vth_e, max_v_e+np.abs(vdrift_e), cfl)
    coll_rate_i = test_helper.LB_collision_freq_cfl_limit(dt, dv_i, vth_i, max_v_i, cfl)
    print('coll rate cfl', coll_rate_e, coll_rate_i)
    collision_config.base_rate_e = coll_rate_e
    collision_config.base_rate_i = coll_rate_i
    print('cfl limit scaled collision rates', collision_config.coll_rate_e, collision_config.coll_rate_i)

num_tsteps = int(T/dt)
print('dt',dt,num_tsteps)


fstr = get_fstr()
print('filename', fstr)

try:
    if not load_data:  raise IOError
    fe_t = pickle.load(open(fdir+'fe_'+fstr+'.pkl','rb'))
    fi_t = pickle.load(open(fdir+'fi_'+fstr+'.pkl','rb'))
    # Ex_t  = pickle.load(open(fdir+'Ext_' +fstr+'.pkl','rb'))
    max_bond_V  = pickle.load(open(fdir+'maxD_V_'+fstr+'.pkl','rb'))
    max_bond_fe = pickle.load(open(fdir+'maxD_fe_'+fstr+'.pkl','rb'))
    max_bond_fi = pickle.load(open(fdir+'maxD_fi_'+fstr+'.pkl','rb'))
    nrg_fe_ts = pickle.load(open(fdir+'nrg_e_'+fstr+'.pkl','rb'))
    nrg_fi_ts = pickle.load(open(fdir+'nrg_i_'+fstr+'.pkl','rb'))
    nrg_E_ts = pickle.load(open(fdir+'nrg_E_'+fstr+'.pkl','rb'))
    ts = pickle.load(open(fdir+'ts_'+fstr+'.pkl','rb'))
    init_E  = None
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
            # Ex_t  = pickle.load(open(fdir+'Ext_' +restart_fstr+'.pkl','rb'))
            max_bond_V  = pickle.load(open(fdir+'maxD_V_'+restart_fstr+'.pkl','rb'))
            max_bond_fe = pickle.load(open(fdir+'maxD_fe_'+restart_fstr+'.pkl','rb'))
            max_bond_fi = pickle.load(open(fdir+'maxD_fi_'+restart_fstr+'.pkl','rb'))
            nrg_fe_ts = pickle.load(open(fdir+'nrg_e_'+fstr+'.pkl','rb'))
            nrg_fi_ts = pickle.load(open(fdir+'nrg_i_'+fstr+'.pkl','rb'))
            nrg_E_ts = pickle.load(open(fdir+'nrg_E_'+fstr+'.pkl','rb'))
            ts = pickle.load(open(fdir+'ts_'+fstr+'.pkl','rb'))
            print('restarting from ',restart_fstr)
            nt = len(ts)
        else:
            if not restart:  raise(ValueError)
            restart_fstrs = glob.glob(sdir+'restart/fe_'+fstr+'*.pkl')
            print(sdir+'restart/')
            print('restart?', len(restart_fstrs))
            nts = []
            for f in restart_fstrs:
                nt_ind = f.find('nt')
                nt_str = f[nt_ind+2:-4]
                nts += [int(nt_str)]
            nt = max(nts)
            print('nt',nt)
            nts.remove(nt)

            try:
                raise NameError
                for nt_ in nts:   ## remove old restart files
                    if nt_ > nt:
                        print('removing nt_', nt_)
                        restart_files = glob.glob(sdir+'restart/*'+fstr+f'-nt{nt_}.pkl')
                        for f in restart_files:  os.remove(f)
            except NameError:   pass

            fe_t = pickle.load(open(sdir+'restart/fe_'+fstr+f'-nt{nt}.pkl','rb'))
            fi_t = pickle.load(open(sdir+'restart/fi_'+fstr+f'-nt{nt}.pkl','rb'))
            # Ex_t = pickle.load(open(sdir+'restart/Ext_'+fstr+f'.pkl','rb'))
            max_bond_V  = pickle.load(open(sdir+'restart/maxD_V_'+fstr+f'.pkl','rb'))
            max_bond_fe = pickle.load(open(sdir+'restart/maxD_fe_'+fstr+f'.pkl','rb'))
            max_bond_fi = pickle.load(open(sdir+'restart/maxD_fi_'+fstr+f'.pkl','rb'))
            print('loaded',nt,len(max_bond_fi))
            nrg_fe_ts = pickle.load(open(sdir+'restart/nrg_e_'+fstr+f'.pkl','rb'))
            nrg_fi_ts = pickle.load(open(sdir+'restart/nrg_i_'+fstr+f'.pkl','rb'))
            nrg_E_ts = pickle.load(open(sdir+'restart/nrg_E_'+fstr+f'.pkl','rb'))
            ts = pickle.load(open(sdir+'restart/ts_'+fstr+f'.pkl','rb'))
            print('loaded',nt,len(nrg_fe_ts))

        max_bond_V  = max_bond_V [:nt+1]
        max_bond_fe = max_bond_fe[:nt+1]
        max_bond_fi = max_bond_fi[:nt+1]
        nrg_fe_ts = nrg_fe_ts[:nt+1]
        nrg_fi_ts = nrg_fi_ts[:nt+1]
        nrg_E_ts = nrg_E_ts[:nt+1]
        ts = ts[:nt+1]
        total_E0 = nrg_E_ts[0] + nrg_fi_ts[0] + nrg_fe_ts[0]

        init_fe_gtn.data = fe_t
        init_fi_gtn.data = fi_t
        init_V_gtn.data = None

        print('init fe norm', init_fe_gtn.norm(), init_fe_gtn.data.exponent)
        print('init fi norm', init_fi_gtn.norm(), init_fi_gtn.data.exponent)

        restarted = True
        print('loaded restart',nt)


    except(ValueError,FileNotFoundError):
        nt = 0
        ts = [0]
        restarted = False

        init_fe_data = init_fe_gtn.get_data()
        init_fi_data = init_fi_gtn.get_data()
        init_E = E_field(x_vals)

        nrg_fe_ts = [np.sum(init_fe_data*(ve_vals**2))*dx*dv_e*mass_e/2 / tot_x]
        nrg_fi_ts = [np.sum(init_fi_data*(vi_vals**2))*dx*dv_i*mass_i/2 / tot_x]
        nrg_E_ts = [np.sum(init_E*np.conj(init_E))*dx*1./2 / tot_x]   # electric field energy

        print('init nrgs', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1])

        norm_fe = np.sum(init_fe_data) * dx * dv_e
        norm_fi = np.sum(init_fi_data) * dx * dv_i
        print('norm', norm_fe, norm_fi)

        print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
        max_bond_V = [init_V_gtn.max_bond()]
        max_bond_fe = [init_fe_gtn.max_bond()]
        max_bond_fi = [init_fi_gtn.max_bond()]

    init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, compress_config=te_compress_e)
    init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, compress_config=te_compress_i)
    init_V_field = ScalarField('V', grid_V, data=init_V_gtn, compress_config=te_compress_V)

    print('norm', init_fe_field.norm(), init_fi_field.norm())

    ## boundary conditions on V:
    ## V(x=edge) = 0
    bc00 = np.array([0.]*(x_npts-1)+[1.]).reshape(1,x_npts) # * 1./ax_x.dx**2

    ### BUILD PDE
    vp_sys = VlasovPoisson(init_fe_field, init_fi_field, init_V_field, coords_x=coords_x, coords_ve=coords_ve,
                           coords_vi=coords_vi, elc_params=elc_config, ion_params=ion_config, upwind=upwind,
                           potential_boundary_conditions=(bc00, None), te_order=te_order, compress_levels=comp_levels)
    vp_sys.collision = collision_config
    print('colls', collision_config.coll_params)

    vp_sys.compress_F = compress_E
    vp_sys.compress_F_opts = E_compress_opts
    print('compress E?', compress_E, E_compress_opts)

    vp_sys.v_from_v0_e = vdrift_x_e
    vp_sys.v_from_v0_i = vdrift_x_i

    print('before update', vp_sys.V.component.ax_deriv_configs[ax_x].bc, ax_x)
    E_field = vp_sys.V.gradient()
    E_data = E_field.get_comp_data(X)
    if E_data is not None:
        print('V nrg', np.sum(E_data * np.conj(E_data)) * dx * 1. / 2 / tot_x)

    if restarted:
        vp_sys.update_V(inplace=True)

    print('after update', vp_sys.V.component.ax_deriv_configs[ax_x].bc)
    E_field = vp_sys.V.gradient()
    E_data = E_field.get_comp_data(X)
    print('V nrg', np.sum(E_data * np.conj(E_data)) * dx * 1. / 2 / tot_x)


    if nt == 0:
        fig1,ax1 = plt.subplots(2,1)
        time_ = nt*dt
        im1 = ax1[0].imshow(init_fi_data.T, extent=(x_vals[0], x_vals[-1], vi_vals[0], vi_vals[-1]),
                            aspect='auto', origin='lower')
        im2 = ax1[1].imshow(init_fe_data.T, extent=(x_vals[0], x_vals[-1], ve_vals[0], ve_vals[-1]),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
        ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
        if save_figs:   fig1.savefig(fdir+f'plot_f_'+fstr+f'_nt{nt}.png')
        plt.show()
        plt.close(fig1)


    if restarted:
        if do_adapt_dt:
            E_field = vp_sys.V.gradient()
            E_data = E_field.get_comp_data(X)

            e_params = ((dx, dv_e), (max_v_e, np.max(E_data)*charge/mass_e))
            i_params = ((dx, dv_i), (max_v_i, np.max(E_data)*charge/mass_i))
            if not do_adapt_coll or coll_type is None:
                Ce_params = ((dv_e, dv_e**2/2), ((max_v_e+np.abs(vdrift_e))*collision_config.coll_rate_e,
                                                 (vth_e**2)*collision_config.coll_rate_e))
                Ci_params = ((dv_i, dv_i**2/2), (max_v_i*collision_config.coll_rate_i,
                                                 (vth_i**2)*collision_config.coll_rate_i))
                dt = cfl * test_helper.cfl_limit( e_params, i_params, Ce_params, Ci_params )
            else:
                dt = cfl * test_helper.cfl_limit(e_params, i_params)  # , Ce_params, Ci_params )
                coll_rate_e = test_helper.LB_collision_freq_cfl_limit(dt, dv_e, vth_e, max_v_e + np.abs(vdrift_e), cfl)
                coll_rate_i = test_helper.LB_collision_freq_cfl_limit(dt, dv_i, vth_i, max_v_i, cfl)
                print('max collision rates', coll_rate_e, coll_rate_i)
                collision_config.base_rate_e = coll_rate_e
                collision_config.base_rate_i = coll_rate_i

            print('restarted dt', dt)
            print('restarted collision rates', collision_config.coll_rate_e, collision_config.coll_rate_i)


    time1 = time.time()

    while ts[-1] < T:

        walltime = time.time()
        vp_sys = vp_sys.next_time_step(dt, compress_level=1)  # , verbose_plot=(nt > 18 or nt<=5))
        print('wall time', time.time() - walltime)

        pot_field = vp_sys.V
        max_bond_V += [pot_field.max_bond()]
        elec_field = pot_field.gradient()
        E_data = elec_field.get_comp_data(X)

        # plt.figure()
        # plt.plot(x_vals, E_data, label='E', marker='x')
        # plt.plot(x_vals, pot_field.get_field_data([0])[0], label='V', marker='x')
        # plt.show()

        fe_field = vp_sys.fe
        max_bond_fe += [fe_field.max_bond()]
        fe_data = fe_field.get_comp_data()

        fi_field = vp_sys.fi
        max_bond_fi += [fi_field.max_bond()]
        fi_data = fi_field.get_comp_data()

        nrg_fe_ts += [np.sum(fe_data * (ve_vals ** 2)) * dx * dv_e * mass_e / 2 / tot_x]
        nrg_fi_ts += [np.sum(fi_data * (vi_vals ** 2)) * dx * dv_i * mass_i / 2 / tot_x]
        energy_E_t = np.sum(E_data * np.conj(E_data)) * dx * 1. / 2 / tot_x
        nrg_E_ts += [energy_E_t]

        ts += [ts[-1] + dt]
        nt += 1

        if do_adapt_dt:
            e_params = ((dx, dv_e), (max_v_e, np.max(E_data)*charge/mass_e))  ## can use sqrt(sum(E**2))
            i_params = ((dx, dv_i), (max_v_i, np.max(E_data)*charge/mass_i))

            if not do_adapt_coll or coll_type is None:
                Ce_params = ((dv_e, dv_e ** 2 / 2), ((max_v_e + np.abs(vdrift_e)) * collision_config.coll_rate_e,
                                                     (vth_e ** 2) * collision_config.coll_rate_e))
                Ci_params = ((dv_i, dv_i ** 2 / 2), (max_v_i * collision_config.coll_rate_i,
                                                     (vth_i ** 2) * collision_config.coll_rate_i))
                dt = cfl * cfl_limit( e_params, i_params, Ce_params, Ci_params )
            else:
                dt = cfl * cfl_limit( e_params, i_params )
                coll_rate_e = test_helper.LB_collision_freq_cfl_limit(dt, dv_e, vth_e, (max_v_e+np.abs(vdrift_e)), cfl)
                coll_rate_i = test_helper.LB_collision_freq_cfl_limit(dt, dv_i, vth_i, max_v_i, cfl)
                # print('cfl limit collision rates', coll_rate_e, coll_rate_i)
                collision_config.base_rate_e = coll_rate_e
                collision_config.base_rate_i = coll_rate_i
                # print('cfl limit scaled collision rates', collision_config.coll_rate_e, collision_config.coll_rate_i)

        norm_e = vp_sys.fe.norm()
        norm_i = vp_sys.fi.norm()
        print('result', nt, ts[-1], dt, norm_e - norm_i, energy_E_t,
              pot_field.max_bond(), fe_field.max_bond(), 2 ** ((Lx + Lve) / 2))

        if nt%500 == 0:
            fig1,ax1 = plt.subplots(2,1)
            time_ = ts[-1]
            im1 = ax1[0].imshow(fi_data.T, extent=(x_vals[0], x_vals[-1], vi_vals[0], vi_vals[-1]),
                                aspect='auto', origin='lower')
            im2 = ax1[1].imshow(fe_data.T, extent=(x_vals[0], x_vals[-1], ve_vals[0], ve_vals[-1]),
                                aspect='auto', origin='lower')

            fig1.colorbar(im1, ax=ax1[0])
            fig1.colorbar(im2, ax=ax1[1])
            ax1[0].set_title(f'time {time_:3.2f}')
            ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
            ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
            if save_figs:   fig1.savefig(fdir+f'plot_f_'+fstr+f'_t{nt}.png')
            else:           plt.show()
            plt.close()

            V_data = pot_field.get_comp_data()
            plt.figure()
            plt.plot(x_vals, E_data, label='E field')
            plt.plot(x_vals, V_data, label='V field')
            plt.legend()

            if not save_figs:
                plt.show()
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
                # pickle.dump(Ex_t ,open(sdir+'restart/'+'Ext_' +fstr+'.pkl','wb'))
                pickle.dump(max_bond_V ,open(sdir+'restart/'+'maxD_V_'+fstr+'.pkl','wb'))
                pickle.dump(max_bond_fe,open(sdir+'restart/'+'maxD_fe_'+fstr+'.pkl','wb'))
                pickle.dump(max_bond_fi,open(sdir+'restart/'+'maxD_fi_'+fstr+'.pkl','wb'))
                pickle.dump(nrg_fe_ts,open(sdir+'restart/nrg_e_'+fstr+'.pkl','wb'))
                pickle.dump(nrg_fi_ts,open(sdir+'restart/nrg_i_'+fstr+'.pkl','wb'))
                pickle.dump(nrg_E_ts,open(sdir+'restart/nrg_E_'+fstr+'.pkl','wb'))
                pickle.dump(ts,open(sdir+'restart/ts_'+fstr+'.pkl','wb'))


    if save_data:
        pickle.dump(vp_sys.fe.component.data,open(fdir+'fe_'+fstr+'.pkl','wb'))
        pickle.dump(vp_sys.fi.component.data,open(fdir+'fi_'+fstr+'.pkl','wb'))
        # pickle.dump(Ex_t ,open(fdir+'Ext_' +fstr+'.pkl','wb'))
        pickle.dump(max_bond_V ,open(fdir+'maxD_V_'+fstr+'.pkl','wb'))
        pickle.dump(max_bond_fe,open(fdir+'maxD_fe_'+fstr+'.pkl','wb'))
        pickle.dump(max_bond_fi,open(fdir+'maxD_fi_'+fstr+'.pkl','wb'))
        pickle.dump(nrg_fe_ts,open(fdir+'nrg_e_'+fstr+'.pkl','wb'))
        pickle.dump(nrg_fi_ts,open(fdir+'nrg_i_'+fstr+'.pkl','wb'))
        pickle.dump(nrg_E_ts,open(fdir+'nrg_E_'+fstr+'.pkl','wb'))
        pickle.dump(ts,open(fdir+'ts_'+fstr+'.pkl','wb'))

    print('wall time', time.time()-time1)

    # ts = np.array(ts)
    # Ex_t = np.array(Ex_t)
    # pxv_e_t = np.array(pxv_e_t)
    # pxv_i_t = np.array(pxv_i_t)


fig1,ax1 = plt.subplots(2,1)
time_ = ts[-1]
vmax = max(np.max(fe_data),np.max(fi_data))
vmin = min(np.min(fe_data),np.min(fi_data))
im1 = ax1[0].imshow(fi_data.T, extent=(x_vals[0], x_vals[-1], vi_vals[0], vi_vals[-1]),
                    aspect='auto', origin='lower')
                    # vmin=vmin,vmax=vmax,aspect='auto')
im2 = ax1[1].imshow(fe_data.T, extent=(x_vals[0], x_vals[-1], ve_vals[0], ve_vals[-1]),
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
ax2.plot(np.log(max_bond_fe)/np.log(q**K),label='f_e')
ax2.plot(np.log(max_bond_fi)/np.log(q**K),label='f_i')
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



##############

fig7,ax7 = plt.subplots()
# ts = np.arange(num_tsteps+1)*dt
ax7.semilogy(ts,np.abs(energy_fe_t),label='elec')
ax7.semilogy(ts,np.abs(energy_fi_t),label='ion')
ax7.semilogy(ts,np.abs(energy_E_t ),label='E field')
ax7.semilogy(ts,np.abs(total_nrg_t),label='total')
# decay = np.exp(2*theory_gamma*np.array(ts)) * energy_E_t[0]   # x 2 bc it's ~|Ex|^2
# ax7.semilogy(ts,decay,'k:',linewidth=2,label='theor. damping')

# try:
#     ts_, energy_E_t = pickle.load(open('../gkeyll_tests/vm-buneman/m25_k010_256/Ex_t_m25_k010.pkl','rb'))
#     ax7.semilogy(ts_, energy_E_t/2., 'k-',linewidth=1.0, label='ref (gkeyll)')
# except(NameError, IOError):
#     pass

ax7.legend()
ax7.set_xlabel('t')
ax7.set_ylabel('energy')
if save_figs:
    fig7.savefig(fdir+'plot_nrg_'+fstr+'.png')
    print(fdir+'plot_nrg_'+fstr+'.png')

plt.show()
print('done')

