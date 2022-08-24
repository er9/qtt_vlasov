import os, sys, pickle, time, glob

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
two beams propagating into each other
http://ammar-hakim.org/sj/je/je29/je29-es-shock.html
"""

save_figs = True
save_data = True
load_data = True
restart   = True
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

if save_data or save_figs:
    sdir = '/pool001/erikaye/tns_data_v3/VP1-shock-final/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    
    fdir = 'data/VP1-shock-final/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

#####################
# useful functions  #
#####################

def maxwellian(*vs, density=1.0, flow=0.0, mass=1.0, temp=1.0):
    vth2 = eV*temp/mass
    return test_helper.maxwellian(*vs, vth2=vth2, density=density, flow=flow)


def get_grid_params(idx):
    if idx == 0:
        x_bounds = -250 * lamD, 250 * lamD
        ve_bounds = -6 * vth_e, 6 * vth_e
        vi_bounds = -10.0 * vth_i - vdrift_i, 10.0 * vth_i + vdrift_i
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


def get_fstr():
    """ get filename
    """
    ## parameters string
    fstr_params = f'mach{mach_num}_TeTi{Te_Ti}_mr{mass_ratio}'

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
    fstr_grid = f'_g{grid_idx}_TN{TN_layout.value}{ax_map_key}' + \
                f'_Ls{Lx:02d},{Lve:02d},{Lvi:02d}_o{order}'

    if do_adapt_dt:
        fstr_t = f'_cfl{cfl}_T{int(T):03d}_te{te_order}'
    else:
        fstr_t = f'_dt{dt:1.3f}_T{int(T):03d}_te{te_order}'

    fstr1 = fstr_params + fstr_comp + fstr_compE + fstr_coll + fstr_grid + fstr_t
    return fstr1


#######################
## system parameters ##
#######################

Te_Ti = 9.0
mass_ratio = 1836.2
mach_num = 1.5


#####################
# plasma parameters #
#####################

plasma_config = UnitsConfiguration()
charge = plasma_config.e  # unit of charge
eV = plasma_config.eV     # unit of kB*T

T_e  = 1.0             # electron temperature [eV]
T_i  = T_e/Te_Ti       # ion temperature [eV]
mass_e = 1.0                  # mass of electron
mass_i = mass_ratio*mass_e    # mass of ion
n0_e = 1
n0_i = n0_e

ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config.vth    # electron thermal speed
wp_e  = elc_config.wp     # plasma e' frequency
lamD  = elc_config.lamD   # Debye length
vth_i = ion_config.vth    # ion thermal speed
wp_i  = ion_config.wp     # plasma i+ frequency

sound_speed = np.sqrt(eV*T_e/mass_i)

####################

### define collisions
do_adapt_coll = True
coll_type = None
coll_coeff_e = 0.01     ## scale collision freq if using adaptive collisions
coll_coeff_i = 0.01/np.sqrt(mass_i/mass_e)
coll_rate_e = 1.0e-3*wp_e if coll_type is not None else 0.0
coll_rate_i = 1.0e-3*wp_i if coll_type is not None else 0.0
collision_config = CollisionConfiguration(coll_type,
                                          coeff_e = coll_coeff_e, coeff_i = coll_coeff_i,
                                          rate_e = coll_rate_e, rate_i=coll_rate_i)

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
DMAX = 32
compress_style = 1
compress_style_mod = 0

te_compress_i = CompressionConfiguration()
te_compress_e = CompressionConfiguration()
te_compress_V = CompressionConfiguration()

for ci in range(1,compress_style+1):
    te_compress_i.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_e.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_V.set_compress_opts(ci, max_bond=None, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)

print('te compress fe', te_compress_e)
print('te compress fi', te_compress_i)
print('te compress V',  te_compress_V)


comp_levels = list(range(1,6))  # compress at all levels
if compress_style_mod == 1:     # don't compress inside the derivative
    comp_levels = [1,2,3,4,0]
elif compress_style_mod == 2:   # don't compress derivative either. slow
    comp_levels = [1,2,3,0,0]
elif compress_style_mod == 3:
    comp_levels = [1,2,0,4,0]
elif compress_style_mod == 4:   # super slow
    comp_levels = [1,0,0,4,0]

print('compress', DMAX, compress_style, compress_style_mod)


#### compression of E before dot
compress_E = True
DMAX_E = None
E_compress_opts = {'max_bond':DMAX_E,'cutoff':CUTOFF,'cutoff_mode':CUTOFF_MODE}


#############################
## start building PDE system
#############################

## run-time parameters ##
T = 1000.0     # total/final time T
cfl = 0.9
order = 1
te_order = 4 # 2
do_adapt_dt = True

### define simulation grid ###
q = 2
K = 1
Lx, Lve, Lvi = 9, 9, 9
TN_layout = LayoutType.SEQUENTIAL
grid_idx = 0
ax_map_key = 'FB'  # 'FB'

vdrift_e = mach_num*sound_speed
vdrift_i = vdrift_e

(x_min, x_max), (ve_min, ve_max), (vi_min,vi_max) = get_grid_params(grid_idx)
tot_x = x_max - x_min
x_npts, ve_npts, vi_npts = q**Lx, q**Lve, q**Lvi
x_vals  = np.linspace(x_min, x_max, x_npts, endpoint=True)
ve_vals = np.linspace(ve_min, ve_max, ve_npts, endpoint=False)
vi_vals = np.linspace(vi_min, vi_max, vi_npts, endpoint=False)

X = Coordinate('X', CoordinateType.X)
V = Coordinate('V', CoordinateType.X)

ax_map_x, ax_map_v = get_maps(ax_map_key)
ax_x  = Axis(Lx,  q, coordinate=X, xpts=x_vals,  ax_map=ax_map_x)
ax_vi = Axis(Lvi, q, coordinate=V, xpts=vi_vals, ax_map=ax_map_v)
ax_ve = Axis(Lve, q, coordinate=V, xpts=ve_vals, ax_map=ax_map_v)
print('dx',ax_x.dx, 'dv e', ax_ve.dx, 'dv i', ax_vi.dx)
print('x', ax_map_x, 'v', ax_map_v)

coords_x  = CartesianCoordinateSpace('X',ax_x)
coords_vi = CartesianCoordinateSpace('Vi',ax_vi)
coords_ve = CartesianCoordinateSpace('Ve',ax_ve)

## configure derivatives
# deriv_x = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)
deriv_v = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)
deriv_x = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)
# deriv_v = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)

## define grids
grid_i = Grid1D('XVi',(ax_x,ax_vi), layout_type=TN_layout)
grid_e = Grid1D('XVe',(ax_x,ax_ve), layout_type=TN_layout)
grid_V = Grid1D('X',(ax_x,), layout_type=TN_layout)


## define initial grid1DTNs
f = lambda x:  np.heaviside(x,0.0) # * dx

init_fe = np.outer(f(x_vals), maxwellian(ve_vals, density=n0_e, flow=-1*vdrift_e, mass=mass_e, temp=T_e)) + \
          np.outer((1-f(x_vals)), maxwellian(ve_vals, density=n0_e, flow=vdrift_e, mass=mass_e, temp=T_e))
init_fi = np.outer(f(x_vals), maxwellian(vi_vals, density=n0_i, flow=-1*vdrift_i, mass=mass_i, temp=T_i)) + \
          np.outer((1-f(x_vals)), maxwellian(vi_vals, density=n0_i, flow=vdrift_i, mass=mass_i, temp=T_i))

init_fe_gtn = grid_e.map_state_to_mps(init_fe, ax_deriv_configs={ax_x: deriv_x, ax_ve: deriv_v})
init_fi_gtn = grid_i.map_state_to_mps(init_fi, ax_deriv_configs={ax_x: deriv_x, ax_vi: deriv_v})
init_V_gtn  = grid_V.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_x})

dist0_e = init_fe_gtn
dist0_i = init_fi_gtn
vdrift_x_e = np.outer( f(x_vals), ve_vals+vdrift_e ) + np.outer( 1-f(x_vals), ve_vals-vdrift_e )
vdrift_x_i = np.outer( f(x_vals), ve_vals+vdrift_i ) + np.outer( 1-f(x_vals), vi_vals-vdrift_i )

# data_x, data_ve = np.meshgrid(x_vals,ve_vals,indexing='ij')
# data_x, data_vi = np.meshgrid(x_vals,vi_vals,indexing='ij')
# vdrift_x_e = data_ve - ( f(data_x) * -vdrift_e + (1-f(data_x)) * vdrift_e )
# vdrift_x_i = data_vi - ( f(data_x) * -vdrift_i + (1-f(data_x)) * vdrift_i )


## define time step
dx, dv_e, dv_i = ax_x.dx, ax_ve.dx, ax_vi.dx
max_v_e = np.max(np.abs(ve_vals))
max_v_i = np.max(np.abs(vi_vals))
e_params = ((dx, dv_e), (max_v_e, 0.0*eV/mass_e))   # E field is initially zero
i_params = ((dx, dv_i), (max_v_i, 0.0*eV/mass_i))

if not do_adapt_coll or coll_type is None:
    Ce_params = ((dv_e, dv_e**2/2), (max_v_e*coll_rate_e, (vth_e**2)*coll_rate_e))
    Ci_params = ((dv_i, dv_i**2/2), (max_v_i*coll_rate_i, (vth_i**2)*coll_rate_i))
    dt = cfl * test_helper.cfl_limit( e_params, i_params, Ce_params, Ci_params )
else:
    dt = cfl * test_helper.cfl_limit( e_params, i_params)
    coll_rate_e = test_helper.LB_collision_freq_cfl_limit(dt, dv_e, vth_e, max_v_e, cfl)
    coll_rate_i = test_helper.LB_collision_freq_cfl_limit(dt, dv_i, vth_i, max_v_i, cfl)
    print('cfl limit collision rates', coll_rate_e, coll_rate_i)
    coll_rate_e *= coll_coeff_e
    coll_rate_i *= coll_coeff_i
    print('cfl limit scaled collision rates', collision_config.coll_rate_e, collision_config.coll_rate_i)

num_tsteps = int(T/dt)
print('dt',dt,num_tsteps)

print('norm e',np.sum(init_fe)*dv_e*dx)
print('norm i',np.sum(init_fi)*dv_i*dx)

#################3

fstr = get_fstr()
print('filename', fstr)
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
            print('nt',nt)
            nts.remove(nt)

            try: 
                raise NameError
                for nt_ in nts:   ## remove old restart filesa
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
            ts  = pickle.load(open(sdir+'restart/ts_'+re_fstr+f'.pkl','rb'))
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
        restarted = True
        print('loaded restart',nt)

    except(ValueError,FileNotFoundError):
        nt = 0
        ts = [0]
        restarted = False

        init_fe_data = init_fe_gtn.get_data()
        init_fi_data = init_fi_gtn.get_data()

        nrg_fe_ts = [np.sum(init_fe_data * (ve_vals ** 2)) * dx * dv_e * mass_e / 2 / tot_x]
        nrg_fi_ts = [np.sum(init_fi_data * (vi_vals ** 2)) * dx * dv_i * mass_i / 2 / tot_x]
        nrg_E_ts  = [0.0] # [np.sum(init_E * np.conj(init_E)) * dx * 1. / 2]
        print('init nrgs', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1])

        norm_fe = np.sum(init_fe_data) * dx * dv_e
        norm_fi = np.sum(init_fi_data) * dx * dv_i
        print('norms', norm_fe, norm_fi)
        print(np.sum(f(x_vals)) * dx)
        fe_v = maxwellian(ve_vals, density=n0_e, flow=0.0, mass=mass_e, temp=T_e)
        fi_v = maxwellian(vi_vals, density=n0_i, flow=0.0, mass=mass_i, temp=T_i)
        print(np.sum(fe_v * dv_e), np.sum(fe_v * ve_vals**2 * dv_e) * mass_e / 2, eV*T_e)
        print(np.sum(fi_v * dv_i), np.sum(fi_v * vi_vals**2 * dv_i) * mass_i / 2, eV*T_i)

        print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
        max_bond_V  = [init_V_gtn.max_bond()]
        max_bond_fe = [init_fe_gtn.max_bond()]
        max_bond_fi = [init_fi_gtn.max_bond()]


    init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, compress_config=te_compress_e)
    init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, compress_config=te_compress_i)
    init_V_field  = ScalarField('V',  grid_V, data=init_V_gtn, compress_config=te_compress_V)

    print('norm',init_fe_field.norm(), init_fi_field.norm())

    ## boundary conditions on V:
    ## V(x=right edge) = 0
    bc00 = np.array([0.] * (x_npts - 1) + [1.]).reshape(1, x_npts) * 1. / ax_x.dx ** 2

    ### BUILD PDE
    vp_sys = VlasovPoisson(init_fe_field, init_fi_field, init_V_field,
                           coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                           elc_params=elc_config, ion_params=ion_config, upwind=True,
                           potential_boundary_conditions=(bc00, None),
                           te_order=te_order, compress_levels=comp_levels)
    vp_sys.collision = collision_config
    print('colls', collision_config.coll_params)

    vp_sys.compress_F = compress_E
    vp_sys.compress_F_opts = E_compress_opts
    print('compress E?', compress_E, E_compress_opts)

    print('before update', vp_sys.V.component.ax_deriv_configs[ax_x].bc, ax_x)
    print('V norm', vp_sys.V.norm())
    if restarted:
        vp_sys.update_V(inplace=True)
    print('after update', vp_sys.V.component.ax_deriv_configs[ax_x].bc)
    print('V norm', vp_sys.V.norm())

    vp_sys.v_from_v0_e = grid_e.map_state_to_mps(vdrift_x_e)
    vp_sys.v_from_v0_i = grid_i.map_state_to_mps(vdrift_x_i)


    if nt == 0:

        fig1,ax1 = plt.subplots(2,1)
        time_ = nt*dt
        im1 = ax1[0].imshow(init_fi_data.T,extent=(x_vals[0], x_vals[-1], vi_vals[0], vi_vals[-1]),
                            aspect='auto', origin='lower')
        im2 = ax1[1].imshow(init_fe_data.T,extent=(x_vals[0], x_vals[-1], ve_vals[0], ve_vals[-1]),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
        ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
        if save_figs:   fig1.savefig(fdir+f'plot_f_'+fstr+f'_nt{nt}.png')
        plt.show()
        plt.close(fig1)

        # V_data = vp_sys.V.get_field_data(comps=0)
        # plt.figure()
        # plt.plot(x_vals, V_data, label='V field')
        # plt.title('V')
        # plt.legend()

        # plt.show()

    if restarted:
        # vp_sys.update_V(inplace=True)
        if do_adapt_dt:
            V_field = vp_sys.V
            E_field = V_field.gradient()
            E_data = E_field.get_comp_data(X)
        
            e_params = ((dx, dv_e), (max_v_e, np.max(np.abs(E_data))*eV/mass_e))  ## can use sqrt(sum(E**2))
            i_params = ((dx, dv_i), (max_v_i, np.max(np.abs(E_data))*eV/mass_i))

            if not do_adapt_coll or coll_type is None:
                Ce_params = ((dv_e, dv_e**2/2), (max_v_e*coll_rate_e, (vth_e**2)*coll_rate_e))
                Ci_params = ((dv_i, dv_i**2/2), (max_v_i*coll_rate_i, (vth_i**2)*coll_rate_i))
                dt = cfl * cfl_limit( e_params, i_params, Ce_params, Ci_params )
            else:
                dt = cfl * cfl_limit( e_params, i_params)
                coll_rate_e = LB_collision_freq_cfl_limit(dt, dv_e, vth_e, max_v_e, cfl)
                coll_rate_i = LB_collision_freq_cfl_limit(dt, dv_i, vth_i, max_v_i, cfl)
                print('cfl limit collision rates', coll_rate_e, coll_rate_i)
                collision_config.base_rate_e = coll_rate_e
                collision_config.base_rate_i = coll_rate_i

            print('restarted dt', dt)


    time1 = time.time()
    
    while ts[-1] < T:

        walltime = time.time()
        vp_sys = vp_sys.next_time_step(dt, compress_level=1,) # verbose_plot=(nt > 100 or nt<=5))
        print('wall time', time.time() - walltime)

        pot_field = vp_sys.V
        max_bond_V += [pot_field.max_bond()]
        elec_field = pot_field.gradient()
        E_data = elec_field.get_comp_data(ax_x.coordinate)

        # plt.figure()
        # plt.plot(x_vals, E_data, label='E', marker='x')
        # plt.plot(x_vals, V_field.get_field_data([0])[0], label='V', marker='x')
        # plt.show()

        fe_field = vp_sys.fe
        max_bond_fe += [fe_field.max_bond()]
        fe_data = fe_field.get_comp_data()

        fi_field = vp_sys.fi
        max_bond_fi += [fi_field.max_bond()]
        fi_data = fi_field.get_comp_data()

        nrg_fe_ts += [np.sum(fe_data * (ve_vals ** 2)) * dx * dv_e * mass_e / 2 / tot_x]
        nrg_fi_ts += [np.sum(fi_data * (vi_vals ** 2)) * dx * dv_i * mass_i / 2 / tot_x]

        energy_E_t = np.sum(E_data * np.conj(E_data)) * dx * 1. / 2 / tot_x  # electric field energy density
        nrg_E_ts += [energy_E_t]

        ts += [ts[-1] + dt]
        nt += 1
    
        if do_adapt_dt:
            e_params = ((dx, dv_e), (max_v_e, np.max(np.abs(E_data))*eV/mass_e))  ## can use sqrt(sum(E**2)) too
            i_params = ((dx, dv_i), (max_v_i, np.max(np.abs(E_data))*eV/mass_i))

            if not do_adapt_coll or coll_type is None:
                Ce_params = ((dv_e, dv_e**2/2), (max_v_e*coll_rate_e, (vth_e**2)*coll_rate_e))
                Ci_params = ((dv_i, dv_i**2/2), (max_v_i*coll_rate_i, (vth_i**2)*coll_rate_i))
                dt = cfl * cfl_limit( e_params, i_params, Ce_params, Ci_params )
            else:
                dt = cfl * cfl_limit( e_params, i_params)
                coll_rate_e = LB_collision_freq_cfl_limit(dt, dv_e, vth_e, max_v_e, cfl)
                coll_rate_i = LB_collision_freq_cfl_limit(dt, dv_i, vth_i, max_v_i, cfl)
                collision_config.base_rate_e = coll_rate_e
                collision_config.base_rate_i = coll_rate_i

        norm = vp_sys.fe.norm() - vp_sys.fi.norm()
        print('result norm', nt, ts[-1], dt, norm, energy_E_t,
                            pot_field.max_bond(), fe_field.max_bond(), 2 ** ((Lx+Lve)/2))

        if nt%50 == 0:
            fig1,ax1 = plt.subplots(2,1)
            time_ = ts[-1]
            im1 = ax1[0].imshow(np.real(fi_data).T,extent=(x_vals[0],x_vals[-1],vi_vals[0],vi_vals[-1]),
                                aspect='auto', origin='lower')
            im2 = ax1[1].imshow(np.real(fe_data).T,extent=(x_vals[0],x_vals[-1],ve_vals[0],ve_vals[-1]),
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
            plt.plot(x_vals, E_data, label='E field')
            plt.plot(x_vals, V_data, label='V field')
            plt.legend()

            fig7, ax7 = plt.subplots()
            ax7.semilogy(ts, np.abs(nrg_fe_ts), label='elec')
            ax7.semilogy(ts, np.abs(nrg_fi_ts), label='ion')
            ax7.semilogy(ts, np.abs(nrg_E_ts), label='E field')

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
                pickle.dump(max_bond_V ,open(sdir+'restart/'+'maxD_V_'+fstr+'.pkl','wb'))
                pickle.dump(max_bond_fe,open(sdir+'restart/'+'maxD_fe_'+fstr+'.pkl','wb'))
                pickle.dump(max_bond_fi,open(sdir+'restart/'+'maxD_fi_'+fstr+'.pkl','wb'))
                pickle.dump(nrg_fe_ts,open(sdir+'restart/nrg_e_'+fstr+'.pkl','wb'))
                pickle.dump(nrg_fi_ts,open(sdir+'restart/nrg_i_'+fstr+'.pkl','wb'))
                pickle.dump(nrg_E_ts ,open(sdir+'restart/nrg_E_'+fstr+'.pkl','wb'))
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
        pickle.dump(ts ,open(fdir+'ts_'+fstr+'.pkl','wb'))

    print('wall time', time.time()-time1)

    # ts = np.array(ts)
    # Ex_t = np.array(Ex_t)
    # pxv_e_t = np.array(pxv_e_t)
    # pxv_i_t = np.array(pxv_i_t)


fig1,ax1 = plt.subplots(2,1)
time_ = ts[-1]
vmax = max(np.max(fe_data),np.max(fi_data))
vmin = min(np.min(fe_data),np.min(fi_data))
im1 = ax1[0].imshow((fi_data).T,extent=(x_vals[0],x_vals[-1],vi_vals[0],vi_vals[-1]),
                    aspect='auto', origin='lower')
                    # vmin=vmin,vmax=vmax,aspect='auto')
im2 = ax1[1].imshow((fe_data).T,extent=(x_vals[0],x_vals[-1],ve_vals[0],ve_vals[-1]),
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
plt.show()


# # print('max_bond_fe', max_bond_fe)
# fig2,ax2 = plt.subplots()
# # ax2.plot(max_bond_fe,label=r'$\displaystyle f_e$')
# # ax2.plot(max_bond_fi,label=r'$\displaystyle f_i$')
# ax2.plot(np.log(max_bond_fe)/np.log(q**K),label=r'$f_e$')
# ax2.plot(np.log(max_bond_fi)/np.log(q**K),label=r'$f_i$')
# ax2.plot(np.log(max_bond_V )/np.log(q**K),label='V')
# ax2.plot([0,num_tsteps],[(Lx+Lve)//2]*2, ':', label='max fe')
# # ax2.plot([0,num_tsteps],[(Lx+Lvi)//2]*2, ':', label='max fi')
# ax2.plot([0,num_tsteps],[Lx//2]*2, ':', label='max V')
# ax2.legend()
# ax2.set_xlabel('t')
# ax2.set_ylabel('log_{q}(max bond dimension)')
# if save_figs:   fig2.savefig(fdir+'plot_maxbond_'+fstr+'.png')

energy_E_t  = np.array(nrg_E_ts)
energy_fi_t = np.array(nrg_fi_ts)
energy_fe_t = np.array(nrg_fe_ts)
total_nrg_t = energy_fe_t + energy_fi_t + energy_E_t


##############

fig7,ax7 = plt.subplots()
ax7.semilogy(ts,np.abs(energy_fe_t),label='elec')
ax7.semilogy(ts,np.abs(energy_fi_t),label='ion')
ax7.semilogy(ts,np.abs(energy_E_t ),label='E field')
ax7.semilogy(ts,np.abs(total_nrg_t),label='total')

ax7.legend()
ax7.set_xlabel('t')
ax7.set_ylabel('energy')
if save_figs: fig7.savefig(fdir+'plot_nrg_'+fstr+'.png')


plt.show()
   
   
print('done')

