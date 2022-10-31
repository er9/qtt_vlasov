import os, sys, time, glob
sys.path.append('../')


from setup.configs import *

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_EM import Maxwell
from pde_vlasovEM import VlasovMaxwell

"""
2D TM EM modes
"""

save_figs = False
save_data = False
load_data = False
restart   = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_data/VM-TM/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    
    fdir = 'data/VM-TM/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir',fdir)

#######################
# standard parameters #
#######################

permittivity = 1.0    # permittivity of free space
permeability = 1.0    # permeability of free space
eV = 1.0              # elementary charge or Joule-eV conversion factor

matl_config = UnitsConfiguration(eps0=permittivity, mu0=permeability)


##################
# 2D TM EM modes #
##################

q = 2
L = 6
K = 2
npts = q**L
TN_layout = LayoutType.PARALLEL_GROUP
T = 5.0

## initial conditions
Lx = 1.0
Ly = 0.5

E0 = 1.
mx, my = 8, 5
a = mx * np.pi/Lx
b = my * np.pi/Ly
Ez = lambda x, y: E0*np.sin(a*x)*np.sin(b*y)

c = 1/np.sqrt(permittivity*permeability)
omega = c*np.sqrt(a**2 + b**2)  # speed of light c = 1
print('period', 2*np.pi/omega)


cfl = 0.9
order = 4
te_order = 4
do_adapt_dt = True

fstr1 = f'test{order}_TN{TN_layout.value}_Lx{Lx}_Ly{Ly}_mx{mx}_my{my}'

#### compression options
""" comp1:   compress final state
    comp2:   + compress intermediate rk4 states
    comp3:   + compress sum of derivatives
    comp4:   + compress derivatives
    comp5:   + compress during calculation of derivative

    mod0:  no modifications 
    mod1:  comp5 = 0 (don't actively compress in interior of calculating derivative)
"""
DMAX = 64
compress_style = 1
compress_style_mod = 0

te_compress_E = CompressionConfiguration()
te_compress_phi = CompressionConfiguration()

for ci in range(1,compress_style+1):
    te_compress_E.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_phi.set_compress_opts(ci, max_bond=None, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)

te_compress_psi = te_compress_phi
te_compress_B = te_compress_E

print('te compress E', te_compress_E)
print('te compress phi', te_compress_phi)


comp_levels = list(range(1,6))
if compress_style_mod == 1:   # don't compress inside the derivative
    comp_levels = [1,2,3,4,0]
elif compress_style_mod == 2:   # don't compress inside the derivative
    comp_levels = [1,2,3,0,0]

fstr_comp = f'_C{compress_style}{compress_style_mod}_mb{DMAX}'

print('compress', DMAX, compress_style, compress_style_mod)

fstr1 += fstr_comp

#############################
## start building PDE system 
#############################

## define coordinates (will label axes and vector components)
X = Coordinate('X', CoordinateType.X)
Y = Coordinate('Y', CoordinateType.Y)
Z = Coordinate('Z', CoordinateType.Z)

## define axis objects
x_vals = np.linspace(0,Lx,npts,endpoint=True) # False)
y_vals = np.linspace(0,Ly,npts,endpoint=True) # False)
## include endpoint if using reflecting bc; exclude if using periodic/antiperiodic bc

ax_x = Axis(L, q, coordinate=X, xpts=x_vals)
ax_y = Axis(L, q, coordinate=Y, xpts=y_vals)

coord_sys = CartesianCoordinateSpace('cart', ax_x, ax_y, coords=[X,Y,Z])

## configure derivatives
deriv_zg = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)
deriv_zv = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, order=order, fd_type=FDType.CENTER)
### following boundary conditions are unstable!
# deriv_zg = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)
# deriv_zv = DerivativeConfiguration(left_bc=BCType.ZEROVALUE, order=order, fd_type=FDType.CENTER)



## define grids
grid_xy = Grid1D('XY', (ax_x,ax_y), layout_type=TN_layout)

## define initial grid1DTNs
data_x, data_y = np.meshgrid(x_vals,y_vals,indexing='ij')
init_Ez  = Ez(data_x,data_y)

init_Ez_gtn = grid_xy.map_state_to_mps(init_Ez)
init_Bx_gtn = grid_xy.make_empty_gridTN()
init_By_gtn = grid_xy.make_empty_gridTN()
init_phi_gtn = grid_xy.make_empty_gridTN()
init_psi_gtn = grid_xy.make_empty_gridTN()

init_Ez_gtn.ax_deriv_configs  = {ax_x: deriv_zv, ax_y: deriv_zv}
init_Bx_gtn.ax_deriv_configs  = {ax_x: deriv_zv, ax_y: deriv_zg}
init_By_gtn.ax_deriv_configs  = {ax_x: deriv_zg, ax_y: deriv_zv}
init_phi_gtn.ax_deriv_configs  = {ax_x: deriv_zv, ax_y: deriv_zv}
init_psi_gtn.ax_deriv_configs  = {ax_x: deriv_zv, ax_y: deriv_zv}

print('init E', init_Ez_gtn.ax_deriv_configs[ax_x].bc)

init_E_field = Field('E', grid_xy, data={Z: init_Ez_gtn}, compress_config=te_compress_E)
init_B_field = Field('B', grid_xy, data={X: init_Bx_gtn, Y:init_By_gtn}, compress_config=te_compress_B)
init_phi_field = ScalarField('phi', grid_xy, data=init_phi_gtn, compress_config=te_compress_phi)
init_psi_field = ScalarField('psi', grid_xy, data=init_psi_gtn, compress_config=te_compress_psi)


## define time step
dx, dy = ax_x.dx, ax_y.dx
cfl_params = ((dx, dy), (c, c))
dt = cfl * cfl_limit( cfl_params )

num_tsteps = int(T/dt)
print('dt',dt,num_tsteps)
print('dx', dx, 'dy', dy)

if do_adapt_dt:
    fstr_t = f'cfl{cfl}_T{int(T):03d}'
else:
    fstr_t = f'dt{dt:1.3f}_T{int(T):03d}'

fstr = fstr1 + f'_L{L:02d}_{fstr_t}_te{te_order}' # '_m{mass_i}'
print(fstr)

try:
    if not load_data:  raise IOError
    E_t = optional_load(fdir + 'E_' + fstr + '.pkl')
    B_t = optional_load(fdir + 'B_' + fstr + '.pkl')
    phi_t = optional_load(fdir + 'E_' + fstr + '.pkl')
    psi_t = optional_load(fdir + 'E_' + fstr + '.pkl')
    max_bond_E = pickle.load(open(fdir + 'maxD_E_' + fstr + '.pkl', 'rb'))
    max_bond_B = pickle.load(open(fdir + 'maxD_B_' + fstr + '.pkl', 'rb'))
    max_bond_phi = pickle.load(open(fdir + 'maxD_phi_' + fstr + '.pkl', 'rb'))
    max_bond_psi = pickle.load(open(fdir + 'maxD_psi_' + fstr + '.pkl', 'rb'))
    nrg_fe_ts = pickle.load(open(fdir+'nrg_e_'+fstr+'.pkl','rb'))
    nrg_fi_ts = pickle.load(open(fdir+'nrg_i_'+fstr+'.pkl','rb'))
    nrg_E_ts  = pickle.load(open(fdir+'nrg_E_'+fstr+'.pkl','rb'))
    nrg_B_ts = pickle.load(open(fdir + 'nrg_B_' + fstr + '.pkl', 'rb'))
    ts = pickle.load(open(fdir+'ts_'+fstr+'.pkl','rb'))

    nt = num_tsteps
    print('loaded data')

    E_data = grid_xy.map_mps_to_state(E_t) if E_t is not None else None
    B_data = grid_xy.map_mps_to_state(B_t) if B_t is not None else None
    phi_data = grid_xy.map_mps_to_state(phi_t) if phi_t is not None else None
    psi_data = grid_xy.map_mps_to_state(psi_t) if psi_t is not None else None

except(IOError,EOFError):

    try:
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
        # nts.remove(nt)

        try:
            raise NameError
            for nt_ in nts:   ## remove old restart files
                if nt_ > nt:
                    print('removing restart', nt_)
                    restart_files = glob.glob(sdir+'restart/*'+re_fstr+f'-nt{nt_}.pkl')
                    for f in restart_files:  os.remove(f)
        except NameError:   pass

        E_t = optional_load(sdir + 'restart/E_' + re_fstr + f'-nt{nt}.pkl')
        B_t = optional_load(sdir + 'restart/B_' + re_fstr + f'-nt{nt}.pkl')
        phi_t = optional_load(sdir + 'restart/phi_' + re_fstr + f'-nt{nt}.pkl')
        psi_t = optional_load(sdir + 'restart/psi_' + re_fstr + f'-nt{nt}.pkl')
        max_bond_E = pickle.load(open(sdir + 'restart/maxD_E_' + re_fstr + f'.pkl', 'rb'))
        max_bond_B = pickle.load(open(sdir + 'restart/maxD_B_' + re_fstr + f'.pkl', 'rb'))
        max_bond_phi = pickle.load(open(sdir + 'restart/maxD_phi_' + re_fstr + f'.pkl', 'rb'))
        max_bond_psi = pickle.load(open(sdir + 'restart/maxD_psi_' + re_fstr + f'.pkl', 'rb'))
        print('loaded', nt, len(max_bond_E))
        nrg_E_ts = pickle.load(open(sdir + 'restart/nrg_E_' + re_fstr + f'.pkl', 'rb'))
        nrg_B_ts = pickle.load(open(sdir + 'restart/nrg_B_' + re_fstr + f'.pkl', 'rb'))
        ts = pickle.load(open(sdir + 'restart/ts_' + re_fstr + f'.pkl', 'rb'))
        ts = list(ts)
        print('loaded',nt,len(nrg_fe_ts))

        max_bond_E = max_bond_E[:nt + 1]
        max_bond_B = max_bond_B[:nt + 1]
        max_bond_phi = max_bond_phi[:nt + 1]
        max_bond_psi = max_bond_psi[:nt + 1]
        nrg_E_ts = nrg_E_ts[:nt + 1]
        nrg_B_ts = nrg_B_ts[:nt + 1]
        ts = ts[:nt + 1]
        total_E0 = nrg_E_ts[0] + nrg_fi_ts[0] + nrg_fe_ts[0]

        ## technically would want to reconstruct X, Y, Z coordinates from saved data

        # init_fe_gtn = None
        # init_fi_gtn = None
        # init_Ez_gtn.data = E_t[Z]
        # init_Bx_gtn.data = B_t[X]
        # init_By_gtn.data = B_t[Y]
        # init_phi_gtn.data = phi_t
        # init_phi_gtn.data = psi_t

        init_E_field.initialize_field(E_t)
        init_B_field.initialize_field(B_t)
        init_phi_field.initialize_field(phi_t)
        init_psi_field.initialize_field(psi_t)

        restarted = True
        print('loaded restart',nt)

    except(ValueError,FileNotFoundError):
        nt = 0
        ts = [0]
        restarted = False

        nrg_E_ts  = [np.sum(init_Ez * np.conj(init_Ez)) * dx * dy * 1. / 2]
        nrg_B_ts  = [0.0]

        max_bond_E = [init_Ez_gtn.max_bond()]
        max_bond_B = [(np.nan,np.nan)]
        max_bond_phi = [np.nan]
        max_bond_psi = [np.nan]


    ### BUILD PDE
    em_sys = Maxwell(init_E_field, init_B_field, phi=init_phi_field, psi=init_psi_field,
                     coords_x=coord_sys, matl_params=matl_config, 
                     is_electrostatic=False, clean=True, normalize=False,
                     te_order=te_order, compress_levels=comp_levels )

    # vm_sys = VlasovMaxwell(None, None, init_E_field, init_B_field, init_phi_field, init_psi_field,
    #                        coords_x=coord_sys, coords_ve=None, coords_vi=None,
    #                        matl_params=matl_config,
    #                        te_order=te_order, compress_levels=comp_levels,
    #                        )


    if restarted:
        if do_adapt_dt:
            cfl_params = ((dx, dy), (c, c))
            dt = cfl * cfl_limit( cfl_params)
            print('restarted dt', dt)



    time1 = time.time()
    Exy_t = [init_E_field.get_comp_data(Z)]
    t0 = ts[-1]
    nt0 = nt
    
    while ts[-1] < T:

        walltime = time.time()
        em_sys = em_sys.next_time_step(dt, compress_level=1)
        print('wall time', time.time() - walltime)

        E_field = em_sys.E
        max_bond_E += [E_field[Z].max_bond()]
        E_data = E_field.get_comp_data(Z)
        Exy_t += [E_data]

        B_field = em_sys.B
        max_bond_B += [(B_field[X].max_bond(), B_field[Y].max_bond())]
        Bx_data = B_field.get_comp_data(X)
        By_data = B_field.get_comp_data(Y)

        max_bond_phi += [em_sys.phi.max_bond() if em_sys.phi is not None else np.nan]
        max_bond_psi += [em_sys.psi.max_bond() if em_sys.psi is not None else np.nan]

        nrg_E = np.sum(E_data * np.conj(E_data)) * dx * dy * 1. / 2
        nrg_E_ts += [nrg_E]

        nrg_Bx = np.sum(Bx_data * np.conj(Bx_data)) if Bx_data is not None else 0.0
        nrg_By = np.sum(By_data * np.conj(By_data)) if By_data is not None else 0.0
        nrg_B_ts += [(nrg_Bx + nrg_By) * dx * dy * 1. / 2 ]

        ts += [ts[-1]+dt]
        nt += 1

        if do_adapt_dt:
            cfl_params = ((dx, dy), (c, c))
            dt = cfl * cfl_limit(cfl_params)

        nrg_norm = nrg_E_ts[-1] + nrg_B_ts[-1]
        print('result norm', nt, ts[-1], dt, nrg_norm,
              E_field[Z].max_bond(), B_field[X].max_bond(), B_field[Y].max_bond(),
              em_sys.phi.max_bond(), em_sys.psi.max_bond(), 2 ** L)

        divB = B_field.divergence(coord_sys=coord_sys)
        divB_data = divB.get_comp_data()
        print('norm divB', np.linalg.norm(divB_data))


        if nt%250 == 0:
            fig1,ax1 = plt.subplots()
            time_ = ts[-1]
            im1 = ax1.imshow(E_data.T, extent=(x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]),
                                aspect='auto', origin='lower')

            fig1.colorbar(im1, ax=ax1)
            ax1.set_title(f'time {time_:3.2f}')
            ax1.set_xlabel('x'), ax1.set_ylabel('y')
            if save_figs:   fig1.savefig(fdir+f'plot_E_'+fstr+f'_nt{nt}.png')

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
                optional_dump(em_sys.E_func._components, sdir + 'restart/' + 'E_' + fstr + f'-nt{nt}.pkl')
                optional_dump(em_sys.B._components, sdir + 'restart/' + 'B_' + fstr + f'-nt{nt}.pkl')
                optional_dump(em_sys.phi._components, sdir + 'restart/' + 'phi_' + fstr + f'-nt{nt}.pkl')
                optional_dump(em_sys.psi._components, sdir + 'restart/' + 'psi_' + fstr + f'-nt{nt}.pkl')
                pickle.dump(max_bond_E, open(sdir + 'restart/' + 'maxD_E_' + fstr + '.pkl', 'wb'))
                pickle.dump(max_bond_B, open(sdir + 'restart/' + 'maxD_B_' + fstr + '.pkl', 'wb'))
                pickle.dump(max_bond_phi, open(sdir + 'restart/' + 'maxD_phi_' + fstr + '.pkl', 'wb'))
                pickle.dump(max_bond_psi, open(sdir + 'restart/' + 'maxD_psi_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_E_ts, open(sdir + 'restart/nrg_E_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_B_ts, open(sdir + 'restart/nrg_B_' + fstr + '.pkl', 'wb'))
                pickle.dump(ts, open(sdir + 'restart/ts_' + fstr + '.pkl', 'wb'))


    if save_data:
        optional_dump(em_sys.E._components, fdir + 'E_' + fstr + '.pkl')
        optional_dump(em_sys.B._components, fdir + 'B_' + fstr + '.pkl')
        optional_dump(em_sys.phi._components, fdir + 'phi_' + fstr + '.pkl')
        optional_dump(em_sys.psi._components, fdir + 'psi_' + fstr + '.pkl')
        pickle.dump(max_bond_E, open(fdir + 'maxD_E_' + fstr + '.pkl', 'wb'))
        pickle.dump(max_bond_B, open(fdir + 'maxD_B_' + fstr + '.pkl', 'wb'))
        pickle.dump(max_bond_phi, open(fdir + 'maxD_phi_' + fstr + '.pkl', 'wb'))
        pickle.dump(max_bond_psi, open(fdir + 'maxD_psi_' + fstr + '.pkl', 'wb'))
        pickle.dump(nrg_E_ts, open(fdir + 'nrg_E_' + fstr + '.pkl', 'wb'))
        pickle.dump(nrg_B_ts, open(fdir + 'nrg_B_' + fstr + '.pkl', 'wb'))
        pickle.dump(ts, open(fdir + 'ts_' + fstr + '.pkl', 'wb'))

    print('wall time', time.time()-time1)

Exy_t = np.array(Exy_t)

fig4, ax4 = plt.subplots()
im4 = ax4.imshow(E_data)
ax4.set_xlabel('y')
ax4.set_ylabel('x')
fig4.colorbar(im4)

ts_ = ts[nt0:]

E_theory = lambda x,y,t: E0*np.sin(a*x)*np.sin(b*y)*np.cos(omega*t)
Exy_th = E_theory(data_x,data_y,ts[-1])
print('L1 diff E(x,y,t=T)',  np.sum(np.abs(Exy_th - Exy_t[-1]))/npts**2)
print('rms diff E(x,y,t=T)', np.linalg.norm(Exy_th - Exy_t[-1])/npts)

x_pt = np.argmax(np.max(init_Ez,axis=1))
y_pt = np.argmax(np.max(init_Ez,axis=0))
x_, y_ = x_vals[x_pt], y_vals[y_pt]
print('eval t at point',x_,y_)
E_th_data = E_theory(x_,y_,np.array(ts_))
print(E_th_data.shape)

fig2, ax2 = plt.subplots()
ax2.plot(ts_,Exy_t[:,x_pt,y_pt])
ax2.plot(ts_,E_th_data,'k:')
ax2.set_xlabel('t')
ax2.set_ylabel('E(x=X,y=Y)')

max_bond_B = np.array(max_bond_B)
max_bond_E = np.array(max_bond_E)

# fig3, ax3 = plt.subplots()
# ax3.plot(ts, np.log(max_bond_E[:,0])/np.log(q**K),label='Ex')
# ax3.plot(ts, np.log(max_bond_E[:,1])/np.log(q**K),label='Ey')
# ax3.plot(ts, np.log(max_bond_E[:,2])/np.log(q**K),label='Ez')
# ax3.plot(ts, np.log(max_bond_B[:,0])/np.log(q**K),label='Bx')
# ax3.plot(ts, np.log(max_bond_B[:,1])/np.log(q**K),label='By')
# ax3.plot(ts, np.log(max_bond_B[:,2])/np.log(q**K),label='Bz')
# ax3.plot([ts[0],ts[-1]], [L//2]*2, 'k:')
# ax3.set_xlabel('t')
# ax3.set_ylabel('log_q**K(max bond (Ez))')

plt.show()
   
   
print('done')

