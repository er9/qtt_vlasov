from setup.configs import *
import helper_quimb as helper
from axis import Axis
from grid1D import Grid1D
from field import Field, ScalarField
from pde_system import PDE_system

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem


class VlasovPoisson(PDE_system):
    """ df/dt + v grad(f) + q/m (E) grad_v(f) = 0
    """

    def __init__(self, dist_e: Optional['ScalarField'], dist_i: Optional['ScalarField'],
                 potential: Optional['ScalarField'],
                 coords_x:  Optional['CoordinateSystem'] = None,
                 coords_ve: Optional['CoordinateSystem'] = None,
                 coords_vi: Optional['CoordinateSystem'] = None,
                 elc_params: Optional[SpeciesConfiguration] = None,
                 ion_params: Optional[SpeciesConfiguration] = None,
                 matl_params: Optional[UnitsConfiguration] = None,
                 f0_gradv_e=None, f0_gradv_i=None,
                 evolve_ion=True, normalize=True, upwind=False,
                 potential_boundary_conditions=(None,None),
                 te_order=2,
                 compress_levels=None,
                 # init_compress_opts=None, te_compress_opts=None
                 ):
        """ dist_e:  distribution of electrons (scalar Field obj). generally on x,v grid
            dist_i:  distribution of ions (scalar Field obj). generally on x,v grid
            potential:  electric potential (scalar Field obj). generally on x grid
            boundary_conditions: boundary conditions on V
               tuple of (bc_vectors [m vectors or mps], bc_results [m float]) such that
               bc_vectors (dot) V_data = bc_results
        """

        # if init_compress_opts is None:   init_compress_opts = {'cutoff_mode':'rsum2','cutoff':1.0e-20}
        # if te_compress_opts is None:     te_compress_opts = {}

        self.names = {}
        self.names['fe'] = dist_e.name if dist_e is not None else 'fe'
        self.names['fi'] = dist_i.name if dist_i is not None else 'fi'
        self.names['V']  = potential.name if potential is not None else 'V'

        field_names = [self.names[x] for x in ['fe','fi','V']]

        super().__init__(dist_e, dist_i, potential, field_names=field_names, normalize=normalize, te_order=te_order,
                         compress_levels=compress_levels)

        self.coords_x  = coords_x
        self.coords_ve = coords_ve
        self.coords_vi = coords_vi

        # if x_axes is None:  self.x_axes = tuple(range(self.ndim//2))
        # else:               self.x_axes = x_axes
        # self.v_axes = tuple([i for i in range(self.ndim) if i not in self.x_axes])

        # ## create velocity field (not necessary?)
        # if self.fe is not None:
        #     v_dict = {}
        #     for v_ax in self.coords_ve.axes:
        #         try:
        #             x_ax = self.coords_x.coords_ax[v_ax.coord_type]
        #             v_mult = v_ax.get_xmultiply_mpo()
        #             v_dict[x_ax.coordinate] = self.fe.grid.make_mpo_ndim({v_ax.coordinate:v_mult})
        #         except KeyError:
        #             pass
        #     self.velocities_e = self.fe.create_like(v_dict)
        #     self.velocities_e.name = 'vel'
        #
        # if self.fi is not None:
        #     v_dict = {}
        #     for v_ax in self.coords_vi.axes:
        #         try:
        #             x_ax = self.coords_x.coords_ax[v_ax.coord_type]
        #             v_mult = v_ax.get_xmultiply_mpo()
        #             v_dict[x_ax.coordinate] = self.fi.grid.make_mpo_ndim({v_ax.coordinate:v_mult})
        #         except KeyError:
        #             pass
        #     self.velocities_i = self.fi.create_like(v_dict)
        #     self.velocities_i.name = 'vel'

        ## save initial normalization for normalization
        self.init_norm_i = None
        self.init_norm_e = None
        if normalize:
            self.init_norm_i = self.fi.norm() if self.fi is not None else None
            self.init_norm_e = self.fe.norm() if self.fe is not None else None

        ## define for linearized Vlasov equation
        self.df0dv_e = f0_gradv_e
        self.df0dv_i = f0_gradv_i

        self.inv_laplacian_mpo = None
        self.potential_boundary_conditions = potential_boundary_conditions

        if elc_params is None and ion_params is None:
            self.matl_params = UnitsConfiguration() if matl_params is None else matl_params
            self.elc_params = ElcConfiguration(units_config=self.matl_params)
            self.ion_params = IonConfiguration(units_config=self.matl_params)
        else:
            self.elc_params = ElcConfiguration() if elc_params is None else elc_params
            self.ion_params = IonConfiguration() if ion_params is None else ion_params
            self.matl_params = matl_params if matl_params is not None else \
                                    elc_params.units_config if elc_params is not None else ion_params.units_config
            self.elc_params.units_config = self.matl_params
            self.ion_params.units_config = self.matl_params

        self.evolve_ion = evolve_ion
        self.upwind = upwind

        self.compress_F = True
        self.compress_F_opts = {'max_bond':None, 'cutoff':CUTOFF, 'cutoff_mode':CUTOFF_MODE}

        self.collision = CollisionConfiguration(None)

        self.dist0_e = None
        self.dist0_i = None

        self.v_from_v0_e = None   # data_ve - vdrift_e
        self.v_from_v0_i = None   # data_vi - vdrift_i


    @property
    def fe(self) -> Optional['ScalarField']:
        try:   return self.fields[self.names['fe']]
        except KeyError:   return None

    @fe.setter
    def fe(self,new_field):
        self.fields[self.names['fe']] = new_field


    @property
    def fi(self) -> Optional['ScalarField']:
        try:   return self.fields[self.names['fi']]
        except KeyError:   return None

    @fi.setter
    def fi(self,new_field):
        self.fields[self.names['fi']] = new_field


    @property
    def V(self) -> Optional['ScalarField']:
        try:   return self.fields[self.names['V']]
        except KeyError:   return None

    @V.setter
    def V(self,new_field):
        self.fields[self.names['V']] = new_field


    def create_like(self, *new_fields, recalc=True):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 3:
            new_fields = new_fields + (None,) *(3-len(new_fields))

        dist_e, dist_i, potential = new_fields[:3]
        if recalc:
            new_system = VlasovPoisson(dist_e, dist_i, potential, coords_x=self.coords_x, coords_ve=self.coords_ve,
                                       coords_vi=self.coords_vi, elc_params=self.elc_params, ion_params=self.ion_params,
                                       f0_gradv_e=self.df0dv_e, f0_gradv_i=self.df0dv_i, evolve_ion=self.evolve_ion,
                                       normalize=self.do_normalization, upwind=self.upwind,
                                       potential_boundary_conditions=self.potential_boundary_conditions,
                                       te_order=self.te_order, compress_levels=self._comp_levels)
        else:
            new_system = VlasovPoisson(dist_e, dist_i, potential, coords_x=self.coords_x, coords_ve=self.coords_ve,
                                       coords_vi=self.coords_vi, elc_params=self.elc_params, ion_params=self.ion_params,
                                       f0_gradv_e=self.df0dv_e, f0_gradv_i=self.df0dv_i, evolve_ion=self.evolve_ion,
                                       normalize=False,
                                       upwind=self.upwind,
                                       potential_boundary_conditions=self.potential_boundary_conditions,
                                       te_order=self.te_order, compress_levels=self._comp_levels)
            new_system.do_normalization = self.do_normalization
            new_system.init_norm_e = self.init_norm_e
            new_system.init_norm_i = self.init_norm_i

        if dist_e is None:      new_system.names['fe'] = self.names['fe']
        if dist_i is None:      new_system.names['fi'] = self.names['fi']
        if potential is None:   new_system.names['V' ] = self.names['V' ]

        new_system.inv_laplacian_mpo = self.inv_laplacian_mpo

        new_system.collision = self.collision

        new_system.compress_F = self.compress_F
        new_system.compress_F_opts = self.compress_F_opts

        new_system.dist0_i = self.dist0_i
        new_system.dist0_e = self.dist0_e

        # new_system.v_from_v0_i = self.v_from_v0_i
        # new_system.v_from_v0_e = self.v_from_v0_e

        return new_system


    # def copy(self):
    #     """ create a new system like this with new fields
    #     """
    #     fe = self.fe.copy() if self.fe is not None else None
    #     fi = self.fi.copy() if self.fi is not None else None
    #     V = self.V.copy() if self.V is not None else None
    #     new_system = self.create_like(fe,fi,V)
    #     return new_system


    def normalize(self,target_e=None,target_i=None):
        """ normalized density field
        """
        if self.fi is not None:
            if target_i is None:
                target_i = self.init_norm_i if self.init_norm_i is not None else 1.0
            norm_i = self.fi.norm()
            # if self.fi.is_sqrt:  norm_i = np.sqrt(norm_i)
            # self.fi.scalar_multiply( np.abs(target_i/norm_i), inplace=True )
            self.fi.scalar_multiply( target_i/np.conj(norm_i), inplace=True )

        if self.fe is not None:
            if target_e is None:
                target_e = self.init_norm_e if self.init_norm_e is not None else 1.0
            norm_e = self.fe.norm()
            # if self.fe.is_sqrt:  norm_e = np.sqrt(norm_e)
            # self.fe.scalar_multiply( np.abs(target_e/norm_e), inplace=True )
            self.fe.scalar_multiply( target_e/np.conj(norm_e), inplace=True )
        return 


    def get_time_derivative_operator(self,compress=1):
        """ dF/dt = G*prob[t].  Returns G
            NOT DEBUGGED
        """ 
        raise NotImplementedError

        x_dim = v_dim = self.grid.ndim // 2

        convective_term = None
        for i in range(x_dim):
            tmp = self.fe.grid.get_mpo_firstderivative(i).copy()
            tmp = helper.scalar_multiply(tmp, 1./self.grid.dx[i])
            tmp = helper.apply( self.velocities[i], tmp )
            convective_term = helper.add_MPO(convective_term,tmp,compress=max(0,compress-1))
        helper.scalar_multiply(grads,-1,inplace=True)

        # if self.B is not None:  em_term = velocity.cross_product(self[self.B], compress=1)
        # else:                   em_term = None
        # em_term = helper.add_MPO(self['E'],em_term
        em_term = self['V'].gradient()
        if em_term.ndim <= self.fe.grid.ndim:  
            em_term = em_term.pad_to_new_grid(new_grid=self.fe_grid)

        lorentz_term = None
        for i in range(v_dim):
            tmp = self.fe.grid.get_mpo_secondderivative(i+x_dim,i+x_dim).copy()      # velocity derivative
            tmp = helper.scalar_multiply(tmp, 1./self.grid.dx[i+x_dim])
            try:               tmp = helper.apply( em_term[i], tmp )
            except KeyError:  continue
            lorentz_term = helper.add_MPO(lorentz_term,tmp,compress=max(0,compress-1))
            
        tot_op = helper.add_MPO(convective_term, lorentz_term, compress=compress)

        return tot_op  ## an MPO


    def euler(self, dt: Numeric, deriv0: Optional['VlasovPoisson'] = None, inplace: bool = False,
              compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0,
              verbose_plot=False) -> 'VlasovPoisson':
        """ perform explicit Euler time evolution
            compress:  final compression of final state, updateV
            compress1: compression of derivative
            compress2: compress1 of calculate derivative, updateV
        """
        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(compress_level=compress_level1, compress_level1=compress_level2,
                                                    verbose_plot=verbose_plot)

        state1 = self.add(deriv0 * dt, compress_level=0, inplace=inplace)
        # state1 = self.copy()

        x_axes  = self.coords_x.axes
        if self.fe is not None:
            for x_ax in x_axes:
                v_ax = self.coords_ve.get_axis(x_ax.coordinate.type)
                self.fe.apply_absorbing_bc(x_ax, v_ax, inplace=True, compress_level=compress_level1)  ## acts iff bc is absorbing
        if self.fi is not None:
            for x_ax in x_axes:
                v_ax = self.coords_ve.get_axis(x_ax.coordinate.type)
                self.fi.apply_absorbing_bc(x_ax, v_ax, inplace=True, compress_level=compress_level1)  ## acts iff bc is absorbing

        # plt.figure()
        # plt.imshow(np.imag(deriv0.fi.get_comp_data()))
        # plt.title('df/dt i imag ')
        # plt.colorbar()
        #
        # plt.figure()
        # plt.imshow(np.imag(state1.fi.get_comp_data()))
        # plt.title('fi imag pre compress')
        # plt.colorbar()
        # plt.show()

        # print('compress state euler', compress)
        if compress_level:
            state1.compress()

        # plt.figure()
        # plt.imshow(np.imag(state1.fi.get_comp_data()))
        # plt.title('fi imag post compress')
        # plt.colorbar()
        # plt.show()


        if state1.do_normalization:
            state1.normalize()


        if np.abs(state1.matl_params.e) > 0:
            state1.update_V(inplace=True, compress=compress_level, compress1=compress_level2)

        return state1


    def calculate_time_derivative(self, compress_level=0, compress_level1=0, verbose_plot=False) -> 'VlasovPoisson':
        """ df/dt = ...
            compress:  compression of derivative
            compress1: compression of terms summed together to make derivative
        """
        dFdt_e = self._calculate_time_derivative_f(compress=compress_level, compress1=compress_level1,
                                                   verbose_plot=verbose_plot)
        if self.evolve_ion:
            dFdt_i = self._calculate_time_derivative_f(compress=compress_level, compress1=compress_level1, is_ion=True,
                                                       verbose_plot=verbose_plot)
        else:
            dFdt_i = None

        dFdt = self.create_like(dFdt_e, dFdt_i, None)
        return dFdt


    # @profile
    def _calculate_time_derivative_f(self,x_grads=None,v_grads=None,compress=1,compress1=0,compress2=0,is_ion=False,
                                     verbose_plot=False) \
            -> Optional['ScalarField']:
        """ df/dt = ...
            compress:  compression of derivative
            compress1: compression of terms summed together to make derivative
        """
        dist = self.fi if is_ion else self.fe
        spec_params = self.ion_params if is_ion else self.elc_params

        if dist is None:  return None

        x_axes = self.coords_x.axes
        v_coordsys = self.coords_vi if is_ion else self.coords_ve
        v_axes_dict = {ax: v_coordsys.get_axis(ax.coordinate.type) for ax in x_axes}
        v_axes = [v_axes_dict.get(ax, None) for ax in x_axes]

        if x_grads is None:
            # print('v_axes', v_axes_dict)
            if self.upwind:
                x_grads = dist.gradient(deriv_axes=x_axes, upwind_axes=v_axes_dict, compress_level=compress1)
            else:
                x_grads = dist.gradient(deriv_axes=x_axes, upwind_axes=None, compress_level=compress1)
            # print('x_grad', dist.max_bond(), x_grads.max_bonds())

            if verbose_plot:
                dist_data = dist.get_comp_data()
                plt.figure()
                plt.imshow(np.log(np.abs(dist_data))/np.log(10))
                plt.title('dist')
                plt.colorbar()
                # plt.show()
                #
                # # plt.figure()
                # # plt.semilogy(dist_data[:, 0], label='0')
                # # # plt.semilogy(dist_data[:, 1], label='1')
                # # plt.title('dist f(x,v=v_min)')
                # # plt.legend()
                # # plt.xlabel('x')

                x_ax = x_axes[0]
                v_ax = v_axes[0]

                xg_data = x_grads.get_comp_data(x_ax.coordinate)
                plt.figure()
                plt.imshow(np.log(np.abs(xg_data))/np.log(10))
                plt.title('xgrad')
                plt.colorbar()

                # plt.show()

        # dist_data = dist.get_comp_data(v_axes[0].coordinate)
        # print(dist.grid.axes[1].basis)
        # dist_data = dist.grid.axes[0].basis.get_realspace_1D(dist_data, 0, x0=-np.pi / 0.10, xL=np.pi / 0.10, npts=512)
        # dist_data = dist.grid.axes[1].basis.get_realspace_1D(dist_data, 1, x0=-6 / 0.10, xL=6 / 0.10)
        # print('dist', dist_data.shape)
        # plt.figure()
        # plt.imshow(np.real(dist_data))
        # plt.colorbar()
        #
        # plt.figure()
        # plt.imshow(np.imag(dist_data))
        # plt.colorbar()
        # plt.show()
        # # exit()


        df0dv = self.df0dv_i if is_ion else self.df0dv_e    # for linearized Vlasov
        if v_grads is None:
            v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress1) \
                                       if df0dv is None else df0dv
            # print('v_grad', x_grads.max_bond())

            # vgrad_data = v_grads.get_comp_data(v_axes[0].coordinate)
            # print(v_grads.grid.axes[1].basis)
            # vgrad_data = v_grads.grid.axes[0].basis.get_realspace_1D(vgrad_data,0,x0=-np.pi/0.10, xL=np.pi/0.10)
            # vgrad_data = v_grads.grid.axes[1].basis.get_realspace_1D(vgrad_data,1,x0=-6/0.10, xL=6/0.10, npts=512)
            # print('vgrad', vgrad_data.shape)
            # plt.figure()
            # plt.imshow(np.real(vgrad_data))
            # plt.title('vgrad')
            # plt.colorbar()
            #
            # plt.figure()
            # plt.imshow(np.imag(vgrad_data))
            # plt.colorbar()
            # plt.show()

            #### vgrad max: 0.16 for real space, 0.22 for Fourier-Hermite
            #### something with scaling when taking gradient?
            # exit()


        #### convective term  ####

        # print('x grad max bond', x_grads.max_bonds())
        convective_term = x_grads.xdot(v_axes, compress_level=compress1)
        # convective_term = x_grads.xdot(v_axes, compress_level=0)
        # print('convective term', convective_term.max_bond())
        # convective_term.compress(inplace=True, compress_level=compress1)
        convective_term.scalar_multiply(-1, inplace=True)

        # xgrad_data = convective_term.get_comp_data(v_axes[0].coordinate)
        # xgrad_data = v_grads.grid.axes[0].basis.get_realspace_1D(xgrad_data, 0, x0=-np.pi / 0.10, xL=np.pi / 0.10)
        # xgrad_data = v_grads.grid.axes[1].basis.get_realspace_1D(xgrad_data, 1, x0=-6 / 0.10, xL=6 / 0.10)
        # print('vgrad', xgrad_data.shape)
        # plt.figure()
        # plt.imshow(np.real(xgrad_data))
        # plt.colorbar()
        # plt.title('xgrad re')
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(np.imag(xgrad_data))
        # plt.colorbar()
        # plt.title('xgrad im')
        # plt.show()

        # conv_data = convective_term.get_comp_data(v_axes[0].coordinate)
        # conv_data = v_grads.grid.axes[0].basis.get_realspace_1D(conv_data,0,x0=-np.pi/0.10, xL=np.pi/0.10)
        # conv_data = v_grads.grid.axes[1].basis.get_realspace_1D(conv_data,1,x0=-6/0.10, xL=6/0.10)
        # print('vgrad', conv_data.shape)
        # plt.figure()
        # plt.imshow(np.real(conv_data))
        # plt.colorbar()
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(np.imag(conv_data))
        # plt.colorbar()
        # plt.show()
        # # exit()


        if verbose_plot:

            ## seemingly problematic
            tmp = convective_term.get_comp_data()
            plt.figure()
            plt.imshow(np.log(np.abs(tmp))/np.log(10))
            plt.title('conv')
            plt.colorbar()

            plt.show()


        ## (ES) force term
        if np.abs(spec_params.Z) > 0:
            # print('compute E field, no upwind')
            em_term = self.V.gradient(compress_level=0)   # E = -grad(V), but *-1 bc moved from LHS to RHS

            # print('here', self.compress_F)

            if verbose_plot:
                x_ax = x_axes[0]
                print(self.V.component.ax_deriv_configs[x_ax].bc)

                plt.figure()
                tmp = self.V.get_comp_data()
                plt.plot(np.real(tmp), label='V', marker='x')
                # plt.plot(np.roll(tmp,20), label='V',marker='x')
                tmp = em_term.get_comp_data(x_ax.coordinate)
                plt.plot(np.imag(tmp),label='E',marker='x')
                # plt.plot(np.roll(tmp,20),label='E',marker='x')
                plt.legend()
                plt.ylabel('grad V')
                plt.show()

            # print('V bc', self.V[0].ax_deriv_configs[x_ax].bc)
            # print('dist bc', dist[0].ax_deriv_configs[x_ax].bc)
            # print('dist bc', dist[0].ax_deriv_configs[v_ax].bc)

            if self.compress_F:
                em_term.compress(compress_opts=self.compress_F_opts, inplace=True)

            ## old version
            # em_term = em_term.pad_to_new_grid(new_grid=dist.grid)
            # # print('em term', em_term, em_term[x_axes[0].coordinate].data)
            # # print('v grads', v_grads, v_grads[v_axes[0].coordinate].data)
            #
            # # print('em term max bonds', v_grads.max_bonds(), em_term.max_bonds())
            # lorentz_term = em_term.dot(v_grads, other_comps=[ax.coordinate for ax in v_axes], compress_level=compress1)

            ## new version for compatibility with Hermite bases in v
            ## put em_term into elemental_multiply form here (so are MPOs)
            for compID, comp in em_term.components.items():
                em_term.components[compID] = comp.apply_elemental_multiply_op()

            em_term = em_term.pad_to_new_grid(new_grid=dist.grid)  # pad with iden
            lorentz_term = v_grads.dot(em_term, comps=[ax.coordinate for ax in v_axes],
                                       other_comps=[ax.coordinate for ax in x_axes], compress_level=compress1)

            lorentz_term.scalar_multiply(spec_params.Z * spec_params.e / spec_params.mass, inplace=True)
            # if is_ion:
            #     lorentz_term.scalar_multiply(matl_params.Z * matl_params.e / matl_params.mass, inplace=True)
            # else:
            #     lorentz_term.scalar_multiply(-1 * self.elc_params.e / self.elc_params.mass_e, inplace=True)
        else:
            lorentz_term = None

        if verbose_plot:
            tmp = lorentz_term.get_comp_data()
            plt.figure()
            plt.imshow(np.log(np.abs(tmp))/np.log(10))
            plt.title('lorentz')
            plt.colorbar()
            plt.show()

        # print('add', lorentz_term.max_bond(), convective_term.max_bond())
        # print(lorentz_term.component)
        # print(convective_term.component.data)
        # dFdt_e = convective_term.copy()
        # dFdt_e = lorentz_term.copy()
        dFdt_e = convective_term.add(lorentz_term, compress_level=0)
        dFdt_e.name = dist.name

        # if verbose_plot:
        #     x_ax = x_axes[0]
        #     v_ax = v_axes[0]
        #     grid_S1 = Grid1D('S1', (x_ax, v_ax), layout_type=LayoutType.SEQUENTIAL)
        #     grid_PG = Grid1D('PG', (x_ax, v_ax), layout_type=LayoutType.PARALLEL_GROUP)
        #     conv_data = convective_term.component.get_data()
        #     elec_data = lorentz_term.component.get_data()
        #
        #     conv_S1 = grid_S1.map_state_to_mps(conv_data)
        #     elec_S1 = grid_S1.map_state_to_mps(elec_data)
        #
        #     conv_PG = grid_PG.map_state_to_mps(conv_data)
        #     elec_PG = grid_PG.map_state_to_mps(elec_data)
        #
        #     out_S1 = conv_S1.add(elec_S1)
        #     out_PG = conv_PG.add(elec_PG)
        #
        #     out_S1_data = out_S1.get_data()
        #     out_PG_data = out_PG.get_data()
        #
        #     plt.figure()
        #     plt.plot(out_S1_data[:,0] - out_PG_data[:,0])
        #     plt.xlabel('x')
        #     plt.title('add diff SI-PG f(x,vmin)')
        #     plt.show()

        ### get collisions
        coll_field = self.get_collision_term(is_ion=is_ion, v_axes=v_axes, v_grads=v_grads,
                                             compress1=compress1, compress2=compress2)
        dFdt_e.add(coll_field, inplace=True, compress_level=0)

        if compress:
            dFdt_e.compress()

        # print('dFdt norm', dFdt_e.integrate([0])[0])

        if verbose_plot:
            tmp = dFdt_e.get_comp_data()
            plt.figure()
            plt.imshow((np.real(tmp)))
            plt.title(f'dfdt, real, {is_ion}')
            plt.colorbar()
            plt.figure()
            plt.imshow((np.imag(tmp)))
            plt.title(f'dfdt, imag, {is_ion}')
            plt.colorbar()
            plt.show()

        return dFdt_e


    def update_V(self,inplace=False,compress=1,compress1=0) -> 'ScalarField':
        """ div E = div^2 V = 4*pi*rho
            compress:  compression of V
            compress1: compression of terms to create V
        """
        x_axes = self.coords_x.axes
        ve_axes = self.coords_ve.axes
        vi_axes = self.coords_vi.axes

        new_sys = self if inplace else self.copy()

        ## particles are not charged
        if self.matl_params.e == 0 or (np.abs(self.elc_params.Z) == 0 and np.abs(self.ion_params.Z) == 0):
            new_sys.V = None
            return new_sys

        if self.inv_laplacian_mpo is None:
            ax_deriv_configs = self.V.component.ax_deriv_configs if self.V is not None \
                else {ax: DerivativeConfiguration() for ax in x_axes}
            print('inv lapl mpo bc', ax_deriv_configs[x_axes[0]].bc)
            laplacian_mpo = self.V.grid.laplacian_mpo(store=False, ax_deriv_configs=ax_deriv_configs)
            inv_laplacian_mpo = self.V.grid.get_mpo_inverse(laplacian_mpo,
                                                            self.potential_boundary_conditions[0])
            inv_laplacian_mpo.compress(inplace=True, compress_opts={'cutoff':1.0e-16,'cutoff_mode':'rsum2'})
            self.inv_laplacian_mpo = inv_laplacian_mpo
            print('inv laplacian', inv_laplacian_mpo.max_bond())
        else:
            inv_laplacian_mpo = self.inv_laplacian_mpo

        ## get Field object after integrating over velocity space
        fe = self.fe
        e_density = fe.integrate(axes=ve_axes, new_grid=self.V.grid,
                                 new_ax_deriv_configs=self.V.component.ax_deriv_configs)
        e_density.compress_config = self.V.compress_config
        e_density.scalar_multiply(self.elc_params.Z * self.elc_params.e, inplace=True)

        if self.fi is not None:
            # if self.fi.is_sqrt:   fi = self.fi.mult_cc(compress=0) # compress+1)
            # else:                 fi = self.fi
            fi = self.fi
            i_density = fi.integrate(axes=vi_axes, new_grid=self.V.grid,
                                     new_ax_deriv_configs=self.V.component.ax_deriv_configs)
            i_density.compress_config = self.V.compress_config
            i_density.scalar_multiply(self.ion_params.Z * self.ion_params.e, inplace=True)

            charge_density = e_density.add(i_density, compress_level=compress1, inplace=False)
        else:
            charge_density = e_density.compress()

        # e_dens_dat = e_density.get_comp_data()
        # i_dens_dat = i_density.get_comp_data()
        # plt.figure()
        # plt.plot(np.real(i_dens_dat), label='ion')
        # plt.plot(np.real(e_dens_dat), label='elec')
        # # plt.plot(np.imag(e_dens_dat), label='elec', ls='--')
        # charge_dat = charge_density.get_comp_data()
        # plt.plot(np.real(charge_dat), label='total')
        # # plt.plot(np.imag(charge_dat), label='total', ls='--')
        # plt.title('charge density')
        # plt.legend()
        # plt.show()

        # print('charge density', charge_density.max_bond())
        charge_density.scalar_multiply(-1. / self.elc_params.eps0, inplace=True)


        bc_values = self.potential_boundary_conditions[1]
        if bc_values is not None:  # ie. 0
            if bc_values.grid.ndim < charge_density.grid.ndim:
                bc_values = bc_values.pad_to_new_grid(new_grid=charge_density.grid)  
            soln_gtn = charge_density.component.add(bc_values[0], compress=False)
        else:
            soln_gtn = charge_density.component

        # print('inv_laplacian', inv_laplacian_mpo.max_bond())
        # print('update V', compress, new_sys.V.get_te_compress_opts(compress))
        new_V_mps = soln_gtn.apply(inv_laplacian_mpo, compress=compress,
                                   compress_opts=new_sys.V.compress_config[compress])
        # print('new V', compress, new_sys.V.get_te_compress_opts(compress), new_V_mps.max_bond())

        # V_ = V.copy()
        new_sys.V.component = new_V_mps  ## update potential field (which is a scalar field)

        return new_sys


    def get_collision_term(self, is_ion=False, v_axes:Sequence['Axis']=None, v_grads: 'Field'=None,
                           compress1=0, compress2=0):
        """ obtain collision term
        """
        dist = self.fi if is_ion else self.fe

        if v_axes is None:
            x_axes = self.coords_x.axes
            v_coordsys = self.coords_vi if is_ion else self.coords_ve
            v_axes_dict = {ax: v_coordsys.get_axis(ax.coordinate.type) for ax in x_axes}
            v_axes = [v_axes_dict.get(ax, None) for ax in x_axes]

        if self.collision.coll_type is None:
            return None

        elif self.collision.coll_type == CollisionType.LB:

            if dist.is_sqrt:   raise NotImplementedError('LB operator not implemented for sqrt field')

            if v_grads is None:
                df0dv = self.df0dv_i if is_ion else self.df0dv_e  # for linearized Vlasov
                v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress1) \
                    if df0dv is None else df0dv

            ############  get maxwellian ######
            if is_ion:
                vth2 = self.ion_params.vth ** 2
                coll_rate = self.collision.coll_rate_i
            else:
                vth2 = self.elc_params.vth ** 2
                coll_rate = self.collision.coll_rate_e

            ## alternate method of computing collisions
            # v_drift = self.v_from_v0_i if is_ion else self.v_from_v0_e
            # coll = v_drift.elemental_multiply(dist[0], compress_level=0)  ## components are self.v_axes
            v0 = self.collision.v0_i if is_ion else self.collision.v0_e
            try:
                v_offsets = {v_ax: -v0[v_ax.coordinate] for v_ax in v_axes}
            except (IndexError, TypeError):
                v_offsets = {v_ax: -v0 for v_ax in v_axes}
            coll = dist.xmultiply(v_axes, offsets=v_offsets, )
            # exit()

            coll_1 = v_grads.scalar_multiply(vth2)  ## components are self.v_axes
            coll.add(coll_1, inplace=True, compress_level=0)
            # print('coll deriv params', coll[v_axes[0].coordinate].ax_deriv_configs)
            # coll.update_comp_deriv_params(axes=v_axes)

            coords_v = self.coords_vi if is_ion else self.coords_ve
            coll_field = coll.divergence(coords_v, compress_level=compress1, inner_compress_level=compress2)

            coll_field.scalar_multiply(coll_rate, inplace=True)

            return coll_field


    def next_time_step(state0, dt, deriv0: Optional['VlasovPoisson'] = None, inplace: bool = False,
                       compress_level: int = 1, do_postprocessing=False, process_kwargs=None,
                       verbose_plot=False) -> 'VlasovPoisson':
        """ convenience fct btwn different TE methods?
            process_kwargs:
                'axes':  list of axes to post-process along or 'all'
                'sigma': gaussian smoothing stddev (default = 0.3)

            comp_levels:
            comp1:   compress final state
            comp2:   + compress intermediate rk4 states
            comp3:   + compress sum of derivatives
            comp4:   + compress derivatives
            comp5:   + compress during calculation of derivative
        """
        te_order = state0.te_order
    
        ## update density (potential should remain untouched)
        if te_order == 1:
            state_t = state0.euler(dt, deriv0, inplace, compress_level=compress_level)
        elif te_order == 2:
            state_t = state0.rk2(dt, deriv0, inplace, compress_level=compress_level)
        elif te_order == 3:
            state_t = state0.rk3(dt, deriv0, inplace, compress_level=compress_level)
        elif te_order == 4:
            state_t = state0.rk4(dt, deriv0, inplace, compress_level=compress_level, verbose_plot=verbose_plot)
        # elif te_order==21:
        #     state_t = state0.backwards_euler(dt, deriv0, compress_level, **compress_opts)
        # elif te_order==22:
        #     state_t = state0.crank_nicolson(dt, deriv0, inplace, compress_level, **compress_opts)
        else:
            raise NotImplementedError

        if do_postprocessing:
            if process_kwargs is None:  process_kwargs = {}
            if state_t.fe is not None:  state_t.fe.smooth_data(compIDs=[0],inplace=True,**process_kwargs)
            if state_t.fi is not None:  state_t.fi.smooth_data(compIDs=[0],inplace=True,**process_kwargs)


        # plt.plot(state_t.fe.get_field_data(comps=0)[:,npts//2], label='post')
        # plt.xlabel('x')
        # plt.legend()
        # plt.show()

        # i_tot = state_t.fields[state_t.fi].integrate('all')[0]
        if state_t.do_normalization:
            state_t.normalize() # target=i_tot)

        # plt.figure()
        # fe_data = state_t[state_t.fe].get_field_data(comps=0)
        # fi_data = state_t[state_t.fi].get_field_data(comps=0)
        # diff_data = fe_data - fi_data
        # plt.imshow(fe_data-fi_data)
        # plt.colorbar()
        # plt.xlabel('v')
        # plt.ylabel('x')
        # plt.title('new charge 2')

        # plt.figure()
        # plt.plot(diff_data[:,state0.q**state0.L//2])
        # plt.xlabel('x')
        # plt.ylabel('fe-fi')
        # plt.show()

        return state_t

