from setup.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
from field import Field, ScalarField
from pde_system import PDE_system

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from axis import Axis


class VlasovMaxwell(PDE_system):
    """ df/dt + v grad(f) + q/m (E + v x B) grad_v(f) = 0

        - EM method 1:    hyperbolic Maxwell's equations (Maxwell's eq with equation cleaning)
                          http://ammar-hakim.org/sj/je/je6/je6-maxwell-solvers.html
                          http://ammar-hakim.org/maxwell-eigensystem.html

            dB/dt + curl(E) + gamma grad(psi) = 0
            eps0 mu0 dE/dt - curl(B) + chi grad(phi) = 0
            1/chi dphi/dt + div(E) = rho/eps0
            eps0 mu0 /gamma dpsi/dt + div(B) = 0

            choose chi = gamma = 1

        - EM method 2:   Yee cell
            to be implemented. 
    """

    def __init__(self, dist_e: Optional[ScalarField], dist_i: Optional[ScalarField],
                 E: Optional[Field], B: Optional[Field],
                 phi: Optional[ScalarField] = None, psi: Optional[ScalarField] = None,
                 coords_x : Optional['CoordinateSystem'] = None,
                 coords_ve: Optional['CoordinateSystem'] = None,
                 coords_vi: Optional['CoordinateSystem'] = None,
                 # x_grid: Optional[Grid] = None, ve_grid: Optional[Grid] = None, vi_grid: Optional[Grid] = None,
                 # field_compress_config: Optional[FieldCompressionConfiguration] = None, compress_levels = None,
                 elc_params: Optional[SpeciesConfiguration] = None,
                 ion_params: Optional[SpeciesConfiguration] = None,
                 matl_params:  Optional[UnitsConfiguration] = None,
                 f0_gradv_e: Optional['Field'] =None, f0_gradv_i: Optional['Field'] = None,
                 is_electrostatic=False, evolve_ion=True, normalize=True, upwind=False,
                 te_order=2, compress_levels=None):
        """ dist_e:  distribution of electrons (scalar Field obj). generally on x,v grid
            dist_i:  distribution of ions (scalar Field obj). generally on x,v grid
            potential:  electric potential (scalar Field obj). generally on x grid
            velocity_grid: velocity grid (vector Field obj) with components vx, vy, ...
                           generally on v grid
            f0_gradv:   grad_v(f0) (vector Field obj) if doing linearized vlasov. 
            x_axes:  GRID axes (0,...,self.ndim-1) corresponding to spatial positions in fe, fi
        """
        self.names = {}
        self.names['fe']  = dist_e.name if dist_e is not None else 'fe'
        self.names['fi']  = dist_i.name if dist_i is not None else 'fi'
        self.names['E']   = E.name if E is not None else 'E'
        self.names['B']   = B.name if B is not None else 'B'
        self.names['phi'] = phi.name if phi is not None else 'phi'
        self.names['psi'] = psi.name if psi is not None else 'psi'

        field_names = [self.names[x] for x in ['fe','fi','E','B','phi','psi']]

        super().__init__(dist_e, dist_i, E, B, phi, psi, field_names=field_names,
                         normalize=normalize, te_order=te_order, compress_levels=compress_levels,
                         # field_compress_config=field_compress_config,
                         )
        # self.fe = self.fields['fe']
        # self.fi = self.fields['fi']
        # self.E  = self.fields['E']
        # self.B  = self.fields['B']
        # ## correction potentials for hyperbolic Maxwell's equations
        # self.phi = self.fields['phi']
        # self.psi = self.fields['psi']


        self.coords_x  = coords_x
        self.coords_ve = coords_ve
        self.coords_vi = coords_vi

        # self.x_grid  = x_grid
        # self.ve_axes = ve_grid
        # self.vi_axes = vi_grid

        if self.fe is not None:
            v_dict = {}
            for v_ax in self.coords_ve.axes:
                try:
                    x_ax = self.coords_x.get_axis(v_ax.coordinate.type)
                    v_mult = v_ax.get_xmultiply_mpo()
                    v_dict[x_ax.coordinate] = self.fe.grid.make_mpo_ndim({v_ax:v_mult})
                except KeyError:
                    pass
            self.velocities_e = self.fe.create_like_vector(v_dict)
            self.velocities_e.name = 'vel'

        if self.fi is not None:
            v_dict = {}
            for v_ax in self.coords_vi.axes:
                try:
                    x_ax = self.coords_x.get_axis(v_ax.coordinate.type)
                    v_mult = v_ax.get_xmultiply_mpo()
                    v_dict[x_ax.coordinate] = self.fi.grid.make_mpo_ndim({v_ax:v_mult})
                except KeyError:
                    pass
            self.velocities_i = self.fi.create_like_vector(v_dict)
            self.velocities_i.name = 'vel'


        ## save initial normalization for normalization
        self.init_norm_i = None
        self.init_norm_e = None
        if normalize:
            self.init_norm_i = self.fi.norm() if self.fi is not None else None
            self.init_norm_e = self.fe.norm() if self.fe is not None else None

        ## define for linearized Vlasov equation
        self.df0dv_e = f0_gradv_e
        self.df0dv_i = f0_gradv_i

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


        self.is_electrostatic = is_electrostatic
        self.evolve_ion = evolve_ion
        self.upwind = upwind

        self.compress_F = True
        self.compress_F_opts = {'max_bond':None, 'cutoff':CUTOFF, 'cutoff_mode':CUTOFF_MODE}

        self.collision = CollisionConfiguration(None)

        ## divergence cleaning params
        self.chi = 1
        self.gamma = 1


    @property
    def fe(self) -> Optional['ScalarField']:
        try:   return self.fields[self.names['fe']]
        except KeyError:   return None

    @fe.setter
    def fe(self,new_field: 'ScalarField'):
        self.fields[self.names['fe']] = new_field


    @property
    def fi(self) -> Optional['ScalarField']:
        try:   return self.fields[self.names['fi']]
        except KeyError:   return None

    @fi.setter
    def fi(self,new_field: 'ScalarField'):
        self.fields[self.names['fi']] = new_field


    @property
    def E(self) -> Optional['Field']:
        try:   return self.fields[self.names['E']]
        except KeyError:   return None

    @E.setter
    def E(self,new_field: 'Field'):
        self.fields[self.names['E']] = new_field


    @property
    def B(self) -> Optional['Field']:
        try:   return self.fields[self.names['B']]
        except KeyError:   return None

    @B.setter
    def B(self,new_field: 'Field'):
        self.fields[self.names['B']] = new_field


    @property
    def phi(self) -> Optional['ScalarField']:
        try:   return self.fields[self.names['phi']]
        except KeyError:   return None

    @phi.setter
    def phi(self,new_field: 'ScalarField'):
        self.fields[self.names['phi']] = new_field


    @property
    def psi(self) -> Optional['ScalarField']:
        try:   return self.fields[self.names['psi']]
        except KeyError:   return None

    @psi.setter
    def psi(self,new_field: 'ScalarField'):
        self.fields[self.names['psi']] = new_field


    def create_like(self, *new_fields, recalc=True):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 6:
            new_fields = new_fields + (None,) *(6-len(new_fields))

        dist_e, dist_i, E, B, phi, psi = new_fields[:6]
        if recalc:
            new_system = VlasovMaxwell(dist_e, dist_i, E, B, phi=phi, psi=psi,
                                       coords_x=self.coords_x, coords_ve=self.coords_ve, coords_vi=self.coords_vi,
                                       elc_params=self.elc_params, ion_params=self.ion_params,
                                       # field_compress_config=self.field_compress_config, compress_levels=self._comp_levels,
                                       f0_gradv_e=self.df0dv_e, f0_gradv_i=self.df0dv_i,
                                       is_electrostatic=self.is_electrostatic, evolve_ion=self.evolve_ion,
                                       normalize = self.do_normalization, upwind=self.upwind, te_order=self.te_order,
                                       compress_levels=self._comp_levels,
                                       )
        else:
            new_system = VlasovMaxwell(dist_e, dist_i, E, B, phi=phi, psi=psi,
                                       coords_x=self.coords_x, coords_ve=self.coords_ve, coords_vi=self.coords_vi,
                                       elc_params=self.elc_params, ion_params=self.ion_params,
                                       # field_compress_config=self.field_compress_config, compress_levels=self._comp_levels,
                                       f0_gradv_e=self.df0dv_e, f0_gradv_i=self.df0dv_i,
                                       is_electrostatic=self.is_electrostatic, evolve_ion=self.evolve_ion,
                                       normalize=False, upwind=self.upwind, te_order=self.te_order,
                                       compress_levels=self._comp_levels,
                                       )
            new_system.do_normalization = self.do_normalization
            new_system.init_norm_e = self.init_norm_e
            new_system.init_norm_i = self.init_norm_i

        if dist_e is None:      new_system.names['fe'] = self.names['fe']
        if dist_i is None:      new_system.names['fi'] = self.names['fi']
        if E is None:           new_system.names['E'] = self.names['E']
        if B is None:           new_system.names['B'] = self.names['B']
        if phi is None:         new_system.names['phi'] = self.names['phi']
        if psi is None:         new_system.names['psi'] = self.names['psi']

        new_system.collision = self.collision

        new_system.compress_E = self.compress_F
        new_system.E_compress_opts = self.compress_F_opts
        return new_system

    # def copy(self):
    #     """ create a new system like this with new fields
    #     """
    #     fe = self.fe.copy() if self.fe is not None else None
    #     fi = self.fi.copy() if self.fi is not None else None
    #     E = self.E.copy() if self.E is not None else None
    #     B = self.B.copy() if self.B is not None else None
    #     phi = self.phi.copy() if self.phi is not None else None
    #     psi = self.psi.copy() if self.psi is not None else None
    #     new_system = self.create_like(fe,fi,E,B,phi=phi,psi=psi)
    #     return new_system


    def normalize(self,target_e=None,target_i=None):
        """ normalized density field
        """
        # if target is None:   target = self.fi.integrate('all')[0] if self.fi is not None else 1.0

        # if self.fe is None:   return None

        # distribution = self.fe   ## the field
        # norm = distribution.integrate('all')[0]
        # distribution.scalar_multiply( target/norm, inplace=True)
        # # helper.scalar_multiply(distribution, target/norm, inplace=True)

        # return norm

        if self.fi is not None:
            if target_i is None:
                target_i = self.init_norm_i if self.init_norm_i is not None else 1.0
            norm_i = self.fi.norm()   # integrate(0)
            self.fi.scalar_multiply( target_i/norm_i, inplace=True )

        if self.fe is not None:
            if target_e is None:
                target_e = self.init_norm_e if self.init_norm_e is not None else 1.0
            norm = self.fe.norm()     # integrate(0)
            self.fe.scalar_multiply( target_e/norm, inplace=True)
        return 


    def get_time_derivative_operator(self,compress=1,is_ion=False):
        """ dF/dt = G[f(t)].  Returns G
        """
        raise NotImplementedError

        x_dim = v_dim = self.grid.ndim // 2

        convective_term = None
        for i in range(x_dim):
            tmp = self.fe_grid.get_mpo_firstderivative(i).copy()
            tmp = helper.scalar_multiply(tmp, 1./self.grid.dx[i])
            tmp = helper.apply( self.velocities[i], tmp )
            convective_term = helper.add_MPO(convective_term,tmp,compress=max(0,compress-1))
        helper.scalar_multiply(grads,-1,inplace=True)

        if self.B is not None:  em_term = velocity.cross_product(self[self.B], compress_level=compress)
        else:                   em_term = None
        if self.E_func is not None:  em_term = self.E_func.add(em_term)
        else:                   em_term = None

        if is_ion:    em_term.scalar_multiply(self.params['e']/self.params['mass_i']*-1, inplace=True)
        else:         em_term.scalar_multiply(self.params['e']/self.params['mass_e']   , inplace=True)

        if em_term.ndim <= self.fe.grid.ndim:
            em_term = em_term.pad_to_new_grid(new_grid=self.fe.grid)

        lorentz_term = None
        for i in range(v_dim):
            tmp = self.fe.grid.get_mpo_secondderivative(i+x_dim,i+x_dim).copy()     # velocity derivative
            tmp = helper.scalar_multiply(tmp, 1./self.grid.dx[i+x_dim])
            try:               tmp = helper.apply( em_term[i], tmp )
            except(KeyError):  continue
            lorentz_term = helper.add_MPO(lorentz_term,tmp,compress=max(0,compress-1))

        tot_op = helper.add_MPO(convective_term, lorentz_term, compress=compress)

        return tot_op  ## an MPO


    def compute_force_term(self, compress_level=1, compress_level1=0, is_ion=False):
        """ compute EM force:  q/m (E + v x B)
        """
        velocities = self.velocities_i if is_ion else self.velocities_e  ## MPOs; i think it should still work
        matl_params = self.ion_params if is_ion else self.elc_params

        em_term = velocities.cross_product(self.B, compress_level=compress_level1) if self.B is not None else None
        if self.E is not None:
            em_term = self.E.add(em_term, compress_level=compress_level1)

        # plt.figure()
        # x_axes = self.coords_x.axes
        # plt.plot(em_term.get_comp_data(x_axes[0].axID),label='em')
        # plt.legend()
        # plt.show()

        if em_term is not None:
            em_term.scalar_multiply(matl_params.e * matl_params.Z / matl_params.mass, inplace=True)
            if compress_level:
                em_term.compress(compress_opts=self.compress_F_opts, inplace=True)

        # print('E + v x B term', em_term.components)

        return em_term


    def compute_charge_density(self, compress=1):
        """ compute n(x) = \integ dv \sum_s q_s f_s
        """
        charge_density = None

        ve_axes = self.coords_ve.axes if self.fe is not None else None
        vi_axes = self.coords_vi.axes if self.fi is not None else None

        if self.fe is not None:

            e_density = self.fe.integrate(axes=ve_axes, new_grid=self.E.grid,
                                          new_ax_deriv_configs=self.fe.component.ax_deriv_configs)
            e_density.scalar_multiply(self.elc_params.Z * self.elc_params.e / self.elc_params.eps0, inplace=True)
            charge_density = e_density

        if self.fi is not None:

            i_density = self.fi.integrate(axes=vi_axes, new_grid=self.E.grid,
                                          new_ax_deriv_configs=self.fe.component.ax_deriv_configs)
            i_density.scalar_multiply(self.ion_params.Z * self.ion_params.e / self.ion_params.eps0)
            if charge_density is not None:
                charge_density.add(i_density, inplace=True, compress_level=compress)
            else:
                charge_density = i_density.compress()
        else:
            if charge_density is not None:
                charge_density.compress(inplace=True, compress_level=compress)

        # plt.figure()
        # print(type(charge_density))
        # charge_data = charge_density.get_comp_data()
        # plt.plot(charge_data)
        # plt.plot('charge')

        # print('n(x) deriv config', charge_density.component.ax_deriv_configs)

        return charge_density


    def compute_current(self, compress=1, compress1=0):
        """ compute j(x) = \integ dv \sum_s q_s v_s f_s
        """
        j = None

        if self.fe is not None:

            ve_axes = self.coords_ve.axes
            je = self.fe.xmultiply(ve_axes, compress_level=compress1)

            # plt.figure()
            # plt.imshow(self.fe.get_comp_data())

            # plt.figure()
            # plt.imshow(je.get_field_data(comps=0))

            # plt.figure()
            # plt.plot(np.sum(je.get_field_data(comps=0),axis=1)*self.fe.grid.dx[1])

            je = je.integrate(axes=ve_axes, new_grid=self.E.grid,
                              new_ax_deriv_configs=self.fe.component.ax_deriv_configs)
            je.replace_compIDs({v_ax.coordinate: self.coords_x.get_coord(v_ax.coordinate.type) for v_ax in ve_axes})
            je.scalar_multiply(self.elc_params.Z * self.elc_params.e, inplace=True)

            # plt.plot(je.get_field_data(comps=0),'--')
            # plt.show()
            # print('je',je.get_field_data(comps=0)[0])

            j = je.copy()

        if self.fi is not None:

            vi_axes = self.coords_vi.axes
            # ji = self.fi.elemental_multiply(self.velocities_i[0],compress=compress)
            ji = self.fi.xmultiply(vi_axes, compress_level=compress1)

            # plt.figure()
            # plt.imshow(self.fi.get_field_data(comps=0))

            # plt.figure()
            # plt.imshow(ji.get_field_data(comps=0))

            # plt.figure()
            # ji_ = np.sum(ji.get_comp_data(x_axes[0].axID),axis=1)*vi_axes[0].dx
            # plt.plot(ji_)
            # print('ji',ji_[0])

            ji = ji.integrate(axes=vi_axes, new_grid=self.E.grid,
                              new_ax_deriv_configs=self.fi.component.ax_deriv_configs)
            ji.scalar_multiply(self.ion_params.Z * self.ion_params.e)
            ji.replace_compIDs({v_ax.coordinate: self.coords_x.get_coord(v_ax.coordinate.type) for v_ax in vi_axes})
            # print('ji',ji.get_field_data(comps=0)[0])

            # plt.figure()
            # plt.plot(ji.get_comp_data(x_axes[0].axID),'--',label='ion')
            # plt.plot(je.get_comp_data(x_axes[0].axID),'--',label='elec')
            # plt.title('current')

            if j is not None:
                j.add(ji, compress_level=compress, inplace=True)
            else:
                j = ji

            # plt.plot(j.get_comp_data(x_axes[0].axID), ':', label='total')
            # plt.legend()

            # print('j deriv config', j[x_axes[0].axID].ax_deriv_configs[x_axes[0]].bc)

        return j


    def calculate_time_derivative(self, compress_level=0, compress_level1=0, verbose_plot=False) -> 'VlasovMaxwell':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs
            
            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        dFdt_e = self._calculate_time_derivative_f(compress=compress_level, compress1=compress_level1)
        dFdt_i = self._calculate_time_derivative_f(compress=compress_level, compress1=compress_level1, is_ion=True) \
                      if self.fi is not None and self.evolve_ion else None
        dFdt_B = self._calculate_time_derivative_B(compress=compress_level, compress1=compress_level1) \
                      if self.E is not None else None
        dFdt_E = self._calculate_time_derivative_E(compress=compress_level, compress1=compress_level1)
        dFdt_psi = self._calculate_time_derivative_psi(compress=compress_level, compress1=compress_level1)
        dFdt_phi = self._calculate_time_derivative_phi(compress=compress_level, compress1=compress_level1)

        # dFdt_B = None
        # dFdt_E = None
        # dFdt_phi = None
        # dFdt_psi = None

        # print('time deriv', dFdt_e, dFdt_i)

        # if dFdt_e is not None:
        #     dfdt_e_0 = dFdt_e.get_comp_data()
        #     if dfdt_e_0 is not None:
        #         plt.figure()
        #         plt.imshow(dfdt_e_0)
        #         plt.colorbar()
        #         plt.title('fe')
        #     plt.show()
        #
        # if dFdt_i is not None:
        #     dfdt_i_0 = dFdt_i.get_comp_data()
        #     if dfdt_i_0 is not None:
        #         plt.figure()
        #         plt.imshow(dfdt_i_0)
        #         plt.colorbar()
        #         plt.title('fi')
        #     plt.show()

        # X, = self.coords_x.coords
        #
        # if dFdt_B is not None:
        #     dfdt_b_x = dFdt_B.get_comp_data(X)
        #     # dfdt_b_y = dFdt_B.get_comp_data(Y)
        #     # dfdt_b_z = dFdt_B.get_comp_data(Z)
        #     if dfdt_b_x is not None:
        #         plt.figure()
        #         plt.plot(dfdt_b_x)
        #         # plt.imshow(dfdt_b_x)
        #         # plt.colorbar()
        #         plt.title('Bx')
        #     # if dfdt_b_y is not None:
        #     #     plt.figure()
        #     #     plt.imshow(dfdt_b_y)
        #     #     plt.colorbar()
        #     #     plt.title('By')
        #     # if dfdt_b_z is not None:
        #     #     plt.figure()
        #     #     plt.imshow(dfdt_b_z)
        #     #     plt.colorbar()
        #     #     plt.title('Bz')
        #     plt.show()
        #
        # if dFdt_E is not None:
        #     dfdt_e_x = dFdt_E.get_comp_data(X)
        #     # dfdt_e_y = dFdt_E.get_comp_data(Y)
        #     # dfdt_e_z = dFdt_E.get_comp_data(Z)
        #     if dfdt_e_x is not None:
        #         plt.figure()
        #         plt.plot(dfdt_e_x)
        #         plt.plot(self.E.get_comp_data(X))
        #         # plt.imshow(dfdt_e_x)
        #         # plt.colorbar()
        #         plt.title('Ex')
        #     # if dfdt_e_y is not None:
        #     #     plt.figure()
        #     #     plt.imshow(dfdt_e_y)
        #     #     plt.colorbar()
        #     #     plt.title('Ey')
        #     # if dfdt_e_z is not None:
        #     #     plt.figure()
        #     #     plt.imshow(dfdt_e_z)
        #     #     plt.colorbar()
        #     #     plt.title('Ez')
        #     plt.show()

        # if dFdt_phi is not None:
        #     dfdt_phi_dat = dFdt_phi.get_field_data(comps=0)
        #     plt.figure()
        #     plt.plot(dfdt_phi_dat)
        #     # plt.imshow(dfdt_phi_dat)
        #     # plt.colorbar()
        #     plt.title('phi')
        #     plt.show()

        # if dFdt_psi is not None:
        #     dfdt_psi_dat = dFdt_psi.get_field_data(comps=0)
        #     plt.plot(dfdt_psi_dat)
        #     # plt.imshow(dfdt_psi_dat)
        #     # plt.colorbar()
        #     plt.title('psi')
        #     plt.show()

        # print('d/dt B',dFdt_B, dFdt_B.name)
        # print('d/dt E',dFdt_E)
        # print('d/dt psi',dFdt_psi)
        # print('d/dt phi',dFdt_phi)

        dFdt = self.create_like(dFdt_e, dFdt_i, dFdt_E, dFdt_B, dFdt_phi, dFdt_psi)

        return dFdt


    def _calculate_time_derivative_f(self, compress:int = 1, compress1:int = 0, compress2: int = 0, is_ion=False) \
            -> Optional['Field']:
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            df0dv:   grad_v f0 if doing linearized vlasov. list or dict of each velocity component
        """
        dist = self.fi if is_ion else self.fe

        if dist is None:  return None

        x_axes = self.coords_x.axes
        v_coordsys = self.coords_vi if is_ion else self.coords_ve
        v_axes_dict = {ax: v_coordsys.get_axis(ax.coordinate.type) for ax in x_axes}
        v_axes = [v_axes_dict.get(ax, None) for ax in x_axes]

        # for coord in [ax.coord_type for ax in x_axes]:
        #     try:
        #         v_axes += [v_coordsys.coords_ax[coord]]
        #     except KeyError:
        #         v_axes += [None]
        # v_axes_dict = v_coordsys.coord_axes
        # v_axes = [v_axes_dict.get(ax.coordinate.type,None) for ax in x_axes]


        df0dv = self.df0dv_i if is_ion else self.df0dv_e
        if self.upwind:
            x_grads = dist.gradient(deriv_axes=x_axes, upwind_axes=v_axes_dict, compress_level=compress1)
        else:
            x_grads = dist.gradient(deriv_axes=x_axes, upwind_axes=None, compress_level=compress1)
        v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress1) if df0dv is None else df0dv

        ## intermediate plots
        dist_data = dist.get_comp_data() #  - self.fi.get_comp_data()
        xg_data = x_grads.get_comp_data(x_axes[0].coordinate)
        vg_data = v_grads.get_comp_data(x_axes[0].coordinate)

        # plt.figure()
        # plt.imshow(dist_data)
        # plt.xlabel('v')
        # plt.ylabel('x')
        # plt.title(f'dist, is ion {is_ion}')
        # plt.colorbar()

        # plt.figure()
        # plt.plot(np.sum(dist_data,axis=1)*self.V_grid._dx)
        # plt.xlabel('x')

        # plt.figure()
        # plt.plot(dist_data[0,:])
        # plt.ylabel('v')

        # plt.show()

        # plt.figure()
        # # plt.imshow(x_grads.get_field_data(comps=0))
        # plt.imshow(v_grads.get_field_data(comps=1))
        # plt.xlabel('v')
        # plt.ylabel('x')
        # plt.legend()
        # plt.show()

        # plt.figure()
        # plt.plot(x_grads.get_field_data(comps=0)[:,0],label='gradx')
        # plt.plot(v_grads.get_field_data(comps=1)[0,:],label='gradv')
        # plt.xlabel('v')
        # plt.legend()
        # plt.show()
        # exit()

        ## convective term
        # velocities = self.velocities_i if is_ion else self.velocities_e
        # convective_term = velocities.dot(x_grads)  ## an MPS; update now field
        # convective_term = x_grads.dot(velocities)  ## an MPS; update now field
        convective_term = x_grads.xdot(v_axes, compress_level=compress1)
        convective_term.scalar_multiply(-1, inplace=True)

        # plt.figure()
        # plt.imshow(convective_term.get_comp_data())
        # plt.colorbar()
        # plt.title('convective term')

        ## (EM) force term
        em_term = self.compute_force_term(compress_level=compress1, compress_level1=0, is_ion=is_ion)
        em_term.scalar_multiply(-1, inplace=True)

        # velocities = self.velocities_i if is_ion else self.velocities_e  ## MPOs; i think it should still work
        # # print(velocities.components)
        # # print(self.B.components)
        #
        #
        # em_term = velocities.cross_product(self.B, compress_level=compress) if self.B is not None else None
        # if self.E is not None:
        #     em_term = self.E.add(em_term, compress_level=compress)
        #
        # if em_term is not None:
        #
        #     if is_ion:  em_term.scalar_multiply(self.matl_params.e*self.matl_params.Zi/self.matl_params.mass_i *-1,
        #                                         inplace=True)
        #     else:       em_term.scalar_multiply(self.matl_params.e/self.matl_params.mass_e,inplace=True)
        #
        #     if self.compress_F:
        #         em_term.compress(compress_opts=self.compress_F_opts, inplace=True)

        if em_term is not None:
            em_term = em_term.pad_to_new_grid(new_grid=dist.grid)
            lorentz_term = em_term.dot(v_grads, other_comps=[ax.coordinate for ax in v_axes],
                                       compress_level=compress1, inner_compress_level=0)
        else:
            lorentz_term = None

        # plt.figure()
        # plt.imshow(lorentz_term.get_comp_data())
        # plt.title('lorentz')
        # plt.colorbar()
        # plt.show()

        dFdt_s = convective_term.add(lorentz_term, compress_level=0)
        dFdt_s.name = self.fi.name if is_ion else self.fe.name

        ### get collisions
        coll_field = self.get_collision_term(is_ion=is_ion, v_axes=v_axes, v_grads=v_grads,
                                             compress1=compress1, compress2=compress2)
        dFdt_s.add(coll_field, inplace=True, compress_level=0)

        if compress:
            # print('deriv final comp',compress)
            dFdt_s.compress()

        return dFdt_s


    def _calculate_time_derivative_E(self,compress=1,compress1=0,compress2=0):
        """ 
        e0*mu0 dE/dt - curl(B) + chi*grad(phi) = -mu0 J
        J = sum_s qs ns vs
        """
        x_axes = self.coords_x.axes

        ## current
        j = self.compute_current(compress=compress1, compress1=0)

        if j is not None:

            # plt.figure()
            # plt.plot(je.get_field_data(comps=0),label='je')
            # plt.plot(ji.get_field_data(comps=0),label='ji')
            # plt.legend()

            if self.is_electrostatic and len(x_axes)==1:
                X = x_axes[0]
                # j_ghost_mps = helper.sum_tensornetwork(like_tn=j[X.coordinate].data)
                # j_ghost_mps.view_like(j[X.coordinate].data, inplace=True)
                jg_const = j.norm() * -1/(X.npts * X.dx)  # norm integrates over all axes -> int
                # helper.scalar_multiply(j_ghost_mps,jg_const,inplace=True)
                # j_ghost = j.create_like({x_axes[0].coordinate:j_ghost_mps})
                # print('j',j[X.coordinate].data, j_ghost[X.coordinate].data)
                # j.add(j_ghost,inplace=True)
                j.scalar_add(jg_const, inplace=True)

                # plt.figure()
                # plt.plot(j.get_field_data(comps=0),label='total')
                # plt.plot(j_ghost.get_field_data(comps=0),label='ghost')
                # plt.legend()

            # plt.show()
            j.scalar_multiply(-1 * self.matl_params.mu0, inplace=True)

        dEdt = j
        # print('dEdt', dEdt[x_axes[0].coordinate].ax_deriv_configs[x_axes[0]].bc)

        ## curl(B)
        if self.B is not None:
            # print('B', self.B[x_axes[0].coordinate].ax_deriv_configs[x_axes[0]].bc)
            curlB = self.B.curl(self.coords_x, compress_level=compress1, inner_compress_level=compress2)
            # print('curlB',curlB,curlB.components.keys())
            # curlB.scalar_multiply(self.params['c']**2, inplace=True)
            if dEdt is not None:  dEdt.add(curlB,inplace=True, compress_level=0)
            else:  dEdt = curlB

        ## correction
        if self.phi is not None:
            grad_phi = self.phi.gradient(compress_level=compress1)
            grad_phi.scalar_multiply(-1*self.chi,inplace=True)
            dEdt.add( grad_phi, inplace=True, compress_level=0)

        if dEdt is not None:
            if compress:    dEdt.compress(inplace=True, compress_level=compress)
            dEdt.scalar_multiply(self.elc_params.c ** 2, inplace=True)
            dEdt.name = self.E.name if self.E is not None else 'E'


        # print(dEdt.components)
        # for X in dEdt.components.keys():
        #     print(dEdt[X].data)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(dEdt.get_comp_data('Z'))
        # plt.title('dE/dt')
        # plt.show()

        return dEdt


    def _calculate_time_derivative_B(self,compress=1, compress1=0, compress2=0):
        """ 
        dB/dt + curl(E) + gamma*grad(psi) = 0
        """
        dBdt = None

        ## current
        if self.E is not None:
            curlE = self.E.curl(self.coords_x, compress_level=compress1, inner_compress_level=compress2)
            dBdt = curlE

        if self.psi is not None:
            grad_psi = self.psi.gradient(compress_level=compress1)
            grad_psi.scalar_multiply(self.gamma, inplace=True)

            # plt.figure()
            # plt.imshow(self.psi.get_field_data(comps=0))
            # plt.colorbar()
            # plt.title('psi')
            # plt.show()

            # plt.figure()
            # plt.imshow(grad_psi.get_field_data(comps=0))
            # plt.colorbar()
            # plt.title('grad psi x')
            # plt.show()

            # plt.figure()
            # plt.imshow(grad_psi.get_field_data(comps=1))
            # plt.colorbar()
            # plt.title('grad psi y')
            # plt.show()

            # # plt.figure()
            # # plt.imshow(grad_psi.get_field_data(comps=2))
            # # plt.colorbar()
            # # plt.title('grad psi z')
            # # plt.show()

            dBdt = dBdt.add(grad_psi,inplace=True,compress_level=0) if dBdt is not None else grad_psi

        if dBdt is not None:
            if compress:
                dBdt.compress(inplace=True, compress_level=compress)
            dBdt.scalar_multiply(-1, inplace=True)
            dBdt.name = self.B.name if self.B is not None else 'B'

        # X, Y = dBdt.components.keys()
        # print(dBdt.components, dBdt.components[X].data, dBdt.components[Y].data)
        # exit()

        return dBdt


    def _calculate_time_derivative_phi(self, compress=1, compress1=0, compress2=0):
        """ dphi/dt + div(E) = rho/eps0
        """
        charge_density = self.compute_charge_density(compress=compress1)
        if charge_density is not None:
            charge_density.scalar_multiply(1. / self.matl_params.eps0, inplace=True)
        dPhidt = charge_density

        if self.E is not None:
            divE = self.E.divergence(self.coords_x, compress_level=compress1)
            if divE is not None:    divE.scalar_multiply(-1,inplace=True)
            if dPhidt is not None:  dPhidt.add(divE,inplace=True,compress_level=0)
            else:                   dPhidt = divE
        
        if dPhidt is not None:
            dPhidt.scalar_multiply(self.chi, inplace=True)
            if compress:
                dPhidt.compress(inplace=True, compress_level=compress)
            dPhidt.name = self.phi.name if self.phi is not None else 'phi'

        return dPhidt


    def _calculate_time_derivative_psi(self,compress=1, compress1=0, compress2=0):
        """ eps0*mu0/gamma dpsi/dt + div(B) = 0
        """
        if self.B is not None:
            divB = self.B.divergence(self.coords_x, compress_level=compress, inner_compress_level=compress2)

            if divB is not None:
                divB.scalar_multiply(-self.gamma * self.matl_params.c ** 2, inplace=True)
                divB.name = self.psi.name if self.psi is not None else 'psi'
            return divB
        else:
            return None


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



