from setup.configs import *
import helper_TE
import helper_dlr

if TYPE_CHECKING:
    from axis import Axis
    from gridTN import GridTN
    from field import Field, ScalarField


""" class defining PDE system, containing info about
    - geometry of system (cart[esian], sph[erical], cyl[indrical])
    - fields
      = vector fields, represented as ndim distinct MPS
      = scalar fields, represented as an MPS
    - discretization of spatial grid, represented as MPS
      = basis functions Phi(x) [default Phi_i(x) = delta(x-x_i)]
"""

class FieldCompressionConfiguration:
    """ class containing compression parameters for different levels
        for each field in the
    """
    def __init__(self, field_configs: dict[str, CompressionConfiguration]=None):
        if field_configs is None:
            self.field_configs = {}
        else:
            self.field_configs = field_configs.copy()

    def __getitem__(self, field_name: str):
        return self.field_configs[field_name]

    def __setitem__(self, field_name: str, config: CompressionConfiguration):
        self.field_configs[field_name] = config

    def set_field_compress_opts(self, field_name, compress_level: int, max_bond=None, cutoff_mode=CUTOFF_MODE,
                                cutoff=CUTOFF):
        try:
            config = self.field_configs[field_name]
        except KeyError:
            config = CompressionConfiguration()
            self.field_configs[field_name] = config

        config.set_compress_opts(compress_level, max_bond=max_bond, cutoff_mode=cutoff_mode, cutoff=cutoff)

    def get_field_compress_opts(self, field_name, compress_level: int) -> dict:
        return self.field_configs[field_name][compress_level]



class PDE_system:

    def __init__(self,*fields: Union['Field','ScalarField'], field_names: Sequence[str] = None, normalize: bool = True,
                 background_pde: Optional['PDE_system'] = None, evolve_background: bool = False,
                 te_order: int = 1, # field_compress_config: Optional[CompressionConfiguration] = None,
                 compress_levels = None,
                 # init_compress_opts=None, te_compress_opts=None
                 ):
        """ npts:  number of discretized points along each axis
            ndim:  dimension of system
            vec_fields:  list of strs to act as keys indexing vector fields
            scalar_fields:  list of strs to act as keys indexing scalar fields
            grid_type:  str describing grid_structure
        """

        if compress_levels is None:
            compress_levels = list(range(1,10))     # some arbitrary number; compress at all levels
        self._comp_levels = compress_levels

        self._fields: dict[Any, Optional[Field]] = {field.name: field for field in fields if field is not None}
        self.field_names = [field.name for field in fields if field is not None] \
                              if field_names is None else field_names

        ## pad system with 'None' fields as needed, prescribed by field_names
        for fn in self.field_names:
            if fn not in self._fields:  self._fields[fn] = None

        self.te_order = te_order
        self.time = None    # can use to keep track of time

        self.do_normalization = normalize

        self.background_pde = background_pde
        self.evolve_background = evolve_background if background_pde is not None else False

        ## keep history for two-step TE systems
        self.state_history: dict[int, 'PDE_system'] = {}
        self.deriv_history: dict[int, 'PDE_system'] = {}


    def __getitem__(self,field_name):
        """ return ith component of the field. for convenience
        """
        return self.get_field(field_name)   ## allow KeyError to be raised


    def __setitem__(self, field_name, new_field):
        """ set ith component of the field. for convenience
            new_field is a Field object
        """
        self.set_field(field_name, new_field)

    @property
    def fields(self):
        return {fn: self.get_field(fn) for fn in self.field_names}
        # return self._fields

    def get_field(self, field_name):
        """ return ith component of the field. for convenience
        """
        return self._fields[field_name]

    def set_field(self, field_name, new_field):
        """ set ith component of the field. for convenience
                    new_field is a Field object
        """
        assert (isinstance(new_field, Field) or new_field is None), 'new_field must be Field object or None'
        self._fields[field_name] = new_field


    def _get_compress_levels(self, compress_level, num_levels):

        if compress_level == 0:
            return [0] * num_levels

        comp_levels = self._comp_levels[compress_level-1:compress_level-1+num_levels]
        if len(comp_levels) < num_levels:
            comp_levels += [0]*(num_levels-len(comp_levels))
        return comp_levels


    # def initialize_fields(self,fields_dict):
    #     """ fields_dict: dictionary indexed by field name, yielding dictionaries of field components
    #     """
    #     for name, field in fields_dict.items():
    #         self.fields[name] = field


    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this without defining the fields
        """
        new_system = self.__class__(*new_fields, field_names=self.field_names,
                                    normalize=self.do_normalization,te_order=self.te_order,
                                    compress_levels=self._comp_levels)
                                    # field_compress_config=self.field_compress_config)
        new_system.time = self.time
        return new_system


    def copy(self):
        """ copy self including the fields
        """
        copy_fields = [self.get_field(k).copy() if self.get_field(k) is not None else None
                       for k in self.field_names]
        new_system = self.create_like(*copy_fields, recalc=False)
        new_system.state_history = self.state_history
        new_system.deriv_history = self.deriv_history
        return new_system


    def normalize(self):
        """ normalize fields during time evolution
        """
        raise NotImplementedError


    def __add__(self,other):
        return self.add(other)


    def __mul__(self,scalar):
        return self.scalar_multiply(scalar)


    def add(self, other: 'PDE_system', compress_level: int = 0, inplace=False):
        """ add fields of other to self. other can be a dict with the correct keys
            requires self and other to have the same names for each field
        """
        new_sys = self if inplace else self.copy()

        if other is None:   return new_sys

        # all_fields = set.union(set(self.fields.keys()), set(other.fields.keys()))
        all_fields = set.union(set(self.field_names), set(other.field_names))

        for k in all_fields:

            try:
                other_field = other[k]
                if other_field is None:  raise KeyError
            except KeyError:
                continue       ## other[k] doesn't add anything

            try:
                field = new_sys[k]
                if field is None:  raise KeyError
            except KeyError:                    ## self[k] doesn't exist
                new_sys[k] = other_field        ## replace with field in other (must exist)
                continue

            # new_sys[k] = field.add(other_field,compress,inplace=False)
            field.add(other_field, inplace=True, compress_level=compress_level)

        return new_sys


    def scalar_multiply(self, const: Numeric, field_keys: Sequence[Any] = None, inplace=False):
        if field_keys is None:   field_keys = self.field_names
        new_sys = self if inplace else self.copy()
        for k in field_keys:
            if new_sys[k] is None:  continue
            new_sys[k].scalar_multiply(const,inplace=True)
        return new_sys


    def elemental_multiply(self, mps_vec: qtn.MatrixProductState, field_keys: Sequence[Any] = None,
                           compress_level: int =0, inplace=False):
        if field_keys is None:   field_keys = self.field_names
        new_sys = self if inplace else self.copy()
        for k in field_keys:
            f = new_sys.get_field(k)
            if f is None:  continue
            f.elemental_multiply(mps_vec, inplace=True, compress_level=compress_level)
        return new_sys


    def distances(self, other: 'PDE_system', compress_level: int = 0, total=False, normalize=False,
                  field_norms:dict=None):
        """ add fields of other to self. other can be a dict with the correct keys
            requires self and other to have the same names for each field
        """
        fields_distance = {}
        # all_fields = set.union(set(self.fields.keys()), set(other.fields.keys()))
        all_fields = set.union(set(self.field_names), set(other.field_names))

        for k in all_fields:

            try:
                other_field = other[k]
            except KeyError:
                other_field = None

            try:
                field = self[k]
            except KeyError:
                field = None

            if field is None and other_field is None:
                continue
            elif field is None:
                if normalize:
                    fields_distance[k] = np.array([np.inf]*other_field.ncomp)
                else:
                    comp_norms = other_field.norms()
                    fields_distance[k] = np.array([comp_norms[compID] for compID in other_field.componentIDs])
            elif other_field is None:
                fields_distance[k] = field.norms()
                if normalize:
                    fields_distance[k] = {compID: 1.0 for compID in field.componentIDs}
                else:
                    comp_norms = field.norms()
                    fields_distance[k] = comp_norms
            else:
                comp_norms = None if field_norms is None else field_norms[k]
                distance_vals = field.distance(other_field, normalize=normalize, comp_norms=comp_norms)

                fields_distance[k] = distance_vals
        # print('diffs', fields_distance)

        if total:
            tot_diff = 0.0
            for k in all_fields:
                diff_vals = np.array([val for compID, val in fields_distance[k].items()])
                tot_diff += np.sum( diff_vals**2 )
            return np.sqrt(tot_diff)
        else:
            return fields_distance


    # def norm2(self):
    #     norm = 0
    #     for k, field in self.fields.items():
    #         norm += field.norm2()
    #     return norm


    def compress(self, inplace=True, compress_level=1, verbose=False):
        new_sys = self if inplace else self.copy()
        for k in new_sys.field_names:
            field = new_sys[k]
            if field is None:  continue
            field.compress(inplace=True, compress_level=compress_level, verbose=verbose)
        return new_sys


    # def time_evolution(self,dt,num_tsteps,order='4'):
    #     nt = 0
    #     new_state = self
    #
    #     while nt < num_tsteps:
    #         if   order=='2':  new_state = new_state.rk2(dt)
    #         elif order=='4':  new_state = new_state.rk4(dt)
    #         else:             raise(NotImplementedError)
    #         nt += 1
    #
    #     self.fields = new_state.fields
    #     return new_state


    def calculate_time_derivative(self, time=None,
                                  compress_level: int = 0, compress_level1: int = 0, compress_level2: int = 0,
                                  do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
                                  update_force = True, verbose_plot: bool =False, **kwargs) -> 'PDE_system':
        raise NotImplementedError


    def get_time_derivative_mpo(self, time=None,
                                compress_level: int = 0, compress_level1: int = 0, compress_level2: int = 0,
                                do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
                                verbose_plot: bool =False, **kwargs) -> 'PDE_system':
        raise NotImplementedError


    def next_time_step(self, dt: Numeric, deriv0: Optional['PDE_system']=None, inplace=False, compress_level: int=1,
                       max_iter = 100, err_tol=None,
                       is_first_time_step=False, is_last_time_step=False,
                       do_postprocessing=False, process_kwargs=None, verbose_plot=False, ):
        """ convenience fct btwn different TE methods?
        """
        te_order = self.te_order

        # mod_compress_opts = state0.get_te_compress_opts(compress).copy()
        # mod_compress_opts.update(compress_opts)
        if   te_order==1:
            state_t = self.euler(dt, deriv0, inplace=inplace, compress_level=compress_level, )
        elif te_order==2:
            state_t = self.rk2(dt, deriv0, compress_level=compress_level, )
        elif te_order==3:
            state_t = self.rk3(dt, deriv0, compress_level=compress_level, )
        elif te_order==4:
            state_t = self.rk4(dt, deriv0, compress_level=compress_level, verbose_plot=verbose_plot, )

        ## implicit methods
        elif te_order==-1:
            if inplace:
                raise NotImplementedError
            state_t = self.backwards_euler(dt, err_tol=err_tol, max_iter=max_iter, compress_level=compress_level,
                                           verbose_plot=verbose_plot, )
        elif te_order == -21:
            if inplace:
                raise NotImplementedError
            state_t = self.implicit_midpoint(dt, err_tol=err_tol, max_iter=max_iter, compress_level=compress_level,
                                             verbose_plot=verbose_plot, )
        elif te_order==-22:
            if inplace:
                raise NotImplementedError
            state_t = self.crank_nicolson(dt, err_tol=err_tol, max_iter=max_iter, compress_level=compress_level,
                                          verbose_plot=verbose_plot, )

        ## two-step methods. note that these require a constant time step
        elif te_order==21:
            state_t = self.two_step(dt, method='two-leap', compress_level=compress_level, verbose_plot=verbose_plot,)
        elif te_order==22:
            state_t = self.two_step(dt, method='two-adams', compress_level=compress_level, verbose_plot=verbose_plot,)
        elif te_order==23:
            state_t = self.two_step(dt, method='two-mag', compress_level=compress_level, verbose_plot=verbose_plot,)
        elif te_order==24:
            ### worse than second order AB?
            state_t = self.two_step(dt, method='two-adams3', compress_level=compress_level, verbose_plot=verbose_plot,)

        ## split-step methods
        elif te_order==31:
            state_t = self.split_step(dt, compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order == 300:
            state_t = self.split_step(dt, method_v='rk4', method_f='rk4', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order == 303:
            state_t = self.split_step(dt, method_v='rk4', method_f='mac', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order == 304:
            state_t = self.split_step(dt, method_v='rk4', method_f='lax', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order == 310:
            state_t = self.split_step(dt, method_v='SL', method_f='rk4', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order==311:
            state_t = self.split_step(dt, method_v='SL', method_f='SL', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order==312:
            state_t = self.split_step(dt, method_v='SL2', method_f='mac', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot,)
        elif te_order==313:
            state_t = self.split_step(dt, method_v='SL', method_f='mac', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order==314:
            state_t = self.split_step(dt, method_v='SL', method_f='lax', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order==316 or 3160 <= te_order < 3170:
            order_ = int(str(te_order)[3:]) if te_order > 3160 else 0
            if is_first_time_step:
                # state_t = self.rk4(dt, deriv0, compress_level=compress_level, verbose_plot=verbose_plot, )
                state_t = self.split_step(dt, method_v='SL', method_f='mac', compress_level=compress_level,
                                          is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                          verbose_plot=verbose_plot, )
            else:
                print('tdvp order', order_)
                state_t = self.split_step(dt, method_v='SL', method_f=f'tdvp{order_}', compress_level=compress_level,
                                          is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                          verbose_plot=verbose_plot, )

        elif te_order==317 or 3170 <= te_order < 3180:
            order_ = int(str(te_order)[3:]) if te_order > 3170 else 0
            if is_first_time_step:
                # state_t = self.rk4(dt, deriv0, compress_level=compress_level, verbose_plot=verbose_plot, )
                state_t = self.split_step(dt, method_v='SL', method_f='mac', compress_level=compress_level,
                                          is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                          verbose_plot=verbose_plot, )
            else:
                print('tdmrg order', order_)
                state_t = self.split_step(dt, method_v='SL', method_f=f'split-tdmrg{order_}', compress_level=compress_level,
                                          is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                          verbose_plot=verbose_plot, )

        elif te_order == 377 or 3770 <= te_order < 3780:
            order_ = int(str(te_order)[3:]) if te_order > 3770 else 0
            if is_first_time_step:
                # state_t = self.rk4(dt, deriv0, compress_level=compress_level, verbose_plot=verbose_plot, )
                state_t = self.split_step(dt, method_v='mac', method_f='mac', compress_level=compress_level,
                                          is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                          verbose_plot=verbose_plot, )
            else:
                print('tdmrg order', order_)
                state_t = self.split_step(dt, method_v=f'tdmrg{order_}', method_f=f'tdmrg{order_}',
                                          compress_level=compress_level,
                                          is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                          verbose_plot=verbose_plot, )


        elif te_order==333:
            state_t = self.split_step(dt, method_v='mac', method_f='mac', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order == 363:
            state_t = self.split_step(dt, method_v='rk2', method_f='mac', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order == 373:
            state_t = self.split_step(dt, method_v='imp-mid', method_f='mac', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order == 383:
            state_t = self.split_step(dt, method_v='imp-cn', method_f='mac', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )

        elif te_order == 403:
            state_t = self.split_step(dt, method_v='rk4', method_f='SL3,mac', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )

        elif te_order==410:
            state_t = self.split_step(dt, method_v='SL', method_f='SL3,rk4', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order==413:
            state_t = self.split_step(dt, method_v='SL', method_f='SL3,mac', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order==433:
            state_t = self.split_step(dt, method_v='mac', method_f='SL3,mac', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order==414:
            state_t = self.split_step(dt, method_v='SL', method_f='SL3,lax', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )
        elif te_order==415:
            state_t = self.split_step(dt, method_v='SL', method_f='SLd3,mac', compress_level=compress_level,
                                      is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                      verbose_plot=verbose_plot, )

        elif te_order==416 or 4160 <= te_order < 4170:
            order_ = int(str(te_order)[3:]) if te_order > 4160 else 0
            print('tdvp order', order_)
            if is_first_time_step:
                # state_t = self.rk4(dt, deriv0, compress_level=compress_level, verbose_plot=verbose_plot, )
                state_t = self.split_step(dt, method_v='SL', method_f='SL3,mac', compress_level=compress_level,
                                          is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                          verbose_plot=verbose_plot, )
            else:
                state_t = self.split_step(dt, method_v='SL', method_f=f'SL,tdvp{order_}', compress_level=compress_level,
                                          is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                          verbose_plot=verbose_plot, )

        elif te_order==417 or 4170 <= te_order < 4180:
            order_ = int(str(te_order)[3:]) if te_order > 4170 else 0
            print('tdmrg order', order_)
            if is_first_time_step:
                # state_t = self.rk4(dt, deriv0, compress_level=compress_level, verbose_plot=verbose_plot, )
                state_t = self.split_step(dt, method_v='SL', method_f='SL3,mac', compress_level=compress_level,
                                          is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                          verbose_plot=verbose_plot, )
            else:
                state_t = self.split_step(dt, method_v='SL', method_f=f'SL,tdmrg{order_}', compress_level=compress_level,
                                          is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                          verbose_plot=verbose_plot, )


        elif te_order==419 or 4190 <= te_order < 4200:
            order_ = int(str(te_order)[3:]) if te_order > 4190 else 0
            if is_first_time_step:
                # state_t = self.rk4(dt, deriv0, compress_level=compress_level, verbose_plot=verbose_plot, )
                state_t = self.split_step(dt, method_v='SL', method_f='SL3,mac', compress_level=compress_level,
                                          is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                                          verbose_plot=verbose_plot, )
            else:
                state_t = self.time_dependent_variational_principle_SL(dt, inplace=False, te_order=order_, do_adapt=True,
                                                                       is_first_time_step=is_first_time_step,
                                                                       is_last_time_step=is_last_time_step,
                                                                       compress_level=compress_level, compress_level_2=4)

        ### dynamical low rank
        elif 50 <= te_order < 60:
            order_ = int(str(te_order)[1:])
            # print('is first time step')
            if is_first_time_step:
                state_t = self.rk4(dt, deriv0, compress_level=compress_level, verbose_plot=verbose_plot, )
            else:
                state_t =  self.dynamical_low_rank(dt, inplace=False, te_order=order_, do_adapt=True,
                                                   is_first_time_step=is_first_time_step,
                                                   is_last_time_step=is_last_time_step,
                                                   compress_level=compress_level,compress_level_2=4)

        ### tdvp
        elif 60 <= te_order < 70:
            order_ = int(str(te_order)[1:])
            if is_first_time_step:
                state_t = self.rk4(dt, deriv0, compress_level=compress_level, verbose_plot=verbose_plot, )
                # state_t = self.split_step(dt, method_v='mac', method_f='mac', compress_level=compress_level,
                #                           is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                #                           verbose_plot=verbose_plot, )
            else:
                state_t = self.time_dependent_variational_principle(dt, inplace=False, te_order=order_, do_adapt=True,
                                                                    is_first_time_step=is_first_time_step,
                                                                    is_last_time_step=is_last_time_step,
                                                                    compress_level=compress_level, compress_level_2=4)

        ### tdmrg
        elif 70 <= te_order < 80:
            order_ = int(str(te_order)[1:])
            if is_first_time_step:
                state_t = self.rk4(dt, deriv0, compress_level=compress_level, verbose_plot=verbose_plot, )
                # state_t = self.split_step(dt, method_v='mac', method_f='mac', compress_level=compress_level,
                #                           is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
                #                           verbose_plot=verbose_plot, )
            else:
                state_t = self.time_dmrg(dt, inplace=False, te_order=order_, do_adapt=True,
                                         is_first_time_step=is_first_time_step,
                                         is_last_time_step=is_last_time_step,
                                         compress_level=compress_level,
                                         compress_level_2=4)

        else:
            raise NotImplementedError

        if state_t.do_normalization:
            state_t.normalize()

        # if self.evolve_background:
        #     self.background_pde = self.background_pde.next_time_step(dt, max_iter=max_iter, err_tol=err_tol,
        #                                                              is_first_time_step=is_first_time_step,
        #                                                              is_last_time_step=is_last_time_step,
        #                                                              verbose_plot=verbose_plot)

        return state_t


    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, method_v=None, method_f=None,
                   is_first_time_step = False, is_last_time_step = False, inplace=False,
                   compress_level: int = 1, verbose_plot=False, ) -> 'PDE_system':
        raise NotImplementedError

    def two_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, inplace=False,
                 method=None, compress_level: int = 1, verbose_plot=False, **deriv_kwargs) -> 'PDE_system':

        if method is None or method[:10] == 'two-adams3':
            return self.two_step_adamsbashforth3(dt, deriv0, inplace=inplace, compress_level=compress_level,
                                          verbose_plot=verbose_plot, **deriv_kwargs)
        elif method[:8] == 'two-leap':
            return self.two_step_leapfrog(dt, deriv0, inplace=inplace, compress_level=compress_level,
                                   verbose_plot=verbose_plot, **deriv_kwargs)
        elif method[:9] == 'two-adams':
            return self.two_step_adamsbashforth(dt, deriv0, inplace=inplace, compress_level=compress_level,
                                   verbose_plot=verbose_plot, **deriv_kwargs)
        elif method[:7] == 'two-mag':
            return self.two_step_magazenkov(dt, deriv0, inplace=inplace, compress_level=compress_level,
                                   verbose_plot=verbose_plot, **deriv_kwargs)
        else:
            raise NotImplementedError


    def _two_step_core(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, a2 = 1, inplace=False,
                       compress_level: int = 1, verbose_plot=False, **deriv_kwargs) -> 'PDE_system':
        """ note: cannot use adaptive time step with these methods!
        """
        new_state = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        a1 = 1 - a2
        b1 = 1./2 * (a2 + 3)
        b2 = 1./2 * (a2 - 1)

        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(time=self.time, compress_level=comp4, compress_level1=comp5,
                                                    verbose_plot=verbose_plot, **deriv_kwargs)

        prev_state = self.state_history.get(-1, None)
        if prev_state is None:
            ## initialize with RK2
            state1 = self.rk2(dt, deriv0=deriv0, compress_level=compress_level, **deriv_kwargs)
        else:
            if np.abs(a1) < 1.0e-16:
                state_sum = prev_state
            elif np.abs(a2) < 1.0e-16:
                state_sum = self.copy()
            else:
                state_sum = prev_state.scalar_multiply(a2, inplace=False)
                state_sum = state_sum.add( self.scalar_multiply(a1, inplace=False),
                                            compress_level=comp2, inplace=True )

            prev_deriv = self.deriv_history.get(-1, None)

            deriv_sum = deriv0.scalar_multiply(dt * b1, inplace=False)
            if not np.abs(b2) < 1.0e-16:
                deriv_sum.add( prev_deriv.scalar_multiply(dt * b2, inplace=False),
                               compress_level=0, inplace=True )
            state1 = state_sum.add( deriv_sum, inplace=True, compress_level=comp1 )

        ## update history
        new_state.state_history[-1] = self.copy()
        new_state.deriv_history[-1] = deriv0

        ## update new_state fields
        for fname in new_state.field_names:
            new_state.set_field(fname, state1.get_field(fname))

        return new_state

    def two_step_adamsbashforth(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, inplace=False,
                               compress_level: int = 1, verbose_plot=False, **deriv_kwargs) -> 'PDE_system':
        print('adams')
        return self._two_step_core(dt, deriv0, a2=0, inplace=inplace, compress_level=compress_level,
                                   verbose_plot=verbose_plot, **deriv_kwargs)

    def two_step_leapfrog(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, inplace=False,
                          compress_level: int = 1, verbose_plot=False, **deriv_kwargs) -> 'PDE_system':
        print('leap')
        return self._two_step_core(dt, deriv0, a2=1, inplace=inplace, compress_level=compress_level,
                                   verbose_plot=verbose_plot, **deriv_kwargs)

    def two_step_magazenkov(self, dt: Numeric, deriv0: Optional['PDE_system'] = None,
                            inplace=False, compress_level: int = 1,
                            verbose_plot=False, **deriv_kwargs) -> 'PDE_system':
        """ note that this is a four-step method; dt -> dt/2 and perform leapfrong and Adams-Bashforth
        """
        print('magzenkov')
        new_state = self if inplace else self.copy()
        new_state.two_step_leapfrog(dt/2, deriv0, inplace=True, compress_level=compress_level,
                                    verbose_plot=verbose_plot, **deriv_kwargs)
        new_state.two_step_adamsbashforth(dt/2, None, inplace=True, compress_level=compress_level,
                                    verbose_plot=verbose_plot, **deriv_kwargs)
        return new_state


    def two_step_adamsbashforth3(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, inplace=False,
                                 compress_level: int = 1, verbose_plot=False, **deriv_kwargs) -> 'PDE_system':
        print('adams3')
        new_state = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(time=self.time, compress_level=comp4, compress_level1=comp5,
                                                    verbose_plot=verbose_plot, **deriv_kwargs)

        prev_state_1 = self.state_history.get(-1, None)
        prev_state_2 = self.state_history.get(-2, None)
        if prev_state_1 is None or prev_state_2 is None:
            ## initialize with RK4
            prev_deriv_1 = self.deriv_history.get(-1, None)
            state1 = self.rk4(dt, deriv0=deriv0, compress_level=compress_level, **deriv_kwargs)
        else:
            prev_deriv_1 = self.deriv_history[-1]
            prev_deriv_2 = self.deriv_history[-2]

            deriv_sum = deriv0.scalar_multiply(dt * 23./12, inplace=False)
            deriv_sum.add(prev_deriv_1.scalar_multiply(dt * (-4./3), inplace=False),
                          compress_level=0, inplace=True)
            deriv_sum.add(prev_deriv_2.scalar_multiply(dt * 5./12, inplace=False),
                          compress_level=0, inplace=True)

            state1 = self.add(deriv_sum, inplace=False, compress_level=comp1)

        ## update history
        new_state.state_history[-2] = prev_state_1
        new_state.deriv_history[-2] = prev_deriv_1

        new_state.state_history[-1] = self.copy()
        new_state.deriv_history[-1] = deriv0

        ## update new_state fields
        for fname in self.field_names:
            new_state.set_field(fname, state1.get_field(fname))

        return new_state


    # @profile
    def euler(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, inplace: bool = False,
              compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0,
              verbose_plot=False, **deriv_kwargs) -> 'PDE_system':
        """ perform explicit Euler time evolution
            compress_opts are for compression after time evolution
        """
        # if deriv0 is None:
        #     deriv0 = self.calculate_time_derivative(time=self.time, compress_level=compress_level,
        #                                             compress_level1=compress_level1, compress_level2=compress_level2,
        #                                             verbose_plot=verbose_plot, **deriv_kwargs)

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(time=self.time, compress_level=comp4, compress_level1=comp5,
                                                    verbose_plot=verbose_plot, **deriv_kwargs)

        # print('self', self.fe.max_bond(), self.fi.max_bond())
        # print('deriv0', deriv0.fe.max_bond(), deriv0.fi.max_bond())
        state1 = self.add(deriv0 * dt, compress_level=0, inplace=inplace)

        if compress_level != 0:
            state1.compress(inplace=True, compress_level=compress_level)

        if state1.do_normalization:
            state1.normalize()

        if self.time is not None:
            self.time += dt

        return state1


    def rk2(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, compress_level: int = 1, verbose_plot=False,
            **deriv_kwargs,) -> 'PDE_system':
        """ perform explicit RK2 time evolution
            Heun's method:
                input: y_n, h=dt
                y1_(n+1) = y_(n) + h F(t_n, y_n)
                y_(n+1) = y_(n) + h/2 [ F(t_n, y_n) + F(t_(n+1), y1_(n+1))
                output: y_(n+1)
        """
        state0 = self.copy() # if inplace else self.copy()
        time = self.time
        time1 = time + dt if time is not None else None

        print('rk2')

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(time=time, compress_level=comp4, compress_level1=comp5,
                                                    verbose_plot=verbose_plot, **deriv_kwargs)
    
        state1 = state0.euler(dt, deriv0=deriv0, compress_level=comp2, compress_level1=comp4, compress_level2=comp5)
        deriv1 = state1.calculate_time_derivative(time=time1, compress_level=comp4, compress_level1=comp5,
                                                  verbose_plot=verbose_plot, **deriv_kwargs)

        deriv_sum = deriv0.add(deriv1,compress_level=0) # comp4)
        new_state = state0.euler(0.5 * dt, deriv0=deriv_sum, inplace=True, compress_level=comp1)
        new_state.time = time + dt if time is not None else None

        return new_state
    

    def rk3(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, compress_level: int = 1,
            verbose_plot=False, **deriv_kwargs) -> 'PDE_system':
        """ perform 4-stage RK3 time evolution
            https://gkeyll.readthedocs.io/en/latest/dev/ssp-rk.html#ssprk
        """
        state0 = self # if inplace else self.copy()

        time = self.time
        time1 = time+dt/2 if time is not None else None
        time2 = time+dt if time is not None else None

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(time=time, compress_level=comp4, compress_level1=comp5,
                                                    verbose_plot=verbose_plot, **deriv_kwargs)

        # state1 = state0.euler(dt, deriv0=deriv0, compress_level=comp2, compress_level1=comp4, compress_level2=comp5)
        state1 = state0.euler(dt, deriv0=deriv0, compress_level=0, compress_level1=comp4, compress_level2=comp5)
        state1.add(state0, inplace=True, compress_level=comp2)
        state1.scalar_multiply(0.5, inplace=True)
        deriv1 = state1.calculate_time_derivative(time=time1, compress_level=comp4, compress_level1=comp5,
                                                  verbose_plot=verbose_plot, **deriv_kwargs)

        # state2 = state1.euler(dt, deriv0=deriv1, compress_level=comp2, compress_level1=comp4, compress_level2=comp5)
        state2 = state1.euler(dt, deriv0=deriv1, compress_level=0, compress_level1=comp4, compress_level2=comp5)
        state2.add(state1, inplace=True, compress_level=comp2)
        state2.scalar_multiply(0.5, inplace=True)
        deriv2 = state2.calculate_time_derivative(time=time2, compress_level=comp4, compress_level1=comp5,
                                                  verbose_plot=verbose_plot, **deriv_kwargs)

        state3 = state2.euler(dt, deriv0=deriv2, compress_level=0, compress_level1=comp4, compress_level2=comp5)
        # state3 = state2.euler(dt, deriv0=deriv2, compress_level=comp2, compress_level1=comp4, compress_level2=comp5)
        state3.add(state2, inplace=True)
        state3.scalar_multiply(1./6, inplace=True)
        state3.add(state0.scalar_multiply(2. / 3), inplace=True, compress_level=comp2)

        deriv3 = state3.calculate_time_derivative(time=time1, compress_level=comp4, compress_level1=comp5,
                                                  verbose_plot=verbose_plot, **deriv_kwargs)
        new_state = state3.euler(dt, deriv0=deriv3, compress_level=0, compress_level1=comp4, compress_level2=comp5)
        # new_state = state3.euler(dt, deriv0=deriv3, compress_level=comp2, compress_level1=comp4, compress_level2=comp5)
        new_state.add(state3, inplace=True, compress_level=comp1)
        new_state.scalar_multiply(0.5,inplace=True)

        new_state.time = time + dt if time is not None else None

        return new_state


    # @profile
    # def rk4(self, dt, deriv0: Optional['PDE_system'] = None, compress_level: int = 1,
    #         verbose_plot=False, **deriv_kwargs) -> 'PDE_system':
    #     """ perform explicit RK4 time evolution
    #         compress = 0:   don't compress at all
    #         compress_level: compress final state
    #         comp2:   + compress intermediate rk4 states
    #         comp3:   + compress sum of derivatives
    #         comp4:   + compress derivatives
    #         comp5:   + compress during calculation of derivative
    #     """
    #     print('rk4', compress_level)
    #     if compress_level == 0:
    #         comp1 = comp2 = comp3 = comp4 = comp5 = 0
    #     else:
    #         comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level,5)
    #
    #     time = self.time
    #
    #     def euler_func(state_: 'PDE_system', dt_, deriv0=None, compress_level=comp1, compress_level1=comp4,
    #                    compress_level2=comp5, **kwargs):
    #         return euler(state_, dt_, deriv0=deriv0, compress_level=compress_level, compress_level1=compress_level1,
    #                      compress_level2=compress_level2, verbose_plot=verbose_plot, **deriv_kwargs)
    #
    #     def deriv_func(state_: 'PDE_system', time=time, compress_level=comp1, compress_level1=comp4,
    #                    compress_level2=comp5, **kwargs):
    #         return calculate_time_derivative(state_, time=time, compress_level=compress_level,
    #                                          compress_level1=compress_level1, compress_level2=compress_level2,
    #                                          verbose_plot=verbose_plot, **deriv_kwargs)
    #
    #     def add_func(state_: 'PDE_system', state2: 'PDE_system', compress_level=1, inplace=False):
    #         return add(state_, state2, compress_level=compress_level, inplace=inplace)
    #
    #     def scale_func(state_: 'PDE_system', scalar_const, inplace=False):
    #         return scalar_multiply(state_, scalar_const, inplace=inplace)
    #
    #     new_state = helper_TE.rk4(self, dt, euler_func, deriv_func, add_func, scale_func,
    #                               deriv0=deriv0, compress_level=compress_level)
    #
    #     return new_state

    def rk4(self, dt, deriv0: Optional['PDE_system'] = None, compress_level: int = 1,
            verbose_plot=False, **deriv_kwargs) -> 'PDE_system':
        """ perform explicit RK4 time evolution
            compress = 0:   don't compress at all
            compress_level: compress final state
            comp2:   + compress intermediate rk4 states
            comp3:   + compress sum of derivatives
            comp4:   + compress derivatives
            comp5:   + compress during calculation of derivative
        """
        print('rk4', compress_level)
        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        time = self.time
        time1 = time + dt / 2 if time is not None else None
        time2 = time + dt if time is not None else None

        state0 = self  # .copy() # if not inplace else self
        # print('state0 norm', state0.fe.component.norm())
        # print('state0 norm', state0.fi.component.norm())
        # print('state0 norm', state0.V.component.norm())

        # print(self, dt)

        if deriv0 is None:
            # print('get deriv0')
            deriv0 = state0.calculate_time_derivative(time=time, compress_level=comp4, compress_level1=comp5,
                                                      compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)
        # print('deriv0 fe diff', deriv0.fe.component.norm())
        # print('deriv0 fi diff', deriv0.fi.component.norm())
        #### what was this for again?
        # state0.dist0_e = deriv0.dist0_e
        # state0.dist0_i = deriv0.dist0_i

        # print('get euler 1')
        state1 = state0.euler(dt * 0.5, deriv0=deriv0, compress_level=comp2, compress_level1=comp4,
                              compress_level2=comp5)
        # print('state1 fe diff', helper.distance(state1.fe.component.data, state0.fe.component.data))
        # print('state1 fi diff', helper.distance(state1.fi.component.data, state0.fi.component.data))
        # print('state1 V diff', helper.distance(state1.V.component.data, state0.V.component.data))

        # print('deriv1', state1)
        deriv1 = state1.calculate_time_derivative(time=time1, compress_level=comp4, compress_level1=comp5,
                                                  compress_level2=comp5, update_force=False,
                                                  verbose_plot=verbose_plot, **deriv_kwargs)
        # print('deriv1 fe diff', deriv1.fe.component.norm())
        # print('deriv1 fi diff', deriv1.fi.component.norm())

        # print('get euler 2')
        state2 = state0.euler(dt * 0.5, deriv0=deriv1, compress_level=comp2, compress_level1=comp4,
                              compress_level2=comp5)

        # print('state2 fe diff', helper.distance(state2.fe.component.data, state0.fe.component.data))
        # print('state2 fi diff', helper.distance(state2.fi.component.data, state0.fi.component.data))
        # print('state2 V diff', helper.distance(state2.V.component.data, state0.V.component.data))

        # print('deriv2', state2)
        deriv2 = state2.calculate_time_derivative(time=time1, update_force=False,
                                                  compress_level=comp4, compress_level1=comp5, compress_level2=comp5,
                                                  verbose_plot=verbose_plot, **deriv_kwargs)

        # print('deriv2 fe diff', deriv2.fe.component.norm())
        # print('deriv2 fi diff', deriv2.fi.component.norm())

        # print('state 3')
        state3 = state0.euler(dt, deriv0=deriv2, compress_level=comp2, compress_level1=comp4, compress_level2=comp5)

        # print('state3 fe diff', helper.distance(state3.fe.component.data, state0.fe.component.data))
        # print('state3 fi diff', helper.distance(state3.fi.component.data, state0.fi.component.data))
        # print('state3 V diff', helper.distance(state3.V.component.data, state0.V.component.data))

        # print('deriv3')
        deriv3 = state3.calculate_time_derivative(time=time2, update_force=False,
                                                  compress_level=comp4, compress_level1=comp5, compress_level2=comp5,
                                                  verbose_plot=verbose_plot, **deriv_kwargs)

        # print('deriv3 fe diff', deriv3.fe.component.norm())
        # print('deriv3 fi diff', deriv3.fi.component.norm())

        # sum_deriv = deriv0 + deriv1*2 + deriv2*2 + deriv3
        # print('sum deriv')
        sum_deriv = deriv0.add(deriv1 * 2, compress_level=0)
        sum_deriv = sum_deriv.add(deriv2 * 2, compress_level=0, inplace=True)
        sum_deriv = sum_deriv.add(deriv3, compress_level=comp4, inplace=True)

        # print('sumderiv fe diff', sum_deriv.fe.component.norm())
        # print('sumderiv fi diff', sum_deriv.fi.component.norm())

        # print('final euler')
        new_state = state0.euler(dt / 6., deriv0=sum_deriv, inplace=True, compress_level=comp1, compress_level1=comp2,
                                 compress_level2=comp3)
        new_state.time = time + dt if time is not None else None
        # print('end euler')

        return new_state

    # def lax_wendroff(self, dt, time: float = None):
    #     raise NotImplementedError
    #
    #
    # def mccormack(self, dt, time: float = None):
    #     raise NotImplementedError
    #
    #
    # def warming_beam(self, dt, time: float = None):
    #     raise NotImplementedError


    def backwards_euler(self, dt, time: float = None, err_tol: float = None, max_iter:int = 100, compress_level: int = 1,
                        verbose_plot=False, **deriv_kwargs):
        """ implicit time differentation:  dy/dt = F(y)
            y_n+1 = y_n + dt*F(y_n+1)       [n specifies time step]
            via fixed point iteration, Jacobi iteration with diagonal = I:
                (I + dt*F) y_n+1 = y_n
                y^(j+1) = dt*F y^(j) + y_n      [j specifies iteration]
                s.t. y^(j=infty) -> y_n+1

            note: this wouldn't work as written if there was a second order derivative term...
                diagonal wouldn't equal I
        """
        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if err_tol is None:
            err_tol = CUTOFF
            if CUTOFF_MODE == 'rsum2':
                err_tol = np.sqrt(err_tol)
            # err_tol *= 10

        time1 = time+dt if time is not None else None

        # deriv0 = self.calculate_time_derivative(compress_level=comp4, compress_level1=comp5,
        #                                         compress_level2=comp5, verbose_plot=verbose_plot)
        # prev_state = self.euler(dt, deriv0=deriv0, compress_level=comp2, compress_level1=comp4, compress_level2=comp5)
        prev_state = self
        fields_norm = {k: field.norms() for k, field in self._fields.items()}

        err = np.inf
        it = 0
        while np.abs(err) > err_tol and it < max_iter:
            prev_err = err
            deriv0 = prev_state.calculate_time_derivative(time=time1, compress_level=comp4, compress_level1=comp5,
                                                          compress_level2=comp5, verbose_plot=verbose_plot,
                                                          **deriv_kwargs)
            new_state = self.euler(dt, deriv0=deriv0, inplace=False, compress_level=comp2, compress_level1=comp4,
                                   compress_level2=comp5)
            # errs = new_state.distances(prev_state, total=False, normalize=False)
            err = new_state.distances(prev_state, total=True, normalize=True, field_norms=fields_norm)
            print('tot err', it, err)

            if err > prev_err or np.abs(err-prev_err)/prev_err < err_tol:
                break

            if comp2 != comp1:
                new_state.compress(inplace=True, compress_level=comp1)
            # print('after comp errs', new_state.distances(prev_state, total=False, normalize=False))

            prev_state = new_state
            it += 1

        # if self.do_normalization:
        #     prev_state.normalize()

        return prev_state


    def implicit_midpoint(self, dt, time: float = None, err_tol: float = None, max_iter:int = 100, compress_level: int = 1,
                          verbose_plot=False, **deriv_kwargs):
        """ implicit time differentation:  dy/dt = F(y)
            y_(n+1) = y_(n) + F( (y_(n+1) + y_(n))/2 )
        """
        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if err_tol is None:
            err_tol = CUTOFF
            if CUTOFF_MODE == 'rsum2':
                err_tol = np.sqrt(err_tol)
            err_tol *= 10

        time1 = time + dt/2 if time is not None else None

        prev_state = self.copy()
        fields_norm = {k: field.norms() for k, field in self._fields.items()}
        ## initial guess of y_(n+1)
        # deriv0 = self.calculate_time_derivative(compress_level=comp4, compress_level1=comp5,
        #                                         compress_level2=comp5, verbose_plot=verbose_plot)
        # prev_state = self.euler(dt, deriv0=deriv0, compress_level=comp2, compress_level1=comp4, compress_level2=comp5)


        err = np.inf
        it = 0
        while err > err_tol and it < max_iter:
            prev_err = err
            midpt_state = self.add(prev_state, inplace=False, compress_level=comp3)
            midpt_state.scalar_multiply(0.5, inplace=True)
            deriv0 = midpt_state.calculate_time_derivative(time=time1, compress_level=comp4, compress_level1=comp5,
                                                           compress_level2=comp5, verbose_plot=verbose_plot,
                                                           **deriv_kwargs)
            new_state = self.euler(dt, deriv0=deriv0, inplace=False, compress_level=comp2, compress_level1=comp4,
                                   compress_level2=comp5)

            err = new_state.distances(prev_state, total=True, normalize=True, field_norms=fields_norm)
            print('tot err', it, err)

            if err > prev_err or np.abs(err-prev_err)/prev_err < err_tol:
                break

            if comp2 != comp1:
                new_state.compress(inplace=True, compress_level=comp1)

            prev_state = new_state
            it += 1

        # if self.do_normalization:
        #     prev_state.normalize()

        return prev_state


    def crank_nicolson(self, dt, time: float = None, err_tol: float = None, max_iter:int = 100, compress_level: int = 1,
                       verbose_plot=False, **deriv_kwargs):
        """ implicit time differentation:  dy/dt = F(y)
            y_(n+1) = y_(n) + h/2 ( F(y_(n+1)) + F(y_(n)) )
        """
        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if err_tol is None:
            err_tol = CUTOFF
            if CUTOFF_MODE == 'rsum2':
                err_tol = np.sqrt(err_tol)
            err_tol *= 10

        time1 = time + dt if time is not None else None

        prev_state = self
        fields_norm = {k: field.norms() for k, field in self._fields.items()}
        deriv0 = self.calculate_time_derivative(time=time, compress_level=comp4, compress_level1=comp5,
                                                compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)
        # prev_state = self.euler(dt, deriv0=deriv0, compress_level=comp2, compress_level1=comp4, compress_level2=comp5)

        err = np.inf
        it = 0
        while err > err_tol and it < max_iter:
            prev_err = err
            deriv1 = prev_state.calculate_time_derivative(time=time1, compress_level=comp4, compress_level1=comp5,
                                                          compress_level2=comp5, verbose_plot=verbose_plot,
                                                          **deriv_kwargs)
            sum_deriv = deriv0.add(deriv1, inplace=False, compress_level=comp3)
            new_state = self.euler(dt / 2, deriv0=sum_deriv, inplace=False, compress_level=comp2, compress_level1=comp4,
                                   compress_level2=comp5)
            err = new_state.distances(prev_state, total=True, normalize=True, field_norms=fields_norm)
            print('tot err', it, err)

            if err > prev_err or np.abs(err-prev_err)/err < err_tol:
                break

            if comp2 != comp1:
                new_state.compress(inplace=True, compress_level=comp1)

            prev_state = new_state
            it += 1

        # if self.do_normalization:
        #     prev_state.normalize()

        return prev_state


    #### dynamical low rank methods
    def dynamical_low_rank(self, dt: Numeric, inplace=False, te_order=4, do_adapt: bool = True,
                           compress_level: int = 1, compress_level_2: int = 0,
                           advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                           verbose_plot: bool = False, **kwargs):
        raise NotImplementedError


    def time_dependent_variational_principle(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace=False,
                                             compress_level: int = 1, compress_level_2: int = 4, direction=1,
                                             advec_axes: Sequence['Axis'] = None, background_force=True,
                                             internal_force=True, verbose_plot: bool = False, **kwargs):
        raise NotImplementedError


    def time_dmrg(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace=False,
                  compress_level: int = 1, compress_level_2: int = 4, direction=1,
                  advec_axes: Sequence['Axis'] = None, background_force=True,
                  internal_force=True, verbose_plot: bool = False, **kwargs):
        raise NotImplementedError


    def time_dependent_variational_principle_SL(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace=False,
                                                advec_axes = None, bg_method = 'SL', bg_split_order = 2,
                                                compress_level: int = 1, compress_level_2: int = 4, direction=1,
                                                verbose_plot: bool = False, **kwargs):
        raise NotImplementedError

    def time_dmrg_SL(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace=False,
                                                advec_axes = None, bg_method = 'SL', bg_split_order = 2,
                                                compress_level: int = 1, compress_level_2: int = 4, direction=1,
                                                verbose_plot: bool = False, **kwargs):
        raise NotImplementedError


    ### maybe can make a general bg/pert/bg or pert/bg/pert method.



def calculate_time_derivative(state: 'PDE_system', time=None, compress_level=1, compress_level1=4,
                              compress_level2=5, verbose_plot=False, **deriv_kwargs):
    return state.calculate_time_derivative(time=time, compress_level=compress_level,
                                           compress_level1=compress_level1, compress_level2=compress_level2,
                                           verbose_plot=verbose_plot, **deriv_kwargs)

def euler(state: 'PDE_system', dt, deriv0=None, inplace=False, compress_level=1, compress_level1=4, compress_level2=5,
          verbose_plot=False, **deriv_kwargs):
    return state.euler(dt, deriv0=deriv0, inplace=inplace, compress_level=compress_level,
                       compress_level1=compress_level1, compress_level2=compress_level2,
                       verbose_plot=verbose_plot, **deriv_kwargs)

def add(state1: 'PDE_system', state2: 'PDE_system', compress_level=1, inplace=False):
    return state1.add(state2, compress_level=compress_level, inplace=inplace)

def scalar_multiply(state: 'PDE_system', scalar_const, field_keys=None, inplace=False):
    return state.scalar_multiply(scalar_const, field_keys=field_keys, inplace=inplace)