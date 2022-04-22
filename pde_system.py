from setup.configs import *

from field import Field, ScalarField
import helper_quimb as helper

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
        # if field_compress_config is None:
        #     field_compress_config = CompressionConfiguration()
        # self.field_compress_config = field_compress_config

        if compress_levels is None:
            compress_levels = list(range(1,10))     # some arbitrary number; compress at all levels
        self._comp_levels = compress_levels

        self.fields: dict[Any, Optional[Field]] = {field.name: field for field in fields if field is not None}
        self.field_names = [field.name for field in fields if field is not None] \
                              if field_names is None else field_names

        # ref_field = self.fields[next(iter(self.fields))]
        # self.L = ref_field.grid.L
        # self.ndim = ref_field.grid.ndim
        # self.q = ref_field.grid.q

        ## pad system with 'None' fields as needed, prescribed by field_names
        for fn in self.field_names:
            if fn not in self.fields:  self.fields[fn] = None

        # self._te_compress_opts_all = {}
        # self.set_te_compress_opts(te_compress_opts)
        self.te_order = te_order

        # ## update all compress opts to that defined by te_compress_opts
        # if len(te_compress_opts) > 0:
        #     for f in self.field_names:
        #         field = self.fields[f]
        #         if field is not None:   field.set_te_compress_opts(te_compress_opts)

        # self.init_compress_opts = init_compress_opts

        self.do_normalization = normalize


    def __getitem__(self,field_name):
        """ return ith component of the field. for convenience
        """
        return self.fields[field_name]   ## allow KeyError to be raised


    def __setitem__(self,field_name,new_field):
        """ set ith component of the field. for convenience
            new_field is a Field object
        """
        assert(isinstance(new_field,Field) or new_field is None), 'new_field must be Field object or None'
        self.fields[field_name] = new_field


    def _get_compress_levels(self, compress_level, num_levels):
        comp_levels = self._comp_levels[compress_level-1:compress_level-1+num_levels]
        if len(comp_levels) < num_levels:
            comp_levels += [0]*(num_levels-len(comp_levels))
        return comp_levels


    # def initialize_fields(self,fields_dict):
    #     """ fields_dict: dictionary indexed by field name, yielding dictionaries of field components
    #     """
    #     for name, field in fields_dict.items():
    #         self.fields[name] = field


    def create_like(self, *new_fields, recalc=True):
        """ create a new system like this without defining the fields
        """
        new_system = self.__class__(*new_fields,field_names=self.field_names,
                                    normalize=self.do_normalization,te_order=self.te_order,
                                    compress_levels=self._comp_levels)
                                    # field_compress_config=self.field_compress_config)
        return new_system


    def copy(self):
        """ copy self including the fields
        """
        copy_fields = [self.fields[k].copy() if self.fields[k] is not None else None 
                       for k in self.field_names]
        new_system = self.create_like(*copy_fields, recalc=False)
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

        all_fields = set.union(set(self.fields.keys()), set(other.fields.keys()))

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
        if field_keys is None:   field_keys = self.fields.keys()
        new_sys = self if inplace else self.copy()
        for k in field_keys:
            if new_sys.fields[k] is None:  continue
            new_sys.fields[k].scalar_multiply(const,inplace=True)
        return new_sys


    def elemental_multiply(self, mps_vec: qtn.MatrixProductState, field_keys: Sequence[Any] = None,
                           compress_level: int =0, inplace=False):
        if field_keys is None:   field_keys = self.fields.keys()
        new_sys = self if inplace else self.copy()
        for k in field_keys:
            f = new_sys.fields[k]
            if f is None:  continue
            f.elemental_multiply(mps_vec, inplace=True, compress_level=compress_level)
        return new_sys


    # def norm2(self):
    #     norm = 0
    #     for k, field in self.fields.items():
    #         norm += field.norm2()
    #     return norm


    def compress(self, inplace=False, compress_level=1):
        new_sys = self if inplace else self.copy()
        for k, field in new_sys.fields.items():
            if field is None:  continue
            field.compress(inplace=True,compress_level=compress_level)
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


    def calculate_time_derivative(self, compress_level: int = 0, compress_level1: int = 0, verbose_plot=False) \
            -> 'PDE_system':
        raise NotImplementedError


    def next_time_step(state0, dt: Numeric, deriv0: Optional['PDE_system']=None, inplace=False, compress_level: int=1,
                       do_postprocessing=False, process_kwargs=None, verbose_plot=False):
        """ convenience fct btwn different TE methods?
        """
        te_order = state0.te_order

        # mod_compress_opts = state0.get_te_compress_opts(compress).copy()
        # mod_compress_opts.update(compress_opts)

        if   te_order==1:
            state_t = state0.euler(dt, deriv0, inplace, compress_level=compress_level)
        elif te_order==2:
            state_t = state0.rk2(dt, deriv0, inplace, compress_level=compress_level)
        elif te_order==3:
            state_t = state0.rk3  (dt, deriv0, inplace, compress_level=compress_level)
        elif te_order==4:
            state_t = state0.rk4(dt, deriv0, inplace, compress_level=compress_level, verbose_plot=verbose_plot)
        # elif te_order==21:
        #     state_t = state0.backwards_euler(dt, deriv0, compress_level)
        # elif te_order==22:
        #     state_t = state0.crank_nicolson(dt, deriv0, compress_level)
        else:
            raise NotImplementedError

        if state_t.do_normalization:
            state_t.normalize()

        return state_t


    def euler(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, inplace: bool = False, compress_level: int = 1,
              compress_level1: int = 0, compress_level2: int = 0, verbose_plot=False) -> 'PDE_system':
        """ perform explicit Euler time evolution
            compress_opts are for compression after time evolution
        """
        if deriv0 is None:   deriv0 = self.calculate_time_derivative(compress_level=compress_level1,
                                                                     compress_level1=compress_level2,
                                                                     verbose_plot=verbose_plot)
    
        state1 = self.add(deriv0 * dt, compress_level=0, inplace=inplace)

        if compress_level != 0:
            state1.compress(inplace=True, compress_level=compress_level)

        if state1.do_normalization:
            state1.normalize()

        return state1


    def rk2(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, inplace: bool = False, compress_level: int = 1,
            verbose_plot=False) \
            -> 'PDE_system':
        """ perform explicit RK2 time evolution
        """
        state0 = self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(compress_level=comp4, compress_level1=comp5,
                                                    verbose_plot=verbose_plot)
    
        state1 = state0.euler(dt, deriv0=deriv0, compress_level=comp2, compress_level1=comp4, compress_level2=comp5)
        deriv1 = state1.calculate_time_derivative(compress_level=comp4, compress_level1=comp5,
                                                  verbose_plot=verbose_plot)

        deriv_sum = deriv0.add(deriv1,compress_level=comp3)
        new_state = state0.euler(0.5 * dt, deriv0=deriv_sum, inplace=inplace, compress_level=comp1)

        return new_state
    

    def rk3(self, dt: Numeric, deriv0: 'PDE_system' = None, inplace: bool = False, compress_level: int = 1,) \
            -> 'PDE_system':
        """ perform 4-stage RK3 time evolution
            https://gkeyll.readthedocs.io/en/latest/dev/ssp-rk.html#ssprk
        """
        raise NotImplementedError

        state0 = self.copy()
    
        if deriv0 is None:   deriv0 = self.calculate_time_derivative(compress + 2, verbose_plot=verbose_plot)
    
        state1 = self.euler(0.5 * dt, deriv0=deriv0, **compress_opts)
        state1.add(state0.scalar_multiply(0.5), inplace=True)
        deriv1 = state1.calculate_time_derivative(compress, verbose_plot=verbose_plot)
    
        state2 = state1.euler(0.5 * dt, deriv0=deriv1, **compress_opts)
        state2.add(state1.scalar_multiply(0.5), inplace=True)
        deriv2 = state2.calculate_time_derivative(compress, verbose_plot=verbose_plot)
    
        state3 = self.euler(dt, deriv0=deriv2, **compress_opts)
        state3.add(state2, inplace=True)
        state3.scalar_multiply(1./6,inplace=True)
        state3.add(state0.scalar_multiply(2. / 3))
        deriv3 = state3.calculate_time_derivative(compress, verbose_plot=verbose_plot)
        
        new_state = state3.euler(0.5 * dt, deriv0=deriv3, **compress_opts)
        new_state.add(state3, inplace=True)
        new_state.scalar_multiply(0.5,inplace=True)
    
        return new_state


    def rk4(self, dt, deriv0: Optional['PDE_system'] = None, inplace: bool = False, compress_level: int = 1,
            verbose_plot=False) \
            -> 'PDE_system':
        """ perform explicit RK4 time evolution
            compress = 0:   don't compress at all
            compress_level: compress final state
            comp2:   + compress intermediate rk4 states
            comp3:   + compress sum of derivatives
            comp4:   + compress derivatives
            comp5:   + compress during calculation of derivative
        """
        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level,5)

        state0 = self.copy()

        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(compress_level=comp4, compress_level1=comp5,
                                                    verbose_plot=verbose_plot)
        #### what was this for again?
        # state0.dist0_e = deriv0.dist0_e
        # state0.dist0_i = deriv0.dist0_i

        state1 = state0.euler(dt * 0.5, deriv0=deriv0, compress_level=comp2, compress_level1=comp4,
                              compress_level2=comp5)
        # print('deriv1')
        deriv1 = state1.calculate_time_derivative(compress_level=comp4, compress_level1=comp5,
                                                  verbose_plot=verbose_plot)
    
        # print('state 2')
        state2 = state0.euler(dt * 0.5, deriv0=deriv1, compress_level=comp2, compress_level1=comp4,
                              compress_level2=comp5)
        # print('deriv2')
        deriv2 = state2.calculate_time_derivative(compress_level=comp4, compress_level1=comp5,
                                                  verbose_plot=verbose_plot)
    
        # print('state 3')
        state3 = state0.euler(dt, deriv0=deriv2, compress_level=comp2, compress_level1=comp4, compress_level2=comp5)
        # print('deriv3')
        deriv3 = state3.calculate_time_derivative(compress_level=comp4, compress_level1=comp5,
                                                  verbose_plot=verbose_plot)
        
        # sum_deriv = deriv0 + deriv1*2 + deriv2*2 + deriv3
        # print('adding')
        sum_deriv = deriv0.add(deriv1*2,compress_level=0)
        sum_deriv = sum_deriv.add(deriv2*2,compress_level=0)
        sum_deriv = sum_deriv.add(deriv3,compress_level=comp3)

        # print('final euler')
        new_state = state0.euler(dt / 6., deriv0=sum_deriv, inplace=inplace, compress_level=comp1)
    
        return new_state


    def backwards_euler(self, dt, deriv0=None, compress=1, **compress_opts):
        """ implicit time differentation:  dy/dt = F(y)
            y_n+1 = y_n + dt*F(y_n+1)
        """
        raise NotImplementedError

        target = self.copy()
        target.scalar_multiply(-1,inplace=True)
        
        ## could potentially re-use info (ie. MPO) if a linear eq.
        # if deriv0 is None:   deriv0 = state.calculate_time_derivative(compress)
        # guess = state.euler(dt, compress=compress, deriv0=deriv0, **compress_opts)
        # guess = state.rk2(dt, compress=compress, deriv0=deriv0, **compress_opts)
        guess = self.copy()  ## don't get increase in bond dimension?

        # deriv_x = state.calculate_time_derivative(compress,get_approx=True)
        # guess = state.next_time_step(deriv_x,dt,compress=compress,**compress_opts)

        guess_tn = helper.state_to_tensornetwork(guess)

        ## artificially expand bond dimension
        current_max = guess_tn.max_bonds()
        max_bond = compress_opts.get('max_bond', int(self.q ** (self.L // 2)))
        new_maxbond = min(current_max+3, max_bond)
        guess_tn.expand_bond_dimension(new_maxbond, inplace=True, rand_strength=1.0e-3)
        print(current_max, guess_tn.max_bonds())

        # function to minimze single component of the field
        def min_func(guess_tn_):
            guess_state = helper.tensornetwork_to_state(guess_tn_, self)
            deriv1 = guess_state.calculate_time_derivative(compress_level=0, verbose_plot=verbose_plot)
            guess_state_t = guess_state.euler(-dt, deriv0=deriv1, **compress_opts)

            diff = guess_state_t.add(target)
            diff_norm = diff.norm2()
            return diff_norm

        # # function to compute L1-norm
        # sum_tn_ = helper.L1_tensornetwork(like_tn=guess_tn,scale=np.prod(state.grid.dx))
        # def norm_func(guess_tn_):
        #     norm = helper.L1norm(guess_tn_,sum_tn_)
        #     return guess_tn_.multiply(1./norm)
            

        optmzr = TNOptimizer(guess_tn, loss_fn=min_func, norm_fn=None, #norm_func, loss_constants={}, 
                             optimizer='L-BFGS-B', autodiff_backend='jax')
        comp_opt = optmzr.optimize(500,tol=1.0e-20)
        new_state = helper.tensornetwork_to_state(comp_opt, self)

        if compress > 1:
            new_state.compress(inplace=True,**compress_opts)
        
        # helper.tensornetwork_to_state(comp_opt,state0)
        print('diff init', min_func(guess_tn))
        print('diff opt', min_func(comp_opt))
        print('new max bond', new_state['probability'][0].max_bonds())

        try:    new_state.normalize()
        except(AttributeError):  pass

        return new_state


    def crank_nicolson(state, dt, deriv0=None, compress=1, **compress_opts):
        """ implicit time differentation:  dy/dt = F(y)
        """
        raise NotImplementedError

        if deriv0 is None:   deriv0 = state.calculate_time_derivative(compress, verbose_plot=verbose_plot)
        target = state.euler(dt * 0.5, deriv0=deriv0, **compress_opts)
        target.scalar_multiply(-1,inplace=True)
        
        ## could potentially re-use info (ie. MPO) if a linear eq.
        # guess = state.euler(dt, compress=compress, deriv0=deriv0, **compress_opts)
        # guess = state.rk2(dt, compress=compress, deriv0=deriv0, **compress_opts)
        guess = state.copy()  ## don't get increase in bond dimension?

        # deriv_x = state.calculate_time_derivative(compress,get_approx=True)
        # guess = state.next_time_step(deriv_x,dt,compress=compress,**compress_opts)

        guess_tn = helper.state_to_tensornetwork(guess)

        ## artificially expand bond dimension
        current_max = guess_tn.max_bonds()
        max_bond = compress_opts.get('max_bond',int(state.q**(state.L//2)))
        new_maxbond = min(current_max+3, max_bond)
        guess_tn.expand_bond_dimension(new_maxbond, inplace=True) # , rand_strength=1.0e-3)
        print(current_max, guess_tn.max_bonds())

        # function to minimze single component of the field
        def min_func(guess_tn_):
            guess_state = helper.tensornetwork_to_state(guess_tn_,state)
            deriv1 = guess_state.calculate_time_derivative(compress_level=0, verbose_plot=verbose_plot)
            guess_state_t = guess_state.euler(-dt * 0.5, deriv0=deriv1, **compress_opts)

            diff = guess_state_t.add(target)
            return diff.norm2()

        # # function to compute L1-norm
        # sum_tn_ = helper.L1_tensornetwork(like_tn=guess_tn,scale=np.prod(state.grid.dx))
        # def norm_func(guess_tn_):
        #     norm = helper.L1norm(guess_tn_,sum_tn_)
        #     return guess_tn_.multiply(1./norm)
            

        optmzr = TNOptimizer(guess_tn, loss_fn=min_func, norm_fn=None, #norm_func, loss_constants={}, 
                             optimizer='L-BFGS-B', autodiff_backend='jax')
        comp_opt = optmzr.optimize(500,tol=1.0e-20)
        # comp_opt = norm_func(comp_opt)
        new_state = helper.tensornetwork_to_state(comp_opt,state)

        if compress > 1:
            # print('compressing')
            # print('pre comp', new_state['probability'][0])
            new_state.compress(inplace=True,**compress_opts)
            # print('new state', new_state['probability'][0].singular_values(state.grid.L//2))
            # print('post comp', new_state['probability'][0])
            # print('post comp', new_state['probability'][0].max_bond())
        
        # helper.tensornetwork_to_state(comp_opt,state0)
        print('diff init', min_func(guess_tn))
        print('diff opt', min_func(comp_opt))
        print('new max bond', new_state['probability'][0].max_bonds())

        try:    new_state.normalize()
        except(AttributeError):  pass

        return new_state

