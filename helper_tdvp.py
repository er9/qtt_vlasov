from setup.configs import *
import scipy.linalg
import helper_quimb as helper
from helper_dmrg import *
import helper_TE
from setup.quimb_TN1D import MatrixProductStateUSVT as MPS_USVT
import gridTN as GTN

if TYPE_CHECKING:
    from axis import Axis
    from gridTN import GridTN


class CompressDirection(IntEnum):
    ## int denotes where the MPS needs to be canonicalized to
    LEFT = -1
    RIGHT = 1

# def dlr_compute_time_derivative_site(dist_gtn: 'GridTN', deriv_mpos: Sequence['GridTN'], site_ind: int, nsites=1,
#                                      canonize=True, **kwargs) -> 'qtn.Tensor':
#     """ compute projection of MPS onto site(s) [specified by nsites] or bond (right of site_ind) [nsites=0]
#     """
#     if canonize:
#         if nsites == 0:
#             dist_gtn.convert_to_USVT(canon_site=site_ind)
#         else:
#             dist_gtn.canonize(i=site_ind, inplace=True)
#
#     proj_total: 'Optional[qtn.Tensor]' = None
#     for mpo in deriv_mpos:
#         if nsites == 0:
#             proj_x = dist_gtn.project_bond(mpo, bond_ind=site_ind)
#         else:
#             proj_x = dist_gtn.project_site(mpo, site_ind, nsites=nsites)
#         if proj_total is None:
#             proj_total = proj_x
#         else:
#             proj_x.transpose_like_(proj_total)
#             proj_total.modify(apply=lambda x: x + proj_x.data)
#
#     return proj_total

def _update_tn_with_tensor(dist_submpx: 'qtn.TensorNetwork', active_tens: 'qtn.Tensor',
                           submpx_tags: Sequence['str'], compress_direction=CompressDirection.RIGHT,
                           compress_opts=None):
    """ inplace operation
    """
    nsites = dist_submpx.num_tensors
    compress_opts = {'cutoff': CUTOFF, 'cutoff_mode': CUTOFF_MODE} if compress_opts is None else compress_opts

    T1 = active_tens
    s_tags_list = submpx_tags if compress_direction > 0 else submpx_tags[::-1]
    # print('update tn', submpx_tags, s_tags_list, compress_direction)
    for ix in range(nsites - 1):
        # print('s_ind', s_ind)
        dist_gtn_tens: 'qtn.Tensor' = next(iter(dist_submpx.select(s_tags_list[ix])))
        neighbor_tens: 'qtn.Tensor' = next(iter(dist_submpx.select(s_tags_list[ix + 1])))
        rix, lix = dist_gtn_tens.filter_bonds(neighbor_tens)
        # rix, lix = dist_submpx[s_tag].filter_bonds(dist_submpx[s_ind + 1 * compress_direction])
        TL, T1 = T1.split(lix, get='tensors', absorb='right', **compress_opts)

        TL.transpose_like(dist_gtn_tens, inplace=True)
        dist_gtn_tens.modify(data=TL.data)

    # print('update', s_tags_list[-1])
    dist_gtn_tens = dist_submpx[s_tags_list[-1]]
    T1 = T1.transpose_like(dist_gtn_tens, inplace=True)
    dist_gtn_tens.modify(data=T1.data)

    return dist_submpx


def _sum_eff_TNs(eff_tns: Sequence['qtn.TensorNetwork'], transpose_bonds=None):

    transpose_bonds = eff_tns[0].outer_inds() if transpose_bonds is None else transpose_bonds

    A_eff_ = None
    for A_eff_tn in eff_tns:
        # print('A eff tn', A_eff_tn, A_eff_tn.exponent)
        # A_eff_tens = A_eff_tn.contract_tags(all)
        A_eff_tens = qtn.tensor_contract(*A_eff_tn.tensors, preserve_tensor=True)
        # print('A eff tens', A_eff_tens)
        A_eff_tens.modify(apply=lambda data: data * 10 ** A_eff_tn.exponent)
        if len(transpose_bonds) > 0:
            A_eff_tens.transpose(*transpose_bonds, inplace=True)

        if A_eff_ is None:
            A_eff_ = A_eff_tens.copy()
        else:
            A_eff_.modify(apply=lambda data: data + A_eff_tens.data)
    return A_eff_

# @profile
def _exactTE_site(dist_submpx: 'qtn.TensorNetwork', dt, deriv_tens: 'qtn.Tensor',
                  submpx_tags: Sequence[str], bonds_i: Sequence['str'], bonds_o: Sequence['str'],
                  inplace=False, compress_direction=CompressDirection.RIGHT, compress_opts=None,
                  verbose_plot=False
                  ) -> Optional['qtn.TensorNetwork1D']:
    """ compute projection of MPS onto site(s) [specified by nsites] or bond (right of site_ind) [nsites=0]
    """
    dist_submpx = dist_submpx if inplace else dist_submpx.copy()

    deriv_tens.transpose(*bonds_o, *bonds_i, inplace=True)

    if len(bonds_o) > 0:
        # dist_cc = dist_submpx.conj(inplace=False, mangle_inner=True)
        # dist_cc.reindex({bi: bo for bi, bo in zip(bonds_i, bonds_o)}, inplace=True)
        # expec = qtn.TensorNetwork([deriv_tens, dist_submpx, dist_cc])
        # print('exact site EXPEC', expec.contract_tags(all))

        # print('exact deriv_tens shape', deriv_tens.shape, deriv_tens.norm())
        fuse_map = {'out': tuple(bonds_o), 'in': tuple(bonds_i)}
        proj_op = deriv_tens.fuse(fuse_map, inplace=False)
        proj_op.transpose('out','in', inplace=True)
    else:
        proj_op = deriv_tens

    # print('exact TE proj op norm', proj_op.norm())

    # print('check antiHermitian', np.linalg.norm(proj_op.data + proj_op.data.T.conj()))
    if verbose_plot:
        plt.figure()
        plt.imshow(np.real(proj_op.data))
        plt.colorbar()
        plt.figure()
        plt.imshow(np.imag(proj_op.data))
        plt.colorbar()
        plt.show()

    # print('proj op', proj_total.shape)
    if len(proj_op.inds) > 0:
        proj_op.modify(apply=lambda x: scipy.linalg.expm(x * -dt))
    else:
        proj_op.modify(apply=lambda x: np.exp(x * -dt))

    if verbose_plot:
        plt.figure()
        plt.imshow(np.real(proj_op.data))
        plt.colorbar()
        plt.figure()
        plt.imshow(np.imag(proj_op.data))
        plt.colorbar()
        plt.title('exponentiated')
        plt.show()

    if len(bonds_o) > 0:
        shape_i = [deriv_tens.ind_size(bi) for bi in bonds_i]
        shape_o = [deriv_tens.ind_size(bo) for bo in bonds_o]
        exp_total = proj_op.unfuse(fuse_map, {'out': shape_o, 'in': shape_i})

        reindex_map = {bo: bi for bo, bi in zip(bonds_o, bonds_i)}
        active_tens: 'qtn.Tensor' = qtn.tensor_contract(*dist_submpx, exp_total)
        active_tens.reindex(reindex_map, inplace=True)

        _update_tn_with_tensor(dist_submpx, active_tens, submpx_tags,
                               compress_direction=compress_direction, compress_opts=compress_opts)
    else:
        ## dist_submpx is a scalar Tensor
        dist_submpx[submpx_tags[0]].modify(apply=lambda x: x * proj_op.data)

    return dist_submpx


def _exactTE_site_2(dist_submpx: 'qtn.TensorNetwork', dt, eff_op: 'qtn.Tensor',
                    sources: Optional[Sequence['qtn.Tensor']], submpx_tags: Sequence[str],
                    bonds_i: Sequence['str'], bonds_o: Sequence['str'],
                    inplace=False, compress_direction=CompressDirection.RIGHT, compress_opts=None
                    ) -> Optional['qtn.TensorNetwork1D']:
    """ solves d^2 x / dt^2 + O * x = b0 + t * b1
        solution:   rotate into diagonal basis U^-1 O U = D;  y = U^-1 x
                    y = yh + yp
                    yh(t) = sum_i a_i exp(-i w_i t) where w_i = sqrt(D_i)
                    yp(t) = m0 + m1 * t
                        yp = D_i^-1 * (b0 + t * b1)
                    a_i = y(0)_i - m0_i
    """
    dist_submpx = dist_submpx if inplace else dist_submpx.copy()

    eff_op.transpose(*bonds_o, *bonds_i, inplace=True)
    fuse_map = {'in': tuple(bonds_i), 'out': tuple(bonds_o)}
    proj_op = eff_op.fuse(fuse_map, inplace=False)
    proj_op.transpose('out','in', inplace=True)

    ## diagonalize operator
    evals, U = np.linalg.eig(proj_op.data)
    U_inv = np.linalg.inv(U)    ## i think proj_op should be diagonal

    ## current state y(0)
    y0 = dist_submpx.contract(inplace=False)
    y0.transpose(*bonds_i, inplace=True)
    y0_vec = y0.fuse({'in': tuple(bonds_i)}, inplace=False)
    y0_vec.modify(apply=lambda x: np.dot(U_inv, x))

    ## sources
    b0, b1, source_vec = None, None, None
    if sources is not None:
        b0 = sources[0]
        b1 = sources[1] if len(sources) > 1 else None

    if b0 is not None:
        b0.transpose(*bonds_o, inplace=True)
        b0_vec = b0.fuse({'out': tuple(bonds_o)})
        b0_vec.modify(apply = lambda x: np.dot(U_inv, x))   # rotate into diag basis
        source_vec = b0_vec.data

    if b1 is not None:
        b1.transpose(*bonds_o, inplace=True)
        b1_vec = b1.fuse({'out': tuple(bonds_o)})
        b1_vec.modify(apply = lambda x: np.dot(U_inv, x))   # rotate into diag basis
        source_vec += b1_vec.data * dt

    ## particular solution yp = m0 +
    m0 = b0_vec.data / evals if b0 is not None else 0.0
    m1 = b1_vec.data / evals if b1 is not None else 0.0
    yp = m0 + dt * m1

    ## homogeneous solution (in rotated space)
    coeffs = y0_vec.data - m0
    yh = coeffs * np.exp(-1.j * np.sqrt(evals) * dt)

    ## total solution
    y_tot = np.dot(U, yh + yp)   ## rotated back
    new_y_tens = qtn.Tensor(data=y_tot, inds='out')
    shape_i = [eff_op.ind_size(bi) for bi in bonds_i]
    new_y_tens.unfuse({'in': tuple(bonds_i)}, {'in': shape_i})

    _update_tn_with_tensor(dist_submpx, new_y_tens, submpx_tags,
                           compress_direction=compress_direction, compress_opts=compress_opts)
    return dist_submpx


def _exactTE_site_0(dist_submpx: 'qtn.TensorNetwork', dt,
                    sources: Optional[Sequence['qtn.Tensor']], submpx_tags: Sequence[str],
                    bonds_i: Sequence['str'], bonds_o: Sequence['str'],
                    inplace=False, compress_direction=CompressDirection.RIGHT, compress_opts=None
                    ) -> Optional['qtn.TensorNetwork1D']:
    """ solves dx / dt = b0 [+ t * b1]
        x = x0 + b0 * dt [+ b1 * t**2]
    """
    dist_submpx = dist_submpx if inplace else dist_submpx.copy()

    ## current state y(0)
    y0 = dist_submpx.contract(inplace=False)
    y0.transpose(*bonds_i, inplace=True)

    ## sources
    b0, b1, data_to_add = None, None, None
    if sources is not None:
        b0 = sources[0]
        b1 = sources[1] if len(sources) > 1 else None

    if b0 is not None:
        b0.transpose(*bonds_o, inplace=True)
        data_to_add = b0.data * dt
    # print('data to add', np.linalg.norm(data_to_add))

    if b1 is not None:
        b1.transpose(*bonds_o, inplace=True)
        data_to_add += b1.data * dt ** 2


    ## total solution
    y_tot = y0.data + data_to_add
    new_y_tens = qtn.Tensor(data=y_tot, inds=bonds_i)

    _update_tn_with_tensor(dist_submpx, new_y_tens, submpx_tags,
                           compress_direction=compress_direction, compress_opts=compress_opts)
    return dist_submpx


# @profile
def _euler_site(dist_submpx: 'qtn.TensorNetwork', dt, deriv: 'qtn.Tensor', submpx_tags=Sequence[str],
                inplace=False, compress_direction=CompressDirection.RIGHT, compress_opts=None
                ) -> Optional['GridTN']:
    """ df/dt = ...
        projected onto qtn manifold defined remaining sites
        for backwards TE, set -dt
        direction: canonicalization direction of output (+1: left canon, -1: right canon)
    """
    dist_submpx = dist_submpx if inplace else dist_submpx.copy()

    active_tens: 'qtn.Tensor' = dist_submpx.contract_tags(all)

    deriv.transpose_like(active_tens, inplace=True)
    active_tens.modify(apply=lambda data: data - dt * deriv.data)  # * 10**(-dist_submpx.exponent))

    _update_tn_with_tensor(dist_submpx, active_tens, submpx_tags,
                           compress_direction=compress_direction, compress_opts=compress_opts)

    return dist_submpx


# @profile
def site_time_evolution(dist_submpx: 'qtn.TensorNetwork', dt, # deriv_tens: 'qtn.Tensor',
                        deriv_tns: Sequence['qtn.TensorNetwork'],
                        submpx_tags: Sequence[str], bonds_i: Sequence[str], bonds_o: Sequence[str],
                        te_order = 4, inplace=False, compress_direction = CompressDirection.RIGHT,
                        compress_level = 1, compress_opts_dict = None) -> 'qtn.TensorNetwork1D':
    """ time integration of specific site, as dictated by te_order. limited to RK TE methods
    """
    dist_submpx = dist_submpx if inplace else dist_submpx.copy()

    # for deriv_tn in deriv_tns:
    #     print('deriv tn', deriv_tn)
    #     print('deriv_tn', helper.norm(deriv_tn), deriv_tn.norm(), deriv_tn.exponent)
    #     for tens in deriv_tn:
    #         print('tens', tens)
    #         print('tens data', tens.data)
    #         print('tens norm', tens.norm())
    # exit()

    if te_order == 0:
        deriv_total = _sum_eff_TNs(deriv_tns, transpose_bonds=list(bonds_o) + list(bonds_i))
        return _exactTE_site(dist_submpx, dt, deriv_total, submpx_tags, bonds_i, bonds_o,
                             inplace=True, compress_direction=compress_direction,
                             compress_opts=compress_opts_dict.get(compress_level, None))

    def euler_func(dist_submpx_, dt_, deriv0=None, compress_level=1, inplace=False, **kwargs):
        return _euler_site(dist_submpx_, dt_, deriv0, submpx_tags, inplace=inplace,
                           compress_direction=compress_direction,
                           compress_opts=compress_opts_dict.get(compress_level, None))

    # @profile
    def deriv_func(dist_submpx_, **kwargs) -> 'qtn.Tensor':

        eff_Ax = []
        for deriv_tn in deriv_tns:
            eff_Ax += [qtn.TensorNetwork([deriv_tn, dist_submpx_])]
            # print('eff_ax', eff_Ax[-1].exponent, deriv_tn.exponent)
            # print('dist_submpx exponent', dist_submpx_.exponent)
            ## dist_submpx exponent should equal 0
        out = _sum_eff_TNs(eff_Ax, transpose_bonds=bonds_o)
        # out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
        out.reindex({bo: bi for bo, bi in zip(bonds_o, bonds_i)}, inplace=True)
        return out

    def add_func(t1, t2, inplace=False, **kwargs):
        return helper.add_tensors(t1, t2, inplace=inplace)

    deriv0 = deriv_func(dist_submpx)
    # print('site time evolution', compress_opts_dict)
    out = helper_TE.time_integration(dist_submpx, dt, euler_func, deriv_func, add_func, helper.scale_tensors,
                                     te_order=te_order, deriv0=deriv0, compress_level=compress_level,
                                     compress_opts_dict=compress_opts_dict)
    if inplace:
        for stags in submpx_tags:
            t1 = next(iter(dist_submpx.select_tensors(stags)))
            t2 = next(iter(out.select_tensors(stags)))
            t1.modify(data=t2.data, inds=t2.inds)
        return dist_submpx
    else:
        return out


def site_time_evolution_2(dist_submpx: 'qtn.TensorNetwork', dt, deriv_tens: 'qtn.Tensor',
                          sources_tens: Optional[Sequence['qtn.Tensor']],
                          submpx_tags: Sequence[str], bonds_i: Sequence[str], bonds_o: Sequence[str],
                          te_order = 4, inplace=False, target_dt: Optional['qtn.TensorNetwork'] = None,
                          compress_direction = CompressDirection.RIGHT,
                          compress_level = 1, compress_opts_dict = None) -> 'qtn.TensorNetwork1D':
    """ time integration of specific site, as dictated by te_order. limited to exact solution at the moment
    """
    dist_submpx = dist_submpx if inplace else dist_submpx.copy()

    if te_order == 0:
        return _exactTE_site_2(dist_submpx, dt, deriv_tens, sources_tens, submpx_tags, bonds_i, bonds_o,
                               inplace=True, compress_direction=compress_direction,
                               compress_opts=compress_opts_dict.get(compress_level, None))

    else:
        raise NotImplementedError


def site_time_evolution_0(dist_submpx: 'qtn.TensorNetwork', dt,
                          sources_tens: Optional[Sequence['qtn.Tensor']],
                          submpx_tags: Sequence[str], bonds_i: Sequence[str], bonds_o: Sequence[str],
                          te_order = 4, inplace=False,
                          compress_direction = CompressDirection.RIGHT,
                          compress_level = 1, compress_opts_dict = None) -> 'qtn.TensorNetwork1D':
    """ time integration of specific site, as dictated by te_order. limited to exact solution at the moment
    """
    dist_submpx = dist_submpx if inplace else dist_submpx.copy()

    if te_order == 0:
        return _exactTE_site_0(dist_submpx, dt, sources_tens, submpx_tags, bonds_i, bonds_o,
                               inplace=True, compress_direction=compress_direction,
                               compress_opts=compress_opts_dict.get(compress_level, None))

    else:
        raise NotImplementedError


#####################################
####        TDVP solver         #####
#####################################

class TDVPSolver(LocalSolver):
    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 operators: Optional[Sequence['MPO_type']] = None,
                 targets: Optional[Sequence['MPS_type']] = None,
                 mps_inds: Sequence[int] = None,
                 norm_env0_Ls: Sequence['qtn.Tensor'] = None,
                 norm_env0_Rs: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 te_order=0, compress_config: CompressionConfiguration=None,
                 backprop_edge=False):

        # operators = [] if operators is None else list(operators)
        self.A_envsLs, self.A_envsRs = [], []
        self.b_envsLs, self.b_envsRs = [], []

        super().__init__(trial_state, targets=targets, operators=operators, mps_inds=mps_inds)

        norm_env0_Ls = [None] * self.num_operators if norm_env0_Ls is None else norm_env0_Ls
        norm_env0_Rs = [None] * self.num_operators if norm_env0_Rs is None else norm_env0_Rs
        self.A_envsLs = [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, self.operators[i],
                                     init_env=norm_env0_Ls[i], mps_inds=self.mps_inds)
                            for i in range(self.num_operators)]
        self.A_envsRs = [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, self.operators[i],
                                     init_env=norm_env0_Rs[i], mps_inds=self.mps_inds)
                            for i in range(self.num_operators)]

        self.b_envsLs, self.b_envsRs = [], []
        for ib in range(self.num_targets):
            b_envL = Environment(self.L, EnvironmentSide.LEFT, self.targets[ib], self.bra, None,
                                 mps_inds=self.mps_inds,
                                 init_env=ovlp_env0_Ls[ib] if ovlp_env0_Ls is not None else None)
            b_envR = Environment(self.L, EnvironmentSide.RIGHT, self.targets[ib], self.bra, None,
                                 mps_inds=self.mps_inds,
                                 init_env=ovlp_env0_Rs[ib] if ovlp_env0_Rs is not None else None)
            self.b_envsLs += [b_envL]
            self.b_envsRs += [b_envR]

        self.te_order = te_order
        self.solve_type = None
        self.is_H = False
        self.compress_config = CompressionConfiguration() if compress_config is None else compress_config
        self.max_bond = self.compress_config.max_bonds.get(1, None)

        self.gammas = None
        self.lambdas = None
        self.backprop_edge = backprop_edge


    @property
    def left_envs(self):
        return self.A_envsLs + self.b_envsLs

    @property
    def right_envs(self):
        return self.A_envsRs + self.b_envsRs


    def reinitialize_envs(self):
        """ if boundary envs are None, uses old values
        """
        for env in self.A_envsLs:
            env.modify(ket=self.ket, bra=self.bra, mps_inds=self.mps_inds)

        for i in range(len(self.b_envsLs)):
            env = self.b_envsLs[i]
            target = self.targets[i]
            env.modify(ket=target, bra=self.bra, mps_inds=self.mps_inds)


        for env in self.A_envsRs:
            env.modify(ket=self.ket, bra=self.bra, mps_inds=self.mps_inds)

        for i in range(len(self.b_envsRs)):
            env = self.b_envsRs[i]
            target = self.targets[i]
            env.modify(ket=target, bra=self.bra, mps_inds=self.mps_inds)

        self.err = np.inf
        self.is_conv = False


    def create_like(self, copy=True, new_ket=None, **kwargs):
        norm_env0_Ls = [A_envL[0] for A_envL in self.A_envsLs]
        norm_env0_Rs = [A_envR[self.L-1] for A_envR in self.A_envsRs]
        ovlp_env0_Ls = [b_envL[0] for b_envL in self.b_envsLs]
        ovlp_env0_Rs = [b_envR[self.L-1] for b_envR in self.b_envsRs]

        new_solver = TDVPSolver((self.ket.copy() if copy else self.ket) if new_ket is None else new_ket,
                                targets=kwargs.get('targets', [t.copy() if copy else t for t in self.targets]),
                                operators=kwargs.get('operators', [o.copy() if copy else o for o in self.operators]),
                                mps_inds=kwargs.get('mps_ind_range', self.mps_inds),
                                norm_env0_Ls=kwargs.get('norm_env0_Ls', norm_env0_Ls),
                                norm_env0_Rs=kwargs.get('norm_env0_Rs', norm_env0_Rs),
                                ovlp_env0_Ls=kwargs.get('ovlp_env0_Ls', ovlp_env0_Ls),
                                ovlp_env0_Rs=kwargs.get('ovlp_env0_Rs', ovlp_env0_Rs),
                                te_order=kwargs.get('te_order', self.te_order),
                                compress_config=kwargs.get('compress_config', self.compress_config),
                                backprop_edge=kwargs.get('backprop_edge', self.backprop_edge),
                                # solve_type=kwargs.get('solve_type', self.solve_type)
                                )
        return new_solver

    def copy(self, deep=True):
        new_solver = self.create_like(copy=deep)

        new_solver.A_envsL = [A_envsL.copy() for A_envsL in self.A_envsLs]
        new_solver.A_envsR = [A_envsR.copy() for A_envsR in self.A_envsRs]
        # new_solver.b_envsLs = [bL.copy() for bL in self.b_envsLs]
        # new_solver.b_envsRs = [bR.copy() for bR in self.b_envsRs]

        # new_solver.left_envs = new_solver.A_envsLs + new_solver.b_envsLs
        # new_solver.right_envs = new_solver.A_envsRs + new_solver.b_envsRs

        # new_solver.solve_type = self.solve_type
        # new_solver.err = self.err
        # new_solver.is_conv = self.is_conv
        return new_solver


    def _get_A_effs(self, left_site_pos, nsites):

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]

        A_effs = []
        for i in range(self.num_operators):
            A_left  = self.A_envsLs[i][left_site_pos]
            A_right = self.A_envsRs[i][left_site_pos + nsites - 1]
            # print('A left', A_left)
            # print('A right', A_right)

            A_eff = qtn.TensorNetwork([])
            A = self.operators[i]
            if isinstance(site_inds, (list,tuple)):
                for si in site_inds:
                    A_eff.add(A[si])
            else:
                A_eff.add(A[site_inds])

            if A_left is not None:
                A_eff.add(A_left)
            if A_right is not None:
                A_eff.add(A_right)

            #### !!! CHANGED HERE
            A_eff.exponent += A.exponent
            A_effs += [A_eff]
            # print('A eff', A_eff.exponent, A.exponent, helper.norm(A_eff))
            # print('A_eff', left_site_pos, nsites, A_eff)

        return A_effs


    def _get_b_effs(self, left_site_pos, nsites) -> Sequence['qtn.TensorNetwork']:

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]

        b_effs = []
        for i in range(self.num_targets):
            b_left = self.b_envsLs[i][left_site_pos]
            b_right = self.b_envsRs[i][left_site_pos + nsites - 1]

            b_eff = qtn.TensorNetwork([])
            b = self.targets[i]
            if isinstance(site_inds, (list, tuple)):
                for si in site_inds:
                    b_eff.add(b[si])
            else:
                b_eff.add(b[site_inds])

            if b_left is not None:
                b_eff.add(b_left)
            if b_right is not None:
                b_eff.add(b_right)

            #### !!! CHANGED HERE
            b_eff.exponent += b.exponent
            b_effs += [b_eff]

        return b_effs


    def _sum_eff_TNs(self, eff_tns: Sequence['qtn.TensorNetwork'], transpose_bonds=None):

        transpose_bonds = eff_tns[0].outer_inds() if transpose_bonds is None else transpose_bonds

        A_eff_ = None
        for A_eff_tn in eff_tns:
            # print('A eff tn', A_eff_tn, A_eff_tn.exponent)
            # A_eff_tens = A_eff_tn.contract_tags(all)
            A_eff_tens = qtn.tensor_contract(*A_eff_tn.tensors, preserve_tensor=True)
            # print('A eff tens', A_eff_tens)
            A_eff_tens.modify(apply=lambda data: data * 10 ** A_eff_tn.exponent)
            if len(transpose_bonds) > 0:
                A_eff_tens.transpose(*transpose_bonds, inplace=True)

            if A_eff_ is None:
                A_eff_ = A_eff_tens.copy()
            else:
                A_eff_.modify(apply=lambda data: data + A_eff_tens.data)
        return A_eff_


    # @profile
    def _site_time_evolution(self, dt, left_site_pos, nsites=1,
                             compress_direction=CompressDirection.RIGHT,
                             ) -> Optional[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        """
        # print('check orthog site TE', left_site_pos)
        # helper.check_orthog(self.ket)
        # helper.check_orthog(self.bra)

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        # A_effs = []
        # for i in range(self.num_operators):
        #     A_left  = self.A_envsLs[i][left_site_pos]
        #     A_right = self.A_envsRs[i][left_site_pos + nsites - 1]
        #
        #     A_eff = qtn.TensorNetwork([])
        #     A = self.operators[i]
        #     if isinstance(site_inds, (list,tuple)):
        #         for si in site_inds:
        #             A_eff.add(A[si])
        #     else:
        #         A_eff.add(A[site_inds])
        #
        #     if A_left is not None:
        #         A_eff.add(A_left)
        #     if A_right is not None:
        #         A_eff.add(A_right)
        #
        #     # A_eff.exponent = self.A_envsLs[i].exponent
        #     A_eff.exponent = A.exponent
        #     A_effs += [A_eff]
        # print('get A_effs')
        A_effs = self._get_A_effs(left_site_pos, nsites)

        ## change indices from bra (missing T*[i]) to ket
        # bra_to_ket_inds = {}
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))
            # bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))


        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]
        # print('ket to bra inds', ket_to_bra_inds)
        # print('bonds o', bonds_o)
        # print('bonds i', bonds_i)
        # deriv_total = sum_Aeffs()
        # deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bonds_o + list(bonds_i))
        # print('deriv total', deriv_total)

        site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                            inplace=True, te_order=self.te_order,
                            compress_direction=compress_direction,
                            compress_level=1, compress_opts_dict=self.compress_config)

        ## i think this is already done in canonize
        self.set_bra_from_ket(sites = list(range(left_site_pos,left_site_pos + nsites)))

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            self.canonize(site_inds[-1], cur_orthog=site_inds[0])     ## also sets bra from ket
            for i in range(left_site_pos, left_site_pos + nsites - 1):
                # print('(site) update envs left', i, 'orthog at', site_inds[-1])
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_left(i, canonize=False)
        else:
            self.canonize(site_inds[0], cur_orthog=site_inds[-1])  ## also sets bra from ket
            for i in range(left_site_pos + nsites - 1, left_site_pos, -1):
                # print('(site) update envs right', i, 'orthog at', site_inds[0])
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_right(i, canonize=False)

        return self.ket


    def _bond_time_evolution(self, dt, cur_orthog,
                             compress_direction=CompressDirection.RIGHT
                             ) -> Sequence[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        """
        # site_pos = list(range(self.L))[bond_ind]
        # bond_ind = self.mps_inds[bond_pos]

        # helper.check_orthog(self.ket)
        ket_usvt = MPS_USVT.from_MPS(self.ket, canon_site=cur_orthog, cur_orthog=cur_orthog, direction=compress_direction)
        bra_usvt = MPS_USVT.from_MPS(self.bra, canon_site=cur_orthog, cur_orthog=cur_orthog, direction=compress_direction)
        Q_tens = ket_usvt[cur_orthog]
        S_tens = ket_usvt.get_S_tensor()
        Q_conj = bra_usvt[cur_orthog]
        S_conj = bra_usvt.get_S_tensor()

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = []
        for i in range(self.num_operators):
            A_left = self.A_envsLs[i][cur_orthog]
            A_right = self.A_envsRs[i][cur_orthog]
            # print('A left', A_left)
            # print('A right', A_right)

            A_eff = qtn.TensorNetwork([])
            A = self.operators[i]
            A_eff.add(A[cur_orthog])

            if A_left is not None:
                A_eff.add(A_left)
            if A_right is not None:
                A_eff.add(A_right)

            ## add Q(R) / (L)Q tensor
            A_eff.add([Q_tens, Q_conj])

            # A_eff.exponent = self.A_envsLs[i].exponent
            ## !!! changed = -> +=
            A_eff.exponent += A.exponent
            A_effs += [A_eff]

        ## change indices from bra (missing T*[i]) to ket
        shared, notshared = S_tens.filter_bonds(Q_tens)
        ket_inds = shared + notshared

        shared, notshared = S_conj.filter_bonds(Q_conj)
        bra_inds = shared + notshared

        # deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bra_inds + ket_inds)
        dist_submpx = qtn.TensorNetwork([S_tens], virtual=True)

        # for A_eff in A_effs:
        #     print('A eff', A_eff)
        # print('ket', ket_usvt)
        # print('bra', bra_usvt)
        # print('bonds_o', bra_inds, 'bonds_i', ket_inds)
        site_time_evolution(dist_submpx, dt, A_effs, [next(iter(S_tens.tags))],
                            bonds_i = ket_inds, bonds_o = bra_inds, te_order=self.te_order, inplace=True,
                            compress_direction=compress_direction, compress_opts_dict=self.compress_config
                            )

        # print('cur orthog to MPS', cur_orthog)
        new_ket = ket_usvt.to_MPS()
        ket_tens = self.ket[cur_orthog]
        new_tens = new_ket[cur_orthog]
        new_tens = new_tens.transpose_like(ket_tens, inplace=True)
        ket_tens.modify(data = new_tens.data)
        self.set_bra_from_ket(sites=[cur_orthog])

        ### envs are reset in main loop
        # ## inplace update of ket, bra
        # if compress_direction == CompressDirection.RIGHT:
        #     self.canonize(cur_orthog + 1, cur_orthog=cur_orthog)  ## also sets bra from ket
        #     self._update_envs_left(cur_orthog, canonize=False)
        # else:
        #     self.canonize(cur_orthog - 1, cur_orthog=cur_orthog)  ## also sets bra from ket
        #     self._update_envs_right(cur_orthog, canonize=False)

        return self.ket


    def canonize(self, i, cur_orthog=None):
        if self.gammas is None:
            super().canonize(i, cur_orthog=cur_orthog)

        else:
            new_mpx = helper.gamma_lambda_to_mps(self.gammas, self.lambdas, canon_site=i, view_like=self.ket)
            print('canonize check orthog', i)
            helper.check_orthog(new_mpx)

            for pos in range(self.ket.L):
                new_tens = new_mpx[pos]
                new_tens = new_tens.transpose_like(self.ket[pos], inplace=True)
                self.ket[pos].modify(data=new_tens.data)
            self.set_bra_from_ket(reinit_envs=False)


    def take_time_step(self, dt, grid=None, do_adapt=False, **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        max_bond = self.max_bond
        # do_adapt = False

        self.canonize(0)   ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        self._build_all_envs_right(1, canonize=False)
        # for tmp_env in self.A_envsRs:
        #     tmp = tmp_env[L // 2 - 1]
        #     mpo = tmp_env.operator
        #     print('right env L//2-1', tmp, tmp.norm())
        #     mpo_tmp = qtn.TensorNetwork([tmp, tmp_env.operator[:L//2]])
        #     print('mpo env', mpo.exponent, mpo_tmp.norm(), mpo_tmp.outer_inds())
        #     print('mpo norm 0 ', mpo[:L // 2].norm())
        #     print('mpo norm 1', mpo[L // 2:].norm())
        #
        #     mps = tmp_env.ket
        #     print('mps norm 0 ', mps[:L // 2].norm())
        #     print('mps norm 1', mps[L // 2:].norm())
        #     print('mps exp', mps.exponent)
        #
        #     bra = mps[L//2:].conj()
        #     bond_l = mps.bond(L//2-1, L//2)
        #     bra.reindex({bond_l: bond_l + '_'}, inplace=True)
        #     print('bra', bra)
        #     print('ket', mps[L//2:])
        #     tmp = helper.expectation_value(mps[L//2:], mpo[L//2:], bra=bra)
        #     print('meas expec 1', tmp, tmp.norm())
        #
        # exit()

        ## right sweep
        self.take_time_step_l2r(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)
        # exit()

        ### right to left sweep
        # self.canonize(L-1)  ## does not scale canonical tensors
        # self._build_all_envs_left(1, canonize=False)
        self.take_time_step_r2l(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)

        print('done sweep')
        # exit()
        return self.ket


    def take_time_step_l2r(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        max_bond = self.max_bond
        # do_adapt = False
        print('l2r tdvp', dt)
        # print('ket L', L)
        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if canonize:
            self.canonize(0)   ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_right(1, canonize=False)

        ## right sweep
        nsites = 0
        direction = CompressDirection.RIGHT
        site_ind = 0
        cur_orthog = 0
        while site_ind < L:

            if site_ind == L - 1 and nsites == 2:
                ## previously updated L-2, L-1
                print('continue', site_ind)
                site_ind += 1
                continue

            ## num sites for update
            if site_ind == L - 1:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                max_bond_ = max_bond  # min(max_bond, 2**(cur_orthog+1), 2**(L-cur_orthog-1))
                # print('max bond', max_bond_, cur_orthog)
                # print('bond size', self.ket.bond_size(cur_orthog, cur_orthog + 1))
                adapt = do_adapt and self.ket.bond_size(cur_orthog, cur_orthog + 1) < max_bond_
                # print('do_adapt', adapt)

            if not adapt:  # one site update of site_ind
                if nsites == 2:  ## previously updated this site
                    nsites = 1
                    site_ind += 1
                    continue

            ## backward propagation (skipped for first iteration)
            if 0 < site_ind < L:
                if nsites == 1:
                    # bond_ind = site_ind - 1
                    print('BACK prop (+1) bond ind', cur_orthog, direction)
                    # print('check orthog', helper.check_orthog(self.ket))
                    self._bond_time_evolution(-dt, cur_orthog, compress_direction=direction )

                    ## move canonicalization to next site (site_ind)
                    self.canonize(site_ind, cur_orthog=cur_orthog)      ## updates bra internally
                    # print('(bond) update left env', cur_orthog, 'canon at', site_ind)
                    # print('check orthog', helper.check_orthog(self.ket))
                    self._update_envs_left(cur_orthog, canonize=False)

                else:
                    # print('BACK prop (+1) site ind', site_ind, direction)
                    # print('check orthog', helper.check_orthog(self.ket))
                    self._site_time_evolution(-dt, site_ind, nsites = 1, compress_direction=direction)


            nsites = 2 if adapt else 1

            ## forward propagation
            # print('self.ket norm', helper.norm(self.ket))
            # print('check orthog', helper.check_orthog(self.ket))
            print('FORWARD prop sites (+1)', list(range(site_ind,site_ind+nsites)) )
            self._site_time_evolution(dt, left_site_pos=site_ind, nsites=nsites, compress_direction=direction)
                ## updates environment within function to site_ind + nsites - 1
            cur_orthog = site_ind + nsites - 1
            site_ind = site_ind + nsites - 1 if nsites > 1 else site_ind + nsites
            # print('check orthog', helper.check_orthog(self.ket))
            # print('next cur orthog', cur_orthog, 'next site ind', site_ind)

        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if self.backprop_edge:
            raise NotImplementedError('backprop of last site in TDVP not implemented')

        print('done left to right sweep')
        return self.ket


    def take_time_step_r2l(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        max_bond = self.max_bond
        print('r2l tdvp', dt)
        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if canonize:
            self.canonize(L-1)  ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_left(1, canonize=False)

        ### right to left sweep
        print('backward sweep')
        nsites = 0
        direction = CompressDirection.LEFT
        site_ind = L - 1
        cur_orthog = L - 1
        while site_ind >= 0:

            if site_ind == 0 and nsites == 2:
                ## previously updated 0, 1
                print('continue', site_ind)
                site_ind -= 1
                continue

            if site_ind == 0:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                # max_bond_ = min(max_bond, 2 ** site_ind, 2 ** (L - site_ind))
                # print('max bond', max_bond_, site_ind)
                # print('bond size', site_ind, site_ind-1, self.ket.bond_size(site_ind, site_ind - 1))
                # adapt = do_adapt and self.ket.bond_size(site_ind, site_ind - 1) < max_bond_

                max_bond_ = max_bond  # min(max_bond, 2 ** cur_orthog, 2 ** (L - cur_orthog))
                # print('max bond', max_bond_, cur_orthog)
                # print('bond size', cur_orthog, cur_orthog - 1, self.ket.bond_size(cur_orthog, cur_orthog - 1))
                adapt = do_adapt and self.ket.bond_size(cur_orthog, cur_orthog - 1) < max_bond_
                # print('do_adapt', adapt)

            if not adapt:  # one site update of site_ind
                if nsites == 2:  ## previously updated this site
                    nsites = 1
                    site_ind -= 1
                    continue

            if 0 <= site_ind < L - 1:    ## skip initial back propagation
                if nsites == 1:
                    print('BACK prop (-1) bond ind', cur_orthog, direction )
                    # print('check orthog', helper.check_orthog(self.ket))
                    self._bond_time_evolution(-dt, cur_orthog, compress_direction=direction)

                    ## move canonicalization to next site (site_ind)
                    self.canonize(site_ind, cur_orthog=cur_orthog)  ## updates bra internally
                    # print('(bond) update right env', cur_orthog, 'canon at', site_ind)
                    # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                    self._update_envs_right(cur_orthog, canonize=False)
                    # print('new site ind', site_ind)
                else:
                    # print('BACK prop (-1) site ind', site_ind, direction)
                    self._site_time_evolution(-dt, site_ind, nsites=1, compress_direction=direction)

                # print('back prop (-1) check orthog', site_ind, cur_orthog)
                # helper.check_orthog(self.ket)
                # helper.check_orthog(self.bra)
                # if grid is not None:
                #     plt.figure()
                #     gtn = grid.make_gridTN(data=self.ket)
                #     plt.imshow(np.abs(gtn.get_data()) ** 2)
                #     plt.colorbar()
                #     plt.show()

            nsites = 2 if adapt else 1
            cur_orthog = site_ind - nsites + 1    ## get left site ind

            # print('self.ket norm', helper.norm(self.ket))
            # print('check orthog', helper.check_orthog(self.ket))
            print('FORWARD prop sites (-1)', list(range(cur_orthog,cur_orthog+nsites)) )
            self._site_time_evolution(dt, cur_orthog, nsites=nsites, compress_direction=direction)
            site_ind = site_ind - nsites + 1 if nsites > 1 else site_ind - nsites
            # print('check orthog', helper.check_orthog(self.ket))
            # print( 'next cur orthog', cur_orthog, 'next site ind', site_ind,)

            # print('forward prop (-1) check orthog', cur_orthog)
            # helper.check_orthog(self.ket)
            # helper.check_orthog(self.bra)
            # if grid is not None:
            #     plt.figure()
            #     gtn = grid.make_gridTN(data=self.ket)
            #     plt.imshow(np.abs(gtn.get_data()) ** 2)
            #     plt.colorbar()
            #     plt.show()

        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if self.backprop_edge:
            raise NotImplementedError('backprop of last site in TDVP not implemented')

        print('done right to left sweep')
        return self.ket


    def take_time_step_v2(self, dt, grid=None, do_adapt=False, **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        max_bond = self.max_bond
        cutoff = self.compress_config.cutoffs.get(1, CUTOFF)

        num_it = 0
        nsites = 2 if do_adapt and self.ket.max_bond() < max_bond else 1      ## don't do mixed 1/2 site updates
        dt_ = dt / nsites
        num_layers = nsites

        while num_it < num_layers:      ## number of iterations (1 for 1 site, 2 for 2 sites)

            site_ind = 0
            while site_ind < L:
                # block_nsites = 1 if site_ind == num_it - 1 else min(L - site_ind, nsites)

                print('self ket norm', self.ket.norm(), helper.norm(self.ket))


                ## compute site, bond derivatives
                forward_derivs = {}

                helper.collect_exponent(self.ket, inplace=True)
                new_ket = self.ket      ## bond dim can change after compress etc
                ket_gammas, ket_lambdas, gl_norm = helper.mps_to_gamma_lambda(new_ket, cutoff=CUTOFF)
                check_mps = helper.gamma_lambda_to_mps(ket_gammas, ket_lambdas, view_like=new_ket)
                check_mps.exponent = new_ket.exponent       ### signs can in Gammas can be different
                check_mps.exponent += gl_norm
                print('gl/mps diff', helper.distance(check_mps, new_ket))
                # print('new mps', new_ket)
                # print('check mps', check_mps)
                self.ket = check_mps
                self.set_bra_from_ket()
                self.gammas = ket_gammas
                self.lambdas = ket_lambdas

                ## get <Ax|x> init envs
                print('canonize L-1')
                self.canonize(L - 1)  ## does not scale canonical tensors
                # self.ket = helper.gamma_lambda_to_mps(ket_gammas, ket_lambdas, canon_site=L-1, view_like=new_ket)
                # self.set_bra_from_ket()
                self._build_all_envs_left(1, canonize=False)

                print('canonize 0')
                self.canonize(0)  ## does not scale canonical tensors
                # self.ket = helper.gamma_lambda_to_mps(ket_gammas, ket_lambdas, canon_site=0, view_like=new_ket)
                # self.set_bra_from_ket()
                self._build_all_envs_right(1, canonize=False)

                #################################
                active_sites = []
                site_pairs = []
                tmp_site_ind = 0
                while tmp_site_ind < L:
                    block_nsites = 1 if tmp_site_ind == num_it - 1 else min(L - tmp_site_ind, nsites)
                    ## to stagger layers if doing nsites==2, catching for last tensor
                    sites = tuple(range(tmp_site_ind, tmp_site_ind + block_nsites))
                    print('tmp sites', sites)
                    site_pairs += [sites]
                    tmp_site_ind = tmp_site_ind + block_nsites

                block_nsites = 1 if site_ind == num_it - 1 else min(L - site_ind, nsites)
                sites = tuple(range(site_ind, site_ind + block_nsites))
                print('sites', sites)
                active_sites += [sites]

                ## forward propagation derivatives
                forward_derivs[sites] = self._get_A_effs(site_ind, block_nsites)

                ## next site_ind
                site_ind = site_ind + block_nsites
                ######################################

                ## update sites
                new_ket = qtn.TensorNetwork([])
                for sites in site_pairs:
                    gs = [ket_gammas[s] for s in sites]
                    ls = [ket_lambdas[s - 1] for s in sites if s >= 1]
                    if sites[-1] < L - 1:
                         ls += [ket_lambdas[sites[-1]]]

                    tens_list = []
                    for i in range(len(gs) - 1):
                        if sites[i] == 0:   #  or sites[i] == L-1:
                            # self.canonize(L-1)
                            tens_list += [gs[i].copy()]
                        else:
                            tens_list += [qtn.tensor_contract(gs[i], ls[i], output_inds=gs[i].inds)]

                    if sites[-1] == 0 or sites[-1] == L - 1:
                        tens_list += [qtn.tensor_contract(gs[-1], ls[-1], output_inds=gs[-1].inds)]
                    else:
                        tens_list += [qtn.tensor_contract(gs[-1], ls[-2], ls[-1], output_inds=gs[-1].inds)]

                    dist_mpx = qtn.TensorNetwork(tens_list)


                    ## gl inds to ket inds
                    sL, sR = sites[0], sites[-1]
                    reindex_gl_to_ket = {}
                    if sL > 0:
                        ind_g_l = ket_lambdas[sL-1].inds[0]  # next(iter(qtn.bonds(ket_lambdas[sL-1], ket_gammas[sL])))
                        ind_k_l = next(iter(qtn.bonds(self.ket[sL - 1], self.ket[sL])))
                        if ind_g_l != ind_k_l:
                            reindex_gl_to_ket[ind_g_l] = ind_k_l
                    if sR < L - 1:
                        ind_g_r = ket_lambdas[sR].inds[0]  # next(iter(qtn.bonds(ket_lambdas[sR], ket_gammas[sR]))) if sR < L - 1 else None
                        ind_k_r = next(iter(qtn.bonds(self.ket[sR], self.ket[sR + 1]))) if sR < L - 1 else None
                        if ind_g_r != ind_k_r:
                            reindex_gl_to_ket[ind_g_r] = ind_k_r
                    reindex_ket_to_gl = {val: k for k, val in reindex_gl_to_ket.items()}
                    dist_mpx.reindex(reindex_gl_to_ket, inplace=True)


                    # ############### check ##############
                    # print('check dist check orthog', sites[-1])
                    # self.canonize(sites[-1])
                    # ket_canon = self.ket.copy()
                    # helper.check_orthog(ket_canon)
                    # ket_dist = ket_canon[sites[-1] - len(sites) + 1:sites[-1] + 1]
                    # # print(ket_dist, sites)
                    # # ket_dist.exponent = 0.0
                    # print('ket dist', helper.distance(dist_mpx, ket_dist))

                    if sites in active_sites:
                        ## directly update self.ket
                        # dist_mpx = self.ket[sites[-1] - len(sites) + 1:sites[-1] + 1]

                        # self.canonize(1)
                        # if sites[0] == 0:
                        #     k0 = self.ket[0].copy()
                        #     k0.reindex(reindex_ket_to_gl, inplace=True)
                        #     k0.transpose_like(ket_gammas[0], inplace=True)
                        #     print('diff 0', np.linalg.norm(k0.data - ket_gammas[0].data))
                        # if sites[-1] == L-1:
                        #     kL = self.ket[L-1].copy()
                        #     kL.reindex(reindex_ket_to_gl, inplace=True)
                        #     kL.transpose_like(ket_gammas[L-1], inplace=True)
                        #     print('diff L-1', np.linalg.norm(kL.data - ket_gammas[-1].data))

                        #####################################


                        ## change indices from bra (missing T*[i]) to ket
                        ket_to_bra_inds = {}
                        for site_p in sites:
                            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

                        bonds_i = dist_mpx.outer_inds()
                        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

                        print('active site', sites)
                        A_effs = forward_derivs[sites]
                        deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bonds_o + list(bonds_i))

                        submpx_tags = [self.ket.site_tag_id.format(s) for s in sites]
                        site_time_evolution(dist_mpx, dt_, deriv_total, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                                            inplace=True, te_order=self.te_order,
                                            compress_direction=CompressDirection.RIGHT,
                                            compress_level=1, compress_opts_dict=self.compress_config)
                        self.set_bra_from_ket()
                        self.gammas = None
                        self.lambdas = None

                    dist_mpx.reindex(reindex_ket_to_gl, inplace=True)

                    # apply inverse lambda to the right end (bc we're taking l-g-l for each site.
                    if sR < L - 1:
                        tens_R = dist_mpx.select_tensors(self.ket.site_tag(sites[-1]))[0]
                        lam_inv = ls[-1].copy()
                        lam_inv.modify(apply = lambda x: 1./ x)
                        new_tens_R = tens_R.contract(lam_inv, output_inds=tens_R.inds)
                        new_tens_R.transpose_like(tens_R, inplace=True)
                        tens_R.modify(data=new_tens_R.data)
                        # tens_R.modify(apply = lambda x: np.tensordot(x, np.diag(1./ls[-1].data), axes=[ind,0]))

                    # print('add dist to new ket')
                    new_ket.add(dist_mpx)

                new_ket.view_like(self.ket, inplace=True)
                new_ket.exponent = self.ket.exponent
                # print('self.ket norm', helper.norm(new_ket), new_ket.copy().norm())
                # print('collect exponent?')
                # helper.collect_exponent(new_ket, inplace=True)
                # print('new ket norm', helper.norm(new_ket), new_ket.copy().norm())

                ## update ket
                self.ket = new_ket
                # new_ket = self.ket.copy()
                self.set_bra_from_ket()
                ket_gammas, ket_lambdas, gl_norm = helper.mps_to_gamma_lambda(self.ket, cutoff=CUTOFF)
                self.ket.exponent += gl_norm
                bra_gammas, bra_lambdas, gl_norm = helper.mps_to_gamma_lambda(self.bra, cutoff=CUTOFF)
                self.bra.exponent += gl_norm
                print('self.ket norm (1)', helper.norm(self.ket), self.ket.copy().norm())

                check_mps = helper.gamma_lambda_to_mps(ket_gammas, ket_lambdas, view_like=self.ket)
                check_mps.exponent = self.ket.exponent  ### signs can in Gammas can be different
                check_mps.exponent += gl_norm
                print('gl/mps diff (1)', helper.distance(check_mps, new_ket))

                self.gammas = ket_gammas
                self.lambdas = ket_lambdas

                print('self.ket norm', helper.norm(self.ket), self.ket.copy().norm())

                print('canonize (1) L-1')
                self.canonize(L - 1)  ## does not scale canonical tensors
                self._build_all_envs_left(1, canonize=False)

                print('canonize (1) 0')
                self.canonize(0)     ## does not scale canonical tensors
                self._build_all_envs_right(1, canonize=False)


                ## backward propagation of inds between active_sites
                new_lambdas = {}
                for sites in active_sites:
                    bond_ind = sites[-1]
                    print('bond ind', bond_ind)
                    if bond_ind == L - 1:
                        continue

                    ## relevant indices
                    ind_kg_r = ket_lambdas[bond_ind].inds[0]
                    ind_bg_r = bra_lambdas[bond_ind].inds[0]
                    ind_k_r = self.ket.bond(bond_ind, bond_ind + 1)  ## to right env
                    ind_b_r = self.bra.bond(bond_ind, bond_ind + 1)  ## to right env

                    ## backward propagation derivatives     (indices)
                    A_effs = self._get_A_effs(bond_ind, 1)  ## missing info at bond_ind; connect via left inds

                    ############### check ##############
                    print('check dist check orthog', bond_ind)
                    self.canonize(bond_ind)

                    ## left canonical site at bond_ind to add to A_eff
                    bra_gamma = bra_gammas[bond_ind].copy()
                    ket_gamma = ket_gammas[bond_ind].copy()
                    bg_reindex_map = {ind_bg_r: ind_bg_r + '_'}
                    kg_reindex_map = {ind_kg_r: ind_kg_r + '_'}

                    print('bra gamma', bra_gamma)
                    print('ket gamma', ket_gamma)

                    if bond_ind > 0:
                        ind_kg_l = ket_lambdas[bond_ind - 1].inds[0]
                        ind_bg_l = bra_lambdas[bond_ind - 1].inds[0]
                        ind_k_l = self.ket.bond(bond_ind - 1, bond_ind)
                        ind_b_l = self.bra.bond(bond_ind - 1, bond_ind)

                        new_bra = bra_gamma.contract(bra_lambdas[bond_ind - 1], output_inds=bra_gamma.inds)
                        new_bra.transpose_like(bra_gamma, inplace=True)
                        bra_gamma = new_bra   # .modify(data=new_bra.data)
                        bg_reindex_map[ind_bg_l] = ind_b_l

                        new_ket = ket_gamma.contract(ket_lambdas[bond_ind - 1], output_inds=ket_gamma.inds)
                        new_ket.transpose_like(ket_gamma, inplace=True)
                        ket_gamma = new_ket   # .modify(data=new_ket.data)
                        kg_reindex_map[ind_kg_l] = ind_k_l

                    bra_gamma.reindex(bg_reindex_map, inplace=True)
                    ket_gamma.reindex(kg_reindex_map, inplace=True)

                    for A_eff in A_effs:
                        A_eff.add([bra_gamma, ket_gamma])
                        # print('A eff', A_eff)

                    # print('lambda', bond_ind, ket_lambdas[bond_ind])
                    lambda_tens = qtn.Tensor(data=np.diag(ket_lambdas[bond_ind].data), inds=(ind_kg_r + '_', ind_k_r))
                    bonds_i = [ind_kg_r + '_', ind_k_r]
                    bonds_o = [ind_bg_r + '_', ind_b_r]
                    print('lambda tens norm', lambda_tens.norm())
                    old_lambda = lambda_tens.copy()
                    # print('bonds_o', bonds_o)
                    # print('bonds_i', bonds_i)

                    s_tag = f'_S_{bond_ind}_'
                    lambda_tens.add_tag(s_tag)
                    dist_mpx = qtn.TensorNetwork([lambda_tens], virtual=True)

                    deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bonds_o + list(bonds_i))

                    # print('dist mpx', dist_mpx)
                    # print('deriv total', deriv_total)

                    ## ket_lambda is updated in place
                    site_time_evolution(dist_mpx, -dt_, deriv_total, [s_tag], bonds_i=bonds_i, bonds_o=bonds_o,
                                        inplace=True, te_order=self.te_order,
                                        compress_level=1, compress_opts_dict=self.compress_config)

                    lambda_tens.drop_tags()
                    new_lambdas[bond_ind] = lambda_tens

                    old_lambda.transpose_like(lambda_tens, inplace=True)
                    print('lambda diffs', np.linalg.norm(old_lambda.data - lambda_tens.data))

                ## inplace update of actual gammas/lambdas
                for bond_ind in new_lambdas.keys():
                    lambda_tens = new_lambdas[bond_ind]
                    print('new lambda norm', lambda_tens.norm())
                    lambda_to_update = ket_lambdas[bond_ind]
                    gamma_to_update_L = ket_gammas[bond_ind]
                    gamma_to_update_R = ket_gammas[bond_ind + 1]

                    ind_kg_r = ket_lambdas[bond_ind].inds[0]
                    ind_k_r = self.ket.bond(bond_ind, bond_ind + 1)  ## to right env

                    # print('gamma to update L', gamma_to_update_L)
                    # print('gamma to update R', gamma_to_update_R)
                    gamma_to_update_L.reindex({ind_kg_r: ind_kg_r + '_'}, inplace=True)
                    gamma_to_update_R.reindex({ind_kg_r: ind_k_r}, inplace=True)
                    tensL, tensS, tensR = lambda_tens.split([ind_kg_r + '_'], method='svd', absorb=None, cutoff=CUTOFF,
                                                            cutoff_mode=CUTOFF_MODE)
                    new_L = gamma_to_update_L.contract(tensL)
                    # new_L.transpose_like(gamma_to_update_L, inplace=True)
                    gamma_to_update_L.modify(data=new_L.data, inds=new_L.inds)
                    new_R = gamma_to_update_R.contract(tensR)
                    # new_R.transpose_like(gamma_to_update_R, inplace=True)
                    gamma_to_update_R.modify(data=new_R.data, inds=new_R.inds)
                    lambda_to_update.modify(data=tensS.data, inds=tensS.inds)
                    # print('gamma', gamma_to_update_L, lambda_to_update, gamma_to_update_L)

                # print('ket gammas', ket_gammas)
                # print('ket lambdas', ket_lambdas)
                new_ket = helper.gamma_lambda_to_mps(ket_gammas, ket_lambdas, view_like=self.ket)
                new_ket.exponent = self.ket.exponent
                helper.collect_exponent(new_ket, inplace=True)

                self.ket = new_ket
                self.set_bra_from_ket()
                self.gammas = None
                self.lambdas = None
                print('self.ket norm (end)', sites, helper.norm(self.ket), self.ket.norm())
                # exit()

            num_it += 1
            # exit()
        print('done sweep')
        return self.ket

    def take_time_step_v3(self, dt, grid=None, do_adapt=False, **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
            parallel forward + back (trotter) at each site
        """
        L = self.ket.L
        max_bond = np.inf if self.max_bond is None else self.max_bond
        cutoff = self.compress_config.cutoffs.get(1, CUTOFF)

        num_it = 0
        nsites = 2  # 2 if do_adapt and self.ket.max_bond() < max_bond else 1  ## don't do mixed 1/2 site updates
        num_layers = nsites
        dt_ = dt / num_layers


        num_inner_its = 1

        while num_it < num_layers:  ## number of iterations (1 for 1 site, 2 for 2 sites)

            print('self ket norm', self.ket.norm(), helper.norm(self.ket))

            ## compute site, bond derivatives
            forward_derivs = {}
            site_ind = 0

            helper.collect_exponent(self.ket, inplace=True)
            new_ket = self.ket  ## bond dim can change after compress etc
            ket_gammas, ket_lambdas, gl_norm = helper.mps_to_gamma_lambda(new_ket, cutoff=cutoff)
            check_mps = helper.gamma_lambda_to_mps(ket_gammas, ket_lambdas, view_like=new_ket)
            check_mps.exponent = new_ket.exponent  ### signs can in Gammas can be different
            check_mps.exponent += gl_norm
            print('gl/mps diff', helper.distance(check_mps, new_ket))
            # print('new mps', new_ket)
            # print('check mps', check_mps)
            self.ket = check_mps
            self.set_bra_from_ket()
            self.gammas = ket_gammas
            self.lambdas = ket_lambdas

            ## get <Ax|x> init envs
            print('canonize L-1')
            self.canonize(L - 1)  ## does not scale canonical tensors
            # self.ket = helper.gamma_lambda_to_mps(ket_gammas, ket_lambdas, canon_site=L-1, view_like=new_ket)
            # self.set_bra_from_ket()
            self._build_all_envs_left(1, canonize=False)

            print('canonize 0')
            self.canonize(0)  ## does not scale canonical tensors
            # self.ket = helper.gamma_lambda_to_mps(ket_gammas, ket_lambdas, canon_site=0, view_like=new_ket)
            # self.set_bra_from_ket()
            self._build_all_envs_right(1, canonize=False)

            active_sites = []
            while site_ind < L:
                block_nsites = 1 if site_ind == (num_it % 2)- 1 else min(L - site_ind, nsites)
                # block_nsites = 1 if site_ind == 1 - 1 else min(L - site_ind, nsites)
                ## to stagger layers if doing nsites==2, catching for last tensor

                sites = tuple(range(site_ind, site_ind + block_nsites))
                print('sites', sites)
                active_sites += [sites]

                ## forward propagation derivatives
                forward_derivs[sites] = self._get_A_effs(site_ind, block_nsites)

                ## next site_ind
                site_ind = site_ind + block_nsites

            ## update sites
            new_ket = qtn.TensorNetwork([])
            for sites in active_sites:
                gs = [ket_gammas[s] for s in sites]
                ls = [ket_lambdas[s - 1] for s in sites if s >= 1]
                if sites[-1] < L - 1:
                    ls += [ket_lambdas[sites[-1]]]

                tens_list = []
                for i in range(len(gs) - 1):
                    if sites[i] == 0: # or sites[i] == L - 1:
                        # self.canonize(L - 1)
                        tens_list += [gs[i].copy()]
                    else:
                        tens_list += [qtn.tensor_contract(gs[i], ls[i], output_inds=gs[i].inds)]

                if sites[-1] == 0 or sites[-1] == L - 1:
                    tens_list += [qtn.tensor_contract(gs[-1], ls[-1], output_inds=gs[-1].inds)]
                else:
                    tens_list += [qtn.tensor_contract(gs[-1], ls[-2], ls[-1], output_inds=gs[-1].inds)]

                dist_mpx = qtn.TensorNetwork(tens_list)

                ## gl inds to ket inds
                sL, sR = sites[0], sites[-1]
                reindex_gl_to_ket = {}
                if sL > 0:
                    ind_g_l = ket_lambdas[sL - 1].inds[0]  # next(iter(qtn.bonds(ket_lambdas[sL-1], ket_gammas[sL])))
                    ind_k_l = next(iter(qtn.bonds(self.ket[sL - 1], self.ket[sL])))
                    if ind_g_l != ind_k_l:
                        reindex_gl_to_ket[ind_g_l] = ind_k_l
                if sR < L - 1:
                    ind_g_r = ket_lambdas[sR].inds[
                        0]  # next(iter(qtn.bonds(ket_lambdas[sR], ket_gammas[sR]))) if sR < L - 1 else None
                    ind_k_r = next(iter(qtn.bonds(self.ket[sR], self.ket[sR + 1]))) if sR < L - 1 else None
                    if ind_g_r != ind_k_r:
                        reindex_gl_to_ket[ind_g_r] = ind_k_r
                reindex_ket_to_gl = {val: k for k, val in reindex_gl_to_ket.items()}
                dist_mpx.reindex(reindex_gl_to_ket, inplace=True)

                ## change indices from bra (missing T*[i]) to ket
                ket_to_bra_inds = {}
                for site_p in sites:
                    ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

                bonds_i = dist_mpx.outer_inds()
                bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

                # A_effs = forward_derivs[sites]

                dist_mpx_copy = dist_mpx.copy()
                submpx_tags = [self.ket.site_tag_id.format(s) for s in sites]

                ## backwards time evolution on evolved state (first half)
                if False:  # sites[-1] < L - 1:
                    print('backwards TE SR', sites)
                    mpx_tens = dist_mpx.contract(output_inds=bonds_i)

                    bR = self.ket.bond(sites[-1], sites[-1] + 1) if sites[-1] < L - 1 else None
                    if sites[-1] < L - 1:
                        tens_S, proj_left_tens = mpx_tens.split(bR, absorb='left', cutoff=cutoff,
                                                                cutoff_mode=CUTOFF_MODE,
                                                                bond_ind=bR + '_')
                    else:
                        tens_S = qtn.Tensor()  ## scalar tensor
                        proj_left_tens = mpx_tens

                    proj_left_cc = proj_left_tens.conj()

                    ## check right orthog
                    if bR is not None:
                        proj_left_cc_ = proj_left_cc.reindex({bR + '_': bR})
                        out = proj_left_tens.contract(proj_left_cc_)
                        print('check left orthog', np.linalg.norm(out.data - np.eye(out.shape[0])))

                    proj_left_cc.reindex(ket_to_bra_inds, inplace=True)
                    if bR is not None:
                        proj_left_cc.reindex({bR + '_': bR + '_b_'}, inplace=True)

                    A_effs_proj = []
                    for A_eff in forward_derivs[sites]:
                        A_eff = A_eff.copy()
                        A_eff.add([proj_left_tens, proj_left_cc])
                        A_effs_proj += [A_eff]

                    if bR is not None:
                        proj_bonds_o = [bR + '_b_', ket_to_bra_inds[bR]]
                        proj_bonds_i = [bR + '_', bR]
                    else:
                        proj_bonds_o, proj_bonds_i = [], []
                    # print('proj bonds o', proj_bonds_o)
                    # print('proj bonds i', proj_bonds_i)
                    deriv_total = self._sum_eff_TNs(A_effs_proj, transpose_bonds=proj_bonds_o + list(proj_bonds_i))
                    if bR is None:
                        print('deriv total L-1', deriv_total.data)

                    S_tag = f'_S_{sites[-1]}_'
                    tens_S.add_tag(S_tag)
                    dist_mpx_S = qtn.TensorNetwork([tens_S], virtual=True)
                    site_time_evolution(dist_mpx_S, -dt_, deriv_total, [S_tag], bonds_i=proj_bonds_i,
                                        bonds_o=proj_bonds_o,
                                        inplace=True, te_order=self.te_order,
                                        compress_direction=CompressDirection.RIGHT,
                                        compress_level=1, compress_opts_dict=self.compress_config)

                    tens_S = dist_mpx_S.tensors[0]
                    tens_S.drop_tags()
                    new_dist_tens = tens_S.contract(proj_left_tens)
                    _update_tn_with_tensor(dist_mpx, new_dist_tens, submpx_tags,
                                           compress_direction=CompressDirection.RIGHT,
                                           compress_opts=self.compress_config[1])

                    print('dist_mpx TE (a)', helper.distance(dist_mpx_copy, dist_mpx))

                num_inner_it = 0
                while num_inner_it < num_inner_its:

                    num_inner_it += 1

                    ### forward TE
                    A_effs = forward_derivs[sites]
                    # deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bonds_o + list(bonds_i))


                    site_time_evolution(dist_mpx, dt_ / num_inner_its, A_effs, submpx_tags,
                                        bonds_i=bonds_i, bonds_o=bonds_o,
                                        inplace=True, te_order=self.te_order,
                                        compress_direction=CompressDirection.RIGHT,
                                        compress_level=1, compress_opts_dict=self.compress_config)
                    print('dist_mpx TE', helper.distance(dist_mpx_copy, dist_mpx))

                    ## backwards time evolution on evolved state
                    if True:  # sites[-1] < L - 1:
                        print('backwards TE SR', sites)
                        mpx_tens = dist_mpx.contract(output_inds=bonds_i)

                        bR = self.ket.bond(sites[-1], sites[-1] + 1) if sites[-1] < L - 1 else None
                        if sites[-1] < L - 1:
                            tens_S, proj_left_tens = mpx_tens.split(bR, absorb='left', cutoff=cutoff,
                                                                    cutoff_mode=CUTOFF_MODE,
                                                                    bond_ind=bR + '_')
                        else:
                            tens_S = qtn.Tensor()   ## scalar tensor
                            proj_left_tens = mpx_tens

                        proj_left_cc = proj_left_tens.conj()

                        ## check right orthog
                        if bR is not None:
                            proj_left_cc_ = proj_left_cc.reindex({bR + '_': bR})
                            out = proj_left_tens.contract(proj_left_cc_)
                            print('check left orthog', np.linalg.norm(out.data - np.eye(out.shape[0])))

                        proj_left_cc.reindex(ket_to_bra_inds, inplace=True)
                        if bR is not None:
                            proj_left_cc.reindex({bR + '_': bR + '_b_'}, inplace=True)

                        A_effs_proj = []
                        for A_eff in forward_derivs[sites]:
                            A_eff = A_eff.copy()
                            A_eff.add([proj_left_tens, proj_left_cc])
                            A_effs_proj += [A_eff]

                        if bR is not None:
                            proj_bonds_o = [bR + '_b_', ket_to_bra_inds[bR]]
                            proj_bonds_i = [bR + '_', bR]
                        else:
                            proj_bonds_o, proj_bonds_i = [], []
                        # print('proj bonds o', proj_bonds_o)
                        # print('proj bonds i', proj_bonds_i)

                        # deriv_total = self._sum_eff_TNs(A_effs_proj, transpose_bonds=proj_bonds_o + list(proj_bonds_i))
                        # if bR is None:
                        #     print('deriv total L-1', deriv_total.data)

                        S_tag = f'_S_{sites[-1]}_'
                        tens_S.add_tag(S_tag)
                        dist_mpx_S = qtn.TensorNetwork([tens_S], virtual=True)
                        site_time_evolution(dist_mpx_S, -dt_/ num_inner_its, A_effs_proj, [S_tag],
                                            bonds_i=proj_bonds_i, bonds_o=proj_bonds_o,
                                            inplace=True, te_order=self.te_order,
                                            compress_direction=CompressDirection.RIGHT,
                                            compress_level=1, compress_opts_dict=self.compress_config)

                        tens_S = dist_mpx_S.tensors[0]
                        tens_S.drop_tags()
                        new_dist_tens = tens_S.contract(proj_left_tens)
                        _update_tn_with_tensor(dist_mpx, new_dist_tens, submpx_tags,
                                               compress_direction=CompressDirection.RIGHT,
                                               compress_opts=self.compress_config[1])

                        print('dist_mpx TE (b)', helper.distance(dist_mpx_copy, dist_mpx))

                # if sites[0] > 0:
                #     print('backwards TE SL')
                #     mpx_tens = dist_mpx.contract(output_inds=bonds_i)
                #     bL = self.ket.bond(sites[0], sites[0] - 1)
                #     tens_S, proj_right_tens = mpx_tens.split(bL, absorb='left', cutoff=cutoff, cutoff_mode=CUTOFF_MODE,
                #                                             bond_ind=bL + '_')
                #     proj_right_cc = proj_right_tens.conj()
                #     proj_right_cc.reindex(ket_to_bra_inds, inplace=True)
                #     proj_right_cc.reindex({bL + '_': bL + '_b_'}, inplace=True)
                #
                #     A_effs_proj = []
                #     for A_eff in forward_derivs[sites]:
                #         A_eff = A_eff.copy()
                #         A_eff.add([proj_right_tens, proj_right_cc])
                #         A_effs_proj += [A_eff]
                #
                #     proj_bonds_o = [bL + '_b_', ket_to_bra_inds[bL]]
                #     proj_bonds_i = [bL + '_', bL]
                #     # print('proj bonds o', proj_bonds_o)
                #     # print('proj bonds i', proj_bonds_i)
                #     deriv_total = self._sum_eff_TNs(A_effs_proj, transpose_bonds=proj_bonds_o + list(proj_bonds_i))
                #
                #     S_tag = f'_S_{sites[0] - 1}_'
                #     tens_S.add_tag(S_tag)
                #     dist_mpx_S = qtn.TensorNetwork([tens_S], virtual=True)
                #     site_time_evolution(dist_mpx_S, -dt_, deriv_total, [S_tag], bonds_i=proj_bonds_i,
                #                         bonds_o=proj_bonds_o,
                #                         inplace=True, te_order=self.te_order,
                #                         compress_direction=CompressDirection.RIGHT,
                #                         compress_level=1, compress_opts_dict=self.compress_config)
                #
                #     tens_S = dist_mpx_S.tensors[0]
                #     tens_S.drop_tags()
                #     new_dist_tens = tens_S.contract(proj_right_tens)
                #     _update_tn_with_tensor(dist_mpx, new_dist_tens, submpx_tags,
                #                            compress_direction=CompressDirection.RIGHT,
                #                            compress_opts=self.compress_config[1])
                #
                # print('dist_mpx TE (c)', helper.distance(dist_mpx_copy, dist_mpx))

                dist_mpx.reindex(reindex_ket_to_gl, inplace=True)

                ## apply inverse lambda to the right end (bc we're taking l-g-l for each site.
                if sR < L - 1:
                    tens_R = dist_mpx.select_tensors(self.ket.site_tag(sites[-1]))[0]
                    lam_inv = ls[-1].copy()
                    lam_inv.modify(apply=lambda x: 1. / x ) # * (1.0 + 1e-6))
                    print('lam inv', lam_inv.data)
                    new_tens_R = tens_R.contract(lam_inv, output_inds=tens_R.inds)
                    new_tens_R = new_tens_R.transpose_like(tens_R, inplace=True)
                    tens_R.modify(data=new_tens_R.data)
                    # tens_R.modify(apply = lambda x: np.tensordot(x, np.diag(1./ls[-1].data), axes=[ind,0]))

                # print('add dist to new ket')
                new_ket.add(dist_mpx.copy())

            new_ket.view_like(self.ket, inplace=True)
            new_ket.exponent = self.ket.exponent
            print('dist', helper.distance(new_ket, self.ket))
            print('old_ket norm', helper.norm(self.ket), self.ket.copy().norm())
            print('new_ket norm', helper.norm(new_ket), new_ket.copy().norm())
            print('collect exponent?')
            helper.collect_exponent(new_ket, inplace=True)
            print('new ket norm', helper.norm(new_ket), new_ket.copy().norm())

            # ## update ket
            self.ket = new_ket
            num_it += 1

        # exit()
        print('done sweep')
        return self.ket


    def take_time_step_v4(self, dt, grid=None, do_adapt=False, **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
            parallel forward + back (projection) at each site
        """
        L = self.ket.L
        max_bond = self.max_bond

        num_it = 0
        nsites = 2 if do_adapt and self.ket.max_bond() < max_bond else 1      ## don't do mixed 1/2 site updates
        dt_ = dt / nsites
        num_layers = nsites

        while num_it < num_layers:      ## number of iterations (1 for 1 site, 2 for 2 sites)

            print('self ket norm', self.ket.norm(), helper.norm(self.ket))

            ## compute site, bond derivatives
            forward_derivs = {}
            site_ind = 0

            helper.collect_exponent(self.ket, inplace=True)
            new_ket = self.ket      ## bond dim can change after compress etc
            ket_gammas, ket_lambdas, gl_norm = helper.mps_to_gamma_lambda(new_ket, cutoff=CUTOFF)
            check_mps = helper.gamma_lambda_to_mps(ket_gammas, ket_lambdas, view_like=new_ket)
            check_mps.exponent = new_ket.exponent       ### signs can in Gammas can be different
            check_mps.exponent += gl_norm
            print('gl/mps diff', helper.distance(check_mps, new_ket))
            # print('new mps', new_ket)
            # print('check mps', check_mps)
            self.ket = check_mps
            self.set_bra_from_ket()
            self.gammas = ket_gammas
            self.lambdas = ket_lambdas

            ## get <Ax|x> init envs
            print('canonize L-1')
            self.canonize(L - 1)  ## does not scale canonical tensors
            # self.ket = helper.gamma_lambda_to_mps(ket_gammas, ket_lambdas, canon_site=L-1, view_like=new_ket)
            # self.set_bra_from_ket()
            self._build_all_envs_left(1, canonize=False)

            print('canonize 0')
            self.canonize(0)  ## does not scale canonical tensors
            # self.ket = helper.gamma_lambda_to_mps(ket_gammas, ket_lambdas, canon_site=0, view_like=new_ket)
            # self.set_bra_from_ket()
            self._build_all_envs_right(1, canonize=False)

            active_sites = []
            while site_ind < L:
                block_nsites = 1 if site_ind == num_it - 1 else min(L - site_ind, nsites)
                ## to stagger layers if doing nsites==2, catching for last tensor

                sites = tuple(range(site_ind, site_ind + block_nsites))
                print('sites', sites)
                active_sites += [sites]

                ## forward propagation derivatives
                forward_derivs[sites] = self._get_A_effs(site_ind, block_nsites)

                ## next site_ind
                site_ind = site_ind + block_nsites

            ## update sites
            new_ket = qtn.TensorNetwork([])
            for sites in active_sites:
                gs = [ket_gammas[s] for s in sites]
                ls = [ket_lambdas[s - 1] for s in sites if s >= 1]
                if sites[-1] < L - 1:
                     ls += [ket_lambdas[sites[-1]]]

                tens_list = []
                for i in range(len(gs) - 1):
                    if sites[i] == 0: # or sites[i] == L-1:
                        # self.canonize(L-1)
                        tens_list += [gs[i].copy()]
                    else:
                        tens_list += [qtn.tensor_contract(gs[i], ls[i], output_inds=gs[i].inds)]

                if sites[-1] == 0 or sites[-1] == L - 1:
                    tens_list += [qtn.tensor_contract(gs[-1], ls[-1], output_inds=gs[-1].inds)]
                else:
                    tens_list += [qtn.tensor_contract(gs[-1], ls[-2], ls[-1], output_inds=gs[-1].inds)]

                dist_mpx = qtn.TensorNetwork(tens_list)


                ## gl inds to ket inds
                sL, sR = sites[0], sites[-1]
                reindex_gl_to_ket = {}
                if sL > 0:
                    ind_g_l = ket_lambdas[sL-1].inds[0]  # next(iter(qtn.bonds(ket_lambdas[sL-1], ket_gammas[sL])))
                    ind_k_l = next(iter(qtn.bonds(self.ket[sL - 1], self.ket[sL])))
                    if ind_g_l != ind_k_l:
                        reindex_gl_to_ket[ind_g_l] = ind_k_l
                if sR < L - 1:
                    ind_g_r = ket_lambdas[sR].inds[0]  # next(iter(qtn.bonds(ket_lambdas[sR], ket_gammas[sR]))) if sR < L - 1 else None
                    ind_k_r = next(iter(qtn.bonds(self.ket[sR], self.ket[sR + 1]))) if sR < L - 1 else None
                    if ind_g_r != ind_k_r:
                        reindex_gl_to_ket[ind_g_r] = ind_k_r
                reindex_ket_to_gl = {val: k for k, val in reindex_gl_to_ket.items()}
                dist_mpx.reindex(reindex_gl_to_ket, inplace=True)

                ## change indices from bra (missing T*[i]) to ket
                ket_to_bra_inds = {}
                for site_p in sites:
                    ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

                bonds_i = dist_mpx.outer_inds()
                bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

                A_effs = forward_derivs[sites]

                ## orthogonal projection
                if True:  # sites[-1] < L - 1:
                    print('backwards TE (S right)', sites)
                    mpx_tens = dist_mpx.contract(output_inds=bonds_i)
                    bR = self.ket.bond(sites[-1], sites[-1] + 1) if sites[-1] < L - 1 else None
                    bL = self.ket.bond(sites[0], sites[0] - 1) if sites[0] > 0 else None
                    bL_bra = self.bra.bond(sites[0], sites[0] - 1) if sites[0] > 0 else None
                    if sites[-1] < L - 1:
                        tens_S, proj_left_tens = mpx_tens.split(bR, absorb='left', cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE,
                                                                bond_ind=bR + '_')
                    else:
                        # tens_S = qtn.Tensor()   ## scalar tensor
                        proj_left_tens = mpx_tens

                    proj_left_cc = proj_left_tens.conj()

                    # ## check right orthog
                    # if bR is not None:
                    #     proj_left_cc_ = proj_left_cc.reindex({bR + '_': bR})
                    #     out = proj_left_tens.contract(proj_left_cc_)
                    #     print('check left orthog', np.linalg.norm(out.data - np.eye(out.shape[0])))

                    proj_left_cc.reindex(ket_to_bra_inds, inplace=True)
                    bra_inds = [self.bra.site_ind_id.format(i) for i in sites]
                    ket_inds = [self.ket.site_ind_id.format(i) for i in sites]
                    proj_left_cc.reindex({bi: bi + '_tmp' for bi in bra_inds}, inplace=True)
                    proj_left_tens.reindex({ki: bi for ki, bi in zip(ket_inds, bra_inds)}, inplace=True)
                    proj_left = proj_left_cc.contract(proj_left_tens)
                    proj_left.modify(apply = lambda x: -1 * x)      ## bc minus projector
                    # print('proj left', proj_left.norm(), proj_left.shape)

                    if bL is not None:
                        proj_left.reindex({bL_bra: bL_bra + '_'}, inplace=True)
                        proj_left.reindex({bL: bL_bra}, inplace=True)

                    A_effs_proj = []
                    for A_eff in forward_derivs[sites]:
                        A_eff = A_eff.copy()
                        A_eff.reindex({bi: bi + '_tmp' for bi in bra_inds}, inplace=True)
                        if bL_bra is not None:
                            A_eff.reindex({bL_bra: bL_bra + '_'}, inplace=True)
                        A_eff.add([proj_left])
                        A_effs_proj += [A_eff]

                    A_effs = A_effs + A_effs_proj


                if False:  # sites[0] > 0:
                    print('backwards TE (S left)', sites)
                    mpx_tens = dist_mpx.contract(output_inds=bonds_i)
                    bR = self.ket.bond(sites[-1], sites[-1] + 1) if sites[-1] < L - 1 else None
                    bR_bra = self.bra.bond(sites[-1], sites[-1] + 1) if sites[-1] < L - 1 else None
                    bL = self.ket.bond(sites[0], sites[0] - 1)
                    # bL_bra = self.bra.bond(sites[0], sites[0] - 1)
                    tens_S, proj_right_tens = mpx_tens.split(bL, absorb='left', cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE,
                                                            bond_ind=bL + '_')
                    proj_right_cc = proj_right_tens.conj()
                    proj_right_cc.reindex(ket_to_bra_inds, inplace=True)
                    bra_inds = [self.bra.site_ind_id.format(i) for i in sites]
                    ket_inds = [self.ket.site_ind_id.format(i) for i in sites]
                    proj_right_cc.reindex({bi: bi + '_tmp' for bi in bra_inds}, inplace=True)
                    proj_right_tens.reindex({ki: bi for ki, bi in zip(ket_inds, bra_inds)}, inplace=True)
                    proj_right = proj_right_cc.contract(proj_right_tens)
                    proj_right.modify(apply = lambda x: -1 * x)      ## bc minus projector
                    print('proj right', proj_right.norm(), proj_right.shape)

                    if bR is not None:
                        proj_right.reindex({bR_bra: bR_bra + '_'}, inplace=True)
                        proj_right.reindex({bR: bR_bra}, inplace=True)

                    A_effs_proj = []
                    for A_eff in forward_derivs[sites]:
                        A_eff = A_eff.copy()
                        A_eff.reindex({bi: bi + '_tmp' for bi in bra_inds}, inplace=True)
                        if bR_bra is not None:
                            A_eff.reindex({bR_bra: bR_bra + '_'}, inplace=True)
                        A_eff.add([proj_right])
                        A_effs_proj += [A_eff]

                    A_effs = A_effs + A_effs_proj

                deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bonds_o + list(bonds_i))

                dist_mpx_copy = dist_mpx.copy()
                submpx_tags = [self.ket.site_tag_id.format(s) for s in sites]
                site_time_evolution(dist_mpx, dt_, deriv_total, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                                    inplace=True, te_order=self.te_order,
                                    compress_direction=CompressDirection.RIGHT,
                                    compress_level=1, compress_opts_dict=self.compress_config)
                print('dist_mpx TE', helper.distance(dist_mpx_copy, dist_mpx))

                dist_mpx.reindex(reindex_ket_to_gl, inplace=True)

                ## apply inverse lambda to the right end (bc we're taking l-g-l for each site.
                if sR < L - 1:
                    tens_R = dist_mpx.select_tensors(self.ket.site_tag(sites[-1]))[0]
                    lam_inv = ls[-1].copy()
                    lam_inv.modify(apply = lambda x: 1./ x)
                    new_tens_R = tens_R.contract(lam_inv, output_inds=tens_R.inds)
                    new_tens_R.transpose_like(tens_R, inplace=True)
                    tens_R.modify(data=new_tens_R.data)
                    # tens_R.modify(apply = lambda x: np.tensordot(x, np.diag(1./ls[-1].data), axes=[ind,0]))

                # print('add dist to new ket')
                new_ket.add(dist_mpx)

            new_ket.view_like(self.ket, inplace=True)
            new_ket.exponent = self.ket.exponent
            print('dist', helper.distance(new_ket, self.ket))
            print('old_ket norm', helper.norm(self.ket), self.ket.copy().norm())
            print('new_ket norm', helper.norm(new_ket), new_ket.copy().norm())
            print('collect exponent?')
            helper.collect_exponent(new_ket, inplace=True)
            print('new ket norm', helper.norm(new_ket), new_ket.copy().norm())

            # ## update ket
            self.ket = new_ket
            num_it += 1

        # exit()
        print('done sweep')
        return self.ket


class TDVPSolver2(TDVPSolver):
    """ solve second order ODE of the form:  y"(x) + O(x)y(x) = b0 + x*b1
        trial_state:  initial state
        operators:  list of MPOs whos sum = O(x)
        targets:  [b0, b1].  if b1 is not None then b0 cannot be None
            NOTE: THIS IS DIFFERENT FROM DMRG/MG SOLVERS
    """
    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 operators: Optional[Sequence['MPO_type']] = None,
                 targets: Optional[Sequence['MPS_type']] = None,
                 mps_inds: Sequence[int] = None,
                 norm_env0_Ls: Sequence['qtn.Tensor'] = None,
                 norm_env0_Rs: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 trial_state_dt: Optional['qtn.MatrixProductState'] = None,
                 te_order=0, compress_config: CompressionConfiguration=None):

        # operators = [] if operators is None else list(operators)
        self.A_envsLs, self.A_envsRs = [], []
        self.b_envsLs, self.b_envsRs = [], []

        super().__init__(trial_state, targets=targets, operators=operators, mps_inds=mps_inds,
                         norm_env0_Ls=norm_env0_Ls, norm_env0_Rs=norm_env0_Rs,
                         ovlp_env0_Ls=ovlp_env0_Ls, ovlp_env0_Rs=ovlp_env0_Rs,
                         te_order=te_order, compress_config=compress_config)

        self.trial_state_dt = trial_state_dt


    def _sum_source_TNs(self, dt, targets: Sequence['qtn.TensorNetwork'], transpose_bonds=None):
        """ sum like b0 + t * b1
        """
        if targets is None:
            return None

        transpose_bonds = targets[0].outer_inds() if transpose_bonds is None else transpose_bonds

        b0 = targets[0].copy()
        b1 = targets[1].copy() if len(targets) > 0 else None
        if b1 is not None:
            b1.exponent += np.log10(dt)

        beff_list = []
        if b0 is not None:
            beff_list += [b0]
        if b1 is not None:
            beff_list += [b1]

        return self._sum_eff_TNs(beff_list, transpose_bonds=transpose_bonds)


    def _site_time_evolution(self, dt, left_site_pos, nsites=1,
                             compress_direction=CompressDirection.RIGHT,
                             ) -> Optional[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        """
        # print('check orthog site TE', left_site_pos)
        # helper.check_orthog(self.ket)
        # helper.check_orthog(self.bra)

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        dist_dt_submpx = None if self.trial_state_dt is None else \
                            qtn.TensorNetwork([self.trial_state_dt[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)
        b_effs = self._get_b_effs(left_site_pos, nsites)

        ## change indices from bra (missing T*[i]) to ket
        # bra_to_ket_inds = {}
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))
            # bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]
        deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bonds_o + list(bonds_i))
        # source_total = self._sum_source_TNs(dt, b_effs, transpose_bonds=bonds_o)
        sources = [b_eff.contract().transpose(*bonds_o) for b_eff in b_effs]

        site_time_evolution_2(dist_submpx, dt, deriv_total, sources, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                              inplace=True, te_order=self.te_order, target_dt=dist_dt_submpx,
                              compress_direction=compress_direction,
                              compress_level=1, compress_opts_dict=self.compress_config)

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            self.canonize(site_inds[-1], cur_orthog=site_inds[0])     ## also sets bra from ket
            for i in range(left_site_pos, left_site_pos + nsites - 1):
                self._update_envs_left(i, canonize=False)
        else:
            self.canonize(site_inds[0], cur_orthog=site_inds[-1])  ## also sets bra from ket
            for i in range(left_site_pos + nsites - 1, left_site_pos, -1):
                self._update_envs_right(i, canonize=False)

        return self.ket


    def _bond_time_evolution(self, dt, cur_orthog,
                             compress_direction=CompressDirection.RIGHT
                             ) -> Sequence[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        """
        # site_pos = list(range(self.L))[bond_ind]
        # bond_ind = self.mps_inds[bond_pos]

        # helper.check_orthog(self.ket)
        ket_usvt = MPS_USVT.from_MPS(self.ket, canon_site=cur_orthog, cur_orthog=cur_orthog, direction=compress_direction)
        bra_usvt = MPS_USVT.from_MPS(self.bra, canon_site=cur_orthog, cur_orthog=cur_orthog, direction=compress_direction)
        Q_tens = ket_usvt[cur_orthog]
        S_tens = ket_usvt.get_S_tensor()
        Q_conj = bra_usvt[cur_orthog]
        S_conj = bra_usvt.get_S_tensor()

        if self.trial_state_dt is not None:
            dt_ket_usvt = MPS_USVT.from_MPS(self.trial_state_dt, canon_site=cur_orthog, cur_orthog=cur_orthog,
                                            direction=compress_direction)
            dt_Q_tens = dt_ket_usvt[cur_orthog]
            dt_S_tens = dt_ket_usvt.get_S_tensor()
        else:
            dt_Q_tens, dt_S_tens = None, None

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = []
        for i in range(self.num_operators):
            A_left = self.A_envsLs[i][cur_orthog]
            A_right = self.A_envsRs[i][cur_orthog]

            A_eff = qtn.TensorNetwork([])
            A = self.operators[i]
            A_eff.add(A[cur_orthog])

            if A_left is not None:
                A_eff.add(A_left)
            if A_right is not None:
                A_eff.add(A_right)

            ## add Q(R) / (L)Q tensor
            A_eff.add([Q_tens, Q_conj])

            # A_eff.exponent = self.A_envsLs[i].exponent
            A_eff.exponent = A.exponent
            A_effs += [A_eff]


        ### bL * b * bR = d/dT*[i] <x|b>
        b_effs = []
        for i in range(self.num_targets):
            b_left = self.b_envsLs[i][cur_orthog]
            b_right = self.b_envsRs[i][cur_orthog]

            b_eff = qtn.TensorNetwork([])
            b = self.targets[i]
            b_eff.add(b[cur_orthog])

            if b_left is not None:
                b_eff.add(b_left)
            if b_right is not None:
                b_eff.add(b_right)

            ## add Q(R) / (L)Q tensor
            b_eff.add([Q_conj])

            # A_eff.exponent = self.A_envsLs[i].exponent
            b_eff.exponent = b.exponent
            b_effs += [b_eff]

        ## change indices from bra (missing T*[i]) to ket
        shared, notshared = S_tens.filter_bonds(Q_tens)
        ket_inds = shared + notshared

        shared, notshared = S_conj.filter_bonds(Q_conj)
        bra_inds = shared + notshared
        deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bra_inds+ket_inds)
        sources = [b_eff.contract().transpose(*bra_inds) for b_eff in b_effs]
        dist_submpx = qtn.TensorNetwork([S_tens], virtual=True)
        dist_dt_submpx = None if dt_S_tens is None else qtn.TensorNetwork([dt_S_tens], virtual=True)

        site_time_evolution_2(dist_submpx, dt, deriv_total, sources, [next(iter(S_tens.tags))],
                              bonds_i = ket_inds, bonds_o = bra_inds, te_order=self.te_order, inplace=True,
                              target_dt=dist_dt_submpx,
                              compress_direction=compress_direction, compress_opts_dict=self.compress_config
                              )

        # print('cur orthog to MPS', cur_orthog)
        new_ket = ket_usvt.to_MPS()
        ket_tens = self.ket[cur_orthog]
        new_tens = new_ket[cur_orthog]
        new_tens.transpose_like(ket_tens, inplace=True)
        ket_tens.modify(data = new_tens.data)
        self.set_bra_from_ket(sites=[cur_orthog])

        if self.trial_state_dt is not None:
            new_dt_ket = dt_ket_usvt.to_MPS()
            ket_tens = self.trial_state_dt[cur_orthog]
            new_tens = new_dt_ket[cur_orthog]
            new_tens.transpose_like(ket_tens, inplace=True)
            ket_tens.modify(data=new_tens.data)

        return self.ket


class TDVPSolver0(TDVPSolver):
    """ solve second order ODE of the form:  y"(x) + O(x)y(x) = b0 + x*b1
        trial_state:  initial state
        operators:  list of MPOs whos sum = O(x)
        targets:  [b0, b1].  if b1 is not None then b0 cannot be None
            NOTE: THIS IS DIFFERENT FROM DMRG/MG SOLVERS
    """
    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 # operators: Optional[Sequence['MPO_type']] = None,
                 targets: Optional[Sequence['MPS_type']] = None,
                 mps_inds: Sequence[int] = None,
                 norm_env0_Ls: Sequence['qtn.Tensor'] = None,
                 norm_env0_Rs: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 te_order=0, compress_config: CompressionConfiguration=None):

        # operators = [] if operators is None else list(operators)
        self.A_envsLs, self.A_envsRs = [], []
        self.b_envsLs, self.b_envsRs = [], []

        super().__init__(trial_state, targets=targets, operators=None, mps_inds=mps_inds,
                         norm_env0_Ls=norm_env0_Ls, norm_env0_Rs=norm_env0_Rs,
                         ovlp_env0_Ls=ovlp_env0_Ls, ovlp_env0_Rs=ovlp_env0_Rs,
                         te_order=te_order, compress_config=compress_config)

        ## match inner bonds of targets
        for ix in range(self.L - 1):
            bond_ind = self.ket.bond(ix, ix + 1)
            for t in self.targets:
                t.reindex({t.bond(ix, ix + 1): bond_ind}, inplace=True)


    def _sum_source_TNs(self, dt, targets: Sequence['qtn.TensorNetwork'], transpose_bonds=None):
        """ sum like b0 + t * b1
        """
        if targets is None:
            return None

        transpose_bonds = targets[0].outer_inds() if transpose_bonds is None else transpose_bonds

        b0 = targets[0].copy()
        b1 = targets[1].copy() if len(targets) > 0 else None
        if b1 is not None:
            b1.exponent += np.log10(dt)

        beff_list = []
        if b0 is not None:
            beff_list += [b0]
        if b1 is not None:
            beff_list += [b1]

        return self._sum_eff_TNs(beff_list, transpose_bonds=transpose_bonds)


    def _site_time_evolution(self, dt, left_site_pos, nsites=1,
                             compress_direction=CompressDirection.RIGHT,
                             ) -> Optional[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        """
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        # dist_dt_submpx = None if self.trial_state_dt is None else \
        #                     qtn.TensorNetwork([self.trial_state_dt[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        # A_effs = self._get_A_effs(left_site_pos, nsites)
        b_effs = self._get_b_effs(left_site_pos, nsites)

        ## change indices from bra (missing T*[i]) to ket
        # bra_to_ket_inds = {}
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))
            # bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]
        # deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bonds_o + list(bonds_i))
        # source_total = self._sum_source_TNs(dt, b_effs, transpose_bonds=bonds_i)
        sources = [b_eff.contract().transpose(*bonds_o) for b_eff in b_effs]

        site_time_evolution_0(dist_submpx, dt, sources, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                              inplace=True, te_order=self.te_order,
                              compress_direction=compress_direction,
                              compress_level=1, compress_opts_dict=self.compress_config)

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            self.canonize(site_inds[-1], cur_orthog=site_inds[0])     ## also sets bra from ket
            for i in range(left_site_pos, left_site_pos + nsites - 1):
                self._update_envs_left(i, canonize=False)
        else:
            self.canonize(site_inds[0], cur_orthog=site_inds[-1])  ## also sets bra from ket
            for i in range(left_site_pos + nsites - 1, left_site_pos, -1):
                self._update_envs_right(i, canonize=False)

        return self.ket


    def _bond_time_evolution(self, dt, cur_orthog,
                             compress_direction=CompressDirection.RIGHT
                             ) -> Sequence[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        """
        # site_pos = list(range(self.L))[bond_ind]
        # bond_ind = self.mps_inds[bond_pos]

        # helper.check_orthog(self.ket)
        ket_usvt = MPS_USVT.from_MPS(self.ket, canon_site=cur_orthog, cur_orthog=cur_orthog, direction=compress_direction)
        bra_usvt = MPS_USVT.from_MPS(self.bra, canon_site=cur_orthog, cur_orthog=cur_orthog, direction=compress_direction)
        Q_tens = ket_usvt[cur_orthog]
        S_tens = ket_usvt.get_S_tensor()
        Q_conj = bra_usvt[cur_orthog]
        S_conj = bra_usvt.get_S_tensor()

        # if self.trial_state_dt is not None:
        #     dt_ket_usvt = MPS_USVT.from_MPS(self.trial_state_dt, canon_site=cur_orthog, cur_orthog=cur_orthog,
        #                                     direction=compress_direction)
        #     dt_Q_tens = dt_ket_usvt[cur_orthog]
        #     dt_S_tens = dt_ket_usvt.get_S_tensor()
        # else:
        #     dt_Q_tens, dt_S_tens = None, None

        # ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        # A_effs = []
        # for i in range(self.num_operators):
        #     A_left = self.A_envsLs[i][cur_orthog]
        #     A_right = self.A_envsRs[i][cur_orthog]
        #
        #     A_eff = qtn.TensorNetwork([])
        #     A = self.operators[i]
        #     A_eff.add(A[cur_orthog])
        #
        #     if A_left is not None:
        #         A_eff.add(A_left)
        #     if A_right is not None:
        #         A_eff.add(A_right)
        #
        #     ## add Q(R) / (L)Q tensor
        #     A_eff.add([Q_tens, Q_conj])
        #
        #     # A_eff.exponent = self.A_envsLs[i].exponent
        #     A_eff.exponent = A.exponent
        #     A_effs += [A_eff]


        ### bL * b * bR = d/dT*[i] <x|b>
        b_effs = []
        for i in range(self.num_targets):
            b_left = self.b_envsLs[i][cur_orthog]
            b_right = self.b_envsRs[i][cur_orthog]

            b_eff = qtn.TensorNetwork([])
            b = self.targets[i]
            b_eff.add(b[cur_orthog])

            if b_left is not None:
                b_eff.add(b_left)
            if b_right is not None:
                b_eff.add(b_right)

            ## add Q(R) / (L)Q tensor
            b_eff.add([Q_conj])

            # A_eff.exponent = self.A_envsLs[i].exponent
            b_eff.exponent = b.exponent
            b_effs += [b_eff]

        ## change indices from bra (missing T*[i]) to ket
        shared, notshared = S_tens.filter_bonds(Q_tens)
        ket_inds = shared + notshared

        shared, notshared = S_conj.filter_bonds(Q_conj)
        bra_inds = shared + notshared
        # deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bra_inds+ket_inds)
        sources = [b_eff.contract().transpose(*bra_inds) for b_eff in b_effs]
        dist_submpx = qtn.TensorNetwork([S_tens], virtual=True)
        # dist_dt_submpx = None if dt_S_tens is None else qtn.TensorNetwork([dt_S_tens], virtual=True)

        site_time_evolution_0(dist_submpx, dt, sources, [next(iter(S_tens.tags))],
                              bonds_i=ket_inds, bonds_o=bra_inds, te_order=self.te_order, inplace=True,
                              compress_direction=compress_direction, compress_opts_dict=self.compress_config
                              )

        # print('cur orthog to MPS', cur_orthog)
        new_ket = ket_usvt.to_MPS()
        ket_tens = self.ket[cur_orthog]
        new_tens = new_ket[cur_orthog]
        new_tens.transpose_like(ket_tens, inplace=True)
        ket_tens.modify(data = new_tens.data)
        self.set_bra_from_ket(sites=[cur_orthog])

        # if self.trial_state_dt is not None:
        #     new_dt_ket = dt_ket_usvt.to_MPS()
        #     ket_tens = self.trial_state_dt[cur_orthog]
        #     new_tens = new_dt_ket[cur_orthog]
        #     new_tens.transpose_like(ket_tens, inplace=True)
        #     ket_tens.modify(data=new_tens.data)

        return self.ket



#####################################
####      t-DMRG solver         #####
#####################################

class TDMRGSolver(TDVPSolver):
    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 operators: Optional[Sequence['MPO_type']] = None,
                 targets: Optional[Sequence['MPS_type']] = None,
                 mps_inds: Sequence[int] = None,
                 norm_env0_Ls: Sequence['qtn.Tensor'] = None,       # for A envs
                 norm_env0_Rs: Sequence['qtn.Tensor'] = None,       # for A envs
                 ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 ket_env0_L: Optional['qtn.Tensor'] = None,
                 ket_env0_R: Optional['qtn.Tensor'] = None,
                 backprop_edge = False,
                 te_order=0, compress_config: CompressionConfiguration=None):

        # super().__init__(trial_state, targets=targets, operators=operators, mps_inds=mps_inds)
        super().__init__(trial_state, targets=targets, operators=operators, mps_inds=mps_inds,
                         norm_env0_Ls=norm_env0_Ls, norm_env0_Rs=norm_env0_Rs,
                         ovlp_env0_Ls=ovlp_env0_Ls, ovlp_env0_Rs=ovlp_env0_Rs,
                         te_order=te_order, compress_config=compress_config)

        self.ket_env0_L = ket_env0_L
        self.ket_env0_R = ket_env0_R
        self.ovlp_proj = None
        self.backprop_edge = backprop_edge    ## do back prop at end of chain if False


    def create_like(self, copy=True, new_ket=None, **kwargs):
        norm_env0_Ls = [A_envL[0] for A_envL in self.A_envsLs]
        norm_env0_Rs = [A_envR[self.L-1] for A_envR in self.A_envsRs]
        ovlp_env0_Ls = [b_envL[0] for b_envL in self.b_envsLs]
        ovlp_env0_Rs = [b_envR[self.L-1] for b_envR in self.b_envsRs]

        new_solver = TDMRGSolver((self.ket.copy() if copy else self.ket) if new_ket is None else new_ket,
                                 targets=kwargs.get('targets', [t.copy() if copy else t for t in self.targets]),
                                 operators=kwargs.get('operators', [o.copy() if copy else o for o in self.operators]),
                                 mps_inds=kwargs.get('mps_ind_range', self.mps_inds),
                                 norm_env0_Ls=kwargs.get('norm_env0_Ls', norm_env0_Ls),
                                 norm_env0_Rs=kwargs.get('norm_env0_Rs', norm_env0_Rs),
                                 ovlp_env0_Ls=kwargs.get('ovlp_env0_Ls', ovlp_env0_Ls),
                                 ovlp_env0_Rs=kwargs.get('ovlp_env0_Rs', ovlp_env0_Rs),
                                 te_order=kwargs.get('te_order', self.te_order),
                                 compress_config=kwargs.get('compress_config', self.compress_config),
                                 backprop_edge=kwargs.get('backprop_edge', self.backprop_edge),
                                 # solve_type=kwargs.get('solve_tyep', self.solve_type)
                                 )
        new_solver.ket_env0_L = self.ket_env0_L
        new_solver.ket_env0_R = self.ket_env0_R
        new_solver.ovlp_proj = self.ovlp_proj

        return new_solver


    def _site_time_evolution_dmrg(self, dt, left_site_pos, ovlp_projector, nsites=2,
                                  compress_direction=CompressDirection.RIGHT, ) \
            -> tuple[Optional[qtn.MatrixProductState], qtn.Tensor]:
        if nsites == 2:
            return self._site_time_evolution_dmrg2(dt, left_site_pos, ovlp_projector,
                                                   compress_direction=compress_direction)
            # return self._site_time_evolution_tdvp2(dt, left_site_pos,
            #                                        compress_direction=compress_direction)
        elif nsites == 1:
            raise NotImplementedError
            # return self._site_time_evolution_dmrg1(dt, left_site_pos, ovlp_projector,
            #                                        compress_direction=compress_direction)


    # @profile
    def _site_time_evolution_dmrg2(self, dt, left_site_pos, ovlp_projector,
                                   compress_direction = CompressDirection.RIGHT, )\
            -> tuple[Optional[qtn.MatrixProductState], qtn.Tensor]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        """
        nsites = 2

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]
        site0 = site_inds[0] if compress_direction is CompressDirection.RIGHT else site_inds[1]
        site1 = site_inds[1] if compress_direction is CompressDirection.RIGHT else site_inds[0]
        pos0 = left_site_pos if compress_direction is CompressDirection.RIGHT else left_site_pos + 1
        pos1 = left_site_pos + 1 if compress_direction is CompressDirection.RIGHT else left_site_pos

        if compress_direction is CompressDirection.RIGHT:
            at_end = pos1 == self.L - 1
        else:
            at_end = pos1 == 0
        at_end = at_end and not self.backprop_edge


        old_M = self.ket[site1].copy()     ## assumes that this is orthogonality center
        ## M(t) projected onto t + dt env


        ## update ovlp projector <psi(t+dt)|psi(t)> before TE
        if ovlp_projector is None:      # no env (end of chain without ancilla)
            ovlp_projector_site = self.ket[site0].copy()
        else:
            # prev_A = qtn.tensor_contract(self.bra[site0], ovlp_projector.conj())
            #     ## bra (t) on L(t + dt) projected to L(t)
            # print('old M init', site1, old_M)
            # ovlp_projector_site = qtn.tensor_contract(ovlp_projector, prev_A)

            ovlp_projector_site = self.ket[site0].copy()        ## site0 (before TE) in L(t+dt) basis [= proj * site0 in L(t) basis]
            # ind_bra = next(iter(qtn.bonds(ovlp_projector, self.bra[site0])))
            # ind_ket = next(iter(qtn.bonds(ovlp_projector, self.ket[site0])))
            # ovlp_projector_site.reindex({ind_ket: ind_bra}, inplace=True)
            inds_bra = qtn.bonds(ovlp_projector, self.bra[site0])
            inds_ket = qtn.bonds(ovlp_projector, self.ket[site0])
            # ovlp_projector_site.reindex({ik: ib for ik, ib in zip(inds_ket, inds_bra)}, inplace=True)
            ovlp_projector_site.reindex({ik: ik + '_' for ik in inds_ket}, inplace = True)

            # print('ovlp proj + site', ovlp_projector_site)
        # print('ovlp proj + site', ovlp_projector_site)


        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                            inplace=True, te_order=self.te_order,
                            compress_direction=compress_direction,
                            compress_level=1, compress_opts_dict=self.compress_config)
        ## ket canonicalized to site1

        # self.set_bra_from_ket(sites=list(range(left_site_pos, left_site_pos + nsites)))
        self.set_bra_from_ket(sites=[pos0, pos1])
        # print('check orthog after TE', pos1)
        # print(helper.check_orthog(self.ket))
        # print(helper.check_orthog(self.bra))

        # print('at end?', at_end, pos0, pos1, self.backprop_edge)
        if not at_end:
            ### update old M onto new t+dt basis
            ## bra, ket have different site_ind_ids
            ovlp_projector_site.reindex({self.ket.site_ind(site0): self.bra.site_ind(site0)}, inplace=True)
            # print('ovlp projector', ovlp_projector)
            # print('ovlp projector site', ovlp_projector_site)
            # print('self.bra', site0, self.bra[site0])
            ovlp_projector = qtn.tensor_contract(ovlp_projector_site, self.bra[site0])      ## TE'd bra
            # print('new overlap proj', site0, ovlp_projector)
            # print('old M', old_M)
            new_M = qtn.tensor_contract(ovlp_projector, old_M)
            # print('new M', new_M)
            # print('self.bra site', site0, self.bra[site0])
            # print('new M', new_M)
            # print('old M', old_M)

            ## update ovlp with old M
            # old_M_left_inds = list(qtn.bonds(old_M, ovlp_projector)) + [self.ket.site_ind_id.format(site1)]
            # old_M_right_ind = self.ket.bond(site1, site1 + (pos1-pos0))
            # old_Q, old_R = qtn.tensor_split(old_M, old_M_left_inds, absorb='right', method='qr',
            #                                 bond_ind=old_M_right_ind)
            # ovlp_projector_site = qtn.tensor_contract(ovlp_projector, old_Q)
            # print('new ovlp proj+site', ovlp_projector_site)

            current_M = self.ket[site1]
            # print('current M', site1, current_M)
            # print('new M', new_M)
            new_M.transpose_like(current_M, inplace=True)
            current_M.modify(data = new_M.data)
            # print('updated current M', site1, current_M)
            self.set_bra_from_ket(sites=[pos1])
            self.ovlp_proj = ovlp_projector
        else:
            ## update ovlp projector in case it's needed later
            ovlp_projector_site.reindex({self.ket.site_ind(site0): self.bra.site_ind(site0)}, inplace=True)
            ovlp_projector = qtn.tensor_contract(ovlp_projector_site, self.bra[site0])  ## TE'd bra
            self.ovlp_proj = ovlp_projector
            # new_M = qtn.tensor_contract(ovlp_projector, old_M)

            self.set_bra_from_ket(sites=[pos1])

        # print('check orthog? should be', site1)
        # print(helper.check_orthog(self.ket))
        # print(helper.check_orthog(self.bra))

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            # self.canonize(site_inds[-1], cur_orthog=site_inds[0])     ## also sets bra from ket
            for i in range(left_site_pos, left_site_pos + nsites - 1):
                # print('(site) update envs left', i, 'orthog at', site_inds[-1])
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_left(i, canonize=False)
        else:
            # self.canonize(site_inds[0], cur_orthog=site_inds[-1])  ## also sets bra from ket
            for i in range(left_site_pos + nsites - 1, left_site_pos, -1):
                # print('(site) update envs right', i, 'orthog at', site_inds[0])
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_right(i, canonize=False)

        return self.ket, ovlp_projector

    # @profile
    def _site_time_evolution_tdvp2(self, dt, left_site_pos,
                                   compress_direction=CompressDirection.RIGHT,) \
            -> tuple[Optional[qtn.MatrixProductState], qtn.Tensor]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        """
        nsites = 2

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]
        site0 = site_inds[0] if compress_direction is CompressDirection.RIGHT else site_inds[1]
        site1 = site_inds[1] if compress_direction is CompressDirection.RIGHT else site_inds[0]
        pos0 = left_site_pos if compress_direction is CompressDirection.RIGHT else left_site_pos + 1
        pos1 = left_site_pos + 1 if compress_direction is CompressDirection.RIGHT else left_site_pos

        if compress_direction is CompressDirection.RIGHT:
            at_end = pos1 == self.L - 1
        else:
            at_end = pos1 == 0

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                            inplace=True, te_order=self.te_order,
                            compress_direction=compress_direction,
                            compress_level=1, compress_opts_dict=self.compress_config)
        ## ket canonicalized to site1

        # self.set_bra_from_ket(sites=list(range(left_site_pos, left_site_pos + nsites)))
        self.set_bra_from_ket(sites=[pos0, pos1])
        if compress_direction is CompressDirection.RIGHT:
            self._update_envs_left(pos0, canonize=False)
        else:
            self._update_envs_right(pos0, canonize=False)

        # print('at end?', at_end, pos0, pos1)
        if not at_end:
            ### backpropagate site1 in time
            A_effs = self._get_A_effs(pos1, 1)

            dist_submpx = qtn.TensorNetwork([self.ket[site1]], virtual=True)
            submpx_tags = [self.ket.site_tag_id.format(site1)]
            bonds_i = dist_submpx.outer_inds()
            bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

            site_time_evolution(dist_submpx, -dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                                inplace=True, te_order=self.te_order,
                                compress_direction=compress_direction,
                                compress_level=1, compress_opts_dict=self.compress_config)

            self.set_bra_from_ket(sites=[pos1])
        else:
            self.set_bra_from_ket(sites=[pos1])

        # print('check orthog? should be', site1)
        # print(helper.check_orthog(self.ket))
        # print(helper.check_orthog(self.bra))

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            self._update_envs_left(pos1, canonize=False)
            # # self.canonize(site_inds[-1], cur_orthog=site_inds[0])     ## also sets bra from ket
            # for i in range(left_site_pos, left_site_pos + nsites - 1):
            #     self._update_envs_left(i, canonize=False)
        else:
            self._update_envs_right(pos1, canonize=False)
            # # self.canonize(site_inds[0], cur_orthog=site_inds[-1])  ## also sets bra from ket
            # for i in range(left_site_pos + nsites - 1, left_site_pos, -1):
            #     # print('(site) update envs right', i, 'orthog at', site_inds[0])
            #     # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
            #     self._update_envs_right(i, canonize=False)

        return self.ket, None


    # # @profile
    # def take_time_step(self, dt, grid=None, do_adapt=False, **kwargs):
    #     """ A: self.operators:  dictates time evolution
    #         x: self.ket (updated in place)
    #     """
    #     L = self.ket.L
    #     max_bond = self.max_bond
    #
    #     self.canonize(0)   ## does not scale canonical tensors
    #
    #     ## ovlp <Ax|b>, <Ax|x> init envs
    #     self._build_all_envs_right(1, canonize=False)
    #
    #     ## old to new basis projector
    #     proj = self.ket_env0_L
    #
    #     if proj is None:      # no env (end of chain without ancilla)
    #         proj_and_site = self.ket[0].copy()
    #     else:
    #         proj_and_site = qtn.tensor_contract(proj, self.ket[0])
    #         print('ovlp proj + site', proj_and_site)
    #
    #     ## right sweep
    #     direction = CompressDirection.RIGHT
    #     nsites = 0
    #     site_ind = 0
    #     cur_orthog = 0
    #     while site_ind < L:
    #
    #         if site_ind == L - 1 and nsites == 2:
    #             ## previously updated L-2, L-1
    #             print('continue', site_ind)
    #             site_ind += 1
    #             continue
    #
    #         ## num sites for update
    #         if site_ind == L - 1:
    #             adapt = False
    #         elif max_bond is None:
    #             adapt = do_adapt
    #         else:
    #             max_bond_ = min(max_bond, 2**(cur_orthog+1), 2**(L-cur_orthog-1))
    #             # print('max bond', max_bond_, cur_orthog)
    #             # print('bond size', self.ket.bond_size(cur_orthog, cur_orthog + 1))
    #             adapt = do_adapt # and self.ket.bond_size(cur_orthog, cur_orthog + 1) < max_bond_
    #             # print('do_adapt', adapt)
    #
    #         # if not adapt:  # one site update of site_ind
    #         #     if nsites == 2:  ## previously updated this site
    #         #         nsites = 1
    #         #         site_ind += 1
    #         #         continue
    #
    #         nsites = 2 # if adapt else 1
    #
    #         ## initial canonicalization
    #         if site_ind == 0:
    #             self.canonize(nsites - 1, cur_orthog=0)
    #
    #         ## 2 site propagation
    #         # print('FORWARD prop sites (+1)', list(range(site_ind,site_ind+nsites)) )
    #         # print('next site ind', site_ind)
    #         new_ket, proj = self._site_time_evolution_dmrg(dt / 2, site_ind, proj, nsites=nsites,
    #                                                        compress_direction=direction)
    #             ## updates environment within function to site_ind + nsites - 1
    #             ## proj now includes site_ind
    #
    #         cur_orthog = site_ind + nsites - 1
    #         site_ind = site_ind + nsites - 1 if nsites > 1 else site_ind + nsites
    #
    #         # print('forward prop (+1) check orthog', cur_orthog)
    #         # helper.check_orthog(self.ket)
    #         # helper.check_orthog(self.bra)
    #         # if grid is not None:
    #         #     plt.figure()
    #         #     gtn = grid.make_gridTN(data=self.ket)
    #         #     plt.imshow(np.abs(gtn.get_data())**2)
    #         #     plt.colorbar()
    #         #     plt.show()
    #
    #     #######################
    #     print('backward sweep')
    #
    #     ## old to new basis projector
    #     proj = self.ket_env0_R
    #     # if proj is None:      # no env (end of chain without ancilla)
    #     #     proj_and_site = self.ket[self.L-1].copy()
    #     # else:
    #     #     proj_and_site = qtn.tensor_contract(proj, self.ket[self.L-1])
    #     #     print('ovlp proj + site', proj_and_site)
    #
    #     ### right to left sweep
    #     nsites = 0
    #     direction = CompressDirection.LEFT
    #     site_ind = L - 1
    #     cur_orthog = L - 1
    #     while site_ind >= 0:
    #
    #         if site_ind == 0 and nsites == 2:
    #             ## previously updated 0, 1
    #             print('continue', site_ind)
    #             site_ind -= 1
    #             continue
    #
    #         if site_ind == 0:
    #             adapt = False
    #         elif max_bond is None:
    #             adapt = do_adapt
    #         else:
    #             # max_bond_ = min(max_bond, 2 ** site_ind, 2 ** (L - site_ind))
    #             # print('max bond', max_bond_, site_ind)
    #             # print('bond size', site_ind, site_ind-1, self.ket.bond_size(site_ind, site_ind - 1))
    #             # adapt = do_adapt and self.ket.bond_size(site_ind, site_ind - 1) < max_bond_
    #
    #             max_bond_ = min(max_bond, 2 ** cur_orthog, 2 ** (L - cur_orthog))
    #             # print('max bond', max_bond_, cur_orthog)
    #             # print('bond size', cur_orthog, cur_orthog - 1, self.ket.bond_size(cur_orthog, cur_orthog - 1))
    #             adapt = do_adapt # and self.ket.bond_size(cur_orthog, cur_orthog - 1) < max_bond_
    #             # print('do_adapt', adapt)
    #
    #         # if not adapt:  # one site update of site_ind
    #         #     if nsites == 2:  ## previously updated this site
    #         #         nsites = 1
    #         #         site_ind -= 1
    #         #         continue
    #
    #         nsites = 2  # if adapt else 1
    #
    #         ## initial canonicalization
    #         if site_ind == L - 1:
    #             self.canonize(L - 1 - (nsites - 1), cur_orthog=L-1)
    #
    #         cur_orthog = site_ind - nsites + 1    ## get left site ind
    #         new_ket, proj = self._site_time_evolution_dmrg(dt / 2, cur_orthog, proj, nsites=nsites,
    #                                                                 compress_direction=direction)
    #         site_ind = site_ind - nsites + 1 if nsites > 1 else site_ind - nsites
    #         # print('next site ind', site_ind)
    #
    #         # print('forward prop (-1) check orthog', cur_orthog)
    #         # helper.check_orthog(self.ket)
    #         # helper.check_orthog(self.bra)
    #         # if grid is not None:
    #         #     plt.figure()
    #         #     gtn = grid.make_gridTN(data=self.ket)
    #         #     plt.imshow(np.abs(gtn.get_data()) ** 2)
    #         #     plt.colorbar()
    #         #     plt.show()
    #
    #     print('done sweep')
    #     return self.ket


    def take_time_step_l2r(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        max_bond = self.max_bond
        # do_adapt = False
        print('l2r tdmrg', dt)

        if canonize:
            self.canonize(0)   ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_right(1, canonize=False)

        proj = self.ket_env0_L
        # print('proj', proj)
        # if proj is None:  # no env (end of chain without ancilla)
        #     proj_and_site = self.ket[0].copy()
        # else:
        #     proj_and_site = qtn.tensor_contract(proj, self.ket[0])
        #     print('ovlp proj + site', proj_and_site)

        ## right sweep
        direction = CompressDirection.RIGHT
        nsites = 0
        site_ind = 0
        cur_orthog = 0
        while site_ind < L:

            if site_ind == L - 1 and nsites == 2:
                ## previously updated L-2, L-1
                print('continue', site_ind)
                site_ind += 1
                continue

            ## num sites for update
            if site_ind == L - 1:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                max_bond_ = min(max_bond, 2 ** (cur_orthog + 1), 2 ** (L - cur_orthog - 1))
                # print('max bond', max_bond_, cur_orthog)
                # print('bond size', self.ket.bond_size(cur_orthog, cur_orthog + 1))
                adapt = do_adapt  # and self.ket.bond_size(cur_orthog, cur_orthog + 1) < max_bond_
                # print('do_adapt', adapt)

            # if not adapt:  # one site update of site_ind
            #     if nsites == 2:  ## previously updated this site
            #         nsites = 1
            #         site_ind += 1
            #         continue

            nsites = 2  # if adapt else 1

            ## initial canonicalization
            if site_ind == 0:
                self.canonize(nsites - 1, cur_orthog=0)

            ## 2 site propagation
            # print('FORWARD prop sites (+1)', list(range(site_ind,site_ind+nsites)) )
            # print('next site ind', site_ind)
            new_ket, proj = self._site_time_evolution_dmrg(dt, site_ind, proj, nsites=nsites,
                                                           compress_direction=direction)
            ## updates environment within function to site_ind + nsites - 1
            ## proj now includes site_ind

            cur_orthog = site_ind + nsites - 1
            site_ind = site_ind + nsites - 1 if nsites > 1 else site_ind + nsites

            # print('forward prop (+1) check orthog', cur_orthog)
            # helper.check_orthog(self.ket)
            # helper.check_orthog(self.bra)
            # if grid is not None:
            #     plt.figure()
            #     gtn = grid.make_gridTN(data=self.ket)
            #     plt.imshow(np.abs(gtn.get_data())**2)
            #     plt.colorbar()
            #     plt.show()

        self.ovlp_proj = proj
        print('done tdmrg left to right sweep')
        return self.ket


    def take_time_step_r2l(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        max_bond = self.max_bond
        # do_adapt = False
        print('r2l tdmrg', dt)

        if canonize:
            self.canonize(L-1)  ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_left(1, canonize=False)

        proj = self.ket_env0_R
        # if proj is None:      # no env (end of chain without ancilla)
        #     proj_and_site = self.ket[self.L-1].copy()
        # else:
        #     proj_and_site = qtn.tensor_contract(proj, self.ket[self.L-1])
        #     print('ovlp proj + site', proj_and_site)

        ### right to left sweep
        nsites = 0
        direction = CompressDirection.LEFT
        site_ind = L - 1
        cur_orthog = L - 1
        while site_ind >= 0:

            if site_ind == 0 and nsites == 2:
                ## previously updated 0, 1
                print('continue', site_ind)
                site_ind -= 1
                continue

            if site_ind == 0:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                # max_bond_ = min(max_bond, 2 ** site_ind, 2 ** (L - site_ind))
                # print('max bond', max_bond_, site_ind)
                # print('bond size', site_ind, site_ind-1, self.ket.bond_size(site_ind, site_ind - 1))
                # adapt = do_adapt and self.ket.bond_size(site_ind, site_ind - 1) < max_bond_

                max_bond_ = min(max_bond, 2 ** cur_orthog, 2 ** (L - cur_orthog))
                # print('max bond', max_bond_, cur_orthog)
                # print('bond size', cur_orthog, cur_orthog - 1, self.ket.bond_size(cur_orthog, cur_orthog - 1))
                adapt = do_adapt  # and self.ket.bond_size(cur_orthog, cur_orthog - 1) < max_bond_
                # print('do_adapt', adapt)

            # if not adapt:  # one site update of site_ind
            #     if nsites == 2:  ## previously updated this site
            #         nsites = 1
            #         site_ind -= 1
            #         continue

            nsites = 2  # if adapt else 1

            ## initial canonicalization
            if site_ind == L - 1:
                self.canonize(L - 1 - (nsites - 1), cur_orthog=L - 1)

            cur_orthog = site_ind - nsites + 1  ## get left site ind
            new_ket, proj = self._site_time_evolution_dmrg(dt, cur_orthog, proj, nsites=nsites,
                                                           compress_direction=direction)
            site_ind = site_ind - nsites + 1 if nsites > 1 else site_ind - nsites
            # print('next site ind', site_ind)

            # print('forward prop (-1) check orthog', cur_orthog)
            # helper.check_orthog(self.ket)
            # helper.check_orthog(self.bra)
            # if grid is not None:
            #     plt.figure()
            #     gtn = grid.make_gridTN(data=self.ket)
            #     plt.imshow(np.abs(gtn.get_data()) ** 2)
            #     plt.colorbar()
            #     plt.show()

        self.ovlp_proj = proj
        print('done tdmrg right to left sweep')
        return self.ket
