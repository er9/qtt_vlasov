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
    compress_opts = {} if compress_opts is None else compress_opts

    T1 = active_tens
    s_tags_list = submpx_tags if compress_direction > 0 else submpx_tags[::-1]
    for ix in range(nsites - 1):
        # print('s_ind', s_ind)
        dist_gtn_tens: 'qtn.Tensor' = next(iter(dist_submpx.select(s_tags_list[ix])))
        neighbor_tens: 'qtn.Tensor' = next(iter(dist_submpx.select(s_tags_list[ix + 1])))
        rix, lix = dist_gtn_tens.filter_bonds(neighbor_tens)
        # rix, lix = dist_submpx[s_tag].filter_bonds(dist_submpx[s_ind + 1 * compress_direction])
        T2, T1 = T1.split(lix, get='tensors', absorb='right', **compress_opts)

        T2.transpose_like(dist_gtn_tens, inplace=True)
        dist_gtn_tens.modify(data=T2.data)

    dist_gtn_tens = dist_submpx[s_tags_list[-1]]
    T1.transpose_like(dist_gtn_tens, inplace=True)
    dist_gtn_tens.modify(data=T1.data)

    return dist_submpx


def _exactTE_site(dist_submpx: 'qtn.TensorNetwork', dt, deriv_tens: 'qtn.Tensor',
                  submpx_tags: Sequence[str], bonds_i: Sequence['str'], bonds_o: Sequence['str'],
                  inplace=False, compress_direction=CompressDirection.RIGHT, compress_opts=None
                  ) -> Optional['qtn.TensorNetwork1D']:
    """ compute projection of MPS onto site(s) [specified by nsites] or bond (right of site_ind) [nsites=0]
    """
    dist_submpx = dist_submpx if inplace else dist_submpx.copy()

    deriv_tens.transpose(*bonds_o, *bonds_i, inplace=True)
    # print('exact deriv_tens shape', deriv_tens.shape, deriv_tens.norm())
    fuse_map = {'out': tuple(bonds_o), 'in': tuple(bonds_i)}
    proj_op = deriv_tens.fuse(fuse_map, inplace=False)
    proj_op.transpose('out','in', inplace=True)
    # plt.figure()
    # plt.imshow(proj_op.data)
    # plt.colorbar()
    # plt.show()
    # print('proj op', proj_total.shape)
    proj_op.modify(apply=lambda x: scipy.linalg.expm(x * -dt))
    shape_i = [deriv_tens.ind_size(bi) for bi in bonds_i]
    shape_o = [deriv_tens.ind_size(bo) for bo in bonds_o]
    exp_total = proj_op.unfuse(fuse_map, {'out': shape_o, 'in': shape_i})

    reindex_map = {bo: bi for bo, bi in zip(bonds_o, bonds_i)}
    active_tens: 'qtn.Tensor' = qtn.tensor_contract(*dist_submpx, exp_total)
    active_tens.reindex(reindex_map, inplace=True)

    _update_tn_with_tensor(dist_submpx, active_tens, submpx_tags,
                           compress_direction=compress_direction, compress_opts=compress_opts)
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

    eff_op.transpose(*bonds_i, *bonds_o, inplace=True)
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


def site_time_evolution(dist_submpx: 'qtn.TensorNetwork', dt, deriv_tens: 'qtn.Tensor',
                        submpx_tags: Sequence[str], bonds_i: Sequence[str], bonds_o: Sequence[str],
                        te_order = 4, inplace=False, compress_direction = CompressDirection.RIGHT,
                        compress_level = 1, compress_opts_dict = None) -> 'qtn.TensorNetwork1D':
    """ time integration of specific site, as dictated by te_order. limited to RK TE methods
    """
    dist_submpx = dist_submpx if inplace else dist_submpx.copy()

    if te_order == 0:
        return _exactTE_site(dist_submpx, dt, deriv_tens, submpx_tags, bonds_i, bonds_o,
                             inplace=True, compress_direction=compress_direction,
                             compress_opts=compress_opts_dict.get(compress_level, None))

    def euler_func(dist_submpx_, dt_, deriv0=None, compress_level=1, inplace=False, **kwargs):
        return _euler_site(dist_submpx_, dt_, deriv0, submpx_tags, inplace=inplace,
                           compress_direction=compress_direction,
                           compress_opts=compress_opts_dict.get(compress_level, None))

    def deriv_func(dist_submpx_, **kwargs) -> 'qtn.Tensor':
        out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
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
                          te_order = 4, inplace=False, compress_direction = CompressDirection.RIGHT,
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
                 te_order=0, compress_config: CompressionConfiguration=None):

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
                                # solve_type=kwargs.get('solve_tyep', self.solve_type)
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

            # A_eff.exponent = self.A_envsLs[i].exponent
            A_eff.exponent = A.exponent
            A_effs += [A_eff]

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

            b_eff.exponent = b.exponent
            b_effs += [b_eff]

        return b_effs


    def _sum_eff_TNs(self, eff_tns: Sequence['qtn.TensorNetwork'], transpose_bonds=None):

        transpose_bonds = eff_tns[0].outer_inds() if transpose_bonds is None else transpose_bonds

        A_eff_ = None
        for A_eff_tn in eff_tns:
            A_eff_tens = A_eff_tn.contract_tags(all)
            A_eff_tens.modify(apply=lambda data: data * 10 ** A_eff_tn.exponent)
            A_eff_tens.transpose(*transpose_bonds, inplace=True)

            if A_eff_ is None:
                A_eff_ = A_eff_tens
            else:
                A_eff_.modify(apply=lambda data: data + A_eff_tens.data)
        return A_eff_


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
        A_effs = self._get_A_effs(left_site_pos, nsites)

        ## change indices from bra (missing T*[i]) to ket
        # bra_to_ket_inds = {}
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))
            # bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))


        # def sum_Aeffs():
        #     A_eff_ = None
        #     for A_eff_tn in A_effs:
        #         A_eff_tens = A_eff_tn.contract_tags(all)
        #         A_eff_tens.modify(apply=lambda data: data * 10 ** A_eff_tn.exponent)
        #         A_eff_tens.transpose(*bonds_o, *bonds_i, inplace=True)
        #
        #         if A_eff_ is None:
        #             A_eff_ = A_eff_tens
        #         else:
        #             A_eff_.modify(apply=lambda data: data + A_eff_tens.data)
        #     return A_eff_

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]
        # deriv_total = sum_Aeffs()
        deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bonds_o + list(bonds_i))

        site_time_evolution(dist_submpx, dt, deriv_total, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
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

        ## change indices from bra (missing T*[i]) to ket
        shared, notshared = S_tens.filter_bonds(Q_tens)
        ket_inds = shared + notshared

        shared, notshared = S_conj.filter_bonds(Q_conj)
        bra_inds = shared + notshared

        # def sum_Aeffs():
        #     A_eff_ = None
        #     for A_eff_tn in A_effs:
        #         A_eff_tens = A_eff_tn.contract_tags(all)
        #         A_eff_tens.modify(apply=lambda data: data * 10 ** A_eff_tn.exponent)
        #         A_eff_tens.transpose(*bra_inds, *ket_inds, inplace=True)
        #
        #         if A_eff_ is None:
        #             A_eff_ = A_eff_tens
        #         else:
        #             A_eff_.modify(apply=lambda data: data + A_eff_tens.data)
        #     return A_eff_

        # deriv_total = sum_Aeffs()
        deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bra_inds + ket_inds)
        dist_submpx = qtn.TensorNetwork([S_tens], virtual=True)

        site_time_evolution(dist_submpx, dt, deriv_total, [next(iter(S_tens.tags))],
                            bonds_i = ket_inds, bonds_o = bra_inds, te_order=self.te_order, inplace=True,
                            compress_direction=compress_direction, compress_opts_dict=self.compress_config
                            )

        # print('cur orthog to MPS', cur_orthog)
        new_ket = ket_usvt.to_MPS()
        ket_tens = self.ket[cur_orthog]
        new_tens = new_ket[cur_orthog]
        new_tens.transpose_like(ket_tens, inplace=True)
        ket_tens.modify(data = new_tens.data)
        self.set_bra_from_ket(sites=[cur_orthog])
        return self.ket


    def take_time_step(self, dt, do_adapt=False, **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        max_bond = self.max_bond

        self.canonize(0)   ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
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
                max_bond_ = min(max_bond, 2**(cur_orthog+1), 2**(L-cur_orthog-1))
                print('max bond', max_bond_, cur_orthog)
                print('bond size', self.ket.bond_size(cur_orthog, cur_orthog + 1))
                adapt = do_adapt and self.ket.bond_size(cur_orthog, cur_orthog + 1) < max_bond_
                print('do_adapt', do_adapt)

            if not adapt:  # one site update of site_ind
                if nsites == 2:  ## previously updated this site
                    nsites = 1
                    site_ind += 1
                    continue

            ## backward propagation (skipped for first iteration)
            if 0 < site_ind < L:
                if nsites == 1:
                    # bond_ind = site_ind - 1
                    # print('back prop (+1) bond ind', cur_orthog)
                    self._bond_time_evolution(-dt / 2, cur_orthog, compress_direction=direction )

                    ## move canonicalization to next site (site_ind)
                    self.canonize(site_ind, cur_orthog=cur_orthog)      ## updates bra internally
                    self._update_envs_left(cur_orthog, canonize=False)

                else:
                    # print('back prop (+1) site ind', site_ind)
                    self._site_time_evolution(-dt / 2, site_ind, nsites = 1, compress_direction=direction)

            nsites = 2 if adapt else 1

            ## forward propagation
            # print('forward prop sites (+1)', site_ind, nsites)
            self._site_time_evolution(dt / 2, left_site_pos=site_ind, nsites=nsites, compress_direction=direction)
                ## updates environment within function to site_ind + nsites - 1
            cur_orthog = site_ind + nsites - 1
            site_ind = site_ind + nsites - 1 if nsites > 1 else site_ind + nsites



        ### right to left sweep
        print('backward sweep')
        nsites = 0
        direction = CompressDirection.LEFT
        site_ind = L - 1
        cur_orthog = L - 1
        while site_ind >= 0:

            if site_ind == 0 and nsites == 2:
                ## previously updated 0, 1
                site_ind -= 1
                continue

            if site_ind == 0:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                max_bond_ = min(max_bond, 2 ** cur_orthog, 2 ** (L - cur_orthog))
                print('max bond', max_bond_, cur_orthog)
                print('bond size', self.ket.bond_size(cur_orthog, cur_orthog - 1))
                adapt = do_adapt and self.ket.bond_size(cur_orthog, cur_orthog - 1) < max_bond_
                print('do_adapt', do_adapt)

            if not adapt:  # one site update of site_ind
                if nsites == 2:  ## previously updated this site
                    nsites = 1
                    site_ind -= 1
                    continue

            if 0 <= site_ind < L - 1:    ## skip initial back propagation
                if nsites == 1:
                    # print('back prop (-1) bond ind', cur_orthog )
                    self._bond_time_evolution(-dt / 2, cur_orthog, compress_direction=direction)

                    ## move canonicalization to next site (site_ind)
                    self.canonize(site_ind, cur_orthog=cur_orthog)  ## updates bra internally
                    self._update_envs_right(cur_orthog, canonize=False)
                    # print('new site ind', site_ind)
                else:
                    # print('back prop (-1) site ind', site_ind)
                    self._site_time_evolution(-dt / 2, site_ind, nsites=1, compress_direction=direction)

            nsites = 2 if adapt else 1
            cur_orthog = site_ind - nsites + 1    ## get left site ind
            # print('forward prop sites (-1)', cur_orthog, nsites)
            self._site_time_evolution(dt / 2, cur_orthog, nsites=nsites, compress_direction=direction)
            site_ind = site_ind - nsites + 1 if nsites > 1 else site_ind - nsites
            # print('next site ind', site_ind)
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
                 te_order=0, compress_config: CompressionConfiguration=None):

        # operators = [] if operators is None else list(operators)
        self.A_envsLs, self.A_envsRs = [], []
        self.b_envsLs, self.b_envsRs = [], []

        super().__init__(trial_state, targets=targets, operators=operators, mps_inds=mps_inds,
                         norm_env0_Ls=norm_env0_Ls, norm_env0_Rs=norm_env0_Rs,
                         ovlp_env0_Ls=ovlp_env0_Ls, ovlp_env0_Rs=ovlp_env0_Rs,
                         te_order=te_order, compress_config=compress_config)


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

        site_time_evolution_2(dist_submpx, dt, deriv_total, sources, [next(iter(S_tens.tags))],
                              bonds_i = ket_inds, bonds_o = bra_inds, te_order=self.te_order, inplace=True,
                              compress_direction=compress_direction, compress_opts_dict=self.compress_config
                              )

        # print('cur orthog to MPS', cur_orthog)
        new_ket = ket_usvt.to_MPS()
        ket_tens = self.ket[cur_orthog]
        new_tens = new_ket[cur_orthog]
        new_tens.transpose_like(ket_tens, inplace=True)
        ket_tens.modify(data = new_tens.data)
        self.set_bra_from_ket(sites=[cur_orthog])
        return self.ket
