from helper_tdvp import _sum_eff_TNs
from setup.configs import *
import scipy.linalg
import helper_quimb as helper
from helper_dmrg import *
from helper_tdvp import *
import helper_TE
from setup.quimb_TN1D import MatrixProductStateUSVT as MPS_USVT
import gridTN as GTN


""" TDVP but with corrections to equation of motion
    corrections use (canonical) basis from previous iteration; vary weights along bond
    follow Einkemmer's method of modifying equation of motion
"""

class TDVPSolver_v2(TDVPSolver):

    # def __init__(self, trial_state: 'qtn.MatrixProductState',
    #              operators: Optional[Sequence['MPO_type']] = None,
    #              targets: Optional[Sequence['MPS_type']] = None,
    #              mps_inds: Sequence[int] = None,
    #              norm_env0_Ls: Sequence['qtn.Tensor'] = None,
    #              norm_env0_Rs: Sequence['qtn.Tensor'] = None,
    #              ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
    #              ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
    #              te_order=0, compress_config: CompressionConfiguration=None,
    #              backprop_edge=False):
    #
    #     super().__init__(trial_state, targets=targets, operators=operators,
    #                      mps_inds=mps_inds, norm_env0_Ls=norm_env0_Ls,
    #                      norm_env0_Rs=norm_env0_Rs, ovlp_env0_Ls=ovlp_env0_Ls,
    #                      ovlp_env0_Rs=ovlp_env0_Rs, te_order=te_order, compress_config=compress_config,
    #                      backprop_edge=backprop_edge)


    def _site_time_evolution_v1(self, dt, left_site_pos, nsites=1,
                                compress_direction=CompressDirection.RIGHT,
                                ) -> Optional[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        with original trotterized back prop
        """
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)

        ## A_eff * site projector (the back TE for repeated bond/site)
        if compress_direction == CompressDirection.RIGHT:
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

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

        ## back prop
        if not at_end:
            # print('project self.ket', ind1)
            right_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
            bra_left_inds = [ket_to_bra_inds[k_ind] for k_ind in left_inds]

            Q, R = qtn.tensor_split(self.ket[ind1], left_inds=left_inds, method='qr', absorb='right')
            # print('Q', Q.inds)
            # print('self.ket[ind1]', self.ket[ind1].inds)
            # Q.transpose_like(self.ket[ind1], inplace=True)
            # Q.modify(data = self.ket[ind1].data)
            # Q = Q.reindex(ket_to_bra_inds)
            QT = Q.conj()
            QT.reindex(ket_to_bra_inds, inplace=True)
            bond_m = next(iter(Q.bonds(R)))
            QT.reindex({bond_m: bond_m + '_'}, inplace=True)
            # Q.reindex({ind_l: ind_l + 'o' for ind_l in bra_left_inds}, inplace=True)
            # Q.add_tag('Q')

            ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
            A_effs = self._get_A_effs(left_site_pos, nsites)

            A_effs_bp = []
            for A_eff in A_effs:
                A_eff_bp = qtn.TensorNetwork([A_eff, QT, Q], virtual=False)
                A_eff_bp.multiply(-1, inplace=True)
                A_effs_bp += [A_eff_bp]

            bonds_i = R.inds
            bonds_o = [(ket_to_bra_inds[ind] if ind != bond_m else bond_m + '_') for ind in bonds_i]

            site_time_evolution(qtn.TensorNetwork([R], virtual=True),
                                dt, A_effs_bp, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                                inplace=True, te_order=self.te_order,
                                compress_direction=compress_direction,
                                compress_level=1, compress_opts_dict=self.compress_config)

            new_QR = qtn.tensor_contract(Q, R)
            new_QR.transpose_like(self.ket[ind1], inplace=True)
            self.ket[ind1].modify(data=new_QR.data)

            ## i think this is already done in canonize
            self.set_bra_from_ket(sites=list(range(left_site_pos, left_site_pos + nsites)))

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            # print('site inds', site_inds[-1], at_end, self.L)
            if at_end:
                # print('canonize', site_inds[0], site_inds[-1])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)     ## also sets bra from ket
            for i in range(ind1, ind2):
                print('(site) update envs left', i, 'orthog at', site_inds[-1])
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_left(i, canonize=False)
        else:
            if at_end:
                # print('canonize', site_inds[-1], site_inds[0])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket
            # for i in range(left_site_pos + nsites, left_site_pos, -1):
            for i in range(ind1, ind2, -1):
                print('(site) update envs right', i, 'orthog at', site_inds[0] - 1)
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_right(i, canonize=False)

        return self.ket


    def _site_time_evolution(self, dt, left_site_pos, nsites=1,
                                compress_direction=CompressDirection.RIGHT,
                                ) -> Optional[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        with original trotterized back prop
        """
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)

        ## A_eff * site projector (the back TE for repeated bond/site)
        if compress_direction == CompressDirection.RIGHT:
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

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

        ## back prop
        if not at_end:
            # print('project self.ket', ind1)
            right_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
            bra_left_inds = [ket_to_bra_inds[k_ind] for k_ind in left_inds]

            Q, R = qtn.tensor_split(self.ket[ind1], left_inds=left_inds, method='qr', absorb='right')
            QTi = Q.conj()
            QTi.reindex({ind_l: ind_l + 'i' for ind_l in left_inds}, inplace=True)
            QTi.add_tag('QTi')
            proj_i = qtn.TensorNetwork([Q, QTi])
            # print('proj_i', proj_i)

            Q = Q.reindex(ket_to_bra_inds)
            QT = Q.conj()
            Q.reindex({ind_l: ind_l + 'o' for ind_l in bra_left_inds}, inplace=True)
            Q.add_tag('Qo')
            proj_o = qtn.TensorNetwork([Q, QT])
            # print('proj_o', proj_o)


            ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
            A_effs = self._get_A_effs(left_site_pos, nsites)

            A_effs_bp = []
            for A_eff in A_effs:
                # A_eff_bp = qtn.TensorNetwork([A_eff, proj_i, proj_o], virtual=False)
                A_eff_bp = qtn.TensorNetwork([A_eff, proj_o], virtual=False)
                # print('A eff bp', A_eff_bp)
                A_eff_bp.multiply(-1, inplace=True)
                A_eff_bp.mangle_inner_()
                Q_tens = A_eff_bp.select_tensors('Qo')[0]
                Q_tens.reindex({ind_l + 'o': ind_l for ind_l in bra_left_inds}, inplace=True)
                # Q_tens = A_eff_bp.select_tensors('QTi')[0]
                # Q_tens.reindex({ind_l + 'i': ind_l for ind_l in left_inds}, inplace=True)
                # print('A eff bp', A_eff_bp)
                # print('A eff bp', A_eff_bp.outer_inds(), A_eff.outer_inds())
                A_effs_bp += [A_eff_bp]

            A_effs = A_effs_bp

            site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                                inplace=True, te_order=self.te_order,
                                compress_direction=compress_direction,
                                compress_level=1, compress_opts_dict=self.compress_config)

            ## i think this is already done in canonize
            self.set_bra_from_ket(sites=list(range(left_site_pos, left_site_pos + nsites)))

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            # print('site inds', site_inds[-1], at_end, self.L)
            if at_end:
                # print('canonize', site_inds[0], site_inds[-1])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)     ## also sets bra from ket
            for i in range(ind1, ind2):
                print('(site) update envs left', i, 'orthog at', site_inds[-1])
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_left(i, canonize=False)
        else:
            if at_end:
                # print('canonize', site_inds[-1], site_inds[0])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket
            # for i in range(left_site_pos + nsites, left_site_pos, -1):
            for i in range(ind1, ind2, -1):
                print('(site) update envs right', i, 'orthog at', site_inds[0] - 1)
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_right(i, canonize=False)

        return self.ket

    def _site_time_evolution_v3(self, dt, left_site_pos, nsites=1,
                                compress_direction=CompressDirection.RIGHT,
                                ) -> Optional[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        not trottered TE
        """
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)
        # print('A eff', left_site_pos, nsites)
        # print('A eff', A_effs[-1].outer_inds())

        ## A_eff * site projector (the back TE for repeated bond/site)
        if compress_direction == CompressDirection.RIGHT:
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

        if True:  # not at_end:
            # print('project self.ket', ind1)
            right_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
            bra_left_inds = [ket_to_bra_inds[k_ind] for k_ind in left_inds]

            Q, R = qtn.tensor_split(self.ket[ind1], left_inds=left_inds, method='qr', absorb='right')

            # Q = Q.reindex(ket_to_bra_inds)
            # QT = Q.conj()
            # Q.reindex({ind_l: ind_l + 'o' for ind_l in bra_left_inds}, inplace=True)
            # Q.add_tag('Q')

            QTi = Q.conj()
            QTi.reindex({ind_l: ind_l + 'i' for ind_l in left_inds}, inplace=True)
            QTi.add_tag('QTi')
            proj_i = qtn.TensorNetwork([Q, QTi])
            # print('proj_i', proj_i)

            Q = Q.reindex(ket_to_bra_inds)
            QT = Q.conj()
            Q.reindex({ind_l: ind_l + 'o' for ind_l in bra_left_inds}, inplace=True)
            Q.add_tag('Qo')
            proj_o = qtn.TensorNetwork([Q, QT])
            # print('proj_o', proj_o)

            A_effs_bp = []
            for A_eff in A_effs:
                # A_eff_bp = qtn.TensorNetwork([A_eff, Q, QT], virtual=False)
                A_eff_bp = qtn.TensorNetwork([A_eff, proj_i, proj_o], virtual=False)
                # print('A eff bp', A_eff_bp)
                A_eff_bp.multiply(-1, inplace=True)
                A_eff_bp.mangle_inner_()
                # Q_tens = A_eff_bp.select_tensors('Q')[0]
                # Q_tens.reindex({ind_l + 'o': ind_l for ind_l in bra_left_inds}, inplace=True)
                Q_tens = A_eff_bp.select_tensors('Qo')[0]
                Q_tens.reindex({ind_l + 'o': ind_l for ind_l in bra_left_inds}, inplace=True)
                Q_tens = A_eff_bp.select_tensors('QTi')[0]
                Q_tens.reindex({ind_l + 'i': ind_l for ind_l in left_inds}, inplace=True)
                # print('A eff bp', A_eff_bp)
                # print('A eff bp', A_eff_bp.outer_inds(), A_eff.outer_inds())
                A_effs_bp += [A_eff_bp]

            A_effs = A_effs + A_effs_bp

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
            # print('site inds', site_inds[-1], at_end, self.L)
            if at_end:
                # print('canonize', site_inds[0], site_inds[-1])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                # print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)     ## also sets bra from ket
            for i in range(ind1, ind2):
                # print('(site) update envs left', i, 'orthog at', site_inds[-1])
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_left(i, canonize=False)
        else:
            if at_end:
                # print('canonize', site_inds[-1], site_inds[0])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                # print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket
            # for i in range(left_site_pos + nsites, left_site_pos, -1):
            for i in range(ind1, ind2, -1):
                # print('(site) update envs right', i, 'orthog at', site_inds[0] - 1)
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_right(i, canonize=False)

        return self.ket


    def _site_time_evolution_v4(self, dt, left_site_pos, nsites=1,
                                compress_direction=CompressDirection.RIGHT,
                                ) -> Optional[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        trotter TE with |new><old| H |old><new|
        """
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)

        ## A_eff * site projector (the back TE for repeated bond/site)
        if compress_direction == CompressDirection.RIGHT:
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]
        # print('ket to bra inds', ket_to_bra_inds)
        # print('bonds o', bonds_o)
        # print('bonds i', bonds_i)
        # deriv_total = sum_Aeffs()
        # deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bonds_o + list(bonds_i))
        # print('deriv total', deriv_total)

        right_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
        Q_old, R_old = qtn.tensor_split(self.ket[ind1], left_inds=left_inds, method='qr', absorb='right')
        bond_m = next(iter(Q_old.bonds(R_old)))

        site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                            inplace=True, te_order=self.te_order,
                            compress_direction=compress_direction,
                            compress_level=1, compress_opts_dict=self.compress_config)

        ## i think this is already done in canonize
        self.set_bra_from_ket(sites=list(range(left_site_pos, left_site_pos + nsites)))

        ## back prop
        if not at_end:
            # print('project self.ket', ind1)
            right_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
            bra_left_inds = [ket_to_bra_inds[k_ind] for k_ind in left_inds]

            Q, R = qtn.tensor_split(self.ket[ind1], left_inds=left_inds, method='qr', absorb='right',
                                    bond_ind=bond_m)
            # QTi = Q.conj()
            QTi = Q_old.conj()
            QTi.reindex({ind_l: ind_l + 'i' for ind_l in left_inds}, inplace=True)
            QTi.add_tag('QTi')
            proj_i = qtn.TensorNetwork([Q_old, QTi])
            # print('proj_i', proj_i)

            QT_old = Q_old.conj().reindex(ket_to_bra_inds)
            # Q = Q.reindex(ket_to_bra_inds)
            Q = Q_old.reindex(ket_to_bra_inds)
            Q.reindex({ind_l: ind_l + 'o' for ind_l in bra_left_inds}, inplace=True)
            Q.add_tag('Qo')
            proj_o = qtn.TensorNetwork([Q, QT_old])
            # print('proj_o', proj_o)

            ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
            A_effs = self._get_A_effs(left_site_pos, nsites)

            A_effs_bp = []
            for A_eff in A_effs:
                # A_eff_bp = qtn.TensorNetwork([A_eff, Q, QT], virtual=False)
                A_eff_bp = qtn.TensorNetwork([A_eff, proj_i, proj_o], virtual=False)
                # print('A eff bp', A_eff_bp)
                A_eff_bp.multiply(-1, inplace=True)
                A_eff_bp.mangle_inner_()
                Q_tens = A_eff_bp.select_tensors('Qo')[0]
                Q_tens.reindex({ind_l + 'o': ind_l for ind_l in bra_left_inds}, inplace=True)
                Q_tens = A_eff_bp.select_tensors('QTi')[0]
                Q_tens.reindex({ind_l + 'i': ind_l for ind_l in left_inds}, inplace=True)
                # print('A eff bp', A_eff_bp)
                # print('A eff bp', A_eff_bp.outer_inds(), A_eff.outer_inds())
                A_effs_bp += [A_eff_bp]

            A_effs = A_effs_bp

            site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                                inplace=True, te_order=self.te_order,
                                compress_direction=compress_direction,
                                compress_level=1, compress_opts_dict=self.compress_config)

            ## i think this is already done in canonize
            self.set_bra_from_ket(sites=list(range(left_site_pos, left_site_pos + nsites)))

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            if at_end:
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket
            for i in range(ind1, ind2):
                self._update_envs_left(i, canonize=False)
        else:
            if at_end:
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket
            for i in range(ind1, ind2, -1):
                self._update_envs_right(i, canonize=False)

        return self.ket



    def take_time_step_l2r(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, verbose=False,
                           **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        max_bond = self.max_bond
        do_adapt = False
        print('l2r tdvp V2', dt, 'do adapt', do_adapt)
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
                if verbose:  print('continue', site_ind)
                site_ind += 1
                continue

            ## num sites for update
            if site_ind == L - 1:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                # max_bond_ = max_bond  # min(max_bond, 2**(cur_orthog+1), 2**(L-cur_orthog-1))
                max_bond_ = min(max_bond, 2**(site_ind+1), 2**(L-site_ind-1))
                # print('mod max bond:', 'site', site_ind, 'cur', cur_orthog, L, max_bond_)
                # print('max bond', max_bond_, cur_orthog)
                # print('bond size', self.ket.bond_size(cur_orthog, cur_orthog + 1))
                # adapt = do_adapt and self.ket.bond_size(cur_orthog, cur_orthog + 1) < max_bond_
                adapt = do_adapt and self.ket.bond_size(site_ind, site_ind + 1) < max_bond_
                # print('bond size', self.ket.bond_size(site_ind, site_ind + 1))
                # print('do_adapt', adapt)

            # if not adapt:  # one site update of site_ind
            #     if nsites == 2:  ## previously updated this site
            #         nsites = 1
            #         site_ind += 1
            #         continue

            ## backward propagation (skipped for first iteration)
            # print('self.ket norm', helper.norm(self.ket))
            # if 0 < site_ind < L:
            #     if nsites == 1:
            #         # bond_ind = site_ind - 1
            #         if verbose:  print('BACK prop (LR) bond ind', cur_orthog, direction)
            #         # print('check orthog', helper.check_orthog(self.ket))
            #         self._bond_time_evolution(-dt, cur_orthog, compress_direction=direction )
            #
            #         ## move canonicalization to next site (site_ind)
            #         self.canonize(site_ind, cur_orthog=cur_orthog)      ## updates bra internally
            #         # print('(bond) update left env', cur_orthog, 'canon at', site_ind)
            #         # print('check orthog', helper.check_orthog(self.ket))
            #         self._update_envs_left(cur_orthog, canonize=False)
            #
            #     else:
            #         if verbose:  print('BACK prop (LR) site ind', site_ind, direction)
            #         # print('check orthog', helper.check_orthog(self.ket))
            #         self._site_time_evolution(-dt, site_ind, nsites = 1, compress_direction=direction)


            nsites = 2 if adapt else 1

            ## forward propagation
            # print('before cur orthog', cur_orthog)
            # print('self.ket norm', helper.norm(self.ket))
            # print('check orthog', helper.check_orthog(self.ket))
            if verbose:  print('FORWARD prop sites (LR)', list(range(site_ind,site_ind+nsites)) )
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


    def take_time_step_r2l(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, verbose=False,
                           **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        do_adapt = False
        max_bond = self.max_bond
        print('r2l tdvp V2', dt, 'do adapt', do_adapt)
        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if canonize:
            self.canonize(L-1)  ## does not scale canonical tensors
            # print('ket canon')
            # helper.check_orthog(self.ket)
            # print('bra canon')
            # helper.check_orthog(self.bra)

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_left(1, canonize=False)

        ### right to left sweep
        nsites = 0
        direction = CompressDirection.LEFT
        site_ind = L - 1
        cur_orthog = L - 1
        while site_ind >= 0:

            if site_ind == 0 and nsites == 2:
                ## previously updated 0, 1
                if verbose:   print('continue', site_ind)
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

                # max_bond_ = max_bond  # min(max_bond, 2 ** cur_orthog, 2 ** (L - cur_orthog))
                max_bond_ = min(max_bond, 2 ** site_ind, 2 ** (L - site_ind))  # if site_ind != L-1 else 2
                # print('mod max bond: site', site_ind, 'cur', cur_orthog, L, max_bond_,)
                # print('max bond', max_bond_, cur_orthog)
                # print('bond size', cur_orthog, cur_orthog - 1, self.ket.bond_size(cur_orthog, cur_orthog - 1))
                # adapt = do_adapt and self.ket.bond_size(cur_orthog, cur_orthog - 1) < max_bond_
                adapt = do_adapt and self.ket.bond_size(site_ind, site_ind - 1) < max_bond_
                # print('bond size', self.ket.bond_size(site_ind, site_ind - 1))
                # print('do_adapt', adapt)

            # if not adapt:  # one site update of site_ind
            #     if nsites == 2:  ## previously updated this site
            #         nsites = 1
            #         site_ind -= 1
            #         continue

            # if 0 <= site_ind < L - 1:    ## skip initial back propagation
            #     if nsites == 1:
            #         if verbose:  print('BACK prop (RL) bond ind', cur_orthog, direction )
            #         # print('check orthog', helper.check_orthog(self.ket))
            #         self._bond_time_evolution(-dt, cur_orthog, compress_direction=direction)
            #
            #         ## move canonicalization to next site (site_ind)
            #         self.canonize(site_ind, cur_orthog=cur_orthog)  ## updates bra internally
            #         # print('(bond) update right env', cur_orthog, 'canon at', site_ind)
            #         # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
            #         self._update_envs_right(cur_orthog, canonize=False)
            #         # print('new site ind', site_ind)
            #     else:
            #         if verbose:  print('BACK prop (RL) site ind', site_ind, direction)
            #         self._site_time_evolution(-dt, site_ind, nsites=1, compress_direction=direction)

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
            if verbose:  print('FORWARD prop sites (RL)', list(range(cur_orthog,cur_orthog+nsites)) )
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


class TDDMRGSolver_v2(TDMRGSolver):

    ### interesting; this does |k><k| = |k1><k1| + |k2><k2| + ...
    ### as opposed to |k> = |k1> + |k2> + ...

    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 operators: Optional[Sequence['MPO_type']] = None,
                 targets: Optional[Sequence['MPS_type']] = None,
                 mps_inds: Sequence[int] = None,
                 norm_env0_Ls: Sequence['qtn.Tensor'] = None,  # for A envs
                 norm_env0_Rs: Sequence['qtn.Tensor'] = None,  # for A envs
                 ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 ket_env0_L: Optional['qtn.Tensor'] = None,
                 ket_env0_R: Optional['qtn.Tensor'] = None,
                 backprop_edge=False,
                 te_order=0, compress_config: CompressionConfiguration = None):

        # super().__init__(trial_state, targets=targets, operators=operators, mps_inds=mps_inds)
        super().__init__(trial_state, targets=targets, operators=operators, mps_inds=mps_inds,
                         norm_env0_Ls=norm_env0_Ls, norm_env0_Rs=norm_env0_Rs,
                         ovlp_env0_Ls=ovlp_env0_Ls, ovlp_env0_Rs=ovlp_env0_Rs,
                         te_order=te_order, compress_config=compress_config)

        self.ket_env0_L = ket_env0_L
        self.ket_env0_R = ket_env0_R
        self.ovlp_proj = None
        self.backprop_edge = backprop_edge  ## do back prop at end of chain if False

        self.orig_sys = self.copy()


    def get_rk4_eff(self, dt, left_site_pos, nsites=1, compress_direction=CompressDirection.RIGHT,):
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]


        def deriv_func(dist_submpx_, dt=1) -> 'qtn.Tensor':

            eff_Ax = []
            for deriv_tn in A_effs:
                eff_Ax += [qtn.TensorNetwork([deriv_tn, dist_submpx_])]
                # print('eff_ax', eff_Ax[-1].exponent, deriv_tn.exponent)
                # print('dist_submpx exponent', dist_submpx_.exponent)
                ## dist_submpx exponent should equal 0
            out = _sum_eff_TNs(eff_Ax, transpose_bonds=bonds_o)
            # out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
            out.reindex({bo: bi for bo, bi in zip(bonds_o, bonds_i)}, inplace=True)
            out.modify(apply=lambda x: x * dt)
            return out

        k1 = deriv_func(dist_submpx, dt=dt)

        T1 = qtn.tensor_contract(dist_submpx)
        T1.transpose_like(k1, inplace=True)
        s1 = k1.copy().modify(apply= lambda x: x + T1.data * 0.5)
        k2 = deriv_func(s1, dt=dt)      ## assumes constant A; should be A(t+dt/2)

        s2 = k2.copy().modify(apply= lambda x: x + T1.data * 0.5)
        k3 = deriv_func(s2, dt=dt)      ## assumes constant A; should be A(t+dt/2)

        s3 = k3.copy().modify(apply= lambda x: x + T1.data)
        k4 = deriv_func(s3, dt=dt)

        return k1, k2, k3, k4


    def _site_time_evolution(self, dt, left_site_pos, nsites=1,
                             compress_direction=CompressDirection.RIGHT,
                             ) -> Optional[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time via RK4 according to Fenguin + White
        """
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)

        ## A_eff * site projector (the back TE for repeated bond/site)
        if compress_direction == CompressDirection.RIGHT:
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        ### build rk4 vectors
        def deriv_func(dist_submpx_, dt=1) -> 'qtn.Tensor':

            eff_Ax = []
            for deriv_tn in A_effs:
                eff_Ax += [qtn.TensorNetwork([deriv_tn, dist_submpx_])]
                # print('eff_ax', eff_Ax[-1].exponent, deriv_tn.exponent)
                # print('dist_submpx exponent', dist_submpx_.exponent)
                ## dist_submpx exponent should equal 0
            out = _sum_eff_TNs(eff_Ax, transpose_bonds=bonds_o)
            # out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
            out.reindex({bo: bi for bo, bi in zip(bonds_o, bonds_i)}, inplace=True)
            out.modify(apply=lambda x: x * dt)
            return out

        T1 = qtn.tensor_contract(dist_submpx)

        k1 = deriv_func(dist_submpx, dt=dt)
        k1.transpose_like(T1, inplace=True)

        s1 = k1.copy().modify(apply= lambda x: x + T1.data * 0.5)
        k2 = deriv_func(s1, dt=dt)      ## assumes constant A; should be A(t+dt/2)
        k2.transpose_like(T1, inplace=True)

        s2 = k2.copy().modify(apply= lambda x: x + T1.data * 0.5)
        k3 = deriv_func(s2, dt=dt)      ## assumes constant A; should be A(t+dt/2)
        k3.transpose_like(T1, inplace=True)

        s3 = k3.copy().modify(apply= lambda x: x + T1.data)
        k4 = deriv_func(s3, dt=dt)
        k4.transpose_like(T1, inplace=True)

        ## targeted time steps
        psi13 = T1.copy().modify(data=lambda x: x + 1./162 * (31 * k1.data + 14 * k2.data + 14 * k3.data - 5 * k4.data))
        psi23 = T1.copy().modify(data=lambda x: x + 1./81  * (16 * k1.data + 20 * k2.data + 20 * k3.data - 2 * k4.data))
        psi33 = T1.copy().modify(data=lambda x: x + 1./6 * (k1.data + 2 * k2.data + 2 * k3.data + k4.data))

        ## bond to contract over
        if compress_direction == CompressDirection.RIGHT:
            x_bond = self.ket.bond(ind1 + nsites - 1, ind1 + nsites)
        else:
            x_bond = self.ket.bond(ind1 - nsites + 1, ind1 - nsites)

        def compute_denmat(ket: 'qtn.Tensor'):
            bra = ket.conj()
            bra.reindex({k: b for k,b in ket_to_bra_inds.items() if k != x_bond}, inplace=True)
            return qtn.tensor_contract(bra, ket)

        rho_bonds_i = [bi for bi in bonds_i if bonds_i != x_bond]
        rho_bonds_o = [ket_to_bra_inds[ind] for ind in rho_bonds_i]

        rho0 = compute_denmat(T1)
        rho0.transpose(*rho_bonds_o, *rho_bonds_i, inplace=True)
        rho1 = compute_denmat(psi13)
        rho1.transpose_like(rho0, inplace=True)
        rho2 = compute_denmat(psi23)
        rho2.transpose_like(rho0, inplace=True)
        rho3 = compute_denmat(psi33)
        rho3.transpose_like(rho0, inplace=True)
        tot_denmat = rho0.copy().modify(apply = lambda x: x * 1./3 + rho1.data * 1./6 +
                                                          rho2.data * 1./6 + rho3.data * 1./3)
        denmat_shape = tot_denmat.shape

        eigvec, eigval = np.linalg.eig(tot_denmat.reshape(np.prod(denmat_shape[:2]),-1))
        sort_inds = np.sort(eigval)[::-1]
        if self.max_bond is not None:
            eigvec = eigvec[:,sort_inds[:self.max_bond]]
        if not at_end:
            eigvec = eigvec.reshape(*denmat_shape[:2],-1)
            eigvec = qtn.Tensor(data=eigvec, inds=tuple(rho_bonds_i) + (x_bond,))

            # eigvec_T = eigvec.conj()
            # eigvec_T.reindex(ket_to_bra_inds, inplace=True)
            # next_site = qtn.tensor_contract( eigvec_T, self.ket[ind1], self.ket[ind2] )

        else:
            eigvec = eigvec.reshape(*denmat_shape[:2])

        if nsites > 1:
            raise NotImplementedError

        eigvec.transpose_like(self.ket[ind1], inplace=True)
        self.ket[ind1].modify(data=eigvec)

        # site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
        #                     inplace=True, te_order=self.te_order,
        #                     compress_direction=compress_direction,
        #                     compress_level=1, compress_opts_dict=self.compress_config)

        ## i think this is already done in canonize
        self.set_bra_from_ket(sites = list(range(left_site_pos,left_site_pos + nsites)))

        ## back prop
        if not at_end:
            # print('project self.ket', ind1)
            right_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
            bra_left_inds = [ket_to_bra_inds[k_ind] for k_ind in left_inds]

            Q, R = qtn.tensor_split(self.ket[ind1], left_inds=left_inds, method='qr', absorb='right')
            # print('Q', Q.inds)
            # print('self.ket[ind1]', self.ket[ind1].inds)
            # Q.transpose_like(self.ket[ind1], inplace=True)
            # Q.modify(data = self.ket[ind1].data)
            # Q = Q.reindex(ket_to_bra_inds)
            QT = Q.conj()
            QT.reindex(ket_to_bra_inds, inplace=True)
            bond_m = next(iter(Q.bonds(R)))
            QT.reindex({bond_m: bond_m + '_'}, inplace=True)
            # Q.reindex({ind_l: ind_l + 'o' for ind_l in bra_left_inds}, inplace=True)
            # Q.add_tag('Q')

            ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
            A_effs = self._get_A_effs(left_site_pos, nsites)

            A_effs_bp = []
            for A_eff in A_effs:
                A_eff_bp = qtn.TensorNetwork([A_eff, QT, Q], virtual=False)
                A_eff_bp.multiply(-1, inplace=True)
                A_effs_bp += [A_eff_bp]

            bonds_i = R.inds
            bonds_o = [(ket_to_bra_inds[ind] if ind != bond_m else bond_m + '_') for ind in bonds_i]

            site_time_evolution(qtn.TensorNetwork([R], virtual=True),
                                dt, A_effs_bp, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                                inplace=True, te_order=self.te_order,
                                compress_direction=compress_direction,
                                compress_level=1, compress_opts_dict=self.compress_config)

            new_QR = qtn.tensor_contract(Q, R)
            new_QR.transpose_like(self.ket[ind1], inplace=True)
            self.ket[ind1].modify(data=new_QR.data)

            ## i think this is already done in canonize
            self.set_bra_from_ket(sites=list(range(left_site_pos, left_site_pos + nsites)))

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            # print('site inds', site_inds[-1], at_end, self.L)
            if at_end:
                # print('canonize', site_inds[0], site_inds[-1])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)     ## also sets bra from ket
            for i in range(ind1, ind2):
                print('(site) update envs left', i, 'orthog at', site_inds[-1])
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_left(i, canonize=False)
        else:
            if at_end:
                # print('canonize', site_inds[-1], site_inds[0])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket
            # for i in range(left_site_pos + nsites, left_site_pos, -1):
            for i in range(ind1, ind2, -1):
                print('(site) update envs right', i, 'orthog at', site_inds[0] - 1)
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_right(i, canonize=False)

        return self.ket


